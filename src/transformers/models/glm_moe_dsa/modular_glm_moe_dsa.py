# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...models.llama.modeling_llama import rotate_half
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import is_flash_attention_requested
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
)
from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from ..glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteDecoderLayer,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to a single tensor.

    This is the transformers equivalent of DeepSeek V3.2's `apply_rotary_emb(x, freqs_cis, interleaved)`.
    Instead of using complex-number `freqs_cis`, we use pre-split `(cos, sin)` tensors from RotaryEmbedding.

    Args:
        x (`torch.Tensor`): Input tensor of shape `[..., head_dim]`.
        cos (`torch.Tensor`): Cosine part from RotaryEmbedding, shape `[batch, seq_len, head_dim]`.
        sin (`torch.Tensor`): Sine part from RotaryEmbedding, shape `[batch, seq_len, head_dim]`.
        unsqueeze_dim (`int`): Dimension along which to unsqueeze cos/sin for broadcasting.
            Use `1` when x is `[B, H, S, D]` (BHSD) and `2` when x is `[B, S, H, D]` (BSHD).

    Returns:
        `torch.Tensor`: Tensor with rotary embeddings applied, same shape as input.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Split-half (NeoX/Llama style): (x[:d/2], x[d/2:])
    # This matches llama's apply_rotary_pos_emb logic.
    x_rotated = (x * cos) + (rotate_half(x) * sin)
    return x_rotated


@auto_docstring(checkpoint="zai-org/GLM-5")
@strict
class GlmMoeDsaConfig(Glm4MoeLiteConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int | None`, *optional*, defaults to 32):
        Number of heads for the indexer projections (DSA).
    indexer_types (`list[str]`, *optional*):
        Indexer mode for each layer (`"full"` or `"shared"`). Defaults to first layer full, then every `index_topk_freq`-th layer full, rest shared.

    ```python
    >>> from transformers import GlmMoeDsaConfig, GlmMoeDsaModel

    >>> # Initializing a GLM-MoE-DSA configuration
    >>> configuration = GlmMoeDsaConfig()

    >>> # Initializing a model from the configuration
    >>> model = GlmMoeDsaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    base_model_tp_plan = {
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "mla_kv_a_proj",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    q_lora_rank: int = 2048
    num_experts_per_tok: int = 8
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    pretraining_tp = AttributeError()
    rope_interleave = AttributeError()
    indexer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # MLP layer types: first 3 dense, rest sparse
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )

        # Indexer layer types
        if self.indexer_types is None:
            pattern = kwargs.pop("index_topk_pattern", None)
            freq = kwargs.pop("index_topk_freq", 1)
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
                )
            else:
                # First layer full, then every freq-th layer full, rest shared
                self.indexer_types = [
                    "full" if (max(i - 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]
        PreTrainedConfig.__post_init__(self, **kwargs)


class GlmMoeDsaRMSNorm(Glm4MoeRMSNorm):
    pass


class GlmMoeDsaIndexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) indexer for selecting top-k tokens.

    The Indexer has its own lightweight projections (wq_b, wk) separate from the
    main MLA attention. It uses non-interleaved (NeoX/Llama) RoPE, unlike the main attention
    which uses interleaved RoPE.

    **Cache strategy**: The Indexer manages its own key cache (`_cached_keys`) separately
    from the DynamicCache used by MLA attention, since DynamicCache is sized for exactly
    `num_hidden_layers` attention layers. Keys are concatenated along the sequence dimension
    during autoregressive decode.
    """

    def __init__(self, config: "GlmMoeDsaConfig", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size: int = config.hidden_size
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.qk_rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank

        # Named to match checkpoint: wq_b, wk, k_norm
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        # Named to match checkpoint: weights_proj
        # In the reference, this is fp32; the HF FP8 checkpoint stores a bf16 tensor.
        # Keeping it as a plain Linear prevents FP8 conversion (see `_keep_in_fp32_modules`).
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5

        # Indexer maintains its own key cache (not in DynamicCache, which is sized for attention layers only)
        self.register_buffer("_cached_keys", None, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden]
        q_resid: torch.Tensor,  # [B, S, q_lora_rank]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        use_cache: bool = False,
    ) -> torch.LongTensor:
        """
        Computes top-k token indices for sparse attention (DSA).

        This is the bf16 equivalent of the reference Indexer which uses `rotate_activation` (Hadamard transform)
        and `fp8_index` (FP8 quantized scoring kernel). Since the Hadamard transform is orthogonal (dot products
        are preserved: Hq·Hk = q·k), and FP8 quantization is a precision optimization, we skip both and compute
        scores directly in bf16/fp32.

        The scoring logic computes:
            index_score[b,s,t] = Σ_h (weight[b,s,h] · softmax_scale · q[b,s,h,:] · k[b,t,:])

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            use_cache: Whether to store/update the indexer's own key cache for autoregressive decode.

        Returns:
            `torch.LongTensor`: Top-k token indices of shape `[B, S, topk]`.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # === Queries ===
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)  # [B, S, H, rope_D]
        q = torch.cat([q_pe, q_nope], dim=-1)  # [B, S, H, D]

        # === Keys ===
        k = self.k_norm(self.wk(hidden_states))  # [B, S, D]
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)  # [B, S, rope_D]
        k = torch.cat([k_pe, k_nope], dim=-1)  # [B, S, D]

        # === Key cache (managed by the indexer, not DynamicCache) ===
        # Reset cache on prefill (new prompt) to avoid stale keys / batch-size mismatch
        if seq_len > 1:
            self._cached_keys = None

        if use_cache:
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)  # [B, T, D]
            else:
                k_cached = k
            self._cached_keys = k_cached
        else:
            k_cached = k

        # === Scoring ===
        # Reference: weights = weights_proj(x.float()) * n_heads^(-0.5)
        # Reference: weights = weights.unsqueeze(-1) * q_scale * softmax_scale
        # Reference: index_score = fp8_index(q_fp8, weights, k_cache, k_scale_cache)
        #
        # In bf16 mode (no FP8), q_scale = 1. The fp8_index kernel computes:
        #   score[b,s,t] = sum_h(weights[b,s,h] * dot(q[b,s,h,:], k[b,t,:]))
        # where weights already absorbs n_heads^(-0.5) and softmax_scale.

        # Don't force fp32 inputs here: the checkpoint stores `weights_proj.weight` in bf16.
        # Use native dtype for matmul, then upcast the result for scoring stability.
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]

        # q·k^T per head: [B, S, H, D] @ [B, T, D]^T → [B, S, H, T]
        scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
        scores = F.relu(scores)
        # Weight per head and sum across heads → [B, S, T]
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        topk_indices = index_scores.topk(topk, dim=-1).indices  # [B, S, topk]
        return topk_indices


class GlmMoeDsaAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) with DeepSeek Sparse Attention (DSA) indexer.

    This follows the same architecture as DeepSeek V3.2's MLA:
      - Query: x → q_a_proj → RMSNorm → q_b_proj → split(q_nope, q_pe) → RoPE(q_pe)
      - KV:    x → kv_a_proj → split(kv_compressed, k_pe) → RMSNorm(kv_compressed) → kv_b_proj
                                                           → RoPE(k_pe)
      - Cache: fully expanded key_states [B, H, T, qk_head_dim] and value_states [B, H, T, v_head_dim]
      - Indexer: selects top-k tokens via DSA, applied as an additive -inf mask on attention scores

    **Caching strategy**: follows the DeepSeek V3 transformers convention of fully expanding K/V
    before caching. This ensures compatibility with DynamicCache, StaticCache, flash attention,
    and SDPA backends. The reference's compressed-cache decode path (which avoids the kv_b_proj
    expansion at decode time) is a future optimization that would require a dedicated MLA cache class.

    **FP8 compatibility**: all weight accesses use standard nn.Linear forward calls (never
    raw `.weight` access), so FP8-quantized checkpoints work transparently.
    """

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads

        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True

        # Query projection (with optional LoRA)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = GlmMoeDsaRMSNorm(config.q_lora_rank)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # Key-Value projections (MLA compressed path)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = GlmMoeDsaRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)

        self.indexer = GlmMoeDsaIndexer(config, layer_idx)

        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        # skip_topk: when True, this layer will skip computation and reuse previous layer's topk indices.
        # next_skip_topk: when True, the next layer will skip computation and reuse this layer's topk indices.
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.next_skip_topk = (
            config.indexer_types[layer_idx + 1] == "shared" if layer_idx < len(config.indexer_types) - 1 else False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        # ===== Query path =====
        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, S, q_lora_rank]
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        # Split nope/rope, apply RoPE, recombine — layout: [B, H, S, D]
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)  # BHSD format

        # ===== KV path =====
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_rank + rope_D]
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)  # [B, S, kv_rank]

        # Expand KV through kv_b_proj
        kv_expanded = self.kv_b_proj(k_compressed)  # [B, S, H * (nope_D + v_D)]
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)  # [B, H, S, nope_D]
        value_states = value_states.transpose(1, 2)  # [B, H, S, v_D]

        # RoPE on k_pe (single-head rope stream)
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)  # [B, 1, S, rope_D]
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)  # BHSD format
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)  # [B, H, S, rope_D]

        # Assemble full Q and K
        query_states = torch.cat([q_nope, q_pe], dim=-1)  # [B, H, S, qk_head_dim]
        key_states = torch.cat([k_nope, k_pe], dim=-1)  # [B, H, S, qk_head_dim]

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # ===== Indexer (DSA sparse mask) =====
        # attention_mask is [B, 1, S, T] (4D) for eager and (2D) otherwise but indexer works with [B, S, T] (3D)
        if not self.skip_topk or prev_topk_indices is None:
            indexer_mask = (
                attention_mask[:, 0, :, :]
                if attention_mask is not None and attention_mask.dim() == 4
                else attention_mask.unsqueeze(1)
                if attention_mask is not None
                else None
            )
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                indexer_mask,
                use_cache=past_key_values is not None,
            )  # [B, S, topk]
        else:
            topk_indices = prev_topk_indices  # [B, S, topk]

        # Build combined DSA + causal mask: -inf everywhere except selected top-k positions
        total_len = key_states.shape[2]
        index_mask = torch.full(
            (batch_size, seq_length, total_len),
            float("-inf"),
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        index_mask.scatter_(-1, topk_indices, 0.0)  # [B, S, T]
        index_mask = index_mask.unsqueeze(1)  # [B, 1, S, T]
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask[..., :total_len]
            combined_mask = index_mask + causal_mask
        else:
            combined_mask = (
                attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
                if attention_mask is not None
                else index_mask
            )

        # Flash attention head_dim padding (qk_head_dim != v_head_dim)
        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            indices=topk_indices,  # flash_mla_with_kvcache
            **kwargs,
        )

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices if self.next_skip_topk else None


class GlmMoeDsaDecoderLayer(Glm4MoeLiteDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk_indices


class GlmMoeDsaPreTrainedModel(Glm4MoePreTrainedModel):
    # NOTE: FP8 quantization uses `_keep_in_fp32_modules` (not `_strict`) to decide which modules to NOT convert.
    # We must keep `indexer.weights_proj` as a plain Linear to match the checkpoint (no `weight_scale_inv`).
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_sdpa = True
    _supports_flex_attn = False
    _compatible_flash_implementations = ["kernels-community/flash-mla"]


class GlmMoeDsaModel(Glm4MoeModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        topk_indices = None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prev_topk_indices=topk_indices,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GlmMoeDsaForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "GlmMoeDsaConfig",
    "GlmMoeDsaPreTrainedModel",
    "GlmMoeDsaModel",
    "GlmMoeDsaForCausalLM",
]
