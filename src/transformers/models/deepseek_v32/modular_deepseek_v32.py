# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""DeepSeek-V3.2-Exp: DeepSeek-V3 plus DeepSeek Sparse Attention (DSA).

This is DeepSeek-V3 with a lightning indexer added to each attention layer: the indexer scores every
query against the cached keys and keeps the top-`index_topk` tokens, which become an additive sparse
mask folded into the MLA attention mask. Everything else (MoE, MLA projections, RoPE, the decoder /
model / causal-LM scaffolding) is inherited unchanged from DeepSeek-V3.

The cross-layer top-k *sharing* variant is a GLM-MoE-DSA innovation and lives in that model, which
inherits from this one (see `models/glm_moe_dsa/modular_glm_moe_dsa.py`).
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RotaryEmbeddingConfigMixin
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from ..glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from ..glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteDecoderLayer


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V3.2-Exp")
@strict
class DeepseekV32Config(Glm4MoeLiteConfig, RotaryEmbeddingConfigMixin):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 64):
        Number of heads for the indexer projections (DSA).
    rope_interleave (`bool`, *optional*, defaults to `True`):
        Whether to apply RoPE in the interleaved (GPT-J adjacent-pair) layout.

    ```python
    >>> from transformers import DeepseekV32Config, DeepseekV32Model

    >>> # Initializing a DeepSeek-V3.2 configuration
    >>> configuration = DeepseekV32Config()

    >>> # Initializing a model from the configuration
    >>> model = DeepseekV32Model(configuration)

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

    attribute_map = {"num_local_experts": "num_experts"}

    vocab_size: int = 129280
    hidden_size: int = 7168
    intermediate_size: int = 18432
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_nope_head_dim: int = 128
    n_group: int = 8
    topk_group: int = 4
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 163840
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    mlp_layer_types: list[str] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 64
    mlp_bias: bool = False
    num_experts: int = 256
    head_dim: int = 64
    # DeepSeek V3.2 applies RoPE in the interleaved (GPT-J pair) layout, like DeepSeek V3.
    rope_interleave: bool = True
    pretraining_tp = AttributeError()
    # ``layer_types`` drives cache-class dispatch: every layer is DSA, so each gets a
    # ``DynamicIndexedLayer`` / ``StaticIndexedLayer`` via ``LAYER_TYPE_CACHE_MAPPING``.
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # MLP layer types: first 3 dense, rest sparse.
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )
        # Every layer is DSA — drives cache-class dispatch.
        if self.layer_types is None:
            self.layer_types = ["dynamic_sparse_attention"] * self.num_hidden_layers
        super().__post_init__(**kwargs)


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    @staticmethod
    def compute_default_rope_parameters(
        config: "DeepseekV32Config | None" = None,
        device=None,
        seq_len: int | None = None,
    ):
        base = config.rope_parameters["rope_theta"]
        head_dim = config.qk_rope_head_dim
        attention_factor = 1.0

        if head_dim == 0:
            return torch.empty(0, device=device), attention_factor

        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / head_dim)
        )
        return inv_freq, attention_factor


class DeepseekV32Indexer(nn.Module):
    """
    DeepSeek Sparse Attention (DSA) indexer for selecting top-k tokens.

    The Indexer has its own lightweight projections (wq_b, wk) separate from the main MLA attention,
    and returns the additive top-k sparse mask directly (`0` at the selected tokens, `-inf` elsewhere);
    the raw top-k indices are only ever scattered into that mask, so they are not surfaced.

    **Cache strategy**: the indexer key cache lives on the per-layer `DynamicIndexedLayer` (or the
    `StaticIndexedLayer` for static caches) inside the shared cache, accessed via
    `past_key_values.update_indexer()`.
    """

    def __init__(self, config: "DeepseekV32Config", layer_idx: int):
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

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden]
        q_resid: torch.Tensor,  # [B, S, q_lora_rank]
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

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
            past_key_values: Cache object containing the indexer key cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]`. The eager / SDPA paths
                turn these into an additive sparse mask; the `flash-mla` kernel consumes them directly.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # === Queries ===
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # === Keys ===
        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # Interleaved RoPE on the rope slice of q / k (single-head k), then recombine.
        q_pe, k_pe = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_pe, k_nope], dim=-1).squeeze(2)  # [B, S, D]

        # The indexer key cache lives on the per-layer indexed cache layer (not on the module), so it
        # integrates with the shared cache and offloading / beam-search / crop bookkeeping.
        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        # === Scoring: index_score[b,s,t] = Σ_h weight[b,s,h] · softmax_scale · ReLU(q[b,s,h,:] · k[b,t,:]) ===
        weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)  # [B, S, H]
        # q·k^T per head: [B, S, H, D] @ [B, 1, D, T] → [B, S, H, T]
        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)
        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        # diff with classic attention, sample don't project :wink:
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]


class DeepseekV32Attention(DeepseekV3Attention):
    """DeepSeek-V3 MLA, with a DSA indexer whose top-k sparse mask is folded into the attention mask."""

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # ===== Query / KV projections (MLA) =====
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # ===== DSA: select the top-k tokens per query =====
        # The indexer scores against a 3D `[B, S, T]` mask; the attention mask is 4D `[B, 1, S, T]`.
        indexer_mask = attention_mask[:, 0, :, :] if attention_mask is not None else None
        topk_indices = self.indexer(
            hidden_states, q_resid, position_embeddings, indexer_mask, past_key_values=past_key_values
        )  # [B, S, topk]

        # The dense eager / SDPA paths apply sparsity as an additive `-inf` mask (materialized here, so it
        # shows up in output recorders); the `flash-mla` kernel instead gathers the top-k tokens itself.
        sparse_indices = None
        if self.config._attn_implementation in ("eager", "sdpa"):
            index_mask = torch.full(
                (batch_size, seq_length, key_states.shape[2]),
                float("-inf"),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            index_mask = index_mask.scatter(-1, topk_indices.long(), 0.0).unsqueeze(1)  # [B, 1, S, T]
            attention_mask = attention_mask + index_mask if attention_mask is not None else index_mask
        else:
            sparse_indices = topk_indices

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            indices=sparse_indices,  # consumed by flash_mla_with_kvcache; ignored by eager / SDPA
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DeepseekV32DecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    # NOTE: FP8 quantization uses `_keep_in_fp32_modules` (not `_strict`) to decide which modules to NOT convert.
    # We must keep `indexer.weights_proj` as a plain Linear to match the checkpoint (no `weight_scale_inv`).
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61.*"]
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_sdpa = True
    _supports_flex_attn = False
    _compatible_flash_implementations = ["kernels-community/flash-mla"]


class DeepseekV32Model(DeepseekV3Model):
    pass


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    pass


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
]
