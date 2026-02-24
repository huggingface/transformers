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

from ...cache_utils import Cache
from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import RopeParameters
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...models.llama.modeling_llama import rotate_half
from ...processing_utils import Unpack
from ...utils import logging
from ...utils.generic import is_flash_attention_requested
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
)
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


class GlmMoeDsaConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GlmMoeDsaModel`]. It is used to instantiate a
    GLM-5 model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GLM-5.
    e.g. [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GlmMoeDsaModel`].
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the dense MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE expert representations.
        num_hidden_layers (`int`, *optional*, defaults to 78):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 64):
            Number of key-value heads for Grouped Query Attention. If equal to `num_attention_heads`, uses MHA.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts in MoE layers.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts in MoE layers.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor for routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections (MLA).
        q_lora_rank (`int`, *optional*, defaults to 2048):
            Rank of the LoRA matrices for query projections (MLA).
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that use rotary position embeddings.
        qk_nope_head_dim (`int`, *optional*, defaults to 192):
            Dimension of the query/key heads that don't use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 256):
            Dimension of the value heads.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts selected per token.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 202752):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_parameters (`RopeParameters`, *optional*):
            Configuration parameters for the RoPE embeddings, including `rope_theta` and optional scaling parameters.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings.
        mlp_layer_types (`list`, *optional*):
            MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of top tokens selected by the indexer for sparse attention.
        index_head_dim (`int`, *optional*, defaults to 128):
            Head dimension for the indexer projections (DSA).
        index_n_heads (`int | None`, *optional*, defaults to 32):
            Number of heads for the indexer projections (DSA).
        indexer_rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether the indexer uses interleaved rotary position embeddings.


    ```python
    >>> from transformers import GlmMoeDsaConfig, GlmMoeDsaModel

    >>> # Initializing a GLM-MoE-DSA configuration
    >>> configuration = GlmMoeDsaConfig()

    >>> # Initializing a model from the configuration
    >>> model = GlmMoeDsaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm_moe_dsa"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 154880,
        hidden_size: int | None = 6144,
        intermediate_size: int | None = 12288,
        moe_intermediate_size: int | None = 2048,
        num_hidden_layers: int | None = 78,
        num_attention_heads: int | None = 64,
        num_key_value_heads: int | None = 64,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 256,
        routed_scaling_factor: float | None = 2.5,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 2048,
        qk_rope_head_dim: int | None = 64,
        qk_nope_head_dim: int | None = 192,
        v_head_dim: int | None = 256,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        num_experts_per_tok: int | None = 8,
        norm_topk_prob: bool | None = True,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 202752,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 1,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        mlp_layer_types=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        index_topk: int | None = 2048,
        index_head_dim: int | None = 128,
        index_n_heads: int | None = 32,
        **kwargs,
    ):
        # Model dimensions
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings

        # Attention dimensions (MLA)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.head_dim = qk_rope_head_dim

        # MoE parameters
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # MLP layer types: first 3 dense, rest sparse
        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, num_hidden_layers) + ["sparse"] * (num_hidden_layers - 3)
        layer_type_validation(self.mlp_layer_types, self.num_hidden_layers, attention=False)

        # Indexer (DSA) parameters
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads

        # General config
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class GlmMoeDsaRMSNorm(Glm4MoeRMSNorm):
    pass


class GlmMoeDsaIndexer(nn.Module):
    """
    Dynamic Sparse Attention (DSA) indexer for selecting top-k tokens.

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
        self._cached_keys: torch.Tensor | None = None

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
    Multi-head Latent Attention (MLA) with Dynamic Sparse Attention (DSA) indexer.

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        # ===== Query path =====
        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, S, q_lora_rank]
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.qk_head_dim).transpose(1, 2)
        # Split nope/rope, apply RoPE, recombine — layout: [B, H, S, D]
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)  # BHSD format

        # ===== KV path =====
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_rank + rope_D]
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)  # [B, S, kv_rank]

        # Expand KV through kv_b_proj
        kv_expanded = self.kv_b_proj(k_compressed)  # [B, S, H * (nope_D + v_D)]
        kv_expanded = kv_expanded.view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)  # [B, H, S, nope_D]
        value_states = value_states.transpose(1, 2)  # [B, H, S, v_D]

        # RoPE on k_pe (single-head rope stream)
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)  # [B, 1, S, rope_D]
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)  # BHSD format
        k_pe = k_pe.expand(-1, self.num_heads, -1, -1)  # [B, H, S, rope_D]

        # Assemble full Q and K
        query_states = torch.cat([q_nope, q_pe], dim=-1)  # [B, H, S, qk_head_dim]
        key_states = torch.cat([k_nope, k_pe], dim=-1)  # [B, H, S, qk_head_dim]

        # Cache update
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ===== Indexer (DSA sparse mask) =====
        # attention_mask is [B, 1, S, T] (4D) for eager and (2D) otherwise but indexer works with [B, S, T] (3D)
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
        return attn_output, attn_weights


class GlmMoeDsaDecoderLayer(Glm4MoeLiteDecoderLayer):
    pass


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
    pass


class GlmMoeDsaForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "GlmMoeDsaConfig",
    "GlmMoeDsaPreTrainedModel",
    "GlmMoeDsaModel",
    "GlmMoeDsaForCausalLM",
]
