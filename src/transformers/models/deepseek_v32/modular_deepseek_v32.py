# coding=utf-8
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
"""
DeepSeek V3.2 - Standalone implementation with DeepSeek Sparse Attention (DSA).

This is a complete standalone implementation that does NOT inherit from DeepSeek V2.
Key innovations:
- Lightning Indexer for sparse attention (reduces O(L²) to O(Lk))
- Multi-head Latent Attention (MLA) with LoRA-compressed Q/KV
- Non-interleaved RoPE in indexer, interleaved in main attention
- Hadamard transform for efficient indexer computation
- Sigmoid scoring with noaux_tc routing for MoE
"""
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging


logger = logging.get_logger(__name__)



# =============================================================================
# Configuration
# =============================================================================


class DeepseekV32Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DeepseekV32Model`]. It is used to instantiate a
    DeepSeek V3.2 model according to the specified arguments, defining the model architecture.

    DeepSeek V3.2 introduces DeepSeek Sparse Attention (DSA) which reduces attention complexity from O(L²) to O(Lk)
    using a Lightning Indexer that selects top-k tokens.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the DeepSeek V3.2 model.
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the MLP intermediate layer (for dense layers).
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP intermediate layer for MoE experts.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of decoder layers.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 128):
            Number of key-value heads (same as attention heads for MLA).
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 163840):
            Maximum sequence length the model can handle.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            Epsilon for RMS normalization.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to use KV cache for faster generation.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input/output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base for rotary position embeddings.
        rope_scaling (`Dict`, *optional*):
            YaRN scaling configuration with keys: `type` ("yarn"), `factor`, `original_max_position_embeddings`,
            `mscale`, `mscale_all_dim`, `beta_fast`, `beta_slow`.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for attention weights.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in MLP layers.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            LoRA rank for query compression in MLA.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            LoRA rank for key-value compression in MLA.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Dimension of query/key heads without position embedding.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of query/key heads with rotary position embedding.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimension of value heads.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Total number of routed experts.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts (always active).
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts activated per token.
        n_group (`int`, *optional*, defaults to 8):
            Number of expert groups for routing.
        topk_group (`int`, *optional*, defaults to 4):
            Number of groups to select in routing.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor for routed expert outputs.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            Scoring function for expert routing ("softmax" or "sigmoid").
        topk_method (`str`, *optional*, defaults to `"noaux_tc"`):
            Top-k selection method for routing.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize top-k probabilities.
        first_k_dense_replace (`int`, *optional*, defaults to 3):
            Number of initial layers that use dense MLP instead of MoE.
        index_n_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for the sparse attention indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each indexer attention head.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of top-k tokens to select for sparse attention.
        use_fp8_indexer (`bool`, *optional*, defaults to `False`):
            Whether to use FP8 quantization in the indexer.

    Example:

    ```python
    >>> from transformers import DeepseekV32Model, DeepseekV32Config

    >>> # Initializing a DeepSeek V3.2 style configuration
    >>> configuration = DeepseekV32Config()

    >>> # Initializing a model from the configuration
    >>> model = DeepseekV32Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "deepseek_v32"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Tensor parallelism plan
    base_model_tp_plan = {
        "layers.*.self_attn.q_a_proj": "colwise",
        "layers.*.self_attn.q_b_proj": "colwise",
        "layers.*.self_attn.kv_a_proj_with_mqa": "colwise",
        "layers.*.self_attn.kv_b_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.experts.*.gate_proj": "colwise",
        "layers.*.mlp.experts.*.up_proj": "colwise",
        "layers.*.mlp.experts.*.down_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 61,
        num_attention_heads: int = 128,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 163840,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: Optional[int] = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        # MLA parameters
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        # MoE parameters
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 2.5,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        first_k_dense_replace: int = 3,
        # Indexer (DSA) parameters
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        use_fp8_indexer: bool = False,
        **kwargs,
    ):
        # Core dimensions
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # For backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        # Ensure rope_scaling values are floats (config.json may have integers)
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)  # Make a copy to avoid mutating input
            for key in ("factor", "beta_fast", "beta_slow", "mscale", "mscale_all_dim"):
                if key in rope_scaling and rope_scaling[key] is not None:
                    rope_scaling[key] = float(rope_scaling[key])
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias

        # MLA parameters
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # MoE parameters
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace

        # Indexer (DSA) parameters
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.use_fp8_indexer = use_fp8_indexer

        # head_dim for RoPE compatibility
        self.head_dim = qk_rope_head_dim

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        if use_fp8_indexer:
            logger.warning_once(
                "FP8 indexer is requested but not fully implemented in HuggingFace transformers. "
                "The indexer will run without FP8 quantization. For optimal performance, use the "
                "reference DeepSeek implementation with custom tilelang kernels."
            )

    @property
    def num_local_experts(self):
        """Alias for n_routed_experts for compatibility with quantizers (e.g., FP8)."""
        return self.n_routed_experts


# =============================================================================
# Helper Functions
# =============================================================================


def apply_rotary_pos_emb_interleaved(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary position embeddings using complex multiplication (interleaved format).

    This matches the reference DeepSeek implementation which treats adjacent pairs
    of elements as complex numbers: [r0, i0, r1, i1, ...] -> [z0, z1, ...]

    Reference: model.py:404-425 (apply_rotary_emb with interleaved=True)

    Args:
        x: Input tensor [..., head_dim] where head_dim has interleaved real/imaginary pairs
        freqs_cis: Complex frequencies [seq_len, head_dim // 2] as complex tensor

    Returns:
        Rotated tensor with same shape as input
    """
    dtype = x.dtype
    # View as complex: [..., head_dim] -> [..., head_dim // 2] as complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Apply rotation via complex multiplication
    # freqs_cis needs to be broadcast to match x dimensions
    y = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return y.to(dtype)


def apply_rotary_pos_emb_non_interleaved(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary position embedding using non-interleaved format.
    Used specifically in the indexer (following reference implementation).

    Non-interleaved format: first half is real, second half is imaginary
    (e.g., [r0, r1, ..., i0, i1, ...] instead of [r0, i0, r1, i1, ...]).

    Reference: model.py:404-425 with interleaved=False

    Args:
        x: Input tensor with shape [..., head_dim] where first half is real, second half is imaginary
        freqs_cis: Complex frequencies tensor

    Returns:
        Rotated tensor with same shape as input, in non-interleaved format
    """
    dtype = x.dtype
    shape = x.shape

    # Non-interleaved: first half is real, second half is imaginary
    # Convert to interleaved format first: [r0, r1, ..., i0, i1, ...] -> [r0, i0, r1, i1, ...]
    x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    # Now apply rotation using complex multiplication (same as interleaved)
    x_complex = torch.view_as_complex(x.float().reshape(*shape[:-1], -1, 2))
    y = torch.view_as_real(x_complex * freqs_cis).flatten(-2)

    # Convert back to non-interleaved format: [r0, i0, r1, i1, ...] -> [r0, r1, ..., i0, i1, ...]
    y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)

    return y.to(dtype)


def hadamard_transform_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard transform to activations for efficient indexer computation.

    This is an optional optimization used in DeepSeek V3.2's sparse attention indexer.
    Requires the `fast_hadamard_transform` package. Falls back to identity if unavailable.
    """
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        # Fallback: skip Hadamard transform if library not installed
        return x

    hidden_size = x.size(-1)
    original_dtype = x.dtype
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    x = hadamard_transform(x, scale=hidden_size**-0.5)
    return x.to(original_dtype)


# =============================================================================
# RMSNorm
# =============================================================================


class DeepseekV32RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# =============================================================================
# Rotary Embedding
# =============================================================================


class DeepseekV32RotaryEmbedding(nn.Module):
    """
    Rotary position embedding with YaRN support for extended context.

    Outputs complex frequencies (freqs_cis) for use with complex multiplication RoPE.

    Reference: model.py:324-402 (precompute_freqs_cis)
    """

    def __init__(self, config: DeepseekV32Config, device=None):
        super().__init__()
        self.config = config
        self.rope_scaling = config.rope_scaling
        self.max_seq_len_cached = config.max_position_embeddings

        # Compute initial frequencies
        inv_freq = self._compute_default_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_default_inv_freq(self, device=None):
        """Compute default RoPE inverse frequencies."""
        dim = self.config.qk_rope_head_dim
        base = self.config.rope_theta
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        return inv_freq

    def _compute_yarn_inv_freq(self, device):
        """
        Compute YaRN-adjusted inverse frequencies for extended context.

        Reference: model.py:324-402 (precompute_freqs_cis)
        """
        if self.rope_scaling is None:
            return self.inv_freq.to(device)

        dim = self.config.qk_rope_head_dim
        base = self.config.rope_theta
        factor = self.rope_scaling.get("factor", 1.0)
        beta_fast = self.rope_scaling.get("beta_fast", 32)
        beta_slow = self.rope_scaling.get("beta_slow", 1)
        original_max_seq_len = self.rope_scaling.get(
            "original_max_position_embeddings", self.config.max_position_embeddings
        )

        def find_correction_dim(num_rotations, dim, base, max_seq_len):
            return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
            low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
            high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001
            linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - min_val) / (max_val - min_val)
            return torch.clamp(linear_func, 0, 1)

        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

        if self.config.max_position_embeddings > original_max_seq_len:
            low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_seq_len)
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        return freqs

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary embeddings as complex frequencies for given positions.

        Args:
            x: Input tensor (used for dtype/device)
            position_ids: Position indices [batch_size, seq_len]

        Returns:
            freqs_cis: Complex frequencies tensor [batch_size, seq_len, head_dim // 2]
                       as complex64 dtype for use with torch.view_as_complex

        Reference: model.py:392-401 - freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        """
        # Use YaRN frequencies if rope_scaling is configured
        if self.rope_scaling is not None and self.rope_scaling.get("type") == "yarn":
            inv_freq = self._compute_yarn_inv_freq(x.device)
        else:
            inv_freq = self.inv_freq.to(x.device)

        # Compute frequencies: freqs[i] = position * inv_freq[i]
        # inv_freq shape: [dim // 2]
        # position_ids shape: [batch_size, seq_len]
        # Result shape: [batch_size, seq_len, dim // 2]

        # Force float32 for precision in trigonometric operations
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            # Outer product: position_ids @ inv_freq
            # position_ids: [batch_size, seq_len] -> [batch_size, seq_len, 1]
            # inv_freq: [dim // 2] -> [1, 1, dim // 2]
            freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)
            # freqs shape: [batch_size, seq_len, dim // 2]

            # Convert to complex exponential: e^(i * freqs) = cos(freqs) + i*sin(freqs)
            # torch.polar(abs, angle) creates complex number with abs * e^(i*angle)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


# =============================================================================
# MLP
# =============================================================================


class DeepseekV32MLP(nn.Module):
    """Standard MLP for dense layers."""

    def __init__(self, config: DeepseekV32Config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use float32 for intermediate SiLU computation for numerical stability (ref: model.py:643)
        return self.down_proj((self.act_fn(self.gate_proj(x).float()) * self.up_proj(x).float()).type_as(x))


# =============================================================================
# MoE Components
# =============================================================================


class DeepseekV32Expert(nn.Module):
    """Single expert MLP for MoE."""

    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use float32 for intermediate SiLU computation for numerical stability (ref: model.py:643)
        return self.down_proj((self.act_fn(self.gate_proj(x).float()) * self.up_proj(x).float()).type_as(x))


class DeepseekV32Gate(nn.Module):
    """
    Gating mechanism for MoE routing with sigmoid scoring and noaux_tc support.
    """

    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.scoring_func = config.scoring_func
        self.routed_scaling_factor = config.routed_scaling_factor
        self.topk_method = config.topk_method

        self.weight = nn.Parameter(torch.zeros(config.n_routed_experts, config.hidden_size))
        # Initialize gate weights with small values (kaiming uniform for linear layers)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Add bias for 7168 hidden size (following reference implementation)
        if config.hidden_size == 7168:
            self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts))
        else:
            self.register_parameter("e_score_correction_bias", None)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and expert indices.

        Args:
            hidden_states: [batch_size * seq_len, hidden_size]

        Returns:
            weights: [batch_size * seq_len, num_experts_per_tok]
            indices: [batch_size * seq_len, num_experts_per_tok]
        """
        batch_size = hidden_states.shape[0]

        # Compute scores
        scores = F.linear(hidden_states.float(), self.weight.float())

        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1)
        else:  # sigmoid
            scores = scores.sigmoid()

        original_scores = scores

        # Apply bias if present
        if self.e_score_correction_bias is not None:
            scores = scores + self.e_score_correction_bias

        # Expert selection based on topk_method
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        elif self.topk_method in ("group_limited_greedy", "noaux_tc"):
            # Group-based selection
            group_scores = scores.view(batch_size, self.n_group, -1)

            if self.e_score_correction_bias is None:
                group_max = group_scores.amax(dim=-1)
            else:
                # Use top-2 sum for group scoring when bias is present
                group_max = group_scores.topk(2, dim=-1)[0].sum(dim=-1)

            group_idx = torch.topk(group_max, k=self.topk_group, dim=-1, sorted=False)[1]

            # Create mask for selected groups
            group_mask = torch.ones(batch_size, self.n_group, dtype=torch.bool, device=scores.device)
            group_mask.scatter_(1, group_idx, False)

            # Apply mask to scores
            scores_masked = scores.view(batch_size, self.n_group, -1)
            scores_masked = scores_masked.masked_fill(group_mask.unsqueeze(-1), float("-inf"))
            scores_masked = scores_masked.view(batch_size, -1)

            topk_weight, topk_idx = torch.topk(scores_masked, k=self.num_experts_per_tok, dim=-1, sorted=False)
        else:
            raise ValueError(f"Unknown topk_method: {self.topk_method}")

        # Get weights from original scores (before bias)
        topk_weight = original_scores.gather(1, topk_idx)

        # Normalize weights for sigmoid scoring
        if self.scoring_func == "sigmoid":
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-8)

        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_weight, topk_idx


class DeepseekV32MoE(nn.Module):
    """
    Mixture of Experts layer with sigmoid scoring and noaux_tc routing.
    """

    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.gate = DeepseekV32Gate(config)
        self.experts = nn.ModuleList([DeepseekV32Expert(config) for _ in range(config.n_routed_experts)])

        # Shared experts (always active)
        self.shared_experts = DeepseekV32MLP(config, intermediate_size=config.n_shared_experts * config.moe_intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MoE layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)

        # Get routing weights and indices
        weights, indices = self.gate(hidden_states_flat)

        # Route tokens to experts
        output = torch.zeros_like(hidden_states_flat, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        for i, expert in enumerate(self.experts):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            output[idx] += expert(hidden_states_flat[idx]) * weights[idx, top, None]

        # Add shared expert output
        output = output + self.shared_experts(hidden_states_flat)

        return output.type_as(hidden_states).view(batch_size, seq_len, hidden_size)


# =============================================================================
# Indexer (Lightning Indexer for DSA)
# =============================================================================


class DeepseekV32Indexer(nn.Module):
    """
    Lightning Indexer for DeepSeek Sparse Attention (DSA).

    Computes index scores: I_{t,s} = Σ w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
    and selects top-k tokens for sparse attention.

    Key differences from main attention:
    - Uses non-interleaved RoPE
    - Single-head keys (broadcast to multi-head queries)
    - Applies Hadamard transform for efficiency
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank

        # Query projection from compressed representation
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # Key projection (single-head)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # Head weights projection
        self.weights_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top-k token indices for sparse attention.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            q_compressed: [batch_size, seq_len, q_lora_rank] - compressed Q from main attention
            freqs_cis: Complex RoPE frequencies [batch_size, seq_len, rope_dim // 2]
            attention_mask: Optional mask for causal attention
            k_cache: Optional cached keys from previous positions

        Returns:
            topk_indices: [batch_size, seq_len, topk] - indices of selected tokens
            k_states: [batch_size, seq_len, head_dim] - keys to cache

        Reference: model.py:456-487
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project queries from compressed representation
        q_states = self.wq_b(q_compressed)
        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Split Q into RoPE and non-RoPE parts
        # Reference: model.py:461 - q_pe, q_nope = torch.split(q, [rope_dim, head_dim - rope_dim])
        q_pe = q_states[..., : self.qk_rope_head_dim]
        q_nope = q_states[..., self.qk_rope_head_dim :]

        # Apply non-interleaved RoPE to Q
        # Reference: model.py:463 - apply_rotary_emb(q_pe, freqs_cis, False)
        # freqs_cis needs to be broadcast to [B, S, H, rope_dim // 2]
        freqs_cis_expanded = freqs_cis.unsqueeze(2)  # [B, S, 1, rope_dim // 2]
        q_pe = apply_rotary_pos_emb_non_interleaved(q_pe, freqs_cis_expanded)
        q_states = torch.cat([q_pe, q_nope], dim=-1)

        # Project and normalize keys (single-head)
        # Reference: model.py:465-466
        k_states = self.k_norm(self.wk(hidden_states))

        # Split K into RoPE and non-RoPE parts
        k_pe = k_states[..., : self.qk_rope_head_dim]
        k_nope = k_states[..., self.qk_rope_head_dim :]

        # Apply non-interleaved RoPE to K (single-head, add head dimension temporarily)
        # Reference: model.py:469 - apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
        k_pe = apply_rotary_pos_emb_non_interleaved(k_pe.unsqueeze(2), freqs_cis_expanded).squeeze(2)
        k_states = torch.cat([k_pe, k_nope], dim=-1)

        # Apply Hadamard transform (optional optimization)
        # Reference: model.py:471-472
        q_states = hadamard_transform_activation(q_states)
        k_states = hadamard_transform_activation(k_states)

        # Use cached keys if available
        if k_cache is not None:
            k_full = torch.cat([k_cache, k_states], dim=1)
        else:
            k_full = k_states

        # Compute head weights
        # Reference: model.py:477 - weights = self.weights_proj(x.float()) * n_heads ** -0.5
        head_weights = F.linear(hidden_states.float(), self.weights_proj.weight.float()) * (self.num_heads**-0.5)

        # Compute attention scores: q @ k^T
        # q_states: [B, S, H, D], k_full: [B, T, D]
        # Reference: model.py:479 (fp8_index does this internally)
        scores = torch.einsum("bshd,btd->bsht", q_states.float(), k_full.float()) * self.softmax_scale

        # Apply ReLU then head weights per formula: I_{t,s} = Σ_j w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
        scores = torch.relu(scores)
        scores = scores * head_weights.unsqueeze(-1)
        index_scores = scores.sum(dim=2)  # [B, S, T]

        # Apply mask
        if attention_mask is not None:
            # attention_mask is [B, 1, S, T] or [B, S, T]
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1)
            index_scores = index_scores + attention_mask

        # Select top-k
        # Reference: model.py:482
        T = index_scores.shape[-1]
        topk = min(self.index_topk, T)
        topk_indices = index_scores.topk(topk, dim=-1).indices

        return topk_indices, k_states


# =============================================================================
# Attention (MLA with DSA)
# =============================================================================


class DeepseekV32Attention(nn.Module):
    """
    Multi-head Latent Attention (MLA) with DeepSeek Sparse Attention (DSA).

    Key features:
    - LoRA-compressed Q and KV projections
    - Split head dims: qk_nope (128) + qk_rope (64) for queries/keys
    - Lightning Indexer for sparse attention
    - YaRN mscale adjustment for extended context
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.attention_dropout = config.attention_dropout

        # Q path (LoRA compressed)
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = DeepseekV32RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # KV path (LoRA compressed + rope stream)
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias
        )
        self.kv_a_layernorm = DeepseekV32RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)

        # Softmax scale with YaRN mscale adjustment
        self.softmax_scale = self.qk_head_dim**-0.5
        rope_scaling = config.rope_scaling or {}
        original_max_pos = rope_scaling.get("original_max_position_embeddings", config.max_position_embeddings)
        if config.max_position_embeddings > original_max_pos:
            mscale = rope_scaling.get("mscale", 1.0)
            rope_factor = rope_scaling.get("factor", 1.0)
            mscale_adjustment = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale_adjustment * mscale_adjustment

        # Lightning Indexer for sparse attention
        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Forward pass for MLA attention with DSA.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            position_embeddings: freqs_cis from rotary embedding [batch_size, seq_len, rope_dim // 2]
            attention_mask: Causal mask [batch_size, 1, seq_len, total_len]
            past_key_value: KV cache
            output_attentions: Whether to output attention weights
            cache_position: Position indices for cache update

        Returns:
            attn_output: [batch_size, seq_len, hidden_size]
            attn_weights: Optional attention weights
            past_key_value: Updated cache

        Reference: model.py:544-608 (MLA.forward)
        """
        batch_size, seq_len, _ = hidden_states.shape
        freqs_cis = position_embeddings

        # Q path: compress -> normalize -> project
        # Reference: model.py:559-561
        q_compressed = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_compressed)
        q_states = q_states.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)

        # Split Q into nope and rope parts
        # Reference: model.py:562 - q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim])
        q_nope, q_pe = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE to Q rope part (interleaved format)
        # Reference: model.py:563 - q_pe = apply_rotary_emb(q_pe, freqs_cis)
        freqs_cis_expanded = freqs_cis.unsqueeze(2)  # [B, S, 1, rope_dim // 2]
        q_pe = apply_rotary_pos_emb_interleaved(q_pe, freqs_cis_expanded)

        # KV path: project -> split -> normalize
        # Reference: model.py:564-566
        kv_all = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_pe = torch.split(kv_all, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_compressed = self.kv_a_layernorm(kv_compressed)

        # Apply RoPE to K rope part (single-head, then broadcast)
        # Reference: model.py:567 - k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        k_pe = apply_rotary_pos_emb_interleaved(k_pe.unsqueeze(2), freqs_cis_expanded)

        # Project KV to full dimensions
        # Reference: model.py:575-577
        kv_proj = self.kv_b_proj(kv_compressed)
        kv_proj = kv_proj.view(batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Combine Q and K
        # Reference: model.py:574, 578
        q_states = torch.cat([q_nope, q_pe], dim=-1)  # [B, S, H, qk_head_dim]
        k_states = torch.cat([k_nope, k_pe.expand(-1, -1, self.num_heads, -1)], dim=-1)  # [B, S, H, qk_head_dim]

        # Transpose for attention: [B, H, S, D]
        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        # Update cache
        if past_key_value is not None:
            # Cache stores (kv_compressed, k_rope) for efficient decode
            cache_kwargs = {"cache_position": cache_position}
            k_states, v_states = past_key_value.update(k_states, v_states, self.layer_idx, cache_kwargs)

        # Compute attention scores
        # Reference: model.py:579
        attn_weights = torch.matmul(q_states.float(), k_states.float().transpose(-1, -2)) * self.softmax_scale

        # Apply causal mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : k_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Apply sparse attention via indexer (during prefill with full mask)
        # Reference: model.py:582-585
        if seq_len > 1:
            # Get top-k indices from indexer
            topk_indices, _ = self.indexer(
                hidden_states, q_compressed, freqs_cis, attention_mask.squeeze(1) if attention_mask is not None else None
            )

            # Build sparse mask
            # Reference: model.py:583-585
            index_mask = torch.full(
                (batch_size, seq_len, k_states.shape[-2]), float("-inf"), device=hidden_states.device, dtype=attn_weights.dtype
            )
            index_mask.scatter_(-1, topk_indices, 0.0)
            attn_weights = attn_weights + index_mask.unsqueeze(1)

        # Softmax and dropout
        # Reference: model.py:587
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Compute output
        # Reference: model.py:588
        attn_output = torch.matmul(attn_weights, v_states)

        # Reshape and project output
        # Reference: model.py:606
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# =============================================================================
# Decoder Layer
# =============================================================================


class DeepseekV32DecoderLayer(nn.Module):
    """
    Transformer decoder layer with MLA attention and MoE/dense MLP.

    First `first_k_dense_replace` layers use dense MLP, rest use MoE.
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Attention
        self.self_attn = DeepseekV32Attention(config, layer_idx)

        # MLP: dense for first k layers, MoE for rest
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DeepseekV32MLP(config)
        else:
            self.mlp = DeepseekV32MoE(config)

        # Layer norms
        self.input_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Forward pass for decoder layer.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Causal attention mask
            position_ids: Position indices
            past_key_value: KV cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            cache_position: Cache position indices
            position_embeddings: freqs_cis from rotary embedding [batch_size, seq_len, rope_dim // 2]

        Returns:
            hidden_states: Output hidden states
            self_attn_weights: Optional attention weights
            present_key_value: Updated cache
        """
        residual = hidden_states

        # Pre-norm + attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Pre-norm + MLP/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


# =============================================================================
# Model
# =============================================================================


class DeepseekV32PreTrainedModel(PreTrainedModel):
    """Base class for DeepSeek V3.2 models."""

    config_class = DeepseekV32Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV32DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    # Ignore MTP (Multi-Token Prediction) layer weights stored as layer 61 in checkpoint
    # MTP is only used for speculative decoding, not needed for standard inference
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.61\..*"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DeepseekV32Model(DeepseekV32PreTrainedModel):
    """
    DeepSeek V3.2 base model outputting raw hidden states.
    """

    def __init__(self, config: DeepseekV32Config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekV32DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV32RotaryEmbedding(config)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPast]:
        """
        Forward pass for base model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            past_key_values: KV cache
            inputs_embeds: Pre-computed input embeddings
            use_cache: Whether to use KV cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output all hidden states
            return_dict: Whether to return ModelOutput

        Returns:
            BaseModelOutputWithPast or tuple
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Initialize cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # Compute rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        output_attentions: bool,
    ) -> Optional[torch.Tensor]:
        """Create causal mask for attention."""
        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]

        # Determine target length for the causal mask
        if past_key_values is not None:
            target_length = past_key_values.get_max_cache_shape()
            if target_length is None or target_length < 0:
                # For DynamicCache without max_cache_shape, use the last position + 1
                target_length = cache_position[-1].item() + 1
        else:
            target_length = sequence_length

        # Create causal mask
        causal_mask = torch.full((sequence_length, target_length), fill_value=torch.finfo(dtype).min, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, torch.finfo(dtype).min
            )

        return causal_mask


class DeepseekV32ForCausalLM(DeepseekV32PreTrainedModel, GenerationMixin):
    """
    DeepSeek V3.2 model for causal language modeling.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepseekV32Config):
        super().__init__(config)
        self.model = DeepseekV32Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        """
        Forward pass for causal LM.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: KV cache
            inputs_embeds: Pre-computed embeddings
            labels: Labels for computing loss
            use_cache: Whether to use KV cache
            output_attentions: Output attention weights
            output_hidden_states: Output all hidden states
            return_dict: Return ModelOutput
            logits_to_keep: Number of logits to keep (0 = all)

        Returns:
            CausalLMOutputWithPast or tuple
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Compute logits
        if logits_to_keep > 0:
            hidden_states = hidden_states[:, -logits_to_keep:, :]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Compute loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> dict:
        """Prepare inputs for generation."""
        # If past_key_values are used, only last tokens needed
        if past_key_values is not None:
            if inputs_embeds is not None:
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # Prepare inputs dict
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.contiguous(), "inputs_embeds": None}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class DeepseekV32ForSequenceClassification(DeepseekV32PreTrainedModel):
    """
    DeepSeek V3.2 model for sequence classification.
    """

    def __init__(self, config: DeepseekV32Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = DeepseekV32Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Forward pass for sequence classification."""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        from transformers.modeling_outputs import SequenceClassifierOutputWithPast

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
    "DeepseekV32ForSequenceClassification",
]
