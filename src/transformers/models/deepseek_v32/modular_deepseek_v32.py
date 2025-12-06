# coding=utf-8
# Copyright 2025 DeepSeek AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on the DeepSeek-V3.2-Exp implementation from DeepSeek AI.
# Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
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
DeepSeek V3.2 model implementation.

DeepSeek V3.2 extends DeepSeek V3 with DeepSeek Sparse Attention (DSA), which uses a
Lightning Indexer to select top-k tokens for sparse attention, reducing complexity
from O(L^2) to O(L*k) where k is the number of selected tokens (default 2048).

Key architectural differences from V3:
1. Lightning Indexer: Computes index scores to select relevant tokens
2. Hadamard Transform: Applied to Q/K in the indexer for activation rotation
3. Non-interleaved RoPE in Indexer: Different from interleaved RoPE in MLA
4. Sparse Attention: Only attends to top-k selected tokens
"""
import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import logging
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3ForSequenceClassification,
    DeepseekV3ForTokenClassification,
    DeepseekV3Model,
    DeepseekV3MLP,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb_interleave,
    yarn_get_mscale,
)


logger = logging.get_logger(__name__)

# Try to import fast_hadamard_transform, fall back to pure PyTorch if not available
try:
    from fast_hadamard_transform import hadamard_transform

    HAS_FAST_HADAMARD = True
except ImportError:
    HAS_FAST_HADAMARD = False
    logger.warning_once(
        "fast-hadamard-transform not installed. Using slower PyTorch fallback. "
        "For better performance, install with: pip install fast-hadamard-transform"
    )


def hadamard_transform_fallback(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Pure PyTorch Fast Walsh-Hadamard Transform fallback.

    This is significantly slower than the CUDA version but works on CPU and
    doesn't require the fast-hadamard-transform package.

    Args:
        x: Input tensor with shape (..., dim) where dim should be a power of 2
        scale: Multiplier for the output

    Returns:
        Transformed tensor with same shape as input
    """
    orig_dtype = x.dtype
    x = x.float()
    dim = x.shape[-1]

    # Pad to power of 2 if needed
    if dim & (dim - 1) != 0:
        next_pow2 = 1 << (dim - 1).bit_length()
        x = F.pad(x, (0, next_pow2 - dim))
        dim = next_pow2

    # Fast Walsh-Hadamard Transform using butterfly operations
    h = 1
    while h < dim:
        # Reshape for butterfly operation
        x = x.view(*x.shape[:-1], dim // (2 * h), 2, h)
        # Butterfly: [a, b] -> [a + b, a - b]
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.view(*x.shape[:-3], dim)
        h *= 2

    return (x * scale).to(orig_dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Apply Hadamard transform for activation rotation in the indexer.

    This is used in the Lightning Indexer to rotate Q and K activations
    before computing index scores.

    Args:
        x: Input tensor with shape (..., hidden_size)

    Returns:
        Rotated tensor with same shape
    """
    hidden_size = x.size(-1)
    scale = hidden_size**-0.5

    if HAS_FAST_HADAMARD:
        # fast-hadamard-transform requires contiguous bfloat16 input
        return hadamard_transform(x.contiguous(), scale=scale)
    else:
        return hadamard_transform_fallback(x, scale=scale)


def apply_rotary_pos_emb_non_interleave(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding with NON-INTERLEAVED layout.

    This is specifically for the Indexer, which requires non-interleaved RoPE
    (different from the MLA which uses interleaved RoPE).

    The difference is in how dimensions are paired:
    - Interleaved: pairs (0,1), (2,3), (4,5), ...
    - Non-interleaved: pairs (0, dim/2), (1, dim/2+1), ...

    Args:
        q: Query tensor of shape (batch, seq_len, heads, head_dim)
        k: Key tensor of shape (batch, seq_len, heads, head_dim)
        cos: Cosine of rotary angles
        sin: Sine of rotary angles
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple of rotated (query, key) tensors
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Non-interleaved: split in half and rotate
    # q = [q1, q2] where q1 is first half, q2 is second half
    # rotated = [q1 * cos - q2 * sin, q1 * sin + q2 * cos]
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)

    q_embed = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_embed, k_embed


class DeepseekV32Config(DeepseekV3Config):
    """
    Configuration class for DeepSeek V3.2 model.

    Extends DeepseekV3Config with parameters for the Lightning Indexer
    and DeepSeek Sparse Attention (DSA).

    Args:
        index_n_heads (`int`, *optional*, defaults to 64):
            Number of heads in the Lightning Indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each indexer head.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of tokens to select for sparse attention.
        use_sparse_attention (`bool`, *optional*, defaults to True):
            Whether to use sparse attention. Set to False to use dense attention
            (useful for the dense warm-up training stage).
        detach_indexer_input (`bool`, *optional*, defaults to False):
            Whether to detach the indexer input from the computational graph.
            Used in Stage 2 training for separate optimization of indexer.
        **kwargs:
            Additional arguments passed to DeepseekV3Config.
    """

    model_type = "deepseek_v32"

    def __init__(
        self,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        use_sparse_attention: bool = True,
        detach_indexer_input: bool = False,
        **kwargs,
    ):
        # Set V3.2 specific defaults if not provided
        kwargs.setdefault("n_routed_experts", 256)
        kwargs.setdefault("n_shared_experts", 1)
        kwargs.setdefault("num_experts_per_tok", 8)
        kwargs.setdefault("n_group", 8)
        kwargs.setdefault("topk_group", 4)
        kwargs.setdefault("routed_scaling_factor", 2.5)
        kwargs.setdefault("first_k_dense_replace", 3)

        super().__init__(**kwargs)

        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.use_sparse_attention = use_sparse_attention
        self.detach_indexer_input = detach_indexer_input


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class DeepseekV32MLP(DeepseekV3MLP):
    pass


class DeepseekV32TopkRouter(DeepseekV3TopkRouter):
    pass


class DeepseekV32MoE(DeepseekV3MoE):
    pass


class DeepseekV32Indexer(nn.Module):
    """
    Lightning Indexer for DeepSeek Sparse Attention (DSA).

    The indexer computes index scores to select which tokens each query should
    attend to, reducing attention complexity from O(L^2) to O(L*k).

    The index score formula is:
        I_{t,s} = sum_j w^I_{t,j} * ReLU(q^I_{t,j} * k^I_s)

    Key implementation details:
    1. Uses Hadamard transform on Q and K before scoring
    2. Uses NON-INTERLEAVED RoPE (different from MLA which uses interleaved)
    3. Uses LayerNorm on K (not RMSNorm)

    Args:
        config: DeepseekV32Config
        layer_idx: Index of the layer this indexer belongs to
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
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # Key projection (single head, broadcast to all heads)
        self.k_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)

        # LayerNorm for keys (not RMSNorm, following reference)
        self.k_layernorm = nn.LayerNorm(self.head_dim)

        # Per-head weight projection
        self.weight_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.softmax_scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute top-k token indices for sparse attention.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            q_compressed: Compressed query representation [batch, seq_len, q_lora_rank]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask

        Returns:
            topk_indices: Indices of selected tokens [batch, seq_len, topk]
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Query path
        q = self.q_b_proj(q_compressed)  # [B, S, num_heads * head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Split into RoPE and non-RoPE parts
        q_rope, q_nope = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )

        # Key path
        k = self.k_proj(hidden_states)  # [B, S, head_dim]
        k = self.k_layernorm(k)

        k_rope, k_nope = torch.split(
            k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )

        # Apply NON-INTERLEAVED RoPE (critical difference from MLA!)
        k_rope = k_rope.unsqueeze(2)  # [B, S, 1, rope_dim]
        q_rope, k_rope = apply_rotary_pos_emb_non_interleave(q_rope, k_rope, cos, sin)
        k_rope = k_rope.squeeze(2)  # [B, S, rope_dim]

        # Concatenate back
        q = torch.cat([q_rope, q_nope], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_rope, k_nope], dim=-1)  # [B, S, D]

        # Apply Hadamard transform for activation rotation
        q = rotate_activation(q)
        k = rotate_activation(k)

        # Compute index scores: I_{t,s} = sum_j w_{t,j} * ReLU(q_{t,j} * k_s)
        # q: [B, S, H, D], k: [B, S, D]
        # First compute q * k for all pairs: [B, S_q, H, S_k]
        q = q.transpose(1, 2)  # [B, H, S_q, D]
        k = k.unsqueeze(1)  # [B, 1, S_k, D]

        # Compute attention-like scores
        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, H, S_q, S_k]

        # Apply ReLU
        scores = F.relu(scores)

        # Get per-head weights
        weights = self.weight_proj(hidden_states.float())  # [B, S, H]
        weights = weights * (self.num_heads**-0.5) * self.softmax_scale
        weights = weights.transpose(1, 2).unsqueeze(-1)  # [B, H, S, 1]

        # Weighted sum over heads: [B, S_q, S_k]
        index_scores = (scores * weights).sum(dim=1)  # [B, S_q, S_k]

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask is typically [B, 1, S_q, S_k] or [B, S_q, S_k]
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1)
            index_scores = index_scores + attention_mask

        # Select top-k tokens
        k_select = min(self.index_topk, seq_len)
        topk_indices = index_scores.topk(k_select, dim=-1).indices  # [B, S, topk]

        return topk_indices


class DeepseekV32Attention(DeepseekV3Attention):
    """
    DeepSeek V3.2 Attention with Lightning Indexer for sparse attention.

    Extends DeepseekV3Attention by adding the Lightning Indexer which selects
    top-k tokens for each query position, enabling sparse attention.
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__(config, layer_idx)

        # Add the Lightning Indexer
        self.indexer = DeepseekV32Indexer(config, layer_idx)

        # Update softmax scale with YaRN mscale if needed
        if hasattr(config, "rope_parameters") and config.rope_parameters:
            rope_type = config.rope_parameters.get("rope_type", "default")
            if rope_type != "default":
                mscale_all_dim = config.rope_parameters.get("mscale_all_dim", 0)
                scaling_factor = config.rope_parameters.get("factor", 1.0)
                if mscale_all_dim:
                    mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                    self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with sparse attention via Lightning Indexer.
        """
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # Query path with LoRA compression
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
            q_compressed = None
        else:
            q_compressed = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q_states = self.q_b_proj(q_compressed)

            # Optionally detach for separate indexer optimization (Stage 2 training)
            if self.config.detach_indexer_input:
                q_compressed_for_indexer = q_compressed.detach()
            else:
                q_compressed_for_indexer = q_compressed

        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV path with compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        # Apply RoPE (INTERLEAVED for MLA)
        cos, sin = position_embeddings
        if self.config.rope_interleave:
            q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        else:
            from ..llama.modeling_llama import apply_rotary_pos_emb
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        # Update cache if provided
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Compute attention scores
        # For flash attention with different head dims, pad value states
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        # Standard attention computation
        attn_output, attn_weights = self._compute_attention(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_compressed_for_indexer if self.q_lora_rank else None,
            hidden_states,
            position_embeddings,
            **kwargs,
        )

        # Remove padding if added
        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def _compute_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        q_compressed: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute attention with optional sparse masking from the indexer.
        """
        batch_size, num_heads, seq_len, _ = query_states.shape

        # Check if we should use sparse attention
        use_sparse = (
            self.config.use_sparse_attention
            and q_compressed is not None
            and seq_len > 1  # Only for prefill, not decode
        )

        if use_sparse:
            # Get top-k indices from the indexer
            topk_indices = self.indexer(
                hidden_states,
                q_compressed,
                position_embeddings,
                attention_mask,
            )

            # Create sparse attention mask
            # topk_indices: [B, S, topk]
            # We need to create a mask that only allows attention to selected tokens
            kv_seq_len = key_states.shape[2]
            sparse_mask = torch.full(
                (batch_size, seq_len, kv_seq_len),
                float("-inf"),
                device=query_states.device,
                dtype=query_states.dtype,
            )

            # Scatter 0s at the selected positions
            sparse_mask.scatter_(-1, topk_indices, 0.0)

            # Combine with causal mask if provided
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    # [B, 1, S, S] -> [B, S, S]
                    attention_mask = attention_mask.squeeze(1)
                sparse_mask = sparse_mask + attention_mask

            # Expand for heads: [B, H, S, S]
            attention_mask = sparse_mask.unsqueeze(1).expand(-1, num_heads, -1, -1)

        # Use eager attention for now (can be extended to flash attention)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


class DeepseekV32DecoderLayer(DeepseekV3DecoderLayer):
    """DeepSeek V3.2 decoder layer with sparse attention."""

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        # Call grandparent init to avoid V3 attention
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        # Use V3.2 attention with indexer
        self.self_attn = DeepseekV32Attention(config=config, layer_idx=layer_idx)

        # MLP: dense for first k layers, MoE for rest
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV32MoE(config)
        else:
            self.mlp = DeepseekV32MLP(config)

        self.input_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    """Base class for DeepSeek V3.2 models."""

    config_class = DeepseekV32Config
    _no_split_modules = ["DeepseekV32DecoderLayer"]


class DeepseekV32Model(DeepseekV3Model):
    """DeepSeek V3.2 Model with sparse attention."""

    config_class = DeepseekV32Config

    def __init__(self, config: DeepseekV32Config):
        DeepseekV32PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekV32DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV32RotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    """DeepSeek V3.2 for causal language modeling."""

    config_class = DeepseekV32Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepseekV32Config):
        DeepseekV32PreTrainedModel.__init__(self, config)
        self.model = DeepseekV32Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


class DeepseekV32ForSequenceClassification(DeepseekV3ForSequenceClassification):
    """DeepSeek V3.2 for sequence classification."""

    config_class = DeepseekV32Config


class DeepseekV32ForTokenClassification(DeepseekV3ForTokenClassification):
    """DeepSeek V3.2 for token classification."""

    config_class = DeepseekV32Config


__all__ = [
    "DeepseekV32Config",
    "DeepseekV32PreTrainedModel",
    "DeepseekV32Model",
    "DeepseekV32ForCausalLM",
    "DeepseekV32ForSequenceClassification",
    "DeepseekV32ForTokenClassification",
]
