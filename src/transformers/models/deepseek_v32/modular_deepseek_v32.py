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
    unsqueeze_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embedding with NON-INTERLEAVED layout.

    This is specifically for the Indexer, which requires non-interleaved RoPE
    (different from the MLA which uses interleaved RoPE).

    The difference is in how dimensions are paired:
    - Interleaved: pairs (0,1), (2,3), (4,5), ...
    - Non-interleaved: pairs (0, dim/2), (1, dim/2+1), ...

    For non-interleaved RoPE, the rotation is applied to pairs of elements at positions
    (i, i+dim/2) rather than adjacent pairs. This means cos/sin should have dimension
    dim/2 to match the split halves of q and k.

    Args:
        q: Query tensor of shape (batch, seq_len, heads, head_dim)
        k: Key tensor of shape (batch, seq_len, heads, head_dim) or (batch, seq_len, 1, head_dim)
        cos: Cosine of rotary angles, shape (batch, seq_len, rope_dim)
        sin: Sine of rotary angles, shape (batch, seq_len, rope_dim)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting (default 2 for heads dim)

    Returns:
        Tuple of rotated (query, key) tensors
    """
    # Non-interleaved: split in half and rotate
    # q = [q1, q2] where q1 is first half, q2 is second half
    # rotated = [q1 * cos - q2 * sin, q1 * sin + q2 * cos]
    q1, q2 = q.chunk(2, dim=-1)
    k1, k2 = k.chunk(2, dim=-1)

    # For non-interleaved RoPE, cos/sin should match the dimension of q1/q2
    # If cos/sin have full dimension (from standard RoPE), slice to half
    half_dim = q1.shape[-1]
    if cos.shape[-1] != half_dim:
        cos = cos[..., :half_dim]
        sin = sin[..., :half_dim]

    # cos/sin shape: (batch, seq, half_dim)
    # For q/k with shape (batch, seq, heads, half_dim), unsqueeze at dim 2
    # Result: (batch, seq, 1, half_dim) which broadcasts with (batch, seq, heads, half_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_embed, k_embed


class DeepseekV32Config(DeepseekV3Config):
    r"""
    Configuration class for DeepSeek V3.2 model.

    DeepSeek V3.2 extends DeepSeek V3 with DeepSeek Sparse Attention (DSA), which uses a Lightning Indexer
    to select top-k tokens for sparse attention, reducing complexity from O(L^2) to O(L*k).

    This config inherits all parameters from [`DeepseekV3Config`] and adds V3.2-specific parameters
    for the Lightning Indexer.

    Args:
        vocab_size (`int`, *optional*, defaults to 129280):
            Vocabulary size of the DeepSeek V3.2 model.
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 128):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 128):
            Number of key_value heads for Grouped Query Attention.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts (always active).
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor for routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Dimension of query/key heads without rotary position embeddings.
        n_group (`int`, *optional*, defaults to 8):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 4):
            Number of groups selected per token for expert routing.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts activated per token.
        first_k_dense_replace (`int`, *optional*, defaults to 3):
            Number of dense layers before switching to MoE layers.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            Scoring function for expert routing. The official V3.2 config uses "sigmoid".
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings (for MLA).
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in attention layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        index_n_heads (`int`, *optional*, defaults to 64):
            Number of heads in the Lightning Indexer.
        index_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each indexer head.
        index_topk (`int`, *optional*, defaults to 2048):
            Number of tokens to select for sparse attention.
        use_sparse_attention (`bool`, *optional*, defaults to `True`):
            Whether to use sparse attention. Set to `False` for dense attention
            (useful for the dense warm-up training stage).
        detach_indexer_input (`bool`, *optional*, defaults to `False`):
            Whether to detach the indexer input from the computational graph.
            Used in Stage 2 training for separate optimization of indexer.

    Example:

    ```python
    >>> from transformers import DeepseekV32Model, DeepseekV32Config

    >>> # Initializing a DeepSeek V3.2 configuration
    >>> configuration = DeepseekV32Config()

    >>> # Initializing a model from the configuration
    >>> model = DeepseekV32Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    Reference:
        - Technical Report: https://api-docs.deepseek.com/news/news251201
        - Official Code: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
    """

    model_type = "deepseek_v32"

    def __init__(
        self,
        # Inherited from DeepseekV3Config
        vocab_size: int = 129280,
        hidden_size: int = 7168,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 61,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 128,
        n_shared_experts: int = 1,
        n_routed_experts: int = 256,
        routed_scaling_factor: float = 2.5,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_nope_head_dim: int = 128,
        n_group: int = 8,
        topk_group: int = 4,
        num_experts_per_tok: int = 8,
        first_k_dense_replace: int = 3,
        norm_topk_prob: bool = True,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        rope_parameters=None,
        rope_interleave: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        # V3.2 specific: Lightning Indexer parameters
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        use_sparse_attention: bool = True,
        detach_indexer_input: bool = False,
        # V3.2 uses sigmoid scoring (explicit in official config: score_func)
        scoring_func: str = "sigmoid",
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            moe_intermediate_size=moe_intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            n_shared_experts=n_shared_experts,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            n_group=n_group,
            topk_group=topk_group,
            num_experts_per_tok=num_experts_per_tok,
            first_k_dense_replace=first_k_dense_replace,
            norm_topk_prob=norm_topk_prob,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            rope_interleave=rope_interleave,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        # DeepSeek V3.2 specific: Lightning Indexer
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.use_sparse_attention = use_sparse_attention
        self.detach_indexer_input = detach_indexer_input
        # V3.2 official config has "score_func": "sigmoid"
        self.scoring_func = scoring_func


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
        # Named to match official weights: wq_b
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # Key projection (single head, broadcast to all heads)
        # Named to match official weights: wk
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)

        # LayerNorm for keys (not RMSNorm, following reference)
        # Named to match official weights: k_norm
        self.k_norm = nn.LayerNorm(self.head_dim)

        # Per-head weight projection
        # Named to match official weights: weights_proj
        self.weights_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        self.softmax_scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_scores: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute top-k token indices for sparse attention.

        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_size]
            q_compressed: Compressed query representation [batch, seq_len, q_lora_rank]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional attention mask
            output_scores: If True, also return the raw index scores

        Returns:
            topk_indices: Indices of selected tokens [batch, seq_len, topk]
            index_scores: (optional) Raw index scores [batch, seq_len, seq_len] if output_scores=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # Query path
        q = self.wq_b(q_compressed)  # [B, S, num_heads * head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Split into RoPE and non-RoPE parts
        q_rope, q_nope = torch.split(
            q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
        )

        # Key path
        k = self.wk(hidden_states)  # [B, S, head_dim]
        k = self.k_norm(k)

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
        weights = self.weights_proj(hidden_states.float())  # [B, S, H]
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

        if output_scores:
            return topk_indices, index_scores
        return topk_indices


def compute_indexer_kl_loss(
    indexer_scores: Tuple[torch.Tensor, ...],
    indexer_kl_targets: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Compute KL-divergence loss between indexer predictions and attention distribution.

    This implements Equation 3 from the DeepSeek V3.2 technical report:
        L_I = sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))

    where:
    - I_{t,s} = raw indexer output scores (from output_indexer_scores=True)
    - p_{t,:} = target distribution (from output_indexer_kl_target=True)

    Args:
        indexer_scores: Tuple of [batch, seq, seq] tensors per layer (raw indexer I_{t,s} scores)
        indexer_kl_targets: Tuple of [batch, seq, seq] tensors per layer (target distribution p_{t,:})

    Returns:
        Scalar tensor with averaged KL loss across layers
    """
    total_loss = 0.0
    num_layers = len(indexer_scores)

    for scores, targets in zip(indexer_scores, indexer_kl_targets):
        # scores: [B, S, S] - raw indexer scores I_{t,s}
        # targets: [B, S, S] - target distribution p_{t,:} (already L1-normalized)

        # Convert indexer scores to log-probabilities
        log_probs = F.log_softmax(scores, dim=-1)

        # KL divergence: D_KL(p || q) = sum(p * log(p/q)) = sum(p * log(p) - p * log(q))
        # Since we have log(q), we compute: sum(p * log(p)) - sum(p * log(q))
        # The first term is negative entropy of p, second is cross-entropy
        # KL = -H(p) - (-CE(p,q)) = CE(p,q) - H(p) = sum(p * (log(p) - log(q)))

        # Compute KL divergence per position
        # targets is p_{t,:}, log_probs is log(softmax(I_{t,:}))
        # We want sum over s: p_{t,s} * (log(p_{t,s}) - log(q_{t,s}))
        # = sum(p * log(p)) - sum(p * log(q))

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        targets_safe = targets + eps

        # KL divergence: sum(p * log(p/q))
        kl_per_position = targets * (torch.log(targets_safe) - log_probs)  # [B, S, S]
        kl_per_query = kl_per_position.sum(dim=-1)  # [B, S] - sum over keys
        kl_loss = kl_per_query.mean()  # Average over batch and query positions

        total_loss = total_loss + kl_loss

    return total_loss / num_layers


class DeepseekV32Attention(DeepseekV3Attention):
    """
    DeepSeek V3.2 Attention with Lightning Indexer for sparse attention.

    Extends DeepseekV3Attention by adding the Lightning Indexer which selects
    top-k tokens for each query position, enabling sparse attention.

    Key differences from V3:
    - Adds Lightning Indexer for token selection
    - Supports sparse attention mask during prefill
    - Falls back to dense attention (V3 behavior) when use_sparse_attention=False
      or during decode (seq_len=1)

    Note: Sparse attention uses eager computation (matching official DeepSeek code).
    When sparse attention is disabled, flash attention and other backends are supported.
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Add the Lightning Indexer (only new component vs V3)
        self.indexer = DeepseekV32Indexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_indexer_scores: bool = False,
        output_indexer_kl_target: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with sparse attention via Lightning Indexer.

        The sparse attention is only applied during prefill (seq_len > 1) when
        use_sparse_attention=True. During decode or when sparse attention is
        disabled, this behaves identically to DeepseekV3Attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Optional causal/padding mask
            past_key_values: Optional KV cache
            cache_position: Optional cache position indices
            output_indexer_scores: If True, return raw indexer scores I_{t,s}
            output_indexer_kl_target: If True, return KL target distribution p_{t,:}

        Returns:
            attn_output: Attention output [batch, seq_len, hidden_size]
            attn_weights: Attention weights (optional)
            indexer_scores: Raw indexer scores [batch, seq, seq] if output_indexer_scores=True
            indexer_kl_target: KL target distribution [batch, seq, seq] if output_indexer_kl_target=True
        """
        batch_size, seq_length = hidden_states.shape[:-1]
        indexer_scores = None
        indexer_kl_target = None

        # Determine if we should use sparse attention
        # Sparse attention only applies during prefill, not decode
        use_sparse = (
            self.config.use_sparse_attention
            and self.q_lora_rank is not None  # Need compressed queries for indexer
            and seq_length > 1  # Only for prefill, not decode
        )

        # If not using sparse attention, delegate to parent (supports flash attention, etc.)
        if not use_sparse:
            result = super().forward(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )
            # Return with None for indexer outputs
            return result[0], result[1] if len(result) > 1 else None, None, None

        # Sparse attention path (eager computation, matching official DeepSeek code)
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # Query path with LoRA compression
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

        # Get top-k indices from the indexer (optionally with scores)
        need_scores = output_indexer_scores or output_indexer_kl_target
        indexer_result = self.indexer(
            hidden_states,
            q_compressed_for_indexer,
            position_embeddings,
            attention_mask,
            output_scores=need_scores,
        )

        if need_scores:
            topk_indices, indexer_scores = indexer_result
        else:
            topk_indices = indexer_result

        # Create sparse attention mask (matching official DeepSeek implementation)
        # topk_indices: [B, S, topk] -> sparse_mask: [B, S, kv_seq_len]
        kv_seq_len = key_states.shape[2]
        sparse_mask = torch.full(
            (batch_size, seq_length, kv_seq_len),
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
        sparse_mask = sparse_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Eager attention computation (matching official DeepSeek code - no flash attention)
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scaling
        attn_weights = attn_weights + sparse_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Compute KL target if requested (before dropout!)
        # Per tech report: "sum across all attention heads, then L1-normalize"
        if output_indexer_kl_target:
            # attn_weights: [B, H, S_q, S_k] - post-softmax attention
            # Sum across heads: [B, S_q, S_k]
            attn_sum = attn_weights.sum(dim=1)
            # L1 normalize along key dimension to get target distribution p_{t,:}
            indexer_kl_target = attn_sum / (attn_sum.sum(dim=-1, keepdim=True) + 1e-10)
            # Detach - target should not receive gradients
            indexer_kl_target = indexer_kl_target.detach()

        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, indexer_scores, indexer_kl_target


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
    "compute_indexer_kl_loss",
]
