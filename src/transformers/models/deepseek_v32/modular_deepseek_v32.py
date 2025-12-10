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

Training API
------------

The model supports two-stage training following the DeepSeek V3.2 technical report:

**Stage 1 (SFT - Supervised Fine-Tuning):**
    Train the main model with frozen indexer. Use standard `outputs.loss`.

    ```python
    model.config.indexer_kl_coef = 0.0  # Disable KL loss
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss  # Pure LM loss
    loss.backward()
    ```

**Stage 2 (Indexer Training):**
    Train the indexer to match the attention distribution using KL divergence loss.
    Two approaches are supported:

    *Option A: Joint training with combined loss*
    ```python
    model.config.indexer_kl_coef = 0.1  # Enable KL loss
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss  # lm_loss + 0.1 * indexer_kl_loss
    loss.backward()
    ```

    *Option B: Dual LoRA with separate backward passes (for PEFT/verl)*
    ```python
    # Create separate optimizers for LLM and indexer parameters
    llm_optimizer = AdamW([p for n, p in model.named_parameters() if "indexer" not in n])
    indexer_optimizer = AdamW([p for n, p in model.named_parameters() if "indexer" in n])

    # Forward pass with explicit indexer output request
    model.config.indexer_kl_coef = 1.0  # Or set output_indexer_* explicitly
    outputs = model(
        input_ids,
        labels=labels,
        output_indexer_scores=True,
        output_indexer_kl_target=True,
    )

    # Backward pass 1: LM loss -> LLM parameters
    llm_optimizer.zero_grad()
    outputs.lm_loss.backward(retain_graph=True)
    llm_optimizer.step()

    # Backward pass 2: KL loss -> Indexer parameters
    indexer_optimizer.zero_grad()
    outputs.indexer_kl_loss.backward()
    indexer_optimizer.step()
    ```

Output Fields
-------------

The model returns `CausalLMOutputWithIndexer` with the following fields:

- `loss`: Combined loss = lm_loss + indexer_kl_coef * indexer_kl_loss (when labels provided)
- `lm_loss`: Pure language modeling cross-entropy loss
- `indexer_kl_loss`: KL divergence loss D_KL(attention_dist || indexer_dist)
- `logits`: Prediction logits [batch, seq, vocab]
- `past_key_values`: KV cache for generation
- `hidden_states`: Optional tuple of hidden states per layer
- `attentions`: Optional tuple of attention weights per layer

Config Options
--------------

- `indexer_kl_coef` (float, default=0.0): Coefficient for KL loss in combined loss.
  Set to 0 for Stage 1 (SFT), > 0 for Stage 2 (indexer training).
- `detach_indexer_input` (bool, default=False): Whether to detach indexer input
  from the computational graph. Used in Stage 2 for separate optimization.
- `use_sparse_attention` (bool, default=True): Whether to use sparse attention.
  Set to False for dense warm-up training.
"""
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from ...processing_utils import Unpack
from ...utils import auto_docstring, logging
from ...utils.import_utils import is_hadamard_available


# Custom argument documentation for the indexer-specific parameters
DEEPSEEK_V32_INDEXER_ARGS = r"""
        output_indexer_scores (`bool`, *optional*):
            Whether to return raw indexer scores I_{t,s} from each layer. These are used
            for computing the KL divergence loss during indexer training. Auto-enabled
            when `config.indexer_kl_coef > 0`.
        output_indexer_kl_target (`bool`, *optional*):
            Whether to return KL target distributions p_{t,:} from each layer. These are
            the L1-normalized attention distributions used as targets for KL loss.
            Auto-enabled when `config.indexer_kl_coef > 0`.
"""


@dataclass
class CausalLMOutputWithIndexer(ModelOutput):
    """
    Causal language model outputs with indexer KL loss for DeepSeek V3.2.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Combined loss (lm_loss + indexer_kl_coef * indexer_kl_loss when indexer_kl_coef > 0).
        lm_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Pure language modeling loss (for next-token prediction).
        indexer_kl_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            KL divergence loss between indexer predictions and attention distribution.
            Used for training the Lightning Indexer in Stage 2.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head.
        past_key_values (`Cache`, *optional*):
            Pre-computed key/value states for fast decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states at each layer output.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights after softmax.
    """

    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    indexer_kl_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BaseModelOutputWithIndexer(ModelOutput):
    """
    Base model outputs with indexer scores for DeepSeek V3.2.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Hidden states at the output of the last layer.
        past_key_values (`Cache`, *optional*):
            Pre-computed key/value states for fast decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states at each layer output.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights after softmax.
        indexer_scores (`tuple(torch.FloatTensor)`, *optional*):
            Raw indexer scores I_{t,s} from each layer.
        indexer_kl_targets (`tuple(torch.FloatTensor)`, *optional*):
            KL target distributions p_{t,:} from each layer.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    indexer_scores: Optional[Tuple[torch.FloatTensor, ...]] = None
    indexer_kl_targets: Optional[Tuple[torch.FloatTensor, ...]] = None


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
    DeepseekV3NaiveMoe,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb_interleave,
    yarn_get_mscale,
)


logger = logging.get_logger(__name__)

# Import fast_hadamard_transform if available, otherwise use fallback
if is_hadamard_available():
    from fast_hadamard_transform import hadamard_transform
else:
    hadamard_transform = None


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

    if is_hadamard_available():
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
    Per the tech report, DSA is the ONLY architectural difference from V3.

    V3.2-specific Args (see [`DeepseekV3Config`] for inherited parameters):
        index_n_heads (`int`, *optional*, defaults to 64): Number of heads in the Lightning Indexer.
        index_head_dim (`int`, *optional*, defaults to 128): Dimension of each indexer head.
        index_topk (`int`, *optional*, defaults to 2048): Number of tokens to select for sparse attention.
        use_sparse_attention (`bool`, *optional*, defaults to `True`): Whether to use sparse attention.
        detach_indexer_input (`bool`, *optional*, defaults to `False`): Detach indexer input from graph.
        indexer_kl_coef (`float`, *optional*, defaults to 0.0): Coefficient for indexer KL loss.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`): V3.2 uses sigmoid scoring.

    Reference: https://api-docs.deepseek.com/news/news251201
    """

    model_type = "deepseek_v32"

    def __init__(
        self,
        # Inherited from DeepseekV3Config (required for modular converter)
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
        indexer_kl_coef: float = 0.0,
        scoring_func: str = "sigmoid",  # V3.2 uses sigmoid
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
        # V3.2 specific: Lightning Indexer
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.use_sparse_attention = use_sparse_attention
        self.detach_indexer_input = detach_indexer_input
        self.indexer_kl_coef = indexer_kl_coef
        self.scoring_func = scoring_func


class DeepseekV32RMSNorm(DeepseekV3RMSNorm):
    """RMSNorm for DeepSeek V3.2, inherited from V3.

    Per the V3.2 tech report, the only architectural difference from V3 is
    DeepSeek Sparse Attention (DSA). All other components are identical.
    """

    pass


class DeepseekV32RotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


class DeepseekV32MLP(DeepseekV3MLP):
    pass


class DeepseekV32TopkRouter(DeepseekV3TopkRouter):
    pass


class DeepseekV32NaiveMoe(DeepseekV3NaiveMoe):
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
        weights = self.weights_proj(hidden_states)  # [B, S, H]
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
    num_layers = len(indexer_scores)
    # Initialize total_loss on same device as scores to avoid device mismatch
    total_loss = torch.tensor(0.0, device=indexer_scores[0].device, dtype=indexer_scores[0].dtype)

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

        # Move kl_loss to same device as total_loss for multi-GPU compatibility
        total_loss = total_loss + kl_loss.to(total_loss.device)

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

        # If not using sparse attention, use dense attention (V3 path)
        # This handles decode (seq_len=1) and when use_sparse_attention=False
        if not use_sparse:
            # Call parent's dense attention and add None for indexer outputs
            attn_output, attn_weights = DeepseekV3Attention.forward(
                self,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            return attn_output, attn_weights, None, None

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_indexer_scores: bool = False,
        output_indexer_kl_target: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for the decoder layer.

        Returns:
            Tuple of (hidden_states, attn_weights, indexer_scores, indexer_kl_target)
            - hidden_states: Output hidden states
            - attn_weights: Attention weights (optional, for output_attentions)
            - indexer_scores: Raw indexer scores I_{t,s} (optional, for KL loss)
            - indexer_kl_target: KL target distribution p_{t,:} (optional, for KL loss)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - V3.2 attention returns 4 values
        hidden_states, attn_weights, indexer_scores, indexer_kl_target = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            output_indexer_scores=output_indexer_scores,
            output_indexer_kl_target=output_indexer_kl_target,
            **kwargs,
        )
        # Let FSDP/ZeRO-3 manage device placement - no explicit .to() calls
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # Let FSDP/ZeRO-3 manage device placement - no explicit .to() calls
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, indexer_scores, indexer_kl_target


class DeepseekV32PreTrainedModel(DeepseekV3PreTrainedModel):
    """Base class for DeepSeek V3.2 models."""

    config_class = DeepseekV32Config
    _no_split_modules = ["DeepseekV32DecoderLayer"]

    # No custom _init_weights needed - V3.2 inherits from V3 which handles
    # RMSNorm (uses torch.ones), TopkRouter, and NaiveMoe initialization


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

    @auto_docstring(custom_args=DEEPSEEK_V32_INDEXER_ARGS)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_indexer_scores: Optional[bool] = None,
        output_indexer_kl_target: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithIndexer:
        from ...cache_utils import DynamicCache
        from ...masking_utils import create_causal_mask

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Auto-enable indexer outputs if KL loss is configured
        if output_indexer_scores is None:
            output_indexer_scores = self.config.indexer_kl_coef > 0
        if output_indexer_kl_target is None:
            output_indexer_kl_target = self.config.indexer_kl_coef > 0

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        # Accumulate outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_indexer_scores = () if output_indexer_scores else None
        all_indexer_kl_targets = () if output_indexer_kl_target else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # V3.2 decoder layer returns 4 values
            hidden_states, attn_weights, indexer_scores, indexer_kl_target = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                output_indexer_scores=output_indexer_scores,
                output_indexer_kl_target=output_indexer_kl_target,
                **kwargs,
            )

            if output_attentions and attn_weights is not None:
                all_attentions = all_attentions + (attn_weights,)

            if output_indexer_scores and indexer_scores is not None:
                all_indexer_scores = all_indexer_scores + (indexer_scores,)

            if output_indexer_kl_target and indexer_kl_target is not None:
                all_indexer_kl_targets = all_indexer_kl_targets + (indexer_kl_target,)

        hidden_states = self.norm(hidden_states)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return BaseModelOutputWithIndexer(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            indexer_scores=all_indexer_scores,
            indexer_kl_targets=all_indexer_kl_targets,
        )


class DeepseekV32ForCausalLM(DeepseekV3ForCausalLM):
    """DeepSeek V3.2 for causal language modeling with indexer KL loss support."""

    config_class = DeepseekV32Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DeepseekV32Config):
        DeepseekV32PreTrainedModel.__init__(self, config)
        self.model = DeepseekV32Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring(custom_args=DEEPSEEK_V32_INDEXER_ARGS)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_indexer_scores: Optional[bool] = None,
        output_indexer_kl_target: Optional[bool] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> CausalLMOutputWithIndexer:
        # Auto-enable indexer outputs if KL loss is configured
        compute_kl_loss = self.config.indexer_kl_coef > 0
        if output_indexer_scores is None:
            output_indexer_scores = compute_kl_loss
        if output_indexer_kl_target is None:
            output_indexer_kl_target = compute_kl_loss

        outputs: BaseModelOutputWithIndexer = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_indexer_scores=output_indexer_scores,
            output_indexer_kl_target=output_indexer_kl_target,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Compute language modeling loss
        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # Compute indexer KL loss if we have the required outputs
        indexer_kl_loss = None
        if (
            outputs.indexer_scores is not None
            and outputs.indexer_kl_targets is not None
            and len(outputs.indexer_scores) > 0
            and len(outputs.indexer_kl_targets) > 0
        ):
            indexer_kl_loss = compute_indexer_kl_loss(
                outputs.indexer_scores,
                outputs.indexer_kl_targets,
            )

        # Compute combined loss
        loss = None
        if lm_loss is not None:
            loss = lm_loss
            if indexer_kl_loss is not None and self.config.indexer_kl_coef > 0:
                # Move indexer_kl_loss to same device as loss for multi-GPU compatibility
                loss = loss + self.config.indexer_kl_coef * indexer_kl_loss.to(loss.device)

        return CausalLMOutputWithIndexer(
            loss=loss,
            lm_loss=lm_loss,
            indexer_kl_loss=indexer_kl_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


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
