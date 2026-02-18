# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Flash-MLA attention integration for sparse attention with Dynamic Sparse Attention (DSA).

This module provides a wrapper around the flash-mla kernel from kernels-community/flash-mla,
with automatic fallback to flash_attention_2 when input tokens < 2048.
"""

import torch

from ..utils import logging
from .flash_attention import flash_attention_forward, get_target_dtype


logger = logging.get_logger(__name__)

# Minimum sequence length to use flash-mla sparse attention
# Below this threshold, we fall back to flash_attention_2
FLASH_MLA_MIN_SEQ_LEN = 2048


def flash_mla_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Flash-MLA attention forward pass with automatic fallback to flash_attention_2.

    This wrapper handles:
    - Fallback to flash_attention_2 when sequence length < 2048
    - Sparse attention via topk_indices when sequence length >= 2048
    - Tensor layout conversion (transformers BHSD to flash-mla BSHD)
    - head_dim_v padding if needed (flash-mla requires specific dimensions)

    Args:
        module (`torch.nn.Module`):
            The attention module containing config and layer information.
        query (`torch.Tensor`):
            Query tensor of shape `[B, H, S, D]` (BHSD format).
        key (`torch.Tensor`):
            Key tensor of shape `[B, H, T, D]` (BHSD format).
        value (`torch.Tensor`):
            Value tensor of shape `[B, H, T, D_v]` (BHSD format).
        attention_mask (`torch.Tensor | None`):
            Combined attention mask (causal + DSA sparse mask). Used for flash_attention_2 fallback.
        dropout (`float`, optional):
            Dropout probability. Defaults to 0.0.
        scaling (`float | None`, optional):
            Scaling factor for attention scores. Defaults to None.
        sliding_window (`int | None`, optional):
            Sliding window size. Defaults to None.
        softcap (`float | None`, optional):
            Soft cap for attention logits. Defaults to None.
        is_causal (`bool | None`, optional):
            Whether attention is causal. Defaults to None.
        **kwargs:
            Additional keyword arguments, including:
            - topk_indices (`torch.Tensor | None`): Indices for sparse attention from DSA indexer.

    Returns:
        `tuple[torch.Tensor, None]`: Attention output tensor and None (no attention weights).
    """
    # Extract topk_indices from kwargs (used for sparse attention)
    topk_indices = kwargs.pop("topk_indices", None)

    # Get total sequence length from key tensor
    # key shape is [B, H, T, D] in BHSD format
    seq_len = key.shape[2]

    # Fallback to flash_attention_2 when sequence length is below threshold
    # This is because flash-mla sparse attention is optimized for longer sequences
    if seq_len < FLASH_MLA_MIN_SEQ_LEN:
        logger.debug_once(
            f"Sequence length {seq_len} < {FLASH_MLA_MIN_SEQ_LEN}, falling back to flash_attention_2"
        )
        return flash_attention_forward(
            module=module,
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            dropout=dropout,
            scaling=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            is_causal=is_causal,
            **kwargs,
        )

    # Use flash-mla sparse attention with topk_indices
    return _flash_mla_sparse_forward(
        module=module,
        query=query,
        key=key,
        value=value,
        topk_indices=topk_indices,
        dropout=dropout,
        scaling=scaling,
        **kwargs,
    )


def _flash_mla_sparse_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Internal function for flash-mla sparse attention computation.

    This function handles the actual flash-mla kernel call with sparse attention
    via topk_indices from the DSA indexer.

    Args:
        module (`torch.nn.Module`):
            The attention module containing config and layer information.
        query (`torch.Tensor`):
            Query tensor of shape `[B, H, S, D]` (BHSD format).
        key (`torch.Tensor`):
            Key tensor of shape `[B, H, T, D]` (BHSD format).
        value (`torch.Tensor`):
            Value tensor of shape `[B, H, T, D_v]` (BHSD format).
        topk_indices (`torch.Tensor | None`):
            Indices for sparse attention from DSA indexer, shape `[B, S, topk]`.
        dropout (`float`, optional):
            Dropout probability. Defaults to 0.0.
        scaling (`float | None`, optional):
            Scaling factor for attention scores. Defaults to None.
        **kwargs:
            Additional keyword arguments.

    Returns:
        `tuple[torch.Tensor, None]`: Attention output tensor and None (no attention weights).
    """
    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "Flash-MLA does not support `output_attentions=True`. "
            "Please set your attention to `eager` if you want this feature."
        )

    # Get batch size and sequence lengths
    batch_size, num_heads, q_len, head_dim = query.shape
    _, _, kv_len, _ = key.shape

    # Convert from BHSD (transformers) to BSHD (flash-mla) format
    # query: [B, H, S, D] -> [B, S, H, D]
    # key: [B, H, T, D] -> [B, T, H, D]
    # value: [B, H, T, D_v] -> [B, T, H, D_v]
    query = query.transpose(1, 2).contiguous()
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    # Handle dtype conversion for flash attention compatibility
    target_dtype = get_target_dtype(query, module)
    if target_dtype is not None:
        query = query.to(target_dtype)
        key = key.to(target_dtype)
        value = value.to(target_dtype)

    # Get the flash-mla kernel function
    # This is loaded via hub_kernels infrastructure
    try:
        from ..integrations.hub_kernels import get_kernel

        flash_mla_kernel = get_kernel("kernels-community/flash-mla")
        flash_mla_sparse_fwd = flash_mla_kernel.flash_mla_sparse_fwd
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            f"Failed to load flash-mla kernel. Make sure kernels-community/flash-mla is available. Error: {e}"
        )

    # Prepare scaling factor
    if scaling is None:
        scaling = head_dim**-0.5

    # Get value head dimension (may differ from query/key head dimension in MLA)
    v_head_dim = value.shape[-1]

    # Flash-MLA may require specific head_dim_v (e.g., 512)
    # Pad if necessary
    flash_mla_v_head_dim = 512
    needs_v_padding = v_head_dim < flash_mla_v_head_dim
    if needs_v_padding:
        value = torch.nn.functional.pad(value, (0, flash_mla_v_head_dim - v_head_dim))

    # Call flash-mla kernel with sparse attention
    # The kernel expects:
    # - q: [B, S, H, D]
    # - k_cache: [B, T, H, D] (or compressed format)
    # - v_cache: [B, T, H, D_v]
    # - topk_indices: [B, S, topk] for sparse attention
    attn_output = flash_mla_sparse_fwd(
        q=query,
        kv=torch.cat([key, value], dim=-1),
        indices=topk_indices,
        sm_scale=scaling,
        topk_length = module.top_k_length if hasattr(module, "top_k_length") else None,
    )

    # Remove padding if we added it
    if needs_v_padding:
        attn_output = attn_output[..., :v_head_dim]

    # Convert back from BSHD to BHSD format
    # attn_output: [B, S, H, D_v] -> [B, H, S, D_v]
    attn_output = attn_output.transpose(1, 2)

    return attn_output, None


__all__ = ["flash_mla_forward"]
