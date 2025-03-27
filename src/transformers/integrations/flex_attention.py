"""
Partially inspired by torchtune's flex attention implementation

Citation:
@software{torchtune,
  title = {torchtune: PyTorch's finetuning library},
  author = {torchtune maintainers and contributors},
  url = {https//github.com/pytorch/torchtune},
  license = {BSD-3-Clause},
  month = apr,
  year = {2024}
}
"""
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

from typing import Optional, Tuple, Union

import torch

from ..utils import is_torch_flex_attn_available


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import (
        BlockMask,
        flex_attention,
    )
    from torch.nn.attention.flex_attention import (
        create_block_mask as create_block_causal_mask_flex,
    )


class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        """
        Initialize or update the singleton instance.
        """
        if self._is_flex_compiled is False:
            self._compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def make_flex_block_causal_mask(attention_mask_2d: torch.Tensor) -> "BlockMask":
    """
    Create a block causal document mask for a batch of sequences, both packed and unpacked.
    Create Block causal logic and passing it into :func:`torch.nn.attention.flex_attention.create_block_mask`.
    The resultant BlockMask is a compressed representation of the full block causal
    mask. BlockMask is essential for performant computation of flex attention.
    See: https://pytorch.org/blog/flexattention/

    Args:
        attention_mask_2d (torch.Tensor): Attention mask for packed and padded sequences
        of shape (batch_size, total_seq_len). e.g.

        For unpacked sequence:
        [[1, 1, 1, 1, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0]]

        For packed sequence:
        [[1, 1, 1, 2, 2, 2, 0],
         [1, 1, 2, 2, 2, 3, 3]]

    Returns:
        BlockMask
    """
    device = attention_mask_2d.device

    document_ids = attention_mask_2d
    batch_size, total_seq_len = document_ids.shape

    # Instead of passing a tensor mask, flex attention requires a mask_mod function
    # that determines which elements of QK^T should be included in the attention
    # computation prior to the softmax. For sample packing, we need both the
    # logic for both causal mask and document mask. See PyTorch's official
    # blog post for more details: https://pytorch.org/blog/flexattention/#mask-mods
    def causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        """
        Defines the logic of a block causal mask by combining both a standard causal mask
        and a block diagonal document mask.

        See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
        for an illustration.
        """
        causal_mask = q_idx >= kv_idx
        document_mask = document_ids[batch_idx, q_idx] == document_ids[batch_idx, kv_idx]
        padding_mask = document_ids[batch_idx, q_idx] > 0
        return causal_mask & document_mask & padding_mask

    return create_block_causal_mask_flex(
        mask_mod=causal_mask_mod,
        B=batch_size,
        H=None,  # attention head
        Q_LEN=total_seq_len,
        KV_LEN=total_seq_len,
        device=device,
    )


@torch.compiler.disable(recursive=False)
def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    flex_attention_compiled = WrappedFlexAttention()()
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_mask = None
    causal_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask
    else:
        causal_mask = attention_mask

    if causal_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        if softcap is not None:
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            score = score + causal_mask[batch_idx][0][q_idx][kv_idx]
        if head_mask is not None:
            score = score + head_mask[batch_idx][head_idx][0][0]
        return score

    attn_output, attention_weights = compile_friendly_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=True,
        scale=scaling,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=True,
    )
    # lse is returned in float32
    attention_weights = attention_weights.to(value.dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attention_weights
