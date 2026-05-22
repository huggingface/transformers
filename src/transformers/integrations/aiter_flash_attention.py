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

"""Flash Attention wrappers for AMD ROCm using AITER's Triton MHA kernels."""

import torch


def aiter_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    s_aux=None,
    **kwargs,
):
    """
    q: (total_q, nheads, headdim)
    k: (total_k, nheads_k, headdim)
    v: (total_k, nheads_k, headdim)
    s_aux: (nheads,) learnable sink scores, or None
    """
    from aiter.ops.triton.attention.mha import flash_attn_varlen_func

    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q.to(torch.int32),
        cu_seqlens_k=cu_seqlens_k.to(torch.int32),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        sink=s_aux,
    )


def aiter_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    s_aux=None,
    **kwargs,
):
    """
    q: (batch_size, seqlen, nheads, headdim)
    k: (batch_size, seqlen, nheads_k, headdim)
    v: (batch_size, seqlen, nheads_k, headdim)
    s_aux: (nheads,) learnable sink scores, or None
    """
    from aiter.ops.triton.attention.mha import flash_attn_func

    return flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        sink=s_aux,
    )
