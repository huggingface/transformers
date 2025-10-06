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

import math
import os

import torch

from ..utils.import_utils import is_torch_npu_available


if is_torch_npu_available():
    from torch_npu import npu_fusion_attention


# FlashAttention2 is supported on Ascend NPU with down-right aligned causal mask by default.
# Set environment variable `NPU_FA2_SPARSE_MODE` to 2 when using top-left aligned causal mask.
TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE = 2
DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE = 3

SPARSE_MODE = int(os.getenv("NPU_FA2_SPARSE_MODE", default=DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE))
if SPARSE_MODE not in [TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE, DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE]:
    raise ValueError(
        "Environment variable `NPU_FA2_SPARSE_MODE` can only be set as 2 (top-left aligned causal mask) "
        "or 3 (down-right aligned causal mask)."
    )

ATTN_MASK_NPU_CACHE = {}


def get_attn_mask_npu(device):
    """Get or create attention mask for the specified device."""
    if device not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[device] = torch.triu(torch.ones([2048, 2048], device=device), diagonal=1).bool()
    return ATTN_MASK_NPU_CACHE[device]


def is_npu_fa2_top_left_aligned_causal_mask():
    return SPARSE_MODE == TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE if is_torch_npu_available() else False


def npu_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    keep_prob = 1.0 - dropout_p

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if not causal:
        head_num = q.shape[2]
        output = npu_fusion_attention(q, k, v, head_num, "BSND", keep_prob=keep_prob, scale=softmax_scale)[0]
    else:
        attn_mask_npu = get_attn_mask_npu(q.device)
        head_num = q.shape[2]
        output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            "BSND",
            keep_prob=keep_prob,
            scale=softmax_scale,
            atten_mask=attn_mask_npu,
            sparse_mode=SPARSE_MODE,
        )[0]

    return output


def npu_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,  # defined for aligning params order with corresponding function in `flash-attn`
    max_seqlen_k=None,  # defined for aligning params order with corresponding function in `flash-attn`
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    keep_prob = 1.0 - dropout_p

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if not causal:
        head_num = q.shape[1]
        output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            atten_mask=None,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
        )[0]
    else:
        attn_mask_npu = get_attn_mask_npu(q.device)
        head_num = q.shape[1]
        output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            padding_mask=None,
            atten_mask=attn_mask_npu,
            scale=softmax_scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
            actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()),
            sparse_mode=SPARSE_MODE,
        )[0]

    return output
