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
"""DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v4 import DeepseekV3RMSNorm
from ..deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4RotaryEmbedding


logger = logging.get_logger(__name__)



def get_grouped_indices(
    valid_keys: torch.Tensor, pool_size: int, partial_pools_are_valid: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given a 2D boolean tensor of valid keys (shape [batch_size, full_cache_len]), returns the grouped indices for
    each pool. This strips out the padding for each sequences and adds dummy groups to make sure all sequences have the
    same number of pools.

    Pooling starts at the first real token, not raw slot 0. This is the part that makes:
        [P, P, A, B, C, D, ...]
    behave like:
        [A, B, C, D, ...]
    for k-pool grouping. For instance, for pool_size = 4, with ✗ as the padding token:

        input                       pool_indices                          valid_pools
        [✗, ✗, A, B, C, D, E, F] -> [ 2,  3,  4,  5], [ 6,  7, -1, -1]    [True, partial_pools_are_valid]
        [A, B, C, D, E, F, G, H] -> [ 0,  1,  2,  3], [ 4,  5,  6,  7]    [True, True]
        [✗, ✗, ✗, ✗, ✗, A, B, C] -> [ 5,  6,  7, -1], [-1, -1, -1, -1]    [True, False]
    """
    batch_size, seq_len = valid_keys.shape
    number_of_pools = (seq_len + pool_size - 1) // pool_size
    device = valid_keys.device

    # Determine the first valid key, accouting for the fact that some sequences may have none (eg. static cache)
    first_valid_index = valid_keys.long().argmax(-1)
    first_valid_index = torch.where(condition=valid_keys.any(-1), input=first_valid_index, other=seq_len)
    # The pool indices are the first valid index + the indices accross all pools
    pool_offsets = torch.arange(number_of_pools * pool_size, device=device)
    pool_offsets = pool_offsets.view(1, number_of_pools, pool_size)
    pool_indices = first_valid_index[:, None, None] + pool_offsets  # [batch_size, num_pools, pool_size]

    # For all indices in the pools, determine if they are valid
    batch_idx = torch.arange(batch_size, device=device)[:, None, None]
    clamped_pool_indices = pool_indices.clamp(0, seq_len - 1)  # avoid index errors
    valid_pool_indices = valid_keys[batch_idx, clamped_pool_indices]
    # ... and within range
    valid_pool_indices = valid_pool_indices & (pool_indices < seq_len)
    # Use this to mask the invalid indices
    pool_indices = pool_indices.masked_fill(~valid_pool_indices, -1)

    # Also a boolean mask indicating which pools are valid. Partial pools validity depend on `partial_pools_are_valid`
    valid_pools = valid_pool_indices.any(-1) if partial_pools_are_valid else valid_pool_indices.all(-1)
    return pool_indices, valid_pools


@auto_docstring(checkpoint="deepseek-ai/DeepSeek-V4.1-Flash")
@strict
class DeepseekV41Config(DeepseekV4Config):
    pass


class DeepseekV41RMSNorm(DeepseekV3RMSNorm):
    pass


class DeepseekV41RotaryEmbedding(DeepseekV4RotaryEmbedding):
    pass


__all__ = [
    "DeepseekV41Config",
]
