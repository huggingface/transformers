# Copyright 2026 The HuggingFace Inc. team.
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

from math import prod

import torch

from ....configuration_utils import PreTrainedConfig
from .cache_allocator import CacheAllocator
from .cache_pool import CachePool


class LinearAttentionCacheAllocator(CacheAllocator):
    """Cache allocator for a group of linear attention layers.
    Each layer has two cached states: a conv state and a recurrent state, both with fixed shapes. The conv state has
    shape [conv_kernel_size - 1, conv_dim] and the recurrent state has shape [num_v_heads, k_head_dim, v_head_dim].
    For a 3 layer model, with 2 blocks per sector, the cache is arranged this way:

        [ L0 CONV | L0 RECURRENT | L1 CONV | L1 RECURRENT | L2 CONV | L2 RECURRENT | L0 CONV | L0 RECURRENT | ... ]
        [ ------- PAGE 0 ------- | ------- PAGE 1 ------- | ------- PAGE 2 ------- | ------- PAGE 3 ------- | ... ]
        [ ------------------------------- BLOCK 0 -------------------------------- | -------- BLOCK 1 ------- ... ]
        [ --------------------------------------------- SECTOR ---------------------------------------------- ... ]

    The states can be accessed independantly through a view in the main cache tensor.
    """

    supports_block_sharing = False
    supports_block_table = True
    layer_type = "linear_attention"

    rows_per_token = 1  # ?

    def __init__(
        self,
        index: int,
        config: PreTrainedConfig,
        cache_dtype: torch.dtype,
        page_size: int,
        layer_indices: list[int],
        **kwargs,
    ) -> None:
        """Initializes the cache allocator for a group of linear attention layers."""
        self.cache_dtype = cache_dtype

        # Size the conv and recurrent states
        text_config = config.get_text_config()
        num_k_heads = text_config.linear_num_key_heads
        num_v_heads = text_config.linear_num_value_heads
        k_head_dim = text_config.linear_key_head_dim
        v_head_dim = text_config.linear_value_head_dim
        conv_kernel_size = text_config.linear_conv_kernel_dim

        conv_dim = 2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim
        self.conv_state_shape = (conv_dim, conv_kernel_size)
        self.recurrent_state_shape = (num_v_heads, k_head_dim, v_head_dim)

        bytes_per_page = cache_dtype.itemsize * (prod(self.conv_state_shape) + prod(self.recurrent_state_shape))
        self._before_cache_tensor_init(
            index=index,
            layer_indices=layer_indices,
            tokens_per_page=page_size,
            bytes_per_page=bytes_per_page,
            allow_block_sharing=False,
        )

    def register_cache_tensor(
        self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor, pool: CachePool
    ) -> None:
        """Registers the cache tensor so the allocator can use it for updates. TODO"""
        self.bytes_per_sector = bytes_per_sector
        # Byte view of the whole tensor as one row per block, used to copy blocks when forking
        self._copy_view = cache_tensor.view(-1, self.bytes_per_block)

        # Reshape the cache. The views span the entire tensor, including the two leading trash sectors
        total_bytes = non_trash_bytes + 2 * bytes_per_sector
        cache_shape = (total_bytes // self.bytes_per_page, self.bytes_per_page)

        # Cast, truncate, reshape
        cache_tensor = cache_tensor.view(self.cache_dtype)
        cache_tensor = cache_tensor[:prod(cache_shape)]
        cache_tensor = cache_tensor.view(*cache_shape)

        self._after_cache_tensor_init(non_trash_bytes, bytes_per_sector, cache_tensor, pool)

        # Compute the strided views for the conv and recurrent states
        numel_conv = prod(self.conv_state_shape)
        self._conv_state_views: dict[int, torch.Tensor] = {
            layer_idx: self.cache_tensor[i:, :numel_conv].reshape(-1, *self.conv_state_shape)
            for i, layer_idx in enumerate(self.layer_indices)
        }
        self._recurrent_state_views: dict[int, torch.Tensor] = {
            layer_idx: self.cache_tensor[i:, numel_conv:].reshape(-1, *self.recurrent_state_shape)
            for i, layer_idx in enumerate(self.layer_indices)
        }

    # _________________________________________________ BLOCK LEVEL __________________________________________________ #

    def needs_new_blocks(self, request_id: str, past_length: int, query_length: int) -> int:
        """Returns the number of new blocks needed to store the new tokens for a given request. It can be zero."""
        return 1 - len(self.block_table.get(request_id, []))

    def allocate_cache_to_request(self, request_id: str, past_length: int, query_length: int) -> None:
        """Allocates enough cache to the request to store the new tokens."""
        if request_id in self.block_table:
            raise RuntimeError(f"Request {request_id} already has a block for linear attention.")
        self.block_table[request_id] = self.pool.get_free_blocks(self.index, 1)

    # ______________________________________________ INPUT PREPARATION _______________________________________________ #

    def get_seqlen_k(self, past_length: int, query_length: int) -> int:
        return -1

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache. For a group of linear attention layers, we
        read the cache, perform linear attention, and write the modified states back to the cache."""
        block_table = self.block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        return block_table[:]

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices for writing to the cache. For a group of linear attention layers, the write
        indices are the same as the read indices.""" # TODO: de-duplicate
        return self.get_read_indices(request_id, past_length, query_length)

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        raise NotImplementedError("Linear attention does not support block table.")

    # ________________________________________________ RUNTIME UPDATE ________________________________________________ #

    def update(
        self,
        key_states: torch.Tensor,  # shape [seqlen_q, num_kv_heads, head_dim]
        value_states: torch.Tensor,  # shape [seqlen_q, num_kv_heads, head_dim]
        layer_idx: int,
        read_index: torch.Tensor,  # shape [seqlen_q + past_length]
        write_index: torch.Tensor,  # shape [seqlen_q]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Linear attention does not support update.")

    def get_conv_state(self, layer_idx: int, state_idx: int, kwargs: dict[str, Any]) -> torch.Tensor:
        if state_idx != 0:
            raise ValueError(f"State index {state_idx} is not supported for CB.")
        return self.conv_state[layer_idx]

    def get_cache_for_block_table(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Linear attention does not support block table.")
