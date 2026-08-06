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

import torch

from ...configuration_utils import PreTrainedConfig
from ..cache import find_head_dim
from .cache_allocator import CacheAllocator


class FullAttentionCacheAllocator(CacheAllocator):
    """Cache allocator for a group of full attention layers."""

    supports_block_sharing = True

    def __init__(
        self,
        config: PreTrainedConfig,
        num_key_value_heads: int,
        cache_dtype: torch.dtype,
        page_size: int,
        layer_indices: list[int],
    ) -> None:
        """Initializes the cache manager for a group of full attention layers.

        Args:
            - config: the configuration of the model, used to determine the number of bytes per token
            - num_key_value_heads: the number of key value heads, accounting for TP if enabled
            - cache_dtype: the dtype of the cache, also used to determine the number of bytes per token
            - page_size: the number of tokens per page
            - layer_indices: the indices of the layers which cache is handled by this allocator
        """
        # Compute the number of bytes in a page
        self.head_dim = find_head_dim(config)
        self.num_key_value_heads = num_key_value_heads
        self.cache_dtype = cache_dtype
        bytes_per_token = 2 * num_key_value_heads * self.head_dim * cache_dtype.itemsize  # 2 for keys + values
        bytes_per_page = bytes_per_token * page_size
        # Used to compute offsets later on
        self.layer_id_to_offset = {}
        for i, layer_idx in enumerate(layer_indices):
            self.layer_id_to_offset[layer_idx] = 2 * page_size * (num_key_value_heads * self.head_dim) * i

        super().__init__(page_size=page_size, bytes_per_page=bytes_per_page, num_layers=len(layer_indices))

    def register_cache_tensor(self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor) -> None:
        """Registers the cache tensor so the allocator can use it for updates. For full attention KV cache allocator
        with 2 layers, the cache is arranged this way:

        [ LAYER 0 KEYS | LAYER 0 VALUES | LAYER 1 KEYS | LAYER 1 VALUES | LAYER 2 KEYS | LAYER 2 VALUES | --------... ]
        [ ---------- PAGE 0 ----------- | ----------- PAGE 1 ---------- | ---------- PAGE 2 ----------- | --------... ]
        [ -------------------------- BLOCK 0 -------------------------- | -------------------------- BLOCK 1 -----... ]

        """
        # Infer cache shape
        num_pages = non_trash_bytes // self.bytes_per_page
        num_blocks = num_pages // self.num_layers
        total_pages = num_pages + self._num_trash_pages  # for trash read and write
        cache_shape = (total_pages * self.tokens_per_page, self.num_key_value_heads, self.head_dim)
        # View the cache with the right dtype and shape
        numel = num_pages * self.tokens_per_page * self.num_key_value_heads * self.head_dim
        cache_tensor = cache_tensor.view(cache_shape)
        cache_tensor = cache_tensor[:numel]
        # Finalize the initialization by register common attributes
        self._finalize_init(num_pages, num_blocks, bytes_per_sector, cache_tensor)

    # _________________________________________________ BLOCK LEVEL __________________________________________________ #

    def _compute_blocks_needed(self, allocated_blocks: int, past_length: int, query_length: int) -> int:
        """Computes the number of blocks needed to store the new tokens for a given request."""
        total_length = past_length + query_length
        total_blocks = (total_length + self.tokens_per_page - 1) // self.tokens_per_page
        return max(0, total_blocks - allocated_blocks)

    def needs_new_blocks(self, request_id: str, past_length: int, query_length: int) -> int:
        """Returns the number of new blocks needed to store the new tokens for a given request. It can be zero."""
        allocated_blocks = len(self.block_table.get(request_id, []))
        return self._compute_blocks_needed(allocated_blocks, past_length, query_length)

    def allocate_cache_to_request(self, request_id: str, past_length: int, query_length: int) -> None:
        """Allocates enough cache to the request to store the new tokens."""
        if request_id not in self.block_table:
            self.block_table[request_id] = []
        block_table = self.block_table[request_id]
        blocks_needed = self._compute_blocks_needed(len(block_table), past_length, query_length)
        block_table.extend([self.free_block_ids.pop() for _ in range(blocks_needed)])

    # ______________________________________________ INPUT PREPARATION _______________________________________________ #

    def get_seqlen_k(self, past_length: int, query_length: int) -> int:
        return past_length + query_length

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache. For a group of full attention layers, we
        first write the new cache to the cache tensor and then read the entire cache from the beginning to the end."""
        # Retrieve the block table for the request and raise an error if it doesn't exist
        block_table = self.block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        # Compute auxiliary variable so we can perform only two loops
        total_length = past_length + query_length
        num_full_pages = total_length // self.tokens_per_page
        remainder = total_length % self.tokens_per_page
        # Compute the physical indices
        physical_indices = []
        for b in range(num_full_pages):
            start = block_table[b] * self.tokens_per_block
            physical_indices.extend(range(start, start + self.tokens_per_page))
        if remainder:
            start = block_table[num_full_pages] * self.tokens_per_block
            physical_indices.extend(range(start, start + remainder))
        return physical_indices

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices for writing to the cache. For a group of full attention layers, we write the new
        cache as a continuation of the existing cache for the same request."""
        block_table = self.block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        # Compute auxiliary variables so we can perform only one loop
        start_page = past_length // self.tokens_per_page
        start_offset = past_length % self.tokens_per_page
        end_pos = past_length + query_length
        end_page = (end_pos - 1) // self.tokens_per_page  # -1 because if end_pos == page_size, we still end on page 0
        # Compute the physical indices
        physical_indices = []
        for b in range(start_page, end_page + 1):
            physical_start = block_table[b] * self.tokens_per_block
            # First block may start mid-block, last block may end mid-block
            local_start = start_offset if b == start_page else 0
            local_end = (end_pos - 1) % self.tokens_per_page + 1 if b == end_page else self.tokens_per_page
            physical_indices.extend(range(physical_start + local_start, physical_start + local_end))
        return physical_indices

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills the block table for a given request_id, past_length and query_length."""
        raise NotImplementedError("Not implemented for full attention cache allocator")

    # ________________________________________________ RUNTIME UPDATE ________________________________________________ #

    def update(
        self,
        key_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_kv, head_dim]
        value_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_kv, head_dim]
        layer_idx: int,
        read_index: torch.Tensor,  # shape [seqlen_kv + past_length]
        write_index: torch.Tensor,  # shape [seqlen_q]
    ) -> tuple[torch.Tensor, torch.Tensor]:  # shape [seqlen_kv + past_length, num_kv_heads, head_dim]
        """
        Update the cache with new key-value states for a specific layer, and retrieves the relevant KV states from
        the cache for attention computation. For full attention layers, new KV states are written to cache, then
        complete sequence is read from cache.

        When the layer's read index is empty, the batch has no cache reads (all requests are non-chunked prefills): we
        only write to the cache and return the input KV states directly, skipping the index_select read-back.

        Returns the complete KV states (cached + new) for attention computation.
        """
        # Select the right offset for this layer
        offset = self.layer_id_to_offset[layer_idx]
        k_cache = self.cache_tensor[offset:]
        v_cache = self.cache_tensor[offset + self.tokens_per_page * self.num_key_value_heads * self.head_dim:]
        # Transpose the key and value states to match the cache shape, after which shape is [seqlen_kv, num_kv_heads, head_dim]
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)

        # Write the newly generated key and value states to the cache
        k_cache.index_copy_(0, write_index, key_states)
        v_cache.index_copy_(0, write_index, value_states)

        # If there is old cache to read, do it afterwards
        if read_index.numel() == 0:
            key_states_with_cache = torch.index_select(k_cache, 0, read_index)
            value_states_with_cache = torch.index_select(v_cache, 0, read_index)

        return key_states_with_cache, value_states_with_cache
