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

from ....configuration_utils import PreTrainedConfig
from ..utils import find_head_dim
from .cache_allocator import CacheAllocator
from .cache_pool import CachePool


class FullAttentionCacheAllocator(CacheAllocator):
    """Cache allocator for a group of full attention layers."""

    supports_block_sharing = True
    supports_block_table = True
    # One row for the keys and one for the values of each token
    rows_per_token = 2

    def __init__(
        self,
        index: int,
        config: PreTrainedConfig,
        num_key_value_heads: int,
        cache_dtype: torch.dtype,
        page_size: int,
        layer_indices: list[int],
        allow_block_sharing: bool,
    ) -> None:
        """Initializes the cache allocator for a group of full attention layers.

        Args:
            - config: the configuration of the model, used to determine the number of bytes per token
            - num_key_value_heads: the number of key value heads, accounting for TP if enabled
            - cache_dtype: the dtype of the cache, also used to determine the number of bytes per token
            - page_size: the number of tokens per page
            - layer_indices: the indices of the layers which cache is handled by this allocator
            - allow_block_sharing: whether to allow block sharing or not. Can be disabled for diagnostics or perfs.
        """
        self.head_dim = find_head_dim(config)
        self.num_key_value_heads = num_key_value_heads
        self.cache_dtype = cache_dtype
        bytes_per_page = self.get_bytes_per_page(num_key_value_heads, self.head_dim, cache_dtype, page_size)
        self._before_cache_tensor_init(
            index=index,
            layer_indices=layer_indices,
            tokens_per_page=page_size,
            bytes_per_page=bytes_per_page,
            allow_block_sharing=allow_block_sharing,
        )

    def register_cache_tensor(
        self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor, pool: CachePool
    ) -> None:
        """Registers the cache tensor so the allocator can use it for updates. For a full attention KV cache allocator
        with 2 layers, the cache is arranged this way:

        [ LAYER 0 KEYS | LAYER 0 VALUES | LAYER 1 KEYS | LAYER 1 VALUES | LAYER 2 KEYS | LAYER 2 VALUES | -------... ]
        [ ----------- PAGE 0 ---------- | ----------- PAGE 1 ---------- | ----------- PAGE 2 ---------- | -------... ]
        [ -------------------------- BLOCK 0 -------------------------- | ------------------ BLOCK 1 ------------... ]

        The FIRST TWO sectors of the tensor are the trash sectors, shared by all allocators and never allocated from.
        """
        self.bytes_per_sector = bytes_per_sector
        # Byte view of the whole tensor as one row per block, used to copy blocks when forking
        self._copy_view = cache_tensor.view(-1, self.bytes_per_block)

        # Reshape the cache. The views span the entire tensor, including the two leading trash sectors
        total_bytes = non_trash_bytes + 2 * bytes_per_sector
        total_tokens = (total_bytes // self.bytes_per_page) * self.tokens_per_page
        cache_shape = (total_tokens, 2, self.num_key_value_heads, self.head_dim)
        numel = torch.Size(cache_shape).numel()

        cache_tensor = cache_tensor.view(self.cache_dtype)
        cache_tensor = cache_tensor[:numel].view(*cache_shape)
        self._after_cache_tensor_init(non_trash_bytes, bytes_per_sector, cache_tensor, pool)

        # Precompute the per-layer shifted views used by update()
        flattened_view = self.cache_tensor.view(total_tokens * 2, self.num_key_value_heads, self.head_dim)
        self._kv_token_views: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer_idx in enumerate(self.layer_indices):
            k_view = flattened_view[(2 * i + 0) * self.tokens_per_page :]
            v_view = flattened_view[(2 * i + 1) * self.tokens_per_page :]
            self._kv_token_views[layer_idx] = (k_view, v_view)

        # Precompute the same views at page granularity for the decode fast path. The block table kernel requires K and
        # V to hold the same number of pages, so both are truncated to the shorter one (which is always V)
        page_shape = (-1, self.tokens_per_page, self.num_key_value_heads, self.head_dim)
        self._kv_page_views: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx, (k_view, v_view) in self._kv_token_views.items():
            k_pages, v_pages = k_view.view(*page_shape), v_view.view(*page_shape)
            num_pages = min(k_pages.shape[0], v_pages.shape[0])
            self._kv_page_views[layer_idx] = (k_pages[:num_pages], v_pages[:num_pages])

    @classmethod
    def get_bytes_per_page(
        cls, num_key_value_heads: int, head_dim: int, cache_dtype: torch.dtype, page_size: int
    ) -> int:
        """Computes the number of bytes in a full attention page: the keys and values of page_size tokens for one
        layer, hence the 2."""
        return 2 * num_key_value_heads * head_dim * cache_dtype.itemsize * page_size

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
        block_table.extend(self.pool.get_free_blocks(self.index, blocks_needed))

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
            start = block_table[b] * self.block_physical_stride
            physical_indices.extend(range(start, start + self.tokens_per_page))
        if remainder:
            start = block_table[num_full_pages] * self.block_physical_stride
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
            physical_start = block_table[b] * self.block_physical_stride
            # First page may start mid-page, last page may end mid-page
            local_start = start_offset if b == start_page else 0
            local_end = (end_pos - 1) % self.tokens_per_page + 1 if b == end_page else self.tokens_per_page
            physical_indices.extend(range(physical_start + local_start, physical_start + local_end))
        return physical_indices

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills the request's row of the kernel block table."""
        block_ids = self.block_table[request_id]
        pages_stride = self.block_physical_stride // self.tokens_per_page
        entries = torch.tensor(block_ids, dtype=block_table.dtype) * pages_stride
        block_table[: len(block_ids)].copy_(entries, non_blocking=True)

    # ________________________________________________ RUNTIME UPDATE ________________________________________________ #

    def update(
        self,
        key_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_q, head_dim]
        value_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_q, head_dim]
        layer_idx: int,
        read_index: torch.Tensor,  # shape [seqlen_q + past_length]
        write_index: torch.Tensor,  # shape [seqlen_q]
    ) -> tuple[torch.Tensor, torch.Tensor]:  # shape [seqlen_q + past_length, num_kv_heads, head_dim]
        """
        Update the cache with new key-value states for a specific layer, and retrieves the relevant KV states from
        the cache for attention computation. For full attention layers, new KV states are written to cache, then the
        complete sequence is read from cache.

        When the layer's read index is empty, the batch has no cache reads (all requests are non-chunked prefills): we
        only write to the cache and return the input KV states directly, skipping the index_select read-back.

        Returns the complete KV states (cached + new) for attention computation.
        """
        # Select the shifted views of this layer's keys and values
        k_cache, v_cache = self._kv_token_views[layer_idx]
        # Transpose the key and value states to match the cache shape, after which shape is [seqlen_q, num_kv_heads, head_dim]
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)

        # Write the newly computed key and value states to the cache
        k_cache.index_copy_(0, write_index, key_states)
        v_cache.index_copy_(0, write_index, value_states)

        # If there is no old cache to read, the input KV states already contain everything the attention needs
        if read_index.numel() == 0:
            return key_states, value_states

        # Otherwise, read the whole sequence back from the cache
        key_states_with_cache = torch.index_select(k_cache, 0, read_index)
        value_states_with_cache = torch.index_select(v_cache, 0, read_index)
        return key_states_with_cache, value_states_with_cache

    def get_cache_for_block_table(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the K and V cache views for a block table update."""
        return self._kv_page_views[layer_idx]
