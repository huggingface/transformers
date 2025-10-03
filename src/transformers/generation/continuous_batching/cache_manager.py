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
from abc import ABC, abstractmethod
from collections import deque
from math import ceil
from typing import Optional

from .requests import logger


class CacheAllocator(ABC):
    """Abstract base class for cache managers. Cache managers keep track of per-request cache allocations, determine
    when a new physical block needs to be allocated and compute physical indices for reading or writing to the cache."""

    _index: int
    _block_table: dict[str, list[int]]  # request_id -> list of block_ids allocated to the request

    @abstractmethod
    def allocate_blocks(self, n_blocks: int, request_id: str, free_blocks: deque[int]) -> Optional[int]:
        """Allocates n_blocks for a given request_id. Returns the num of blocks allocated if successful and None
        otherwise."""
        pass

    def free_blocks(self, request_id: str, free_blocks: deque[int]) -> None:
        """Frees all blocks associated with a request_id."""
        if request_id in self._block_table:
            blocks_to_free = self._block_table.pop(request_id)
            free_blocks.extend(blocks_to_free)
        else:
            logger.warning(
                f"CacheAllocator {self._index} attempted to free blocks for non-existent request_id: {request_id}"
            )

    @abstractmethod
    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache in the cache tensor."""
        pass

    @abstractmethod
    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to write request_id's cache in the cache tensor."""
        pass

    @abstractmethod
    def get_seqlens_k(self, request_id: str, past_length: int, query_length: int) -> tuple[str, int]:
        """Returns the attention type of the cache allocator and the key sequence length for the given request_id."""
        pass


class FullAttentionCacheAllocator(CacheAllocator):
    """Cache manager for a group of full attention layers."""

    def __init__(self, index: int, block_size: int) -> None:
        """Initializes the cache manager for a group of full attention layers.
        Args:
            - index: the index of the associated layer group
            - block_size: the size of the blocks in the cache
        """
        self._index = index
        self.block_size = block_size
        self._block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, free_blocks: deque[int]) -> Optional[int]:
        """Allocate blocks for a given request_id. Returns the number of blocks allocated if successful and None
        otherwise. For group of full attention layers, we always allocate the number of requested blocks."""
        if len(free_blocks) < n_blocks:
            return None
        if request_id not in self._block_table:
            self._block_table[request_id] = []
        self._block_table[request_id].extend(free_blocks.popleft() for _ in range(n_blocks))
        return n_blocks

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache. For a group of full attention layers, we
        first write the new cache to the cache tensor and then read the entire cache from the beginning to the end."""
        # Retrieve the block table for the request and raise an error if it doesn't exist
        block_table = self._block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        # Compute the physical indices
        physical_indices = []
        for i in range(past_length + query_length):
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            physical_index = block_table[block_idx] * self.block_size + block_offset
            physical_indices.append(physical_index)
        return physical_indices

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices for writing to the cache. For a group of full attention layers, we write the new
        cache as a continuation of the existing cache for the same request."""
        block_table = self._block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        # Compute the physical indices
        physical_indices = []
        for i in range(past_length, past_length + query_length):
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            physical_index = block_table[block_idx] * self.block_size + block_offset
            physical_indices.append(physical_index)
        return physical_indices

    def get_seqlens_k(self, request_id: str, past_length: int, query_length: int) -> tuple[str, int]:
        """Returns the attention type of the cache allocator and the key sequence length for the given request_id."""
        seqlens_k = past_length + query_length
        return "full_attention", seqlens_k


class SlidingAttentionCacheAllocator(CacheAllocator):
    """Cache manager for sliding window attention layers."""

    def __init__(self, index: int, block_size: int, sliding_window: int) -> None:
        """Initializes the cache manager for a group of sliding window attention layers.
        Args:
            - index: the index of the associated layer group
            - block_size: the size of the blocks in the cache
            - sliding_window: the size of the sliding window
        """
        self._index = index
        self.block_size = block_size
        self.sliding_window = sliding_window
        self._max_blocks_per_request = ceil(self.sliding_window / self.block_size)
        self._block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, free_blocks: deque[int]) -> Optional[int]:
        """Allocate blocks for a given request_id. Returns the number of blocks allocated if successful and None
        otherwise. For group of sliding window attention layers, we only allocate up to the point where we can fit an
        entire sliding window in the cache tensor."""
        if request_id not in self._block_table:
            self._block_table[request_id] = []
        # Early return if we are already at the max number of blocks per request
        already_allocated = len(self._block_table[request_id])
        if already_allocated == self._max_blocks_per_request:
            return 0
        # Compute actual number of blocks to allocate
        after_allocation = min(already_allocated + n_blocks, self._max_blocks_per_request)
        actual_n_blocks = after_allocation - already_allocated
        # Classic allocation
        if len(free_blocks) < actual_n_blocks:
            return None
        self._block_table[request_id].extend(free_blocks.popleft() for _ in range(actual_n_blocks))
        return actual_n_blocks

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache in the cache tensor.
        For a group of sliding window attention layers, we read from the cache tensor before writing on it, because the
        new cache can overwrite the old one. To form the cache + new key / values states, we read the at most
        sliding_window - 1 cache page and then manually add the new key / values states after. Hence the -1 indices
        which indicate where to store the new key or values indices."""
        # Retrieve the block table for the request and raise an error if it doesn't exist
        block_table = self._block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        # Apply sliding window
        start_index = 0 if past_length < self.sliding_window else past_length % self.sliding_window
        cache_length = min(past_length, self.sliding_window - 1)
        # Compute the physical indices
        physical_indices = []
        for i in range(start_index, start_index + cache_length):
            i %= self.sliding_window
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            physical_index = block_table[block_idx] * self.block_size + block_offset
            physical_indices.append(physical_index)
        return physical_indices + [-1] * query_length

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to write request_id's cache in the cache tensor. For a group of
        sliding window attention layers, we write the new cache in rolling-buffer kind of way: if we reach the end of
        the allocated physical cache, we start writing from the beginning of the physical cache again."""
        # Retrieve the block table for the request and raise an error if it doesn't exist
        block_table = self._block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        # Apply sliding window
        start_index = past_length % self.sliding_window
        cache_length = min(query_length, self.sliding_window)
        padding_length = query_length - cache_length
        # Compute the physical indices
        physical_indices = []
        for i in range(start_index, start_index + cache_length):
            i %= self.sliding_window
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            physical_index = block_table[block_idx] * self.block_size + block_offset
            physical_indices.append(physical_index)
        if padding_length > 0:
            physical_indices = [-1] * padding_length + physical_indices
        return physical_indices

    def get_seqlens_k(self, request_id: str, past_length: int, query_length: int) -> tuple[str, int]:
        """Returns the attention type of the cache allocator and the key sequence length for the given request_id."""
        seqlens_k = query_length + min(past_length, self.sliding_window - 1)
        return "sliding_attention", seqlens_k


# TODO: test the impact of this
# def get_read_indices(self, request_id: str, past_length: int) -> list[int]:
#     # Retrieve the block table for the request and raise an error if it doesn't exist
#     block_table = self._block_table.get(request_id)
#     if block_table is None:
#         raise ValueError(f"No block table found for request {request_id}")
#     # Compute the physical indices
#     physical_indices = []
#     n_left = past_length
#     for block_idx in block_table:
#         block_physical_index = block_idx * self.block_size
#         pages_used = min(self.block_size, n_left)
#         physical_indices.extend(block_physical_index + i for i in range(pages_used))
#         n_left -= pages_used
#         if n_left == 0:
#             return physical_indices
#     raise ValueError(f"Request {request_id} required too many indices: {past_length = } and {len(block_table) = }")
