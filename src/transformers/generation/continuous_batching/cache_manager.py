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
from collections.abc import Iterator
from math import ceil
from typing import Optional, TypeVar

from .requests import logger


T = TypeVar('T')
def reverse_enumerate(xs: list[T]) -> Iterator[tuple[int, T]]:
    index = len(xs) - 1
    for x in xs[::-1]:
        yield index, x
        index -= 1


class Block:
    """A class to represent a block in the hash table of the block manager. We say that a block is completed when the KV
    cache it points to is fully computed, otherwise it is partial. A block can have a parent, which is the block that
    came before in the sequence. Once a block is computed, it is given a hash, which takes into account the tokens ids 
    of the block and its parent's hash."""

    def __init__(self, id_: int, parent_id: int | None) -> None:
        self.id: int = id_
        self.parent_id: int | None = parent_id
        self.hash: int | None = None
        self.ref_count: int = 1

    def __repr__(self) -> str:
        return f"Block(id={self.id}, parent_id={self.parent_id}, hash={self.hash}, ref_count={self.ref_count})"

    @property
    def is_complete(self) -> bool:
        return self.hash is not None


class BlockManager:
    """A class to manage the number of free blocks and block re-use."""

    def __init__(self, num_blocks: int, block_size: int, use_prefix_sharing: bool) -> None:
        """Initializes the block manager with a given number of blocks (num_blocks)"""
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._uninit_block_ids = deque(range(num_blocks))
        self._init_block_ids: dict[int, None] = {}  # effectively act as an ordered set
        self._use_prefix_sharing = use_prefix_sharing
        # TODO: handle de-allocation for those strutures
        self._hash_to_id: dict[int, int] = {}
        self._id_to_block: dict[int, Block] = {}
        # NOTE: one of those may be redundant
        # TODO: handle case where the last block of a finshed request is not complete

    @property
    def num_free_blocks(self) -> int:
        """Returns the number of free blocks left."""
        return len(self._uninit_block_ids) + len(self._init_block_ids)

    def is_enough_free_blocks(self, n_blocks: int) -> bool:
        # Exit early if there are enough uninitialized blocks
        if len(self._uninit_block_ids) >= n_blocks:
            return True
        # Exit early if even after uninitializing all initialized blocks, there are not enough free blocks
        block_to_unintialize = n_blocks - len(self._uninit_block_ids)
        if len(self._init_block_ids) < block_to_unintialize:
            return False
        # Uninitialize the required amount of blocks
        for _ in range(block_to_unintialize):
            id_to_unintialize = self._init_block_ids.popitem()[0]
            block = self._id_to_block[id_to_unintialize]
            self._hash_to_id.pop(block.hash)
            self._uninit_block_ids.append(id_to_unintialize)
        return True

    def get_free_blocks(self, n_blocks: int, last_block_id: int | None) -> list[int] | None:
        """Returns a free block and mark it as used by removing it from the free blocks queue."""
        if not self.is_enough_free_blocks(n_blocks):
            return None
        allocated_block_ids = [self._uninit_block_ids.popleft() for _ in range(n_blocks)]
        # If we use prefix caching, we keep track of the allocated blocks as partial blocks
        if self._use_prefix_sharing:
            for block_id in allocated_block_ids:
                block = Block(block_id, last_block_id)
                self._id_to_block[block_id] = block  # TODO: we can only store partial block here, and keep the parent referenced as a hash once the plck is complete
                last_block_id = block_id
        # In both cases, we return the allocated block ids
        return allocated_block_ids

    def increase_ref_count(self, block_id: int) -> None:
        """Increases the reference count of a block."""
        block = self._id_to_block[block_id]
        block.ref_count += 1
        if block.ref_count == 1:
            self._init_block_ids.pop(block_id)

    def decrease_ref_count(self, block_id: int) -> None:
        """Decreases the reference count of a block."""
        block = self._id_to_block[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.is_complete:
                self._init_block_ids[block_id] = None
            else:
                self._id_to_block.pop(block_id)
                self._uninit_block_ids.append(block_id)

    def free_blocks(self, blocks: list[int]) -> None:
        """Marks a list of blocks as free. If there is no prefix sharing, we simply add them to the uninitialized blocks
        queue. Otherwise, we mark them as initalized but they can be freed in no uninitialized blocks are lefts."""
        if self._use_prefix_sharing:
            for block_id in blocks:
                self.decrease_ref_count(block_id)
        else:
            self._uninit_block_ids.extend(blocks)


    def mark_blocks_as_computed(
        self,
        num_completed_blocks: int,
        allocated_blocks: list[int],
        prompt_ids: list[int]
    ) -> None:
        # Look for the first complete block, starting from the last block
        parent_hash = None
        incomplete_blocks: list[Block] = []
        for i, block_id in reverse_enumerate(allocated_blocks):
            block = self._id_to_block[block_id]
            if block.is_complete:
                parent_hash = block.hash
                break
            incomplete_blocks.append((i, block))

        # Now go through the incomplete blocks and updated them
        new_parent_id = None
        while incomplete_blocks:
            i, block = incomplete_blocks.pop()

            # If the parent id has been updated, we apply the change
            if new_parent_id is not None:
                block.parent_id = new_parent_id
                new_parent_id = None

            # If we have set the hash for all complete blocks, we can stop
            if num_completed_blocks == 0:
                break

            # Otherwise, we compute the hash
            num_completed_blocks -= 1
            tokens = prompt_ids[i * self.block_size : (i + 1) * self.block_size]
            block.hash = hash((parent_hash, tuple(tokens)))

            existing_block_id = self._hash_to_id.get(block.hash)
            # If the block hash is already in the hash to id mapping, we reference the existing block instead
            if existing_block_id is not None:
                allocated_blocks[i] = existing_block_id
                self._id_to_block[existing_block_id].ref_count += 1
                new_parent_id = existing_block_id
                self.free_blocks([block.id])

            # Otherwise, we add the completed block to the hash table
            else:
                self._hash_to_id[block.hash] = block.id

            # Update loop variables
            parent_hash = block.hash

class CacheAllocator(ABC):
    """Abstract base class for cache managers. Cache managers keep track of per-request cache allocations, determine
    when a new physical block needs to be allocated and compute physical indices for reading or writing to the cache."""

    _index: int
    block_table: dict[str, list[int]]  # request_id -> list of block_ids allocated to the request

    @abstractmethod
    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> Optional[int]:
        """Allocates n_blocks for a given request_id. Returns the num of blocks allocated if successful and None
        otherwise."""

    def free_blocks(self, request_id: str, block_manager: BlockManager) -> None:
        """Frees all blocks associated with a request_id."""
        if request_id in self.block_table:
            blocks_to_free = self.block_table.pop(request_id)
            block_manager.free_blocks(blocks_to_free)
        else:
            logger.warning(
                f"CacheAllocator {self._index} attempted to free blocks for non-existent request_id: {request_id}"
            )

    @abstractmethod
    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache in the cache tensor."""

    @abstractmethod
    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to write request_id's cache in the cache tensor."""

    @abstractmethod
    def get_seqlens_k(self, request_id: str, past_length: int, query_length: int) -> tuple[str, int]:
        """Returns the attention type of the cache allocator and the key sequence length for the given request_id."""

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
        self.block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> Optional[int]:
        """Allocate blocks for a given request_id. Returns the number of blocks allocated if successful and None
        otherwise. For group of full attention layers, we always allocate the number of requested blocks."""
        # Make sure the request_id is in the block table and get the first block id
        if request_id not in self.block_table:
            self.block_table[request_id] = []  # TODO: check the impact of making this a deque
            last_block_id = None
        else:
            last_block_id = self.block_table[request_id][-1]
        # Actual allocation, return early if failed
        allocated_blocks = block_manager.get_free_blocks(n_blocks, last_block_id)
        if allocated_blocks is None:
            return None
        self.block_table[request_id].extend(allocated_blocks)
        return n_blocks

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache. For a group of full attention layers, we
        first write the new cache to the cache tensor and then read the entire cache from the beginning to the end."""
        # Retrieve the block table for the request and raise an error if it doesn't exist
        block_table = self.block_table.get(request_id)
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
        block_table = self.block_table.get(request_id)
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
        self.block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> Optional[int]:
        """Allocate blocks for a given request_id. Returns the number of blocks allocated if successful and None
        otherwise. For group of sliding window attention layers, we only allocate up to the point where we can fit an
        entire sliding window in the cache tensor."""
        if request_id not in self.block_table:
            self.block_table[request_id] = []
        # Early return if we are already at the max number of blocks per request
        already_allocated = len(self.block_table[request_id])
        if already_allocated == self._max_blocks_per_request:
            return 0
        # Compute actual number of blocks to allocate
        after_allocation = min(already_allocated + n_blocks, self._max_blocks_per_request)
        actual_n_blocks = after_allocation - already_allocated
        # Classic allocation
        allocated_blocks = block_manager.get_free_blocks(actual_n_blocks, None)  # no prefix caching w/ sliding window
        if allocated_blocks is None:
            return None
        self.block_table[request_id].extend(allocated_blocks)
        return actual_n_blocks

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache in the cache tensor.
        For a group of sliding window attention layers, we read from the cache tensor before writing on it, because the
        new cache can overwrite the old one. To form the cache + new key / values states, we read the at most
        sliding_window - 1 cache page and then manually add the new key / values states after. Hence the -1 indices
        which indicate where to store the new key or values indices."""
        # Retrieve the block table for the request and raise an error if it doesn't exist
        block_table = self.block_table.get(request_id)
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
        block_table = self.block_table.get(request_id)
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
