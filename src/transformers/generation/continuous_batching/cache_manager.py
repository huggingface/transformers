from abc import ABC, abstractmethod
from collections import deque
from math import ceil
from typing import Optional

from .requests import logger


class CacheManager(ABC):
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
            logger.info(
                f"CacheManager {self._index} attempted to free blocks for non-existent request_id: {request_id}"
            )

    @abstractmethod
    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Maps logical sequence indices to physical cache indices to read from."""
        pass

    @abstractmethod
    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Maps logical sequence indices to physical cache indices to write to."""
        pass


class FullAttentionCacheManager(CacheManager):
    def __init__(self, index: int, block_size: int) -> None:
        self._index = index
        self.block_size = block_size
        self._block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, free_blocks: deque[int]) -> Optional[int]:
        if len(free_blocks) < n_blocks:
            return None
        if request_id not in self._block_table:
            self._block_table[request_id] = []
        self._block_table[request_id].extend(free_blocks.popleft() for _ in range(n_blocks))
        return n_blocks

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
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


class SlidingAttentionCacheManager(CacheManager):
    def __init__(self, index: int, block_size: int, sliding_window: int) -> None:
        self._index = index
        self.block_size = block_size
        self.sliding_window = sliding_window
        self._max_blocks_per_request = ceil(self.sliding_window / self.block_size)
        self._block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, free_blocks: deque[int]) -> Optional[int]:
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
        # Retrieve the block table for the request and raise an error if it doesn't exist
        block_table = self._block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        # Apply sliding window
        start_index = past_length % self.sliding_window
        cache_length = min(query_length, self.sliding_window)
        # Compute the physical indices
        physical_indices = []
        for i in range(start_index, start_index + cache_length):
            i %= self.sliding_window
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            physical_index = block_table[block_idx] * self.block_size + block_offset
            physical_indices.append(physical_index)
        return physical_indices
