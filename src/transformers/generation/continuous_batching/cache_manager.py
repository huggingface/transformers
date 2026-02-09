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
from typing import TypeVar

from .requests import logger


T = TypeVar("T")


def reverse_enumerate(xs: list[T]) -> Iterator[tuple[int, T]]:
    index = len(xs) - 1
    for x in xs[::-1]:
        yield index, x
        index -= 1


class Block:  # TODO: rename to ShareableBlock and update the docs
    """A class to represent a block managed by the block manager. We say that a block is complete when the physical KV
    cache it points to is fully computed. A block can have a parent, which is the block that came before in the
    sequence. Once a block is complete, it is given a hash, which takes into account the tokens ids of the block, the
    layer (group_id) it belong to and its parent's hash (if there is a parent)."""

    def __init__(self, id_: int, parent_id: int | None, group_id: int) -> None:
        self.id: int = id_
        self.parent_id: int | None = parent_id
        self.group_id: int = group_id
        self.hash: int | None = None
        self.ref_count: int = 1

    def __repr__(self) -> str:
        return f"Block(id={self.id}, parent_id={self.parent_id}, group_id={self.group_id}, hash={self.hash}, ref_count={self.ref_count})"

    @property
    def is_complete(self) -> bool:
        return self.hash is not None


class BlockManager:
    """A class to manage the number of free blocks and block re-use. When a block becomes in use, a flag is passed to
    determine if the block is shareable or not. If it is, then a Block object is created and kept track of internally.
    It can have the following states:
      - in use: one or more requests references this block, thus it cannot be written over. The number of requests
        referencing this block is stored as ref_count in the Block object.
      - un-initialized: the block points to a space in the KV cache tensor that contains no data yet. Those blocks can
        be given as free blocks to new requests without any overhead.
      - initialized: the block is complete and was used by one or more request that are finished. It contains KV cache
        data and its hash is stored in the hash table. If a new request needs a block with the same hash, we increase
        the ref_count of the block and remove it from the list of initialized blocks, because it is now in use.
        Still, the block can be freed if no un-initialized blocks are left. In that case, we remove its hash from the
        hash table.
    If the block is not shareable, we just use the block manager as a FIFO structure where blocks are either free or in
    use. Sharability is determined by the type of cache allocator: blocks created for full attention layers are
    shareable, while blocks created for sliding window attention layers are not.
    There is no structure to keep track of the blocks in use: if a block is neither un-initialized nor initialized,
    it is in use.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        """Initializes the block manager with a given number of blocks (num_blocks) of size (block_size)."""
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._uninit_block_ids = deque(range(num_blocks))
        self._init_block_ids: dict[int, None] = {}  # effectively act as an ordered set
        self._hash_to_id: dict[int, int] = {}
        self._id_to_block: dict[int, Block] = {}

    @property
    def num_free_blocks(self) -> int:
        """Returns the number of free blocks left. Both initialized and uninitialized blocks are considered free."""
        return len(self._uninit_block_ids) + len(self._init_block_ids)

    def has_enough_free_blocks(self, n_blocks: int) -> bool:
        """Checks if there are enough free blocks to allocate the requested number of blocks (n_blocks). If there are
        not enough uninitialized blocks, we uninitialize the required number of initialized blocks."""
        # Exit early if there are enough uninitialized blocks
        if len(self._uninit_block_ids) >= n_blocks:
            return True
        # Exit early if even after uninitializing all initialized blocks, there are not enough free blocks
        block_to_uninitialize = n_blocks - len(self._uninit_block_ids)
        if len(self._init_block_ids) < block_to_uninitialize:
            return False
        # Uninitialize the required amount of blocks
        for _ in range(block_to_uninitialize):
            id_to_uninitialize = self._init_block_ids.popitem()[0]
            block = self._id_to_block[id_to_uninitialize]
            # Since the block is initialized it must have a hash, thus no need to check .hash is not None
            self._hash_to_id.pop(block.hash)  # ty:ignore[invalid-argument-type]
            self._uninit_block_ids.append(id_to_uninitialize)
        return True

    def get_free_blocks(
        self, n_blocks: int, last_block_id: int | None, shareable: bool, group_id: int
    ) -> list[int] | None:
        """Returns a list of (n_blocks) free block and mark them as no longuer free in the internal data structures.
        If the (shareable) flag is set to True, a Block object is created to keep track of the block, with the
        (last_block_id) to indicate the last block id in the sequence, also named the parent block. If the manager
        cannot find enough free blocks, it returns None."""
        if not self.has_enough_free_blocks(n_blocks):
            return None
        allocated_block_ids = [self._uninit_block_ids.popleft() for _ in range(n_blocks)]
        # If the block is shareable, we keep track of the allocated blocks as partial blocks
        if shareable:
            for block_id in allocated_block_ids:
                block = Block(block_id, last_block_id, group_id)
                self._id_to_block[block_id] = block
                last_block_id = block_id
        # In both cases, we return the allocated block ids
        return allocated_block_ids

    def fork_blocks(
        self, parent_blocks: list[int], num_forks: int, shareable: bool, group_id: int
    ) -> tuple[list[list[int]] | None, list[int], list[int]]:
        """Fork a given list of (parent_blocks) as many times as (num_forks). If the blocks are (shareable), we use
        reference on the blocks that are complete. Otherwise, we allocate new blocks and keep track of their indices to
        later copy the physical cache. For instance, when forking 4 blocks for 2 children:

        Parent blocks: [0, 1, 2, 3], with all blocks being complete except the last one (block 3).

        ----------------------------------------- IF BLOCKS ARE NOT SHAREABLE -----------------------------------------

        Forked blocks lists: [[5, 6, 7, 8], [9, 10, 11, 12]]
        Copy source:          [0, 1, 2, 3,   0,  1,  2,  3]
                               ↓  ↓  ↓  ↓    ↓   ↓   ↓   ↓
        Copy destination:     [5, 6, 7, 8,   9, 10, 11, 12]  → 8 blocks are newly allocated and copied

        ----------------------------------------- IF BLOCKS ARE SHAREABLE ---------------------------------------------

        Forked blocks lists: [[0, 1, 2, 5], [0, 1, 2, 6]]
        Copy source:          [         3,            3]     (block 3 is not complete so it's copied, not referenced)
                                        ↓             ↓
        Copy destination:     [         5,            6]     → only 2 blocks are newly allocated and copied
        """
        # First phase: reference all complete blocks
        forked_by_reference = []

        if shareable:
            for block_id in parent_blocks:
                block = self._id_to_block[block_id]
                if block.is_complete:
                    forked_by_reference.append(block.id)
                    block.ref_count += num_forks
                else:
                    break

        # Early return if we have forked all blocks by reference
        blocks_to_copy = len(parent_blocks) - len(forked_by_reference)
        if blocks_to_copy == 0:
            return [forked_by_reference[:] for _ in range(num_forks)], [], []

        # From now on, each child will have its own list of blocks
        forked_blocks_lists = []
        copy_src = []
        copy_dst = []

        # Second phase: allocate new blocks if needed
        parent_id = forked_by_reference[-1] if forked_by_reference else None
        for _ in range(num_forks):
            allocated_block_ids = self.get_free_blocks(blocks_to_copy, parent_id, shareable, group_id)
            if allocated_block_ids is None:
                return None, [], []
            forked_blocks_lists.append(forked_by_reference + allocated_block_ids)
            copy_src.extend(parent_blocks[-blocks_to_copy:])
            copy_dst.extend(allocated_block_ids)
        return forked_blocks_lists, copy_src, copy_dst

    def increase_ref_count(self, block_id: int) -> None:
        """Increases the reference count of a given (block_id)."""
        block = self._id_to_block[block_id]
        block.ref_count += 1
        if block.ref_count == 1:
            self._init_block_ids.pop(block_id)

    def decrease_ref_count(self, block_id: int) -> None:
        """Decreases the reference count of a given (block_id). If the reference count reaches 0, the block is no longer
        in use, and becomes initialized (if it was complete) or uninitialized (if it was incomplete)."""
        block = self._id_to_block[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.is_complete:
                self._init_block_ids[block_id] = None
            else:
                self._id_to_block.pop(block_id)
                self._uninit_block_ids.append(block_id)

    def free_blocks(self, blocks: list[int], shareable: bool) -> None:
        """Marks a list of (blocks) as free. If the blocks were not (shareable), we simply add them to the uninitialized
        blocks queue. Otherwise, their new state depends on whether they are complete."""
        if shareable:
            for block_id in blocks:
                self.decrease_ref_count(block_id)
        else:
            self._uninit_block_ids.extend(blocks)

    def uninitialize_unshared_block(self, block_id: int) -> None:
        """Marks a block as uninitialized. Raises an error if the block has more than one reference."""
        # Make sure the block has only one reference and remove it from the block table
        block = self._id_to_block.pop(block_id)
        if block.ref_count > 1:
            raise RuntimeError(f"Block {block_id} has more than one reference: {block.ref_count = }")
        # Add the block to the uninitialized blocks queue
        self._uninit_block_ids.append(block_id)

    def mark_shareable_blocks_as_complete(
        self, num_complete_blocks: int, allocated_blocks: list[int], prompt_ids: list[int]
    ) -> None:
        """Among the list of (allocated_blocks), mark (num_complete_blocks) incomplete blocks as now complete. The list
        of (prompt_ids) is used to compute the hash of the new block."""
        # Look for the first complete block, starting from the last block in the sequence
        parent_hash = None
        incomplete_blocks: list[tuple[int, Block]] = []
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
            if num_complete_blocks == 0:
                break

            # Otherwise, we compute the hash
            num_complete_blocks -= 1
            tokens = prompt_ids[i * self.block_size : (i + 1) * self.block_size]
            block.hash = self.compute_hash(parent_hash, tokens, block.group_id)

            existing_block_id = self._hash_to_id.get(block.hash)
            # If their was a different block with the same hash, we reference the existing block instead
            if existing_block_id is not None:
                if existing_block_id == block.id:
                    # This should not happen, but is not a problem in itself, so we just log a warning
                    logger.warning(f"Block {block.id} was marked as complete more than once")
                else:
                    logger.debug(f"Found existing block {existing_block_id} for block {block.id}")
                    allocated_blocks[i] = existing_block_id
                    new_parent_id = existing_block_id
                    self.increase_ref_count(existing_block_id)
                    self.uninitialize_unshared_block(block.id)

            # Otherwise, we add the completed block to the hash table
            else:
                logger.debug(f"Adding new block {block.id} (group {block.group_id}) with hash {block.hash}")
                self._hash_to_id[block.hash] = block.id

            # Update loop variables
            parent_hash = block.hash

    def compute_hash(self, parent_hash: int | None, tokens: list[int], group_id: int) -> int:
        """Computes the hash of a block identified by the (tokens) it contains, its (parent_hash) and the layer
        (group_id) it belong to. If the block has no parent, the parent hash is None."""
        return hash((parent_hash, tuple(tokens), group_id))


class CacheAllocator(ABC):
    """Abstract base class for cache managers. Cache managers keep track of per-request cache allocations, determine
    when a new physical block needs to be allocated and compute physical indices for reading or writing to the cache."""

    _index: int
    block_table: dict[str, list[int]]  # request_id -> list of block_ids allocated to the request
    uses_block_sharing: bool  # flag to determine if the blocks are shareable

    @abstractmethod
    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> int | None:
        """Allocates (n_blocks) for a given (request_id) using the (block_manager). Returns the num of blocks allocated
        if successful and None otherwise."""

    def free_blocks(self, request_id: str, block_manager: BlockManager) -> None:
        """Frees all blocks associated with a (request_id) using the (block_manager)."""
        if request_id in self.block_table:
            blocks_to_free = self.block_table.pop(request_id)
            block_manager.free_blocks(blocks_to_free, shareable=self.uses_block_sharing)
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

    def fork_blocks(
        self, parent_request_id: str, children_request_ids: list[str], block_manager: BlockManager
    ) -> tuple[list[int], list[int]]:
        """Forks the cache blocks of a (parent_request_id) to a list of (children_request_ids). To manage the blocks,
        the (block_manager) is used. When forking, the child's block are either shared with the parent, or they need to
        be copied from the parent. Hence we return two lists of blocks that need to be copied: one for the source and
        one for the destination."""

        # Sanity checks
        if parent_request_id not in self.block_table:
            raise ValueError(f"No block table found for request {parent_request_id}")

        # Actual forking
        parent_blocks = self.block_table[parent_request_id]
        list_forked_blocks, copy_src, copy_dst = block_manager.fork_blocks(
            parent_blocks=parent_blocks,
            num_forks=len(children_request_ids),
            shareable=self.uses_block_sharing,
            group_id=self._index,
        )
        if list_forked_blocks is None:
            raise ValueError(f"Failed to fork blocks for request {parent_request_id}")

        # Update the block table for all children requests
        for children_request_id, forked_blocks in zip(children_request_ids, list_forked_blocks):
            if children_request_id in self.block_table:
                raise ValueError(f"Block table already exists for request {children_request_id}")
            self.block_table[children_request_id] = forked_blocks
        return copy_src, copy_dst


class FullAttentionCacheAllocator(CacheAllocator):
    """Cache manager for a group of full attention layers."""

    def __init__(self, index: int, block_size: int, allow_block_sharing: bool) -> None:
        """Initializes the cache manager for a group of full attention layers.
        Args:
            - index: the index of the associated layer group
            - block_size: the size of the blocks in the cache
        """
        self._index = index
        self.uses_block_sharing = allow_block_sharing
        self.block_size = block_size
        self.block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> int | None:
        """Allocate (n_blocks) for a given (request_id) using the (block_manager). Returns the number of blocks
        allocated if successful and None otherwise. For group of full attention layers, we always allocate the number of
        requested blocks."""
        # Make sure the request_id is in the block table and get the first block id
        block_table = self.block_table.get(request_id, [])
        if block_table:
            last_block_id = block_table[-1]
        else:
            self.block_table[request_id] = block_table  # TODO: check the impact of making this a deque
            last_block_id = None
        # Actual allocation, return early if failed
        allocated_blocks = block_manager.get_free_blocks(n_blocks, last_block_id, self.uses_block_sharing, self._index)
        if allocated_blocks is None:
            return None
        block_table.extend(allocated_blocks)
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
        self.uses_block_sharing = False
        self.block_size = block_size
        self.sliding_window = sliding_window
        self._max_blocks_per_request = ceil(self.sliding_window / self.block_size)
        self.block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> int | None:
        """Allocate (n_blocks) for a given (request_id) using the (block_manager). Returns the number of blocks
        allocated otherwise. For group of sliding window attention layers, we only allocate up to the point where we can
        fit an entire sliding window in the cache tensor."""
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
        allocated_blocks = block_manager.get_free_blocks(
            actual_n_blocks, None, self.uses_block_sharing, self._index
        )  # no block sharing w/ sliding window
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
