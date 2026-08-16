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
import hashlib
from array import array
from collections import deque
from collections.abc import Iterator
from typing import TypeVar

from ..requests import logger


T = TypeVar("T")


def reverse_enumerate(xs: list[T]) -> Iterator[tuple[int, T]]:
    index = len(xs) - 1
    for x in xs[::-1]:
        yield index, x
        index -= 1


class ShareableBlock:
    """This class represents a block managed by a block manager that can be shared between requests. A block is stored
    on one or more pages, depending on the number of layer groups it belongs to. We say that a block is complete when
    the physical KV cache it points to is fully computed. A block can have a parent, which is the block that came before
    in the sequence. Once a block is complete, it is given a hash, which takes into account the tokens ids of the block,
    the layer group it belongs to and its parent's hash (if there is a parent)."""

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
    """A class to manage the number of free blocks and block re-use inside a cache allocator. When a block becomes in
    use, a flag is passed to determine if the block is shareable or not. If it is, then a Block object is created and
    kept track of internally. It can have the following states:
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

    def __init__(self, num_tokens_per_block: int, shareable: bool, tp_on: bool) -> None:
        """Initializes the block manager with a given number of blocks (num_blocks) of size (block_size)."""
        self.num_tokens_per_block = num_tokens_per_block
        self.shareable = shareable
        self.tp_on = tp_on
        self._uninit_block_ids = deque()  # empty, main allocator will fill it up
        self._init_block_ids: dict[int, None] = {}  # effectively act as an ordered set
        self._hash_to_id: dict[int, int] = {}
        self._id_to_block: dict[int, ShareableBlock] = {}

    def add_free_blocks(self, block_ids: list[int]) -> None:
        """Adds a list of new block_ids to the free blocks queue."""
        self._uninit_block_ids.extend(block_ids)

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

    def get_free_blocks(self, n_blocks: int, last_block_id: int | None) -> list[int] | None:
        """Returns a list of (n_blocks) free block and mark them as no longer free in the internal data structures.
        If the (shareable) flag is set to True, a Block object is created to keep track of the block, with the
        (last_block_id) to indicate the last block id in the sequence, also named the parent block. If the manager
        cannot find enough free blocks, it returns None."""
        if not self.has_enough_free_blocks(n_blocks):
            return None
        allocated_block_ids = [self._uninit_block_ids.popleft() for _ in range(n_blocks)]
        # If the block is shareable, we keep track of the allocated blocks as partial blocks
        if self.shareable:
            for block_id in allocated_block_ids:
                block = ShareableBlock(block_id, last_block_id)
                self._id_to_block[block_id] = block
                last_block_id = block_id
        # In both cases, we return the allocated block ids
        return allocated_block_ids

    def fork_blocks(
        self, parent_blocks: list[int], num_forks: int
    ) -> tuple[list[list[int]] | None, list[int], list[int]]:
        """Fork a given list of (parent_blocks) as many times as (num_forks). If the blocks are shareable, we use
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

        if self.shareable:
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
            allocated_block_ids = self.get_free_blocks(blocks_to_copy, parent_id)
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

    def free_blocks(self, blocks: list[int]) -> None:
        """Marks a list of (blocks) as free. If the blocks were not shareable, we simply add them to the uninitialized
        blocks queue. Otherwise, their new state depends on whether they are complete."""
        if self.shareable:
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
        incomplete_blocks: list[tuple[int, ShareableBlock]] = []
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
            tokens = prompt_ids[i * self.num_tokens_per_block : (i + 1) * self.num_tokens_per_block]
            block.hash = self.compute_hash(parent_hash, tokens)

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
                logger.debug(f"Adding new block {block.id} with hash {block.hash}")
                self._hash_to_id[block.hash] = block.id

            # Update loop variables
            parent_hash = block.hash

    def compute_hash(self, parent_hash: int | None, tokens: list[int]) -> int:
        """Computes the hash of a block identified by the (tokens) it contains, its (parent_hash) and the layer
        (group_id) it belong to. If the block has no parent, the parent hash is None."""
        # If TP is on, we cannot use python `hash` because it depends on the process (it's per-process salted)
        # TODO: figure out if this is really a problem. Even if hashes diverge per-process, does that break anything?
        if self.tp_on:
            h = hashlib.blake2b(digest_size=8)
            if parent_hash is not None:
                h.update(parent_hash.to_bytes(8, "little", signed=False))
            h.update(array("i", tokens).tobytes())
            hash_ = int.from_bytes(h.digest(), "little", signed=False)
        # Otherwise, use `hash`
        else:
            hash_ = hash((parent_hash, tuple(tokens)))
        return hash_
