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


class RegularTrees:
    """
    An object to represent a set of trees with depths two and at most "elems_per_tree" leaves.
    Each tree is defined by an int root R, and starts with leaves [(R+1) * elem_per_tree - 1, ..., R * elem_per_tree].
    This is used to represent the free blocks in a cache pool.
    """

    def __init__(self, elems_per_tree: int) -> None:
        self.elems_per_tree = elems_per_tree
        self.trees: dict[int, list[int]] = {}
        self._length = 0

    def __len__(self) -> int:
        """Returns the total number of leaves across all trees."""
        return self._length

    def add_tree(self, root: int) -> None:
        """Adds a tree to the set of trees, with "elems_per_tree" leaves."""
        self.trees[root] = [(root + 1) * self.elems_per_tree - 1 - i for i in range(self.elems_per_tree)]
        self._length += self.elems_per_tree

    def add_leaves(self, leaves: list[int]) -> None:
        """Adds a list of leaves to the set of trees. If some leaves don't have a corresponding tree, it is created."""
        self._length += len(leaves)
        for leaf in leaves:
            root = leaf // self.elems_per_tree
            self.trees.setdefault(root, []).append(leaf)

    def pop_leaves(self, num_leaves: int) -> list[int]:
        """Pops a given number of leaves from the set of trees. If the number of leaves to pop is greater than the total
        number of leaves, it raises an error."""
        if num_leaves > self._length:
            raise ValueError(f"Cannot pop {num_leaves} leaves from a tree with {self._length} leaves")
        self._length -= num_leaves

        popped_trees, leaves = [], []
        for root, tree in self.trees.items():
            # Stopping criteria
            if num_leaves == 0:
                break
            # Case: we pop the whole tree
            if len(tree) <= num_leaves:
                popped_trees.append(root)
                popped_leaves = tree
            # Otherwise, we pop part of the tree
            else:
                self.trees[root], popped_leaves = tree[:-num_leaves], tree[-num_leaves:]
            # Accumulate
            leaves.extend(popped_leaves)
            num_leaves -= len(popped_leaves)

        # Bookkeeping
        for root in popped_trees:
            self.trees.pop(root)

        return leaves

    def delete_full_trees(self) -> list[int]:
        """Deletes all trees with "elems_per_tree" leaves."""
        deleted_roots = [root for root, tree in self.trees.items() if len(tree) == self.elems_per_tree]
        self._length -= len(deleted_roots) * self.elems_per_tree
        for root in deleted_roots:
            self.trees.pop(root)
        return deleted_roots


class CachePool:
    """Pool of cache sectors. The first num_reserved_sectors sectors are never free for allocation. On GPU, this is used
    for the read / write trash sectors. On CPU, we don't need to reserve any sectors."""

    def __init__(self, num_sectors: int, num_allocators: int, num_reserved_sectors: int = 2) -> None:
        self.num_sectors = num_sectors
        self.num_allocators = num_allocators
        self.num_reserved_sectors = num_reserved_sectors
        self.blocks_per_sector = [0 for _ in range(num_allocators)]
        self.reset()

    def reset(self) -> None:
        """Resets the cache pool and returns it to its initial state."""
        self.free_sectors = list(range(self.num_reserved_sectors, self.num_sectors + self.num_reserved_sectors))
        self._free_blocks = [RegularTrees(blocks_per_sector) for blocks_per_sector in self.blocks_per_sector]

    def set_blocks_per_sector(self, index: int, blocks_per_sector: int) -> None:
        """Sets the number of blocks per sector for an allocator (referenced by its index)."""
        self.blocks_per_sector[index] = blocks_per_sector
        self._free_blocks[index] = RegularTrees(blocks_per_sector)

    # _________________________________________________ SECTOR LEVEL _________________________________________________ #

    def allocate_sector(self, index: int) -> None:
        """Allocates a free sector to an allocator (referenced by its index)."""
        sector_id = self.free_sectors.pop()
        self._free_blocks[index].add_tree(sector_id)

    @property
    def num_free_sectors(self) -> int:
        """Returns the number of free sectors in the cache pool."""
        return len(self.free_sectors)

    def try_to_free_sectors(self) -> None:
        """Tries to create new free sectors by releasing the ones that have no assigned blocks."""
        deleted_roots = []
        for _free_blocks in self._free_blocks:
            deleted_roots.extend(_free_blocks.delete_full_trees())
        for root in deleted_roots:
            self.free_sectors.append(root)

    # _________________________________________________ BLOCK LEVEL __________________________________________________ #

    def free_blocks(self, index: int, block_ids: list[int]) -> None:
        """Marks a list of block_ids as free for an allocator (referenced by its index)."""
        self._free_blocks[index].add_leaves(block_ids)

    def get_free_blocks(self, index: int, num_blocks: int) -> list[int]:
        """Gets a given number of free blocks from an allocator (referenced by its index). Those blocks are no longer
        free after this operation."""
        return self._free_blocks[index].pop_leaves(num_blocks)

    def count_free_blocks(self, index: int) -> int:
        """Counts the number of free blocks available for an allocator (referenced by its index)."""
        return len(self._free_blocks[index])
