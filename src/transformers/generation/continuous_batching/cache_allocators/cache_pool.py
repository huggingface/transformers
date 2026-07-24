class RegularTrees:

    def __init__(self, elems_per_tree: int) -> None:
        self.elems_per_tree = elems_per_tree
        self.trees: dict[int, list[int]] = {}
        self._length = 0

    def __len__(self) -> int:
        return self._length

    def add_tree(self, root: int) -> None:
        self.trees[root] = [(root + 1) * self.elems_per_tree - 1 - i for i in range(self.elems_per_tree)]
        self._length += self.elems_per_tree

    def add_leaves(self, leaves: list[int]) -> None:
        self._length += len(leaves)
        for leaf in leaves:
            root = leaf // self.elems_per_tree
            self.trees.setdefault(root, []).append(leaf)

    def pop_leaves(self, num_leaves: int) -> list[int]:
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
        deleted_roots = [
            root for root, tree in self.trees.items() if len(tree) == self.elems_per_tree
        ]
        self._length -= len(deleted_roots) * self.elems_per_tree
        for root in deleted_roots:
            self.trees.pop(root)
        return deleted_roots


class CachePool:
    """Pool of cache sectors."""

    def __init__(self, num_sectors: int, num_allocators: int) -> None:
        self.num_sectors = num_sectors
        self.num_allocators = num_allocators
        self.blocks_per_sector = [0 for _ in range(num_allocators)]
        self.reset()

    def reset(self) -> None:
        self.free_sectors = list(range(2, self.num_sectors + 2))  # first two sectors are trash
        self._free_blocks = [RegularTrees(blocks_per_sector) for blocks_per_sector in self.blocks_per_sector]

    def set_blocks_per_sector(self, index: int, blocks_per_sector: int) -> None:
        self.blocks_per_sector[index] = blocks_per_sector
        self._free_blocks[index] = RegularTrees(blocks_per_sector)

    # _________________________________________________ SECTOR LEVEL _________________________________________________ #

    def allocate_sector(self, index: int) -> None:
        sector_id = self.free_sectors.pop()
        self._free_blocks[index].add_tree(sector_id)

    @property
    def num_free_sectors(self) -> int:
        return len(self.free_sectors)

    def try_to_free_sectors(self) -> None:
        deleted_roots = []
        for _free_blocks in self._free_blocks:
            deleted_roots.extend(_free_blocks.delete_full_trees())
        for root in deleted_roots:
            self.free_sectors.append(root)

    # _________________________________________________ BLOCK LEVEL __________________________________________________ #

    def free_blocks(self, index: int, block_ids: list[int]) -> None:
        self._free_blocks[index].add_leaves(block_ids)

    def get_free_blocks(self, index: int, num_blocks: int) -> list[int]:
        return self._free_blocks[index].pop_leaves(num_blocks)

    def count_free_blocks(self, index: int) -> int:
        return len(self._free_blocks[index])
