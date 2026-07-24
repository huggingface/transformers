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

from abc import ABC, abstractmethod

import torch

from ..utils import exact_div
from .cache_pool import CachePool


class CacheAllocator(ABC):
    """Base class for cache allocators. A cache allocator receives cache sectors from the PagedAttentionCache and
    allocates them to the requests that need them. A single cache allocator takes care of all layers with a given
    attention type (e.g. full attention, sliding attention, MLA, ...).

    PAGE (for one layer, holds the whole cache of PAGE_SIZE tokens, e.g. both keys and values)
    [  TOKEN 0  |  TOKEN 1  |  TOKEN 2  |  ...  |  TOKEN PAGE_SIZE  ]

    BLOCK (for one request, one page per layer)
    [  PAGE 0  |  PAGE 1  |  PAGE 2  |  ...  |  PAGE NUM_LAYERS  ]

    SECTOR (for one attention type)
    [  BLOCK 0  |  BLOCK 1  |  BLOCK 2  |  ...  |  BLOCK N  ]
    """

    # This attribute depends on the attention type
    supports_block_sharing: bool
    rows_per_token: int
    # These attributes are only known once the cache tensor is registered
    blocks_per_sector: int
    num_pages: int
    num_blocks: int
    _copy_view: torch.Tensor  # cache tensor viewed as [num_blocks, bytes_per_block] for copying

    # ________________________________________________ INITIALIZATION ________________________________________________ #

    def _before_cache_tensor_init(
        self, index: int, layer_indices: list[int], tokens_per_page: int, bytes_per_page: int, allow_block_sharing: bool
    ) -> None:
        # Model-related attributes
        self.index = index
        self.layer_indices = layer_indices
        self.pages_per_block = len(layer_indices)
        # Cache dimensions attributes
        self.tokens_per_page = tokens_per_page
        self.tokens_per_block = tokens_per_page * self.pages_per_block
        self.rows_per_block = self.rows_per_token * self.tokens_per_block
        self.bytes_per_page = bytes_per_page
        self.bytes_per_block = bytes_per_page * self.pages_per_block
        # Bookkeeping attributes
        self.block_table: dict[str, list[int]] = {}
        # Reference counts of shared blocks: a block has an entry only when it is referenced by 2+ requests
        self.use_block_sharing = allow_block_sharing and self.supports_block_sharing
        self.ref_counts: dict[int, int] = {}

    @abstractmethod
    def register_cache_tensor(
        self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor, pool: CachePool
    ) -> None:
        """Registers the cache tensor so the allocator can use it for updates."""

    def _after_cache_tensor_init(
        self, non_trash_bytes: int, bytes_per_sector: int, cache_tensor: torch.Tensor, pool: CachePool
    ) -> None:
        # Cache dimensions attributes
        self.num_pages = exact_div(non_trash_bytes, self.bytes_per_page)
        self.num_blocks = exact_div(self.num_pages, self.pages_per_block)
        self.blocks_per_sector = exact_div(bytes_per_sector, self.bytes_per_block)
        # Cache is a static tensor with shape [num_pages * tokens_per_page, ...]
        self.cache_tensor = cache_tensor
        torch._dynamo.mark_static_address(self.cache_tensor)
        # Cache pool to keep track of the free blocks
        self.pool = pool
        self.pool.set_blocks_per_sector(self.index, self.blocks_per_sector)
        # The first two sectors of the tensor are the trash sectors, and no allocator ever allocates from them.
        # Sector 0 holds the read trash, from which padding tokens read their (zeroed, never written) cache, and the
        # sentinel index, marking where to insert the new key or value states for sliding window attention groups.
        # Sector 1 holds the write trash, where padding tokens can safely write their cache (it is never read from).
        self.read_trash_index = 0
        self.sentinel_index = 1
        self.write_trash_index = self.rows_per_token * self.tokens_per_block * self.blocks_per_sector

    # _________________________________________________ SECTOR LEVEL _________________________________________________ #

    def needs_new_sectors(self, request_id: str, past_length: int, query_length: int) -> int:
        """Returns the number of new sectors needed to store the new tokens for a given request. It can be zero."""
        num_blocks_needed = self.needs_new_blocks(request_id, past_length, query_length)
        num_free_blocks = self.pool.count_free_blocks(self.index)
        fresh_blocks_needed = num_blocks_needed - num_free_blocks
        if fresh_blocks_needed <= 0:
            return 0
        # Round up: a partially needed sector is still a whole sector
        return (fresh_blocks_needed + self.blocks_per_sector - 1) // self.blocks_per_sector

    # _________________________________________________ BLOCK LEVEL __________________________________________________ #

    @abstractmethod
    def needs_new_blocks(self, request_id: str, past_length: int, query_length: int) -> int:
        """Returns the number of new blocks needed to store the new tokens for a given request. It can be zero."""

    @abstractmethod
    def allocate_cache_to_request(self, request_id: str, past_length: int, query_length: int) -> None:
        """Allocates the cache to the request."""

    def _can_free_block(self, block_id: int) -> bool:
        """Returns True if the block can be freed, False otherwise. Also decrements the reference count of the block."""
        new_ref_count = self.ref_counts.pop(block_id, 1) - 1
        if new_ref_count > 1:
            self.ref_counts[block_id] = new_ref_count
        return new_ref_count == 0

    def free_blocks(self, request_id: str) -> None:
        """Marks the blocks owned by the request as free, or decrements their reference count if they are shared."""
        blocks_ids = self.block_table.pop(request_id, [])
        freed_blocks = list(filter(self._can_free_block, blocks_ids))
        self.pool.free_blocks(self.index, freed_blocks)

    def count_shareable_blocks(self, source_request_id: str, past_length: int) -> int:
        """Counts the number of the source request's blocks a fork can share instead of copying."""
        if not self.use_block_sharing:
            return 0
        return min(past_length // self.tokens_per_page, len(self.block_table[source_request_id]))

    def count_non_shareable_blocks(self, source_request_id: str, past_length: int) -> int:
        """Counts the number of fresh blocks a fork of the source request would need, i.e. its non-shareable blocks."""
        num_blocks = len(self.block_table[source_request_id])
        return num_blocks - self.count_shareable_blocks(source_request_id, past_length)

    def fork_blocks(
        self, source_request_id: str, dest_request_id: str, past_length: int
    ) -> tuple[list[int], list[int]]:
        """Builds the block table of a fork of the source request: fully-written blocks are shared (their reference
        count increases) if allowed, and others are backed by fresh blocks. Returns the (source, destination) block
        id pairs which content the caller must copy."""
        source_table = self.block_table[source_request_id]
        num_shared = self.count_shareable_blocks(source_request_id, past_length)
        # It's always the first num_shared blocks that are shareable, so the rest we have to copy
        shared_blocks, blocks_to_copy = source_table[:num_shared], source_table[num_shared:]
        for block_id in shared_blocks:
            self.ref_counts[block_id] = self.ref_counts.get(block_id, 1) + 1
        fresh_blocks = self.pool.get_free_blocks(self.index, len(blocks_to_copy))
        self.block_table[dest_request_id] = shared_blocks + fresh_blocks
        return blocks_to_copy, fresh_blocks

    def copy_blocks(self, source_block_ids: list[int], dest_block_ids: list[int]) -> None:
        """Copies whole blocks (keys and values of every layer of the group) inside the cache tensor."""
        device = self._copy_view.device
        source = torch.tensor(source_block_ids, device=device, dtype=torch.long)
        dest = torch.tensor(dest_block_ids, device=device, dtype=torch.long)
        self._copy_view.index_copy_(0, dest, self._copy_view.index_select(0, source))

    def free_all_requests(self) -> None:
        """Mark all blocks owned by all requests as free."""
        request_ids = list(self.block_table.keys())
        for request_id in request_ids:
            self.free_blocks(request_id)

    # ______________________________________________ INPUT PREPARATION _______________________________________________ #

    @abstractmethod
    def get_seqlen_k(self, past_length: int, query_length: int) -> int:
        """Returns the sequence length of the key for a given request_id, past_length and query_length."""

    @abstractmethod
    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache."""

    @abstractmethod
    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices for writing to the cache."""

    @abstractmethod
    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills the block table for a given request_id, past_length and query_length."""

    # ________________________________________________ RUNTIME UPDATE ________________________________________________ #

    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        read_index: torch.Tensor,
        write_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes the new key and value states in the cache for the given layer and retrieves the KV states needed for
        the attention computation, as indicated by the read and write indices."""
