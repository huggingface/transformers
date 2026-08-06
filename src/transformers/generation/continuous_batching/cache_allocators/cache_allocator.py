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
from collections import deque

import torch


class CacheAllocator(ABC):
    """Base class for cache allocators. A cache allocator receives cache sectors from the PagedAttentionCache and
    allocates them to the requests that need them. A single cache allocator takes care of all layer with a given
    attention type (e.g. full attention, sliding attention, MLA, ...).

    PAGE (for one layer)
    [  TOKEN 0  |  TOKEN 1  |  TOKEN 2  |  ...  |  TOKEN PAGE_SIZE  ]

    BLOCK (for one request)
    [  PAGE 0  |  PAGE 1  |  PAGE 2  |  ...  |  PAGE NUM_LAYERS  ]

    SECTOR (for one attention type)
    [  BLOCK 0  |  BLOCK 1  |  BLOCK 2  |  ...  |  BLOCK N  ]
    """

    _num_trash_pages = 2  # constant across all cache allocators
    supports_block_sharing: bool  # depends on the attention type

    # ________________________________________________ INITIALIZATION ________________________________________________ #

    def __init__(self, page_size: int, bytes_per_page: int, num_layers: int):
        self.tokens_per_page = page_size
        self.bytes_per_page = bytes_per_page
        self.num_layers = num_layers

        self.block_table: dict[str, list[int]] = {}
        self.free_block_ids: deque[int] = deque()
        self.bytes_per_block = bytes_per_page * num_layers
        self.tokens_per_block = self.tokens_per_page * num_layers

    @abstractmethod
    def register_cache_tensor(self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor) -> None:
        """Registers the cache tensor so the allocator can use it for updates."""
        pass

    def _finalize_init(self, num_pages: int, num_blocks: int, bytes_per_sector: int, cache_tensor: torch.Tensor) -> None:
        self.num_pages = num_pages
        self.num_blocks = num_blocks

        self.blocks_per_sector = bytes_per_sector // self.bytes_per_block

        self.cache_tensor = cache_tensor
        torch._dynamo.mark_static_address(self.cache_tensor)

        # Mark the cache tensor as static address to allow for better performance
        # We add two extra blocks to the cache as a padding zone that no CacheAllocator ever allocates from.
        # The first one is zeroed and then never written to. Its first index is the read trash, from which padding
        # tokens read their KV cache, and its second index is the sentinel index, to indicate where to store the new key
        # or values indices for sliding window attention groups.
        # The second is the write trash, where padding tokens can safely write their KV cache (it's never read from).
        self.read_trash_index = num_pages * self.tokens_per_page
        self.sentinel_index = num_pages * self.tokens_per_page + 1  # since tokens_per_page >= 4 > 2, this is safe
        self.write_trash_index = (num_pages + 1) * self.tokens_per_page

    # _________________________________________________ SECTOR LEVEL _________________________________________________ #

    def needs_new_sectors(self, request_id: str, past_length: int, query_length: int) -> int:
        """Returns the number of new sectors needed to store the new tokens for a given request. It can be zero."""
        num_blocks_needed = self.needs_new_blocks(request_id, past_length, query_length)
        num_free_blocks = len(self.free_block_ids)
        fresh_blocks_needed = num_blocks_needed - num_free_blocks
        if fresh_blocks_needed <= 0:
            return 0
        # Since we need at least one fresh sector, we round up to the next integer
        return fresh_blocks_needed // self.blocks_per_sector + 1

    def allocate_new_sector(self, sector_id: int) -> None:
        """Allocates a new sector to the allocator, which translates as new free blocks for this allocator."""
        self.free_block_ids.extend(range(sector_id * self.blocks_per_sector, (sector_id + 1) * self.blocks_per_sector))

    # _________________________________________________ BLOCK LEVEL __________________________________________________ #

    @abstractmethod
    def needs_new_blocks(self, request_id: str, past_length: int, query_length: int) -> int:
        """Returns the number of new blocks needed to store the new tokens for a given request. It can be zero."""
        pass

    @abstractmethod
    def allocate_cache_to_request(self, request_id: str, past_length: int, query_length: int) -> None:
        """Allocates the cache to the request."""
        pass

    def free_blocks(self, request_id: str) -> None:
        """Mark the blocks owned by the request as free."""
        block_ids = self.block_table.pop(request_id, [])
        self.free_block_ids.extend(block_ids)

    def free_all_requests(self) -> None:
        """Mark all blocks owned by all requests as free."""
        request_ids = list(self.block_table.keys())
        for request_id in request_ids:
            self.free_blocks(request_id)

    # ______________________________________________ INPUT PREPARATION _______________________________________________ #

    @abstractmethod
    def get_seqlen_k(self, past_length: int, query_length: int) -> int:
        """Returns the sequence length of the key for a given request_id, past_length and query_length."""
        pass

    @abstractmethod
    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache."""
        pass

    @abstractmethod
    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices for writing to the cache."""
        pass

    @abstractmethod
    def fill_block_table(self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor) -> None:
        """Fills the block table for a given request_id, past_length and query_length."""
        pass

    # ________________________________________________ RUNTIME UPDATE ________________________________________________ #

    @abstractmethod
    def update_cache(self, request_id: str, past_length: int, query_length: int) -> None:
        """Updates the cache for a given request_id, past_length and query_length."""
        pass
