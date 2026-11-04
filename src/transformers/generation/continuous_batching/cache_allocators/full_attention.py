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

from .cache_allocator import CacheAllocator
from .cache_pool import CachePool


# The shape and dtype of one row a layer caches per token. A layer declares its rows by name (`"key"` and `"value"` for
# attention) through `cache_rows`; a dtype of None stands for the cache dtype.
RowSpec = tuple[tuple[int, ...], torch.dtype | None]


class FullAttentionCacheAllocator(CacheAllocator):
    """Cache allocator for a group of full attention layers, or of any layers that keep rows per token.

    The layers declare the rows they cache per token, by name (the keys and the values for an attention layer, with
    their shape and dtype), and every token of a request is kept for its whole life. A token lives in a block at a slot
    of the block's pages, and the indices address it as `block * tokens_per_page + slot`, for every row at once.
    """

    supports_block_sharing = True
    supports_block_table = True
    layer_type = "full_attention"

    def __init__(
        self,
        index: int,
        rows: dict[str, tuple[tuple[int, ...], torch.dtype]],
        page_size: int,
        layer_indices: list[int],
        allow_block_sharing: bool,
    ) -> None:
        """Initializes the cache allocator for a group of layers caching rows per token.

        Args:
            - rows: the shape and dtype of each row a layer caches per token, by name
            - page_size: the number of tokens per page
            - layer_indices: the indices of the layers which cache is handled by this allocator
            - allow_block_sharing: whether to allow block sharing or not. Can be disabled for diagnostics or perfs.
        """
        self.rows = rows
        self.row_bytes = [torch.Size(shape).numel() * dtype.itemsize for shape, dtype in rows.values()]
        self.bytes_per_token = sum(self.row_bytes)
        self._before_cache_tensor_init(
            index=index,
            layer_indices=layer_indices,
            tokens_per_page=page_size,
            bytes_per_block=self.bytes_per_token * page_size * len(layer_indices),
            allow_block_sharing=allow_block_sharing,
        )
        # The block table kernels read the rows where they sit in the pages: flash attention a key and a value of the
        # same size, FlashMLA the one latent row of multi-head latent attention, 576 values wide (DeepSeek V3's), in
        # 16-bit pages of 64 tokens
        from ....integrations.hub_kernels import lazy_load_kernel  # the integrations import the cache

        reads_keys_and_values = set(rows) == {"key", "value"} and len(set(self.row_bytes)) == 1
        reads_latents = (
            set(rows) == {"latent"}
            and rows["latent"] in (((576,), torch.bfloat16), ((576,), torch.float16))
            and page_size == 64
            and lazy_load_kernel("flash-mla") is not None
        )
        self.supports_block_table = reads_keys_and_values or reads_latents

    def register_cache_tensor(
        self, bytes_per_sector: int, non_trash_bytes: int, cache_tensor: torch.Tensor, pool: CachePool
    ) -> None:
        """Registers the cache tensor so the allocator can use it for updates. For a full attention cache allocator
        with 2 layers, the cache is arranged this way (L0 = Layer 0), row by row then layer by layer inside a block:

        [ L0 KEYS | L1 KEYS | L0 VALUES | L1 VALUES | L0 KEYS | L1 KEYS | L0 VALUES | L1 VALUES ]
        [ ---------------- BLOCK 0 ---------------- | ---------------- BLOCK 1 ---------------- ]
        [ --------------------------------------- SECTOR -------------------------------------- ]

        Here, a sector is dimensionned to hold 2 full attention blocks. Every "L0 KEYS" is a page: the keys of
        page_size tokens for layer 0.
        """
        self.bytes_per_sector = bytes_per_sector
        # Byte view of the whole tensor as one row per block, used to copy blocks when forking
        self._copy_view = cache_tensor.view(-1, self.bytes_per_block)

        # The views span the entire tensor, including the two leading trash sectors
        total_bytes = non_trash_bytes + 2 * bytes_per_sector
        num_blocks = total_bytes // self.bytes_per_block
        blocks = cache_tensor[: num_blocks * self.bytes_per_block].view(num_blocks, self.bytes_per_block)
        # One view per row, shaped [num_blocks, pages_per_block, tokens_per_page, *shape]
        self._row_views: dict[str, torch.Tensor] = {}
        offset = 0
        for (name, (shape, dtype)), row_bytes in zip(self.rows.items(), self.row_bytes):
            row_region_bytes = row_bytes * self.tokens_per_page * self.pages_per_block
            row_region = blocks[:, offset : offset + row_region_bytes].view(dtype)
            self._row_views[name] = row_region.view(num_blocks, self.pages_per_block, self.tokens_per_page, *shape)
            offset += row_region_bytes
        self._after_cache_tensor_init(non_trash_bytes, bytes_per_sector, cache_tensor, pool)

        # For the block table kernels, every row of a layer is one array of all the pages of the cache, shifted to the
        # layer's first page of that row, so the block table holds the first page of each block. The kernels require
        # the rows to hold the same number of pages, so they are all truncated to the shortest (the last row's).
        self._layer_page_views: dict[int, dict[str, torch.Tensor]] = {}
        if self.supports_block_table:
            row_pages = {
                name: cache_tensor[: num_blocks * self.bytes_per_block]
                .view(dtype)
                .view(-1, self.tokens_per_page, *shape)
                for name, (shape, dtype) in self.rows.items()
            }
            for page, layer_idx in enumerate(self.layer_indices):
                shifted = {
                    name: pages[row * self.pages_per_block + page :]
                    for row, (name, pages) in enumerate(row_pages.items())
                }
                num_pages = min(view.shape[0] for view in shifted.values())
                self._layer_page_views[layer_idx] = {name: view[:num_pages] for name, view in shifted.items()}

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
            start = block_table[b] * self.tokens_per_page
            physical_indices.extend(range(start, start + self.tokens_per_page))
        if remainder:
            start = block_table[num_full_pages] * self.tokens_per_page
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
            physical_start = block_table[b] * self.tokens_per_page
            # First page may start mid-page, last page may end mid-page
            local_start = start_offset if b == start_page else 0
            local_end = (end_pos - 1) % self.tokens_per_page + 1 if b == end_page else self.tokens_per_page
            physical_indices.extend(range(physical_start + local_start, physical_start + local_end))
        return physical_indices

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills the request's row of the kernel block table: the first page of each block in the row's page array,
        the layer's shift being applied through the view the kernel reads."""
        block_ids = self.block_table[request_id]
        entries = torch.tensor(block_ids, dtype=block_table.dtype) * (len(self.rows) * self.pages_per_block)
        block_table[: len(block_ids)].copy_(entries, non_blocking=True)

    # ________________________________________________ RUNTIME UPDATE ________________________________________________ #

    def _row_views_for_layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """The views of the layer's rows, by name, each shaped [num_blocks, tokens_per_page, *shape]."""
        page = self.layer_indices.index(layer_idx)
        return {name: view[:, page] for name, view in self._row_views.items()}

    def update(
        self,
        rows: dict[str, torch.Tensor],  # shapes [seqlen_q, *shape]
        layer_idx: int,
        read_index: torch.Tensor,  # shape [seqlen_q + past_length]
        write_index: torch.Tensor,  # shape [seqlen_q]
    ) -> dict[str, torch.Tensor]:  # shapes [seqlen_q + past_length, *shape]
        """
        Update the cache with the new rows for a specific layer, and retrieves the rows needed for the computation.
        For full attention layers, new rows are written to cache, then the complete sequence is read from cache. The
        rows are given by name, and need not be all the rows of the layer: a layer may write its rows at different
        points of its forward.

        When the layer's read index is empty, the batch has no cache reads (all requests are non-chunked prefills): we
        only write to the cache and return the input rows directly, skipping the read-back.

        Returns the complete rows (cached + new) for the computation.
        """
        views = self._row_views_for_layer(layer_idx)
        write_block, write_slot = write_index // self.tokens_per_page, write_index % self.tokens_per_page
        for name, row in rows.items():
            views[name][write_block, write_slot] = row

        # If there is no old cache to read, the input rows already contain everything the computation needs
        if read_index.numel() == 0:
            return rows

        # Otherwise, read the whole sequence back from the cache
        read_block, read_slot = read_index // self.tokens_per_page, read_index % self.tokens_per_page
        return {name: views[name][read_block, read_slot] for name in rows}

    def get_cache_for_block_table(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Returns the cache views of the layer's rows, by name, for a block table update."""
        return self._layer_page_views[layer_idx]
