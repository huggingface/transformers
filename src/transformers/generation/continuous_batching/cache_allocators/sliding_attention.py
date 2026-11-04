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

from .full_attention import FullAttentionCacheAllocator, RowSpec


class SlidingAttentionCacheAllocator(FullAttentionCacheAllocator):
    """Cache allocator for a group of sliding attention layers.
    The physical cache of a request is a ring buffer over at most `ceil(sliding_window / tokens_per_page)` pages per
    layer: the token at absolute position `position` lives in the slot `position % sliding_window`, so a new token
    overwrites the cache of the token that just slid out of the window.
    """

    supports_block_sharing = False
    supports_block_table = False
    layer_type = "sliding_attention"

    def __init__(
        self,
        index: int,
        rows: list[RowSpec],
        page_size: int,
        layer_indices: list[int],
        allow_block_sharing: bool,
        sliding_window: int,
    ) -> None:
        """Initializes the cache allocator for a group of sliding attention layers.

        Args:
            - rows: the shape and dtype of each row a layer caches per token, in the order the layer reads and writes
              them
            - page_size: the number of tokens per page
            - layer_indices: the indices of the layers which cache is handled by this allocator
            - allow_block_sharing: whether to allow block sharing or not. Can be disabled for diagnostics or perfs.
            - sliding_window: the number of most recent tokens the layers attend to
        """
        if not isinstance(sliding_window, int) or sliding_window <= 0:
            raise ValueError(f"Sliding window must be a positive integer, but got {sliding_window = }")
        self.sliding_window = sliding_window
        super().__init__(
            index=index,
            rows=rows,
            page_size=page_size,
            layer_indices=layer_indices,
            allow_block_sharing=allow_block_sharing,
        )
        # The ring buffer has no block table kernel, whatever the rows
        self.supports_block_table = False

    # _________________________________________________ BLOCK LEVEL __________________________________________________ #

    def _compute_blocks_needed(self, allocated_blocks: int, past_length: int, query_length: int) -> int:
        """Computes the number of blocks needed to store the new tokens for a given request."""
        total_length = min(past_length + query_length, self.sliding_window)
        total_blocks = (total_length + self.tokens_per_page - 1) // self.tokens_per_page
        return max(0, total_blocks - allocated_blocks)

    # ______________________________________________ INPUT PREPARATION _______________________________________________ #

    def get_seqlen_k(self, past_length: int, query_length: int) -> int:
        """Returns the sequence length of the key for a given request_id, past_length and query_length."""
        return query_length + min(past_length, self.sliding_window - 1)

    def _token_indices_to_physical_indices(self, request_id: str, start_index: int, length: int) -> list[int]:
        """Maps a sequence of consecutive token indices, starting at start_index with the given length, to their
        physical indices."""
        block_table = self.block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        physical_indices = []
        for token_idx in range(start_index, start_index + length):
            ring_idx = token_idx % self.sliding_window
            page_idx, page_offset = divmod(ring_idx, self.tokens_per_page)
            physical_indices.append(block_table[page_idx] * self.tokens_per_page + page_offset)
        return physical_indices

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache in the cache tensor.
        For a group of sliding window attention layers, we read from the cache tensor before writing on it, because the
        new cache can overwrite the old one. To form the cache + new key / values states, we read at most the
        sliding_window - 1 most recent cached tokens and then manually add the new key / values states after. Hence the
        sentinel indices which indicate where to store the new key or values states."""
        if past_length < self.sliding_window:
            start_index = 0
            cache_length = past_length
        else:
            start_index = (past_length + 1) % self.sliding_window
            cache_length = self.sliding_window - 1
        # Compute the physical indices
        physical_indices = self._token_indices_to_physical_indices(request_id, start_index, cache_length)
        return physical_indices + [self.sentinel_index] * query_length

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to write request_id's cache in the cache tensor. For a group of
        sliding window attention layers, we write the new cache in rolling-buffer kind of way: if we reach the end of
        the allocated physical cache, we start writing from the beginning of the physical cache again. If there are
        more new tokens than the window can hold, only the last sliding_window ones are stored, and the earlier ones
        are written to the write trash."""
        # When the query is longer than the window, only its last sliding_window tokens are stored, so the writes
        # start at the slot of the first STORED token, (past_length + padding_length) % sliding_window, to preserve
        # the slot = position % sliding_window invariant.
        cache_length = min(query_length, self.sliding_window)
        padding_length = query_length - cache_length
        start_index = (past_length + padding_length) % self.sliding_window
        # Compute the physical indices
        physical_indices = self._token_indices_to_physical_indices(request_id, start_index, cache_length)
        if padding_length > 0:
            physical_indices = [self.write_trash_index] * padding_length + physical_indices
        return physical_indices

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills the block table for a given request_id, past_length and query_length."""
        raise NotImplementedError("Not implemented for sliding attention cache allocator")

    # ________________________________________________ RUNTIME UPDATE ________________________________________________ #

    def update(
        self,
        rows: tuple[torch.Tensor, ...],  # shapes [seqlen_q, *shape]
        layer_idx: int,
        read_index: torch.Tensor,  # shape [seqlen_q + past_length]
        write_index: torch.Tensor,  # shape [seqlen_q]
    ) -> tuple[torch.Tensor, ...]:  # shapes [seqlen_q + past_length, *shape]
        """Update the cache with the new rows for a specific layer, and retrieves the rows needed for the computation.
        For sliding attention layers, old rows are read from cache along with extra spaces for the new rows, then the
        new rows are written to cache. This is because new rows might overwrite the old ones, so we need to read the
        old ones first.

        When the layer's read index is empty, the batch has no cache reads (all requests are non-chunked prefills): we
        only write to the cache and return the input rows directly, skipping the read-back.

        Returns the complete rows (cached + new) for the computation.
        """
        views = self._row_views_for_layer(layer_idx)
        write_block, write_slot = write_index // self.tokens_per_page, write_index % self.tokens_per_page

        # Case: write-only, no cache read. The input rows already contain everything the computation needs.
        if read_index.numel() == 0:
            for view, row in zip(views, rows):
                view[write_block, write_slot] = row
            return rows

        # Sentinel positions in read_index mark new-token slots: the read returns garbage there, which
        # masked_scatter_ overwrites with the actual new rows. The sentinel lives in slot 1 of block 0.
        read_block, read_slot = read_index // self.tokens_per_page, read_index % self.tokens_per_page
        is_sentinel = read_index == self.sentinel_index
        rows_with_cache = []
        for view, row in zip(views, rows):
            row_with_cache = view[read_block, read_slot]
            mask = is_sentinel.view(-1, *([1] * (row.dim() - 1)))
            row_with_cache.masked_scatter_(mask, row)
            rows_with_cache.append(row_with_cache)
        # Write new rows to the cache (padding slots in write_index point to the trash position)
        for view, row in zip(views, rows):
            view[write_block, write_slot] = row

        return tuple(rows_with_cache)

    def get_cache_for_block_table(self, layer_idx: int) -> tuple[torch.Tensor, ...]:
        """Returns the cache views for a block table update."""
        raise NotImplementedError("Not implemented for sliding attention cache allocator")
