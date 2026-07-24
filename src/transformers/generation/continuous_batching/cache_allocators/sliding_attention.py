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

from ....configuration_utils import PreTrainedConfig
from .full_attention import FullAttentionCacheAllocator


class SlidingAttentionCacheAllocator(FullAttentionCacheAllocator):
    """Cache allocator for a group of sliding attention layers.
    The physical cache of a request is a ring buffer over at most `ceil(sliding_window / tokens_per_page)` pages per
    layer: the token at absolute position `position` lives in the slot `position % sliding_window`, so a new token
    overwrites the cache of the token that just slid out of the window.
    """

    supports_block_sharing = False

    def __init__(
        self,
        index: int,
        config: PreTrainedConfig,
        num_key_value_heads: int,
        cache_dtype: torch.dtype,
        page_size: int,
        layer_indices: list[int],
        allow_block_sharing: bool,
    ) -> None:
        """Initializes the cache allocator for a group of sliding attention layers.

        Args:
            - config: the configuration of the model, used to determine the number of bytes per token
            - num_key_value_heads: the number of key value heads, accounting for TP if enabled
            - cache_dtype: the dtype of the cache, also used to determine the number of bytes per token
            - page_size: the number of tokens per page
            - layer_indices: the indices of the layers which cache is handled by this allocator
            - allow_block_sharing: whether to allow block sharing or not. Can be disabled for diagnostics or perfs.
        """
        # Retrieve the sliding window from the config
        sliding_window = getattr(config, "sliding_window", None)
        if not isinstance(sliding_window, int) or sliding_window <= 0:
            raise ValueError(f"Sliding window must be a positive integer, but got {sliding_window = }")
        self.sliding_window = sliding_window
        super().__init__(
            index=index,
            config=config,
            num_key_value_heads=num_key_value_heads,
            cache_dtype=cache_dtype,
            page_size=page_size,
            layer_indices=layer_indices,
            allow_block_sharing=allow_block_sharing,
        )

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
            physical_indices.append(block_table[page_idx] * self.rows_per_block + page_offset)
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
        key_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_q, head_dim]
        value_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_q, head_dim]
        layer_idx: int,
        read_index: torch.Tensor,  # shape [seqlen_q + past_length]
        write_index: torch.Tensor,  # shape [seqlen_q]
    ) -> tuple[torch.Tensor, torch.Tensor]:  # shape [seqlen_q + past_length, num_kv_heads, head_dim]
        """Update the cache with new key-value states for a specific layer, and retrieves the relevant KV states from
        the cache for attention computation. For sliding attention layers, old KV is read from cache along with extra
        spaces for the new KV, then new KV is written to cache. This is because new KV might overwrite the old KV, so we
        need to read the old KV first.

        When the layer's read index is empty, the batch has no cache reads (all requests are non-chunked prefills): we
        only write to the cache and return the input KV states directly, skipping the index_select read-back.

        Returns the complete KV states (cached + new) for attention computation.
        """
        # Select the shifted views of this layer's keys and values
        k_cache, v_cache = self._kv_views[layer_idx]
        # Transpose the KV states to match the cache shape, after which shape is [seqlen_q, num_kv_heads, head_dim]
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)

        # Case: write-only, no cache read. The input KV states already contain everything the attention needs.
        if read_index.numel() == 0:
            k_cache.index_copy_(0, write_index, key_states)
            v_cache.index_copy_(0, write_index, value_states)
            return key_states, value_states

        # Sentinel positions in read_index mark new-token slots; index_select reads garbage there,
        # then masked_scatter_ overwrites them with the actual new key/value states.
        mask = (read_index == self.sentinel_index).unsqueeze(-1).unsqueeze(-1)
        key_states_with_cache = torch.index_select(k_cache, 0, read_index)
        key_states_with_cache.masked_scatter_(mask, key_states)
        value_states_with_cache = torch.index_select(v_cache, 0, read_index)
        value_states_with_cache.masked_scatter_(mask, value_states)
        # Write new KV values to the cache (padding slots in write_index point to the trash position)
        k_cache.index_copy_(0, write_index, key_states)
        v_cache.index_copy_(0, write_index, value_states)

        # Return the new KV values
        return key_states_with_cache, value_states_with_cache
