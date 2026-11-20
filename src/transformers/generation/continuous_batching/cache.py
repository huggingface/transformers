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
import inspect
import warnings
from math import ceil, lcm
from typing import TYPE_CHECKING, Any

import torch

from ...configuration_utils import PreTrainedConfig
from ...generation.configuration_utils import ContinuousBatchingConfig
from .cache_allocators import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    CacheAllocator,
    CachePool,
    FullAttentionCacheAllocator,
    MLACacheAllocator,
    SlidingAttentionCacheAllocator,
)
from .requests import RequestState, RequestStatus
from .utils import logger


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


# Maps each attention type to the allocator class handling its cache
ATTN_TYPE_TO_ALLOCATOR = {
    FULL_ATTENTION: FullAttentionCacheAllocator,
    SLIDING_ATTENTION: SlidingAttentionCacheAllocator,
}


def group_layers_by_attn_type(config: PreTrainedConfig) -> dict[str, list[int]]:
    """Groups layers depending on their attention type.

    For a model with the following layer types: ["sliding", "full", "full", "sliding", "full", "full", "full", "full"]
    We would get two groups: {"sliding_attention": [0, 3], "full_attention": [1, 2, 4, 5, 6, 7]}.
    """
    layer_types = getattr(config, "layer_types", None)

    # If the config has no layer_type attribute, it means all layers are the same attention type
    if layer_types is None:
        # If there is a sliding window, assume all layers are sliding attention
        sliding_window = getattr(config, "sliding_window", None)
        if sliding_window is not None:
            return {SLIDING_ATTENTION: list(range(config.num_hidden_layers))}
        # Otherwise, assume all layers are full attention
        return {FULL_ATTENTION: list(range(config.num_hidden_layers))}

    # Otherwise simply count the number of layers of each type, making sure they are supported at the same time
    layer_counts = {}
    for i, layer_type in enumerate(layer_types):
        if layer_type not in ATTN_TYPE_TO_ALLOCATOR:
            raise ValueError(f"Invalid layer type: {layer_type}")
        layer_counts[layer_type] = layer_counts.get(layer_type, []) + [i]
    return layer_counts


class PagedAttentionCache:
    """
    High-level manager for any cache used by continuous batching. This object own the cache tensors and distributes
    sectors to sub-allocators, each for a different kind of cache (full-attention layers, MSA layers, embeddings cache,
    etc.).

    The cache tensors are the physical memory on which the cache is stored.
    Virtually, the cache is allocated in a tiered system (smaller units first):
        - layer-level units: pages
        - request-level units: blocks
        - attention-type-level units: sectors
    You can find a more detailled description in the CacheAllocator class documentation. The PagedAttentionCache object
    only concerns itself with the sectors, and anything beyond that granularity is handled by the CacheAllocators.

    For a given attention type, there is only one allocator, which is responsible for the cache of all layers with that
    attention type. For instance, in a 4 layers model with layer types ["sliding", "full", "sliding", "sliding"], there
    are two allocators: one FullAttentionCacheAllocator (for layer 1) and one SlidingAttentionCacheAllocator (for layers
    0, 2 and 3).
    Because of this, the block size for the full attention allocator is not the same as the block size for the sliding
    attention allocator. This is an issue: if the paged cache gave out similar-sized blocks to the two allocators, the
    storage for the full attention allocator would be 3x what is necessary. Hence, we need to find a way to grow the
    storage at different rates for the two allocators.

    To do that, we define "sectors" : a sector is a contiguous region of the cache that is allocated to a single
    allocator. The size of a sector is computed so that all cache allocators can divide a sector into blocks. The paged
    attention cache delegates the responasibilty of managing blocks to the cache allocators, which only request sectors.
    In our example, the amount of memory needed to store all sliding pages is x3 the amount of memory needed to store
    one full page. Hence, one block of sliding cache is 3 blocks of full cache. Taking the LCM(1, 3) = 3, we compute the
    size of a sector to be 3 blocks of full cache, or 1 block of sliding cache.

                           [ -------------------------------------- SECTOR -------------------------------------- ]

    used for full attn:    [ ------ FULL BLOCK 0 ------ | ------ FULL BLOCK 1 ------ | ------ FULL BLOCK 2 ------ ]
                           [ ---- PAGE FOR LAYER 1 ---- | ---- PAGE FOR LAYER 1 ---- | ---- PAGE FOR LAYER 1 ---- ]

    used for sliding attn: [ ---------------------------------- SLIDING BLOCK 0 --------------------------------- ]
                           [ ---- PAGE FOR LAYER 0 ---- | ---- PAGE FOR LAYER 2 ---- | ---- PAGE FOR LAYER 3 ---- ]

    For a more in depth figure, check FullAttentionCacheAllocator.register_cache_tensor docstring.

    Physically, the cache is stored on a single flat tensor, whose first two sectors are never-allocated trash
    sectors used by padding tokens (see FullAttentionCacheAllocator.register_cache_tensor).

    The way this cache breaks down global cache into sectors was inspired by the mechanism described in this vLLM's PR
    (https://github.com/vllm-project/vllm/issues/11382) and the Jenga paper (https://arxiv.org/html/2503.18292v1#S5),
    so we would like to thank the vLLM team and the Jenga authors for their work.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        device: torch.device | str,
        cache_tensor_meta: dict[int, list[tuple[torch.Size, torch.dtype, torch.device]]],
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize a paged attention cache for efficient memory usage. Also turns in prefix sharing if the model has
        only full attention layers.

        Args:
            config: Model configuration
            continuous_batching_config: Continuous batching configuration containing cache parameters. The parameter
            `max_batch_tokens` needs to be resolved, and so does `num_blocks` if `num_sectors` is not set, or the
            initialization will fail.
            device: Device for the cache tensors
            cache_tensor_meta: Shape, dtype and device of the tensors each layer passes to `update`, as recorded by
                `hook_cache_tensor_meta`. Formatted as layer_idx -> [(shape, dtype, device), ...]
            dtype: Data type of the cache
        """
        self.max_batch_tokens = continuous_batching_config.max_batch_tokens
        self.config = config
        self.dtype = dtype
        self.device = device
        self.max_blocks_per_request = continuous_batching_config.max_blocks_per_request or 0  # if not resolved, off

        if self.max_batch_tokens is None:
            raise ValueError(f"The continuous batching config must have a resolved {self.max_batch_tokens = }.")

        # Construct the necessary cache allocator for each attention type
        self.cache_allocators: dict[str, CacheAllocator] = {}
        self.layer_to_allocator: dict[int, CacheAllocator] = {}
        for index, (attn_type, layer_indices) in enumerate(group_layers_by_attn_type(config).items()):
            # Create the allocator and register it by name and layers indices
            allocator = self._create_allocator(
                continuous_batching_config, attn_type, index, layer_indices, cache_tensor_meta
            )
            self.cache_allocators[attn_type] = allocator
            self.layer_to_allocator = self.layer_to_allocator | dict.fromkeys(layer_indices, allocator)

        # To have the maximal granularity while ensuring alignment for all cache allocators, we compute the LCM of all
        # cache allocator block sizes AND a default alignment of 128 bytes
        self.bytes_per_sector = lcm(*(ca.bytes_per_block for ca in self.cache_allocators.values()), 128)

        # We plan for the "worst" sector, ie. the one with the most tokens and so the biggest index tensor
        self.tokens_per_sector = max(
            (self.bytes_per_sector // ca.bytes_per_block) * ca.tokens_per_page for ca in self.cache_allocators.values()
        )

        # If num_sectors isn't set, it's computed from num_blocks so that all allocators can hold num_blocks blocks at
        # the same time
        num_sectors = continuous_batching_config.num_sectors
        if num_sectors is None:
            num_blocks = continuous_batching_config.num_blocks
            if num_blocks is None or num_blocks <= 0:
                raise ValueError(
                    f"Without `num_sectors`, the cache needs a positive number of blocks, got {num_blocks = }"
                )
            num_sectors = 0
            for ca in self.cache_allocators.values():
                sectors_per_block = ca.bytes_per_block / self.bytes_per_sector
                num_sectors += ceil(num_blocks * sectors_per_block)
        # Check that the number of sectors is positive
        if num_sectors <= 0:
            raise ValueError(f"The cache needs a positive number of sectors, got {num_sectors = }.")
        self.num_sectors = num_sectors

        mb_per_sector = self.bytes_per_sector // 1024**2
        logger.info(f"Paged cache initialized: {self.max_batch_tokens = }, {self.num_sectors = }, {mb_per_sector = }")

        # TODO: could be reduced to 1 trash sector under certain hit / miss conditions
        # The cache holds two trash sectors followed by the data sectors. The trash sectors are never allocated.
        non_trash_bytes = self.num_sectors * self.bytes_per_sector
        trash_bytes = 2 * self.bytes_per_sector
        self.cache_tensor = torch.zeros(non_trash_bytes + trash_bytes, dtype=torch.uint8, device=self.device)
        # Cache pool, which keeps track of the free sectors and their allocation
        self.pool = CachePool(self.num_sectors, len(self.cache_allocators))
        # Distribute the cache across all allocators
        for allocator in self.cache_allocators.values():
            allocator.register_cache_tensor(self.bytes_per_sector, non_trash_bytes, self.cache_tensor, self.pool)

        # Block and prefix sharing
        self.allow_block_sharing = continuous_batching_config.allow_block_sharing
        allocators_can_share = all(ca.use_block_sharing for ca in self.cache_allocators.values())
        self.use_prefix_sharing = allocators_can_share and self.allow_block_sharing

        # Helper attributes for the scheduler
        self.read_cache_limit = self._infer_read_cache_limit()
        self.max_decode_fast_path_length = self._infer_max_decode_fast_path_length()
        # Helper attribute for the input/output classes
        self.max_tokens_read = max(ca.tokens_per_page * ca.num_blocks for ca in self.cache_allocators.values())

    def _create_allocator(
        self,
        continuous_batching_config: ContinuousBatchingConfig,
        attn_type: str,
        index: int,
        layer_indices: list[int],
        cache_tensor_meta: dict[int, list[tuple[torch.Size, torch.dtype, torch.device]]],
    ) -> CacheAllocator:
        """Creates the cache allocator for a given set of layers with the same attention type."""
        # Catch a special case: MLA layers are created with the same attention type as full attention layers
        if attn_type == FULL_ATTENTION and getattr(self.config, "kv_lora_rank", None) is not None:
            allocator_class = MLACacheAllocator
        else:
            allocator_class = ATTN_TYPE_TO_ALLOCATOR[attn_type]

        allocator = allocator_class(
            index=index,
            layer_indices=layer_indices,
            config=self.config,
            cache_tensor_meta=cache_tensor_meta,
            cache_dtype=self.dtype,
            page_size=continuous_batching_config.page_size,
            allow_block_sharing=continuous_batching_config.allow_block_sharing,
        )
        return allocator

    def max_num_sectors(
        self,
        available_bytes: int,
        probe_static_bytes: int | None,
        bytes_per_cache_token: int,
        decoder_peak: int,
        read_cost: float,
        post_peak: float,
    ) -> int:
        """Returns the largest number of sectors a cache with the same allocators can have, for continuous batching to
        fit in `available_bytes`. This cache is the probe that measured the memory of a batch:
            - probe_static_bytes: the bytes allocated to build the probe (this cache and the IO tensors), None if unknown
            - bytes_per_cache_token: the bytes of the IO tensors that have one column per cache token
            - decoder_peak: the decoder peak of a batch reading no cache
            - read_cost: the growth of the decoder peak per cache token read
            - post_peak: the peak of the LM head onwards (LM head, logits processing and sampling)
        """
        # Each sector costs its own bytes, plus the columns of the IO tensors sized after the cache
        sector_cost = self.bytes_per_sector + self.tokens_per_sector * bytes_per_cache_token

        # If probe measurements are available, adjust them for the cost of the probe's cache
        static_bytes = (probe_static_bytes - self.num_sectors * sector_cost) if probe_static_bytes is not None else 0
        # At least allocate two sectors worth of static bytes for the trash sectors
        static_bytes = max(2 * self.bytes_per_sector, static_bytes)

        budget = available_bytes - static_bytes
        # In the worst case, a step reads the whole cache, which adds to the decoder peak but not to the LM head peak
        sectors_for_decoder = (budget - decoder_peak) / (sector_cost + self.tokens_per_sector * read_cost)
        sectors_for_lm_head = (budget - post_peak) / sector_cost
        num_sectors = int(min(sectors_for_decoder, sectors_for_lm_head))

        mb = 1024**2
        logger.info(
            f"Cache sizing: {available_bytes // mb} MB available, {static_bytes // mb} MB of static tensors, decoder "
            f"peak of {decoder_peak // mb} MB + {read_cost:.0f} B per cache token read, LM head peak of "
            f"{int(post_peak) // mb} MB. Cache of {num_sectors} sectors of {self.bytes_per_sector // mb} MB."
        )
        if num_sectors <= 0:
            batch_bytes = int(static_bytes + max(decoder_peak, post_peak))
            raise MemoryError(
                f"Not enough memory for the cache: {available_bytes // mb} MB are available but a batch of "
                f"{self.max_batch_tokens} tokens already needs {batch_bytes // mb} MB. Lower `max_batch_tokens` or "
                "`max_requests_per_batch`, or raise `max_memory_percent`."
            )
        return num_sectors

    def _infer_read_cache_limit(self) -> int | None:
        """The maximum number of tokens that can be read from the cache for a single request."""
        # There is only a limit if there are only sliding window attention layers: then the limit is the sliding window
        allocators = list(self.cache_allocators.keys())
        if allocators == [SLIDING_ATTENTION]:
            return self.cache_allocators[SLIDING_ATTENTION].sliding_window  # type: ignore
        return None

    def _infer_max_decode_fast_path_length(self) -> int:
        """Infers the maximum length of a request for it to be eligible for the decode fast path. If any of the
        attention type doesn't support block tables, the fast path is not available."""
        if not all(ca.supports_block_table for ca in self.cache_allocators.values()):
            return 0
        acc = float("inf")
        for allocator in self.cache_allocators.values():
            acc = min(acc, allocator.tokens_per_page * self.max_blocks_per_request)
        return int(acc)

    def can_store_request_tokens(self, state: RequestState, request_len: int, dry_run: bool = False) -> bool:
        """Checks if the new tokens for a request can be stored in the cache. If they can, actual cache allocation is
        performed. Otherwise, this has no side effects. If the dry_run flag is set, the cache allocation is not
        performed."""
        sectors_needed = {}
        # Check if any new sectors are needed for the request
        for name, allocator in self.cache_allocators.items():
            new_sectors = allocator.needs_new_sectors(state.request_id, state.current_len(), request_len)
            if new_sectors > 0:
                sectors_needed[name] = new_sectors

        # Stop here if this is a dry run or if there are not enough free sectors
        enough_free_sectors = self.pool.num_free_sectors >= sum(sectors_needed.values())
        if dry_run or not enough_free_sectors:
            return enough_free_sectors

        # For each allocator, allocate the needed sectors (empty loop if no sector is needed)
        for name, new_sectors in sectors_needed.items():
            allocator = self.cache_allocators[name]
            for _ in range(new_sectors):
                self.pool.allocate_sector(allocator.index)

        # For each allocator, allocate the cache to the request
        for allocator in self.cache_allocators.values():
            allocator.allocate_cache_to_request(state.request_id, state.current_len(), request_len)
        return True

    def count_storable_requests(self, list_blocks_needed: list[dict[str, int]]) -> int:
        """Given a list of blocks needed for each allocator, taken in order, counts how many could allocate the cache
        needed if they were admitted one by one. This is a simulation only, no actual allocation is performed."""
        # These track the status of the simluated cache
        free_sectors = self.pool.num_free_sectors
        free_blocks = {name: self.pool.count_free_blocks(ca.index) for name, ca in self.cache_allocators.items()}
        # Loop over the requests in order, stopping when a request cannot be admitted
        for num_storable, blocks_needed in enumerate(list_blocks_needed):
            # Loop over allocators and simulate allocations
            for name, allocator in self.cache_allocators.items():
                missing_blocks = blocks_needed[name] - free_blocks[name]
                if missing_blocks > 0:
                    new_sectors = ceil(missing_blocks / allocator.blocks_per_sector)
                    if new_sectors > free_sectors:
                        return num_storable  # no more free sectors: request cannot be scheduled
                    free_sectors -= new_sectors
                    free_blocks[name] += new_sectors * allocator.blocks_per_sector
                free_blocks[name] -= blocks_needed[name]
        return len(list_blocks_needed)

    def free_blocks(self, request_id: str, no_cache: bool = False) -> None:
        """Signals all cache allocators that a request's cache can be freed. If the `no_cache` flag is passed, the freed
        blocks go straight to the free pool, and cannot be reused later."""
        for allocator in self.cache_allocators.values():
            allocator.free_blocks(request_id, no_cache)

    def free_all_requests(self, clear_ledgers: bool = False) -> None:
        """Signals all cache allocators that all requests' caches can be freed. Also clears the ledgers if requested."""
        for allocator in self.cache_allocators.values():
            allocator.free_all_requests()
            if clear_ledgers:
                allocator.ledger.reset()

    def search_prefix_match(self, request_id: str, prompt_ids: list[int]) -> int:
        """Searches the prompt for a prefix whose blocks are already in the cache. Matched blocks are shared with the
        request instead of being recomputed, and the number of matched tokens is returned."""
        if not self.use_prefix_sharing:
            return 0

        # Loop over all allocators to find the longest prefix match by all
        prefix_len = 2**32 - 1  # ~inf
        all_matched_blocks = {}
        for name, allocator in self.cache_allocators.items():
            matched_blocks = allocator.match_prefix_blocks(prompt_ids)
            # Stop if there was no prefill match
            if not matched_blocks:
                prefix_len = 0
                break
            # Otherwise, update accumulators
            else:
                prefix_len = min(prefix_len, len(matched_blocks) * allocator.tokens_per_page)
                all_matched_blocks[name] = matched_blocks

        # If there was a prefix match, acquire the blocks and return the number of matched tokens
        if prefix_len > 0:
            logger.debug(f"Found a prefix match of {prefix_len} tokens for request {request_id}")
            for name, matched_blocks in all_matched_blocks.items():
                self.cache_allocators[name].acquire_prefix_blocks(request_id, prefix_len, matched_blocks)
        return prefix_len

    def count_new_complete_blocks(self, state: RequestState, request_len: int) -> dict[str, int]:
        """Counts the blocks that the forward pass over request_len new tokens will complete for the request, so they
        can be marked for de-duplication after the forward."""
        if not self.use_prefix_sharing:
            return {}
        complete_blocks = {}
        for name, allocator in self.cache_allocators.items():
            tokens_in_last_block = state.current_len() % allocator.tokens_per_page
            new_complete_blocks = (tokens_in_last_block + request_len) // allocator.tokens_per_page
            if new_complete_blocks:
                complete_blocks[name] = new_complete_blocks
        return complete_blocks

    def mark_complete_blocks(self, state: RequestState, complete_blocks: dict[str, int]) -> None:
        """Registers the content hashes of the blocks the last forward pass completed for the request, making them
        available for de-duplication."""
        # The status can be FINISHED in async mode, because batch N+1 offloaded the request before batch N was over:
        # the block table no longer exists in that case
        if not self.allow_block_sharing or not complete_blocks or state.status == RequestStatus.FINISHED:
            return
        token_ids = state.initial_tokens + state.generated_tokens
        for name, new_complete_blocks in complete_blocks.items():
            self.cache_allocators[name].mark_complete_blocks(state.request_id, token_ids, new_complete_blocks)

    def evict_cached_blocks(self) -> bool:
        """Evicts all the blocks kept cached for de-duplication, returning them to the pool. Returns True if any block
        was evicted. Called under memory pressure: cached blocks are a bonus, never worth starving live requests."""
        evicted_any = False
        for allocator in self.cache_allocators.values():
            evicted_blocks = allocator.ledger.evict_cached_blocks()
            if evicted_blocks:
                logger.debug(f"Evicting {len(evicted_blocks)} cached blocks from allocator {allocator.index}")
                self.pool.free_blocks(allocator.index, evicted_blocks)
                evicted_any = True
        return evicted_any

    def extend_read_and_write_indices(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        read_index: list[list[int]] | None,
        write_index: list[list[int]],
    ) -> None:
        """Retrieve physical cache indices for reading KV states in the cache across all allocators. This method
        coordinates with all cache allocators to build the complete set of read indices needed for attention computation.
        When read_index is None, the batch has no cache reads and we only compute the write indices.
        The i-th list of each accumulator corresponds to the i-th allocator, in the order of the cache_allocators dict.
        """
        # Write indices are always computed
        for allocator, write_indices in zip(self.cache_allocators.values(), write_index):
            write_indices.extend(allocator.get_write_indices(request_id, past_length, query_length))
        # Read indices are only computed if there are cache indices
        if read_index is not None:
            for allocator, read_indices in zip(self.cache_allocators.values(), read_index):
                read_indices.extend(allocator.get_read_indices(request_id, past_length, query_length))

    @torch.compiler.disable  # does not play well with the views of the cache tensor
    def update(
        self,
        key_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_q, head_dim]
        value_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_q, head_dim]
        layer_idx: int,
        kwargs: dict[str, Any],  # updated in place
    ) -> tuple[torch.Tensor, torch.Tensor]:  # shape [1, num_kv_heads, seqlen_q + past_length, head_dim]
        """Updates the cache or the kwargs before entering the flash attention wrapper. The kwargs are updated in place.
        For an index-based update, the keys and values are actually updated to seqlen "seqlen_q + past_length". For a
        block table update, the KV states are untouched but the kwargs are filled with what is needed to call flash.
        """
        allocator = self.layer_to_allocator[layer_idx]

        # Select the right keywords arguments for this layer. These are always needed.
        kwargs["cu_seq_lens_k"] = kwargs["cu_seq_lens_k"][allocator.layer_type].to(torch.int32)
        kwargs["max_length_k"] = kwargs["max_length_k"][allocator.layer_type]

        # Block table "update": no real update, just prepare the kwargs for the flash call, which will update the cache
        block_table = kwargs.get("block_table")
        if block_table is not None:
            k_cache, v_cache = allocator.get_cache_for_block_table(layer_idx)
            kwargs["block_table"] = block_table[allocator.index]
            kwargs["k_cache"] = k_cache
            kwargs["v_cache"] = v_cache

        # Index-based update: actually update the cache and return full KV
        else:
            read_index = kwargs["read_index"][allocator.index]
            write_index = kwargs["write_index"][allocator.index]
            # Allocator update is done with the KV cache shape: [seqlen_q, num_kv_heads, head_dim]. We also return the
            # KV states in the same shape they were passed. It will help when we unify with regular Cache.
            key_states, value_states = (x.squeeze(0).transpose(0, 1) for x in (key_states, value_states))
            key_states, value_states = allocator.update(key_states, value_states, layer_idx, read_index, write_index)
            key_states, value_states = (x.transpose(0, 1).unsqueeze(0) for x in (key_states, value_states))

        return key_states, value_states

    def reset(self) -> None:
        """Frees the cache of all requests and returns all sectors to the global pool."""
        self.free_all_requests(clear_ledgers=True)
        self.pool.reset()

    def prepare_fork_request(
        self, src_state: RequestState, dst_req_ids: list[str], copy_src_and_dst: dict[str, tuple[list[int], list[int]]]
    ) -> list[str]:
        """Forks the cache of the source request into new requests: fully-written blocks are shared when block sharing
        is allowed, the others will be copied. Children are forked in order while the cache has room, and the ids
        actually forked are returned. The caller must handle the remaining ids another way and call perform_cache_copy.
        """
        past_length = src_state.current_len()

        # Count the number of blocks needed for each allocator
        blocks_needed = {
            name: allocator.count_non_shareable_blocks(src_state.request_id, past_length)
            for name, allocator in self.cache_allocators.items()
        }

        # Loop over the destination request ids
        forked_ids = []
        for dest_request_id in dst_req_ids:
            # Check that the child fits, in the same way as can_store_request_tokens but with direct block counts
            # TODO: this could probably be merged with can_store_request_tokens
            sectors_needed = {}
            for name, allocator in self.cache_allocators.items():
                missing_blocks = blocks_needed[name] - self.pool.count_free_blocks(allocator.index)
                if missing_blocks > 0:
                    sectors_needed[name] = ceil(missing_blocks / allocator.blocks_per_sector)
            # Stop here if the child doesn't fit
            if self.pool.num_free_sectors < sum(sectors_needed.values()):
                break
            # Allocate the needed sectors and build the child's block table
            for name, new_sectors in sectors_needed.items():
                for _ in range(new_sectors):
                    self.pool.allocate_sector(self.cache_allocators[name].index)
            for name, allocator in self.cache_allocators.items():
                src_blocks, dst_blocks = allocator.fork_blocks(src_state.request_id, dest_request_id, past_length)
                copy_src_and_dst[name][0].extend(src_blocks)
                copy_src_and_dst[name][1].extend(dst_blocks)
            forked_ids.append(dest_request_id)

        return forked_ids

    def perform_cache_copy(self, copy_src_and_dst: dict[str, tuple[list[int], list[int]]]) -> None:
        """Performs the cache copy for the given source and destination blocks."""
        for name, (src_blocks, dst_blocks) in copy_src_and_dst.items():
            if src_blocks:
                self.cache_allocators[name].copy_blocks(src_blocks, dst_blocks)

    def compute_free_capacity(self, relative: bool = True) -> int | float:
        """Returns the free capacity of the cache in bytes or as a percentage of the total capacity."""
        free_bytes = len(self.pool.free_sectors) * self.bytes_per_sector
        for allocator in self.cache_allocators.values():
            free_bytes += allocator.pool.count_free_blocks(allocator.index) * allocator.bytes_per_block
        return free_bytes / (self.num_sectors * self.bytes_per_sector if relative else 1)

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills each allocator's row of the kernel block table for the given request."""
        for allocator in self.cache_allocators.values():
            allocator.fill_block_table(request_id, past_length, query_length, block_table[allocator.index])

    # DEPRECATED METHODS
    def get_cache_for_block_table(self, layer_idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Deprecated method to get the K and V cache views for a block table update. Now baked in "update"."""
        warnings.warn(
            "The get_cache_for_block_table function is deprecated and will be removed in v5.23 .",
            FutureWarning,
            stacklevel=2,
        )
        allocator = self.layer_to_allocator[layer_idx]
        k_cache, v_cache = allocator.get_cache_for_block_table(layer_idx)
        return allocator.index, k_cache, v_cache

    def get_block_table_key(self, flash_attn_with_kvcache_fn: Any) -> str:
        """Deprecated method to get the name of the block table key for the given flash_attn_with_kvcache_fn. The
        function's signature is only inspected once. This is necessary because different version of flash have different
        names for the block table key."""
        warnings.warn(
            "The get_block_table_key function is deprecated and will be removed in v5.23 .",
            FutureWarning,
            stacklevel=2,
        )
        _block_table_key = getattr(self, "_block_table_key", None)
        if _block_table_key is None:
            kwarg_names = inspect.signature(flash_attn_with_kvcache_fn).parameters.keys()
            if "block_table" in kwarg_names:
                self._block_table_key = "block_table"
            elif "page_table" in kwarg_names:
                self._block_table_key = "page_table"
            else:
                raise ValueError(
                    f"flash_attn_with_kvcache_fn does not have a block_table or page_table argument: "
                    f"{inspect.signature(flash_attn_with_kvcache_fn)}"
                )
        return self._block_table_key


# TODO: can we get rid of this class?
class HookedDummyCache:
    def __init__(self) -> None:
        self.hooked = {}

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.hooked[layer_idx] = [
            (key_states.shape, key_states.dtype, key_states.device),
            (value_states.shape, value_states.dtype, value_states.device),
        ]
        return key_states, value_states

    def get_query_offset(self, layer_idx: int = 0) -> int:
        return 0

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return query_length, 0


def hook_cache_tensor_meta(
    model: "PreTrainedModel", num_tokens: int
) -> dict[int, list[tuple[torch.Size, torch.dtype, torch.device]]]:
    """Runs a forward pass on a hooked model to get the dimensions of the model's cached tensors."""
    cache = HookedDummyCache()
    with torch.no_grad():
        model(
            input_ids=torch.zeros((1, num_tokens), dtype=torch.long, device=model.device),
            position_ids=torch.arange(num_tokens, device=model.device).unsqueeze(0),
            attention_mask=None,
            cache=cache,
        )
    return cache.hooked
