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
from math import ceil, lcm
from typing import Any

import torch

from ...configuration_utils import PreTrainedConfig
from ...generation.configuration_utils import ContinuousBatchingConfig
from ...utils.generic import is_flash_attention_requested
from .cache_allocators import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    CacheAllocator,
    CachePool,
    FullAttentionCacheAllocator,
    SlidingAttentionCacheAllocator,
)
from .distributed import DistributedHelper
from .requests import RequestState, RequestStatus, get_device_and_memory_breakdown, logger
from .utils import find_head_dim, find_num_key_value_heads


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
    supported_attention_types = {FULL_ATTENTION, SLIDING_ATTENTION}

    layer_counts = {}
    for i, layer_type in enumerate(layer_types):
        if layer_type not in supported_attention_types:
            raise ValueError(f"Invalid layer type: {layer_type}")
        layer_counts[layer_type] = layer_counts.get(layer_type, []) + [i]
    return layer_counts


# Maps each attention type to the allocator class handling its cache
ATTN_TYPE_TO_ALLOCATOR = {
    FULL_ATTENTION: FullAttentionCacheAllocator,
    SLIDING_ATTENTION: SlidingAttentionCacheAllocator,
}


class PagedAttentionCache:
    """
    High-level manager for any cache used by continuous batching. This object own the cache tensors and distributes
    sectors to sub-allocators, each for a different kind of cache (full-attention layers, MSA layers, embeddings cache,
    etc.).

    Virtually, the cache is allocated per layer in the form of *pages*: one page holds the whole cache (e.g. both keys
    and values) of one layer for a number N of tokens. When several layers share a similar attention type, say full
    attention, we group them and allocate the cache per *block*. For instance, this is how a block of full-attention
    cache looks like, if there are 3 full-attention layers:

    [ LAYER 0 KEYS | LAYER 0 VALUES | LAYER 1 KEYS | LAYER 1 VALUES | LAYER 2 KEYS | LAYER 2 VALUES ]
    [ ----------- PAGE 0 ---------- | ----------- PAGE 1 ---------- | ----------- PAGE 2 ---------- ]
    [ ------------------------------------------- BLOCK 0 ------------------------------------------]

    For a given attention type, there is only one allocator, which is responsible for the cache of all layers with that
    attention type. For instance, in a 4 layers model with layer types ["full", "sliding", "sliding", "sliding"], there
    are two allocators: one FullAttentionCacheAllocator (for layer 0) and one SlidingAttentionCacheAllocator (for layers
    1, 2 and 3).
    Because of this, the block size for the full attention allocator is not the same as the block size for the sliding
    attention allocator. To compensate for that, we define * sectors * : a sector is a contiguous region of the cache
    that is allocated to a single allocator. The size of a sector is computed so that all cache allocators can divide a
    sector into blocks.
    For instance, if there are 3 sliding layers for 1 full attention layer, then storing a single token would require 3
    sliding pages (one per layer) and 1 full page. The amount of memory needed to store all sliding pages is x3 the
    amount of memory needed to store one full page. Hence, one block of sliding cache is 3 blocks of full cache. Taking
    the LCM(1, 3) = 3, we compute the size of a sector to be 3 blocks of full cache, or 1 block of sliding cache.

                           [ -------------------------------------- SECTOR -------------------------------------- ]
    used for full attn:    [ ------ FULL BLOCK 0 ------ | ------ FULL BLOCK 1 ------ | ------ FULL BLOCK 2 ------ ]
    used for sliding attn: [ ---------------------------------- SLIDING BLOCK 0 --------------------------------- ]

    Physically, the cache is stored on a single flat tensor, whose first two sectors are never-allocated trash
    sectors used by padding tokens (see FullAttentionCacheAllocator.register_cache_tensor).
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        device: torch.device | str,
        distributed_helper: DistributedHelper,
        dtype: torch.dtype = torch.float16,
        model_supports_logits_to_keep: bool = False,
    ) -> None:
        """Initialize a paged attention cache for efficient memory usage. Also turns in prefix sharing if the model has
        only full attention layers.

        Args:
            config: Model configuration
            continuous_batching_config: Continuous batching configuration containing cache parameters
            device: Device for the cache tensors
            distributed_helper: TP-aware helper. Used to dispatch attention heads and ensure coherent cache size
            dtype: Data type of the activation and the cache (for now, these are the same)
            model_supports_logits_to_keep: When True, memory sizing charges the LM head peak per request instead of
                per batch token, since the model slices hidden states before the LM head
        """
        self.config = config
        self.dtype = dtype
        self.device = device
        self.max_blocks_per_request = continuous_batching_config.max_blocks_per_request or 0  # if not resolved, off

        # If the KV heads are TP'ed, each KV head is dispatched to a different GPU, so the effective number of KV heads
        # per GPU is simply divided by the TP size. We need to solve this before we can construct the cache allocators.
        num_key_value_heads = find_num_key_value_heads(config)
        if distributed_helper.tp_size > 1 and distributed_helper.are_kv_heads_tp_ed():
            if num_key_value_heads % distributed_helper.tp_size != 0:
                raise ValueError(
                    f"Number of key value heads {num_key_value_heads} must be divisible by tensor parallel size"
                    f"{distributed_helper.tp_size}."
                )
            num_key_value_heads //= distributed_helper.tp_size

        # Construct the necessary cache allocator for each attention type
        ca_kwargs = {
            "num_key_value_heads": num_key_value_heads,
            "config": config,
            "cache_dtype": self.dtype,
            "page_size": continuous_batching_config.fa_page_size,
            "allow_block_sharing": continuous_batching_config.allow_block_sharing,
            # "is_tp_enabled": distributed_helper.tp_size > 1,
        }
        self.cache_allocators: dict[str, CacheAllocator] = {}
        self.layer_to_allocator: dict[int, CacheAllocator] = {}
        for index, (attn_type, layer_indices) in enumerate(group_layers_by_attn_type(config).items()):
            # Create the allocator and register it by name and indices
            allocator = ATTN_TYPE_TO_ALLOCATOR[attn_type](index=index, layer_indices=layer_indices, **ca_kwargs)
            self.cache_allocators[attn_type] = allocator
            self.layer_to_allocator = self.layer_to_allocator | dict.fromkeys(layer_indices, allocator)

        # To have the maximal granularity while ensuring alignment for all cache allocators, we compute the LCM of all
        # cache allocator block sizes AND a default alignment of 128 bytes
        self.bytes_per_sector = lcm(*(ca.bytes_per_block for ca in self.cache_allocators.values()), 128)

        # We plan for the "worst" sector, ie. the one with the most tokens and so the biggest index tensor
        self.tokens_per_sector = max(
            (self.bytes_per_sector // ca.bytes_per_block) * ca.tokens_per_page for ca in self.cache_allocators.values()
        )
        # Bytes of one full-attention page, used to convert the num_pages config attribute to and from bytes
        bytes_per_fa_page = FullAttentionCacheAllocator.get_bytes_per_page(
            num_key_value_heads, find_head_dim(config), self.dtype, continuous_batching_config.fa_page_size
        )

        max_batch_tokens, num_sectors = PagedAttentionMemoryHandler(
            config=config,
            cb_config=continuous_batching_config,
            dtype=self.dtype,
            bytes_per_sector=self.bytes_per_sector,
            tokens_per_sector=self.tokens_per_sector,
            bytes_per_fa_page=bytes_per_fa_page,
            attn_types=list(self.cache_allocators.keys()),
            model_supports_logits_to_keep=model_supports_logits_to_keep,
        ).infer_max_batch_tokens_and_num_sectors()

        # For TP, align max_batch_tokens and num_blocks to the minimal value across the TP group
        if distributed_helper.tp_size > 1:
            sync = torch.tensor([max_batch_tokens, num_sectors], device=self.device, dtype=torch.int64)
            distributed_helper.tp_all_reduce_min(sync)
            max_batch_tokens, num_sectors = int(sync[0].item()), int(sync[1].item())
        # Add the inferred attributes to the class
        self.max_batch_tokens, self.num_sectors = max_batch_tokens, num_sectors
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

        # For block table support, we lazy init the name of the block table key
        self._block_table_key = None

        # Helper attribute: the cache capacity expressed in full-attention pages
        self.num_pages = non_trash_bytes // bytes_per_fa_page
        # Helper attributes for the scheduler
        self.read_cache_limit = self._infer_read_cache_limit()
        self.max_decode_fast_path_length = self._infer_max_decode_fast_path_length()
        # Helper attribute for the input/output classes
        self.max_tokens_read = max(ca.tokens_per_page * ca.num_blocks for ca in self.cache_allocators.values())

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

    def free_blocks(self, request_id: str) -> None:
        """Signals all cache allocators that a request's cache can be freed."""
        for allocator in self.cache_allocators.values():
            allocator.free_blocks(request_id)

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

    def update(
        self,
        key_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_q, head_dim]
        value_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_q, head_dim]
        layer_idx: int,
        read_index: list[torch.Tensor],  # one tensor per attention group
        write_index: list[torch.Tensor],  # one tensor per attention group
    ) -> tuple[torch.Tensor, torch.Tensor]:  # shape [seqlen_q + past_length, num_kv_heads, head_dim]
        """Updates the cache with new key-value states for a specific layer and retrieves the KV states needed for the
        attention computation. The actual work is dispatched to the allocator in charge of the layer, using the read
        and write indices prepared for its group."""
        allocator = self.layer_to_allocator[layer_idx]
        layer_read_index = read_index[allocator.index]
        layer_write_index = write_index[allocator.index]
        return allocator.update(key_states, value_states, layer_idx, layer_read_index, layer_write_index)

    def get_cache_for_block_table(self, layer_idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Returns the K and V cache views for a block table update."""
        allocator = self.layer_to_allocator[layer_idx]
        k_cache, v_cache = allocator.get_cache_for_block_table(layer_idx)
        return allocator.index, k_cache, v_cache

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

    # def blocks_needed(self, num_requested_blocks: int, allocated_blocks: int) -> int:
    #     """Returns the number of physical blocks needed to allocate (num_requested_blocks) blocks to a request that
    #     already has (allocated_blocks) blocks. The number of newly allocated blocks needed is predicted by the
    #     following rules:
    #     - for full attention groups: since there is no sliding window for full attention layers, one requested block is
    #         always equivalent to one newly allocated block for EACH full attention group
    #     - for sliding window groups: because of the sliding window, the number of blocks allocated to a request is
    #         capped. Using the number of already (allocated_blocks) we can compute the number of new blocks to actually
    #         allocate to the request, which can be lower than the number of requested blocks. That number is the same for
    #         all sliding window groups, as only one sliding window size is supported.
    #     """
    #     # This is not in a branch, because it is very rare to have zero full attention layer
    #     needed_blocks = num_requested_blocks * self.num_full_attention_groups
    #     # Only take this branch if the model has sliding window attention layers
    #     if self.num_sliding_attention_groups:
    #         blocks_left = max(self.max_sliding_window_blocks_per_request - allocated_blocks, 0)
    #         needed_blocks += min(blocks_left, num_requested_blocks) * self.num_sliding_attention_groups
    #     return needed_blocks

    # def will_allocation_be_successful(self, num_requested_blocks: int, allocated_blocks: int) -> bool:
    #     """Returns a boolean indicating if the allocation of (num_requested_blocks) blocks will be successful."""
    #     return self.blocks_needed(num_requested_blocks, allocated_blocks) <= self.get_num_free_blocks()

    # def blocks_in_use(self, request_id: str) -> int:
    #     """Returns the total number of physical blocks currently referenced by a request across all layer groups."""
    #     return sum(len(cm.block_table.get(request_id, ())) for cm in self.group_cache_managers)

    # def allocate_blocks(self, n_blocks: int, request_id: str, allocated_blocks: int) -> int | None:
    #     """Allocate cache blocks across all layer groups for a given request. Actual allocation is done by the cache
    #     managers, and this method only returns the maximum number of blocks actually allocated across all managers."""
    #     # First check allocation will be successful before starting, to avoid partial allocations
    #     if not self.will_allocation_be_successful(n_blocks, allocated_blocks):
    #         return None
    #     # Allocate blocks across all cache managers
    #     max_allocated = 0
    #     for cm in self.group_cache_managers:
    #         num_allocated_blocks = cm.allocate_blocks(n_blocks, request_id, self._block_manager)
    #         if num_allocated_blocks is None:
    #             raise ValueError(f"Failed to allocate {n_blocks} blocks for request {request_id}")
    #         max_allocated = max(max_allocated, num_allocated_blocks)
    #     return max_allocated

    # def free_blocks(self, request_id: str) -> None:
    #     """Free all allocated cache blocks for a given request across all layer groups. Actual deallocation is done
    #     by the cache managers."""
    #     for cm in self.group_cache_managers:
    #         cm.free_blocks(request_id, self._block_manager)

    # def get_num_free_blocks(self) -> int:
    #     """Get the current number of unallocated blocks available for new requests."""
    #     return self._block_manager.num_free_blocks

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills each allocator's row of the kernel block table for the given request."""
        for allocator in self.cache_allocators.values():
            allocator.fill_block_table(request_id, past_length, query_length, block_table[allocator.index])

    def get_block_table_key(self, flash_attn_with_kvcache_fn: Any) -> str:
        """A function to get the name of the block table key for the given flash_attn_with_kvcache_fn. The function's
        signature is only inspected once. This is necessary because different version of flash have different names for
        the block table key."""
        if self._block_table_key is None:
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


# TODO: BUG: check this class in details (moving on rn)
class PagedAttentionMemoryHandler:
    """Determines the max batch tokens (M) and number of cache sectors (S) for the paged attention cache, given the
    available GPU memory.

    M is fixed upfront: it comes from the config, or defaults to 8192. Every input tensor (IO buffers and activation
    peaks) scales linearly with M, so the memory they occupy is then known, and the rest of the available memory is
    turned into cache sectors. Each sector costs its cache bytes plus the linear per-cache-token tensors it entails
    (cache reads and read indices). The attention masks, which depend on both M and S, are planned for last: once S
    is decided, we remove as many sectors as needed to pay for them.
    """

    _default_max_batch_tokens = 8192

    def __init__(
        self,
        config: PreTrainedConfig,
        cb_config: ContinuousBatchingConfig,
        dtype: torch.dtype,
        bytes_per_sector: int,
        tokens_per_sector: int,
        bytes_per_fa_page: int,
        attn_types: list[str],
        model_supports_logits_to_keep: bool = False,
    ) -> None:
        """Initialize the memory handler with the model configuration, the continuous batching configuration, the data
        type of the activation and the cache, and the sector geometry computed by the PagedAttentionCache."""
        self.config = config
        self.cb_config = cb_config
        self.cache_dtype = dtype
        self.activation_dtype = dtype
        self.model_supports_logits_to_keep = model_supports_logits_to_keep

        self.bytes_per_sector = bytes_per_sector
        self.tokens_per_sector = tokens_per_sector
        self.bytes_per_fa_page = bytes_per_fa_page
        self.trash_bytes = 2 * bytes_per_sector  # the two trash sectors at the start of the cache tensor
        self.num_groups = len(attn_types)

        # TODO: when we generalize to allow for block-attn, we can use `num_attention_masks=len(set(attn_types))`
        if is_flash_attention_requested(self.config):
            self.num_attention_masks = 0
        else:
            self.num_attention_masks = 2 if SLIDING_ATTENTION in attn_types else 1

        if cb_config.max_blocks_per_request is None:
            self.max_blocks_per_request = cb_config.fallback_max_blocks_per_request
        else:
            self.max_blocks_per_request = cb_config.max_blocks_per_request

        # This is the number of output rows for the output_ids tensor
        self.num_output_rows = 2 if cb_config.return_logprobs else 1
        # This account for the set of 2 IOs if async batching is used
        self.io_multiplier = 2 if cb_config.use_async_batching else 1
        self.available_memory = self.get_available_memory()

    @property
    def bytes_per_batch_token(self) -> int:
        """The memory cost of one batch token: its share of the IO tensors plus the largest activation peak, which is
        either the LM head peak (hidden states turned into fp32 logits) or the attention peak (hidden states, query
        projection and new key / value states)."""
        i = torch.int32.itemsize
        a = self.activation_dtype.itemsize
        head_dim = find_head_dim(self.config)

        # Only one activation peak is live at a time, so we reserve for the largest
        if self.model_supports_logits_to_keep:  # turns the memory cost constant thanks to slicing
            lm_head_peak = 0
        else:
            lm_head_peak = self.config.hidden_size * a + self.config.vocab_size * torch.float32.itemsize
        attention_peak = a * (
            self.config.hidden_size  # hidden states, shape [M, hidden_size]
            + self.config.num_attention_heads * head_dim  # query projection, shape [M, num_heads * head_dim]
            + 2 * find_num_key_value_heads(self.config) * head_dim  # new K and V states
        )
        io_bytes = self.io_multiplier * (
            7 * i  # bulk_input: [7, M] int32, packed as 7 rows
            + self.num_output_rows * i  # output_ids: [num_output_rows, M] int32
            + self.num_groups * self.max_blocks_per_request * i  # block_table (zero when fast-decode is off)
            + self.num_groups * 8  # write_index: [num_groups, M] int64
            + self.num_groups * 8  # read_index: [num_groups, N + M] (M part, int64)
        )
        return max(lm_head_peak, attention_peak) + io_bytes

    @property
    def bytes_per_cache_token(self) -> int:
        """The memory cost of one readable cache token beside the cache itself: the old key and value states read from
        the cache at the attention peak, plus its share of the read indices."""
        num_key_value_heads = find_num_key_value_heads(self.config)
        kv_read_bytes = 2 * num_key_value_heads * find_head_dim(self.config) * self.activation_dtype.itemsize
        read_index_bytes = self.io_multiplier * self.num_groups * 8  # read_index: [num_groups, N + M] (N part, int64)
        return kv_read_bytes + read_index_bytes

    @property
    def bytes_per_cache_sector(self) -> int:
        """The total memory cost of one cache sector: the sector itself plus the per-cache-token tensors it entails."""
        return self.bytes_per_sector + self.tokens_per_sector * self.bytes_per_cache_token

    @property
    def fixed_overhead_bytes(self) -> int:
        """Reservations that scale with neither batch tokens nor cache size: the trash sectors and, when the model
        slices hidden states with logits_to_keep, the LM head peak of the per-request logit rows."""
        overhead = self.trash_bytes
        if self.model_supports_logits_to_keep:
            a = self.activation_dtype.itemsize
            # 1024 mirrors FALLBACK_DEFAULTS["max_requests_per_batch"] in initialization.py (import would be circular)
            max_logit_rows = self.cb_config.max_requests_per_batch or 1024
            lm_head_row = self.config.hidden_size * a + self.config.vocab_size * torch.float32.itemsize
            overhead += max_logit_rows * lm_head_row
        return overhead

    def attention_mask_bytes(self, max_batch_tokens: int, num_sectors: int) -> int:
        """The memory cost of the attention masks, of shape [1, 1, M, N + M], for given M and number of sectors."""
        num_readable_tokens = num_sectors * self.tokens_per_sector
        mask_numel = max_batch_tokens * (num_readable_tokens + max_batch_tokens)
        return self.io_multiplier * self.num_attention_masks * self.activation_dtype.itemsize * mask_numel

    def get_available_memory(self) -> int:
        """Calculate available GPU memory for cache allocation in bytes, accounting for the maximum memory percent limit
        fixed by the continuous batching config."""
        _, total, reserved, allocated = get_device_and_memory_breakdown()
        available_memory = total - max(allocated, reserved)
        available_memory = int(available_memory * self.cb_config.max_memory_percent)  # type: ignore
        logger.info(f"Memory available for cache allocation: {available_memory // 1024**2} MB")
        return available_memory

    def num_sectors_from_config(self) -> int | None:
        """Converts the `num_pages` config attribute into a number of sectors. `num_pages` is a capacity target:
        the number of full-attention pages the cache should be able to hold."""
        if self.cb_config.num_pages is None:
            return None
        target_bytes = self.cb_config.num_pages * self.bytes_per_fa_page
        return max(1, ceil(target_bytes / self.bytes_per_sector))

    def infer_max_batch_tokens_and_num_sectors(self) -> tuple[int, int]:
        """Infers max_batch_tokens and num_sectors based on the available memory. Max batch tokens comes from the
        config or a default of 8192. Unless the number of sectors is also fixed by the config, the memory left once
        the M-proportional tensors are paid for becomes cache sectors, minus the sectors removed to pay for the
        attention masks."""
        max_batch_tokens = self.cb_config.max_batch_tokens
        if max_batch_tokens is None:
            max_batch_tokens = self._default_max_batch_tokens

        num_sectors = self.num_sectors_from_config()
        if num_sectors is None:
            # Memory left for the cache once the M-proportional tensors and the fixed overhead are paid for
            cache_memory = self.available_memory - self.fixed_overhead_bytes
            cache_memory -= max_batch_tokens * self.bytes_per_batch_token
            num_sectors = cache_memory // self.bytes_per_cache_sector
            # Plan for the attention masks by removing sectors: each removed sector frees its own cost and also
            # shrinks the masks, so the removal denominator includes both and the sizing stays one-shot.
            mask_bytes = self.attention_mask_bytes(max_batch_tokens, num_sectors)
            mask_bytes_per_sector = self.attention_mask_bytes(max_batch_tokens, num_sectors + 1) - mask_bytes
            num_sectors -= ceil(mask_bytes / (self.bytes_per_cache_sector + mask_bytes_per_sector))

        return self._check_footprint(max_batch_tokens, num_sectors)

    def _check_footprint(self, max_batch_tokens: int, num_sectors: int) -> tuple[int, int]:
        """Checks if the footprint of the cache is within the available memory."""
        memory_footprint = self.compute_memory_footprint(max_batch_tokens, num_sectors)
        if memory_footprint > self.available_memory:
            raise MemoryError(
                f"Memory footprint {memory_footprint} is more than available memory {self.available_memory}"
            )
        if max_batch_tokens <= 0 or num_sectors <= 0:
            raise ValueError(f"Invalid values: {max_batch_tokens = }, {num_sectors = }")
        return max_batch_tokens, num_sectors

    def compute_memory_footprint(self, max_batch_tokens: int, num_sectors: int) -> int:
        """Computes the memory footprint of the cache and every tensor that scales with it, as the exact inverse of
        the sizing performed in infer_max_batch_tokens_and_num_sectors."""
        return (
            self.fixed_overhead_bytes
            + max_batch_tokens * self.bytes_per_batch_token
            + num_sectors * self.bytes_per_cache_sector
            + self.attention_mask_bytes(max_batch_tokens, num_sectors)
        )
