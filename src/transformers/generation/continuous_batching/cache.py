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
from math import floor, gcd, lcm, sqrt
from typing import Any

import torch

from ...configuration_utils import PreTrainedConfig
from ...generation.configuration_utils import ContinuousBatchingConfig
from ...utils.generic import is_flash_attention_requested
from .cache_allocators import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    CacheAllocator,
    FullAttentionCacheAllocator,
    SlidingAttentionCacheAllocator,
)
from .distributed import DistributedHelper
from .initialization import resolve_max_memory_percent
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


class PagedAttentionCache:
    """
    High-level manager for any cache used by continuous batching. This object own the cache tensors and distributes
    sectors to sub-allocators, each for a different kind of cache (full-attention layers, MSA layers, embeddings cache,
    etc.).

    Virually, the cache is allocated per layer in the form of * pages* : one page holds the cache for a number N of
    tokens. When several layers share a similar attention type, say full attention, we group them and allocate the cache
    per * block *. For instance, this is how a block of full-attention cache looks like, if there are 3 full-attention
    layers:

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

    Physically, the cache is stored on a single tensor.

    # TODO: BUG: add ascii drawing
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        device: torch.device | str,
        distributed_helper: DistributedHelper,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize a paged attention cache for efficient memory usage. Also turns in prefix sharing if the model has
        only full attention layers.

        Args:
            config: Model configuration
            continuous_batching_config: Continuous batching configuration containing cache parameters
            device: Device for the cache tensors
            distributed_helper: TP-aware helper. Used to dispatch attention heads and ensure coherent cache size
            dtype: Data type of the activation and the cache (for now, these are the same)
        """
        self.config = config
        self.dtype = dtype
        self.device = device
        self.tokens_per_block = continuous_batching_config.block_size
        self.max_blocks_per_request = continuous_batching_config.max_blocks_per_request

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
            "tokens_per_block": self.tokens_per_block,
            # "allow_block_sharing": self.allow_block_sharing,
            # "is_tp_enabled": distributed_helper.tp_size > 1,
        }
        self.cache_allocators: dict[str, CacheAllocator] = {}
        for attn_type, layer_indices in group_layers_by_attn_type(config).items():
            if attn_type == FULL_ATTENTION:
                self.cache_allocators[attn_type] = FullAttentionCacheAllocator(layer_indices=layer_indices, **ca_kwargs)
            elif attn_type == SLIDING_ATTENTION:
                self.cache_allocators[attn_type] = SlidingAttentionCacheAllocator(layer_indices=layer_indices, **ca_kwargs)
            else:
                raise ValueError(f"Invalid attention type: {attn_type}")

        # To have the maximal granularity while ensuring alignment for all cache allocators, we compute the LCM of all
        # cache allocator block sizes AND a default alignment of 128 bytes
        self.bytes_per_sector = lcm([ca.bytes_per_block for ca in self.cache_allocators.values()] + [128])

        # TODO: BUG: update this before merging
        max_batch_tokens, num_sectors = PagedAttentionMemoryHandler(
            config=config,
            continuous_batching_config=continuous_batching_config,
            dtype=self.dtype,
            sector_size=self.bytes_per_sector,
            attn_types=list(self.cache_allocators.keys()),
        ).infer_max_batch_tokens_and_num_blocks()

        # For TP, align max_batch_tokens and num_blocks to the minimal value across the TP group
        if distributed_helper.tp_size > 1:
            sync = torch.tensor([max_batch_tokens, num_sectors], device=self.device, dtype=torch.int64)
            distributed_helper.tp_all_reduce_min(sync)
            max_batch_tokens, num_sectors = int(sync[0].item()), int(sync[1].item())

        # Add the inferred attributes to the class
        self.max_batch_tokens = max_batch_tokens
        self.num_sectors = num_sectors
        mb_per_sector = self.bytes_per_sector / 1024**2
        logger.info(
            f"Paged cache initialized: {self.max_batch_tokens = }, {self.num_sectors = }, {mb_per_sector = }"
        )

        # Cache is dimensionned so that for any allocator, there is a trash zone of at least two pages
        non_trash_bytes = self.num_sectors * self.bytes_per_sector
        trash_bytes = 2 * max(ca.bytes_per_page for ca in self.cache_allocators.values())
        self.cache_tensor = torch.zeros(non_trash_bytes + trash_bytes, dtype=torch.uint8, device=self.device)
        # Distribute the cache across all allocators
        for allocator in self.cache_allocators.values():
            allocator.register_cache_tensor(self.bytes_per_sector, non_trash_bytes, self.cache_tensor)

        # Sector-tracking data structures
        self.free_sectors = list(range(self.num_sectors))
        # Block management data structures
        self.allow_block_sharing = continuous_batching_config.allow_block_sharing
        allocators_can_share = all(ca.supports_block_sharing for ca in self.cache_allocators.values())
        self.use_prefix_sharing = self.allow_block_sharing and allocators_can_share
        # self._block_manager = BlockManager(num_blocks, self.block_size, tp_on=distributed_helper.tp_size > 1)

        # For block table support, we lazy init the name of the block table key
        self._block_table_key = None

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
        """Infers the maximum length of a request for it to be eligible for the decode fast path."""
        acc = float("inf")
        for allocator in self.cache_allocators.values():
            acc = min(acc, allocator.tokens_per_page * self.max_blocks_per_request)
        return int(acc)

    def can_store_request_tokens(self, state: RequestState, request_len: int) -> bool:
        """Checks if the new tokens for a request can be stored in the cache. If they can, actual cache allocation is
        performed. Otherwise, this has no side effects."""
        sectors_needed = {}
        # Check if any new sectors are needed for the request
        for name, allocator in self.cache_allocators.items():
            new_sectors = allocator.needs_new_sectors(state.request_id, state.current_len(), request_len)
            if new_sectors > 0:
                sectors_needed[name] = new_sectors

        if sectors_needed:
            # Stop if there are not enough free sectors
            if len(self.free_sectors) < sum(sectors_needed.values()):
                return False
            # For each allocator, allocate the needed sectors
            for name, new_sectors in sectors_needed.items():
                allocator = self.cache_allocators[name]
                for _ in range(new_sectors):
                    sector_id = self.free_sectors.pop()
                    allocator.allocate_new_sector(sector_id)

        # For each allocator, allocate the cache to the request
        for allocator in self.cache_allocators.values():
            allocator.allocate_cache_to_request(state.request_id, state.current_len(), request_len)
        return True

    def free_blocks(self, request_id: str) -> None:
        """Signals all cache allocators that a request's cache can be freed."""
        for allocator in self.cache_allocators.values():
            allocator.free_blocks(request_id)

    def free_all_requests(self) -> None:
        """Signals all cache allocators that all requests' caches can be freed."""
        for allocator in self.cache_allocators.values():
            allocator.free_all_requests()

    def extend_read_and_write_indices(self, request_id: str, past_length: int, query_length: int, read_index: list[list[int]] | None, write_index: list[list[int]]) -> None:
        """Retrieve physical cache indices for reading KV states in the cache across all allocators. This method
        coordinates with all cache allocators to build the complete set of read indices needed for attention computation.
        When read_index is None, the batch has no cache reads and we only compute the write indices.
        """
        # Write indices are always computed
        for allocator, write_indices in zip(self.cache_allocators.values(), write_index):
            write_indices.extend(allocator.get_write_indices(request_id, past_length, query_length))
        # Read indices are only computed if there are cache indices
        if read_index is not None:
            for allocator, read_indices in zip(self.cache_allocators.values(), read_index):
                read_indices.extend(allocator.get_read_indices(request_id, past_length, query_length))


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

    # def extend_read_and_write_indices(
    #     self,
    #     request_id: str,
    #     past_length: int,
    #     query_length: int,
    #     read_index: list[list[int]] | None,
    #     write_index: list[list[int]],
    # ) -> None:
    #     """Retrieve physical cache indices for reading KV states in the cache across all layer groups. This method
    #     coordinates with all cache managers to build the complete set of read indices needed for attention computation.
    #     When read_index is None, the batch has no cache reads and we only compute the write indices.
    #     """
    #     # Write indices are always computed
    #     for cm, write_indices in zip(self.group_cache_managers, write_index):
    #         write_indices.extend(cm.get_write_indices(request_id, past_length, query_length))
    #     # Read indices are only computed if there are cache indices
    #     if read_index is not None:
    #         for cm, read_indices in zip(self.group_cache_managers, read_index):
    #             read_indices.extend(cm.get_read_indices(request_id, past_length, query_length))

    # def fill_block_table(
    #     self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    # ) -> None:
    #     for i, cm in enumerate(self.group_cache_managers):
    #         cm.fill_block_table(request_id, past_length, query_length, block_table[i])

    # def get_seqlens_k(self, past_length: int, query_length: int) -> dict[str, int]:
    #     """Retrieve the key sequence length for the given request_id across all layer types. Returns a dictionary of
    #     layer types to their corresponding key sequence lengths."""
    #     seqlens_k = {}
    #     if self.num_full_attention_groups > 0:
    #         seqlens_k["full_attention"] = past_length + query_length
    #     if self.num_sliding_attention_groups > 0:
    #         seqlens_k["sliding_attention"] = query_length + min(past_length, self.config.sliding_window - 1)
    #     # NOTE: when we add more attention types / different sliding windows, we can go back to looping over CMs
    #     return seqlens_k

    # def update(
    #     self,
    #     key_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_kv, head_dim]
    #     value_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_kv, head_dim]
    #     layer_idx: int,
    #     read_index: list[torch.Tensor],  # shape [num_layer_groups, seqlen_kv + past_length]
    #     write_index: list[torch.Tensor],  # shape [num_layer_groups, seqlen_q]
    # ) -> tuple[torch.Tensor, torch.Tensor]:  # shape [seqlen_kv + past_length, num_kv_heads, head_dim]
    #     """Update the cache with new key-value states for a specific layer, and retrieves the relevant KV states from
    #     the cache for attention computation. The behavior differs based on the layer's attention type:

    #     - Full attention: New KV states are written to cache, then complete sequence is read from cache
    #     - Sliding window: Old KV is read from cache along with extra spaces for the new KV, then new KV is written to
    #         cache. This is because new KV might overwrite the old KV, so we need to read the old KV first.

    #     When the layer's read index is empty, the batch has no cache reads (all requests are non-chunked prefills): we
    #     only write to the cache and return the input KV states directly, skipping the index_select read-back.

    #     Returns the complete KV states (cached + new) for attention computation.
    #     """
    #     # Retrieve the layer write index and the relevant cache tensors
    #     group_idx, layer_idx_in_group = self.layer_index_to_group_indices[layer_idx]
    #     layer_read_index = read_index[group_idx]
    #     layer_write_index = write_index[group_idx]
    #     k_cache = self.key_cache[layer_idx_in_group]
    #     v_cache = self.value_cache[layer_idx_in_group]
    #     # Transpose the key and value states to match the cache shape, after which shape is [seqlen_kv, num_kv_heads, head_dim]
    #     key_states = key_states.transpose(1, 2).squeeze(0)
    #     value_states = value_states.transpose(1, 2).squeeze(0)

    #     # Case: write-only, no cache read. The input KV states already contain everything the attention needs.
    #     if layer_read_index.numel() == 0:
    #         k_cache.index_copy_(0, layer_write_index, key_states)
    #         v_cache.index_copy_(0, layer_write_index, value_states)
    #         return key_states, value_states

    #     # Case: full attention
    #     sliding_window = self.sliding_windows[layer_idx]
    #     if sliding_window == 1:
    #         k_cache.index_copy_(0, layer_write_index, key_states)
    #         v_cache.index_copy_(0, layer_write_index, value_states)
    #         key_states_with_cache = torch.index_select(k_cache, 0, layer_read_index)
    #         value_states_with_cache = torch.index_select(v_cache, 0, layer_read_index)

    #     # Case: sliding window -- we  need to be careful of read/write order because of chunked prefill, because it's
    #     # the only case where you may write over cache you need to use
    #     else:
    #         # Sentinel positions in read_index mark new-token slots; index_select reads garbage there,
    #         # then masked_scatter_ overwrites them with the actual new key/value states.
    #         mask = (layer_read_index == self.sentinel_index).unsqueeze(-1).unsqueeze(-1)
    #         key_states_with_cache = torch.index_select(k_cache, 0, layer_read_index)
    #         key_states_with_cache.masked_scatter_(mask, key_states)
    #         value_states_with_cache = torch.index_select(v_cache, 0, layer_read_index)
    #         value_states_with_cache.masked_scatter_(mask, value_states)
    #         # Write new KV values to the cache (padding slots in write_index point to the trash position)
    #         k_cache.index_copy_(0, layer_write_index, key_states)
    #         v_cache.index_copy_(0, layer_write_index, value_states)

    #     # Return the new KV values
    #     return key_states_with_cache, value_states_with_cache

    # def get_block_table_key(self, flash_attn_with_kvcache_fn: Any) -> str:
    #     """A function to get the name of the block table key for the given flash_attn_with_kvcache_fn. The function's
    #     signature is only inspected once. This is necessary because different version of flash have different names for
    #     the block table key."""
    #     if self._block_table_key is None:
    #         kwarg_names = inspect.signature(flash_attn_with_kvcache_fn).parameters.keys()
    #         if "block_table" in kwarg_names:
    #             self._block_table_key = "block_table"
    #         elif "page_table" in kwarg_names:
    #             self._block_table_key = "page_table"
    #         else:
    #             raise ValueError(
    #                 f"flash_attn_with_kvcache_fn does not have a block_table or page_table argument: {inspect.signature(flash_attn_with_kvcache_fn)}"
    #             )
    #     return self._block_table_key

    # def search_prefix_match(self, request_id: str, prompt_ids: list[int]) -> int:
    #     """Searches for a prefix match in the cache for the given (prompts_ids). If one is found, we reference the
    #     matching blocks in the (request_id), increase the reference count of the blocks and return the number of blocks
    #     that match. If no prefix match is found, we return 0."""
    #     current_hash = None
    #     allocated_blocks = []
    #     for b in range(len(prompt_ids) // self.block_size):
    #         tokens = prompt_ids[b * self.block_size : (b + 1) * self.block_size]
    #         # Prefix sharing is only supported when there is only one full attention layer group, so group_id=0.
    #         current_hash = self._block_manager.compute_hash(current_hash, tokens, group_id=0)
    #         block_id = self._block_manager._hash_to_id.get(current_hash)
    #         if block_id is not None:
    #             allocated_blocks.append(block_id)
    #             self._block_manager.increase_ref_count(block_id)
    #         else:
    #             break
    #     # If we found a matching prefix, we reference the blocks in the request
    #     if allocated_blocks:
    #         logger.debug(f"Found prefix match for request {request_id} with {len(allocated_blocks)} blocks")
    #         cm = self.group_cache_managers[0]
    #         cm.block_table[request_id] = allocated_blocks

    #     prefix_length = len(allocated_blocks) * self.block_size
    #     self._total_prefix_length += prefix_length
    #     return prefix_length

    # def mark_shareable_blocks_as_complete(self, state: RequestState, num_complete_blocks: int) -> None:
    #     """Marks the blocks allocated to a request (state) as complete if they are shareable and they have been computed
    #     in the forward pass. A complete block is a block where the KV cache has been fully computed: if the block has
    #     enough space to hold the cache for N tokens, the block is marked as complete when the cache data is present for
    #     the N tokens. If block sharing is off, this is a no-op."""
    #     # The status can be FINISHED in async mode, because batch N+1 offloaded the request before batch N was over. So
    #     # we need to check for this case to avoid looking in the block table for blocks that no longer exist.
    #     if num_complete_blocks == 0 or state.status == RequestStatus.FINISHED:
    #         return None
    #     for cm in self.group_cache_managers:
    #         if cm.uses_block_sharing:
    #             self._block_manager.mark_shareable_blocks_as_complete(
    #                 num_complete_blocks=num_complete_blocks,
    #                 allocated_blocks=cm.block_table[state.request_id],
    #                 prompt_ids=(state.initial_tokens + state.generated_tokens),
    #             )

    # def copy_cache(self, list_source_blocks: list[int], list_forked_blocks: list[int]) -> None:
    #     """Copy the cache from the source blocks to the forked blocks."""
    #     source_blocks = torch.tensor(list_source_blocks, device=self.device, dtype=torch.int32)
    #     forked_blocks = torch.tensor(list_forked_blocks, device=self.device, dtype=torch.int32)
    #     for key_cache, value_cache in zip(self.key_cache, self.value_cache):
    #         key_cache = key_cache.view(-1, self.block_size, self.num_key_value_heads, self.head_dim)
    #         value_cache = value_cache.view(-1, self.block_size, self.num_key_value_heads, self.head_dim)
    #         key_cache[forked_blocks] = key_cache[source_blocks]
    #         value_cache[forked_blocks] = value_cache[source_blocks]
    #     # FIXME: consolidate the cache into a single tensor of shape (group_size, 2, *self.k_or_v_cache_shape)
    #     # This will allow for  better .update and a single copy instead of one per cache tensor

    # def compute_max_num_forks(self, source_request_id: str) -> int:
    #     """Computes the maximum number of children requests that can be forked from the source request."""
    #     # Count, across all groups, the new blocks each fork would have to allocate (i.e. non-shareable blocks)
    #     blocks_needed_per_fork = 0
    #     for cm in self.group_cache_managers:
    #         block_ids = cm.block_table[source_request_id]
    #         shareable_blocks = 0
    #         if cm.uses_block_sharing:
    #             for block_id in block_ids:
    #                 if not self._block_manager._id_to_block[block_id].is_complete:
    #                     break
    #                 shareable_blocks += 1
    #         blocks_needed_per_fork += len(block_ids) - shareable_blocks
    #     # If every block can be shared, no new allocations are needed and any number of forks is possible
    #     if blocks_needed_per_fork == 0:
    #         return 2**31  # absurdly large number, virtually infinite number of forks
    #     return self.get_num_free_blocks() // blocks_needed_per_fork

    # def fork_request(self, source_request_id: str, destination_request_ids: list[str]) -> tuple[list[int], list[int]]:
    #     """Fork the cache of a request (state) into the one of a list of requests with the given (dst_request_ids)."""
    #     # These lists will be the accumulators for the source and destination blocks for the cache copy
    #     source_blocks, destination_blocks = [], []
    #     # Main fork loop
    #     for cm in self.group_cache_managers:
    #         src_blocks, dst_blocks = cm.fork_blocks(source_request_id, destination_request_ids, self._block_manager)
    #         source_blocks.extend(src_blocks)
    #         destination_blocks.extend(dst_blocks)
    #     return source_blocks, destination_blocks

    # def free_all_requests(self) -> None:
    #     """Free all blocks allocated to requests across all cache managers. This preserves prefix hashes in the block
    #     manager (blocks become initialized rather than uninitialized if they were complete), allowing prefix sharing
    #     to work across generation sessions."""
    #     all_request_ids = set()
    #     for cm in self.group_cache_managers:
    #         all_request_ids.update(cm.block_table.keys())
    #     for request_id in all_request_ids:
    #         self.free_blocks(request_id)


class PagedAttentionMemoryHandler:
    """Determines the optimal max batch tokens (M) and number of blocks (N) for the paged attention cache, given
    available GPU memory. The relation between N and number of blocks is: num_blocks = N // block_size.

    The memory footprint is a polynomial in M and N, where each term maps to a tensor allocated in
    ``ContinuousBatchingIOs._setup_static_tensors`` or ``PagedAttentionCache.__init__``:

        memory(M, N)  =  coeff_m · M  +  coeff_n · N  +  coeff_mn · M·N  +  coeff_mm · M²

    See ``_equation_coefficients`` for the breakdown.  All three solving modes (auto, fixed-N, fixed-M) reduce to
    solving this equation, which is at most quadratic in one variable.
    """

    _min_max_batch_tokens = 256
    _default_max_batch_tokens = 8192

    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        dtype: torch.dtype,
        sector_size: int,
        attn_types: list[str],
    ) -> None:
        """Initialize the memory handler. Args:
        - config: the model configuration
        - continuous_batching_config: the continuous batching configuration
        - dtype: the data type of the activation and the cache
        """
        self.config = config
        self.cb_config = continuous_batching_config
        self.cache_dtype = dtype
        self.activation_dtype = dtype
        self.block_size = continuous_batching_config.block_size
        self.page_size = find_head_dim(config) * find_num_key_value_heads(config)

        # TODO: when we generalize to allow for block-attn, we can use `num_attention_masks=sum(set(group_types))`
        if is_flash_attention_requested(self.config):
            self.num_attention_masks = 0
        else:
            self.num_attention_masks = 2 if SLIDING_ATTENTION in attn_types else 1

        self.max_blocks_per_request = continuous_batching_config.max_blocks_per_request
        if self.max_blocks_per_request is None:
            self.max_blocks_per_request = continuous_batching_config.fallback_max_blocks_per_request
        # This is the number of output rows for the output_ids tensor
        self.num_output_rows = 2 if continuous_batching_config.return_logprobs else 1
        # This account for the set of 2 IOs if async batching is used
        self.io_multiplier = 2 if continuous_batching_config.use_async_batching else 1
        self.available_memory = self.get_available_memory()

    @property
    def activation_peak(self) -> dict[str, tuple[int, ...]]:
        mem_per_q_token = self.config.num_attention_heads * find_head_dim(self.config)
        mem_per_k_or_v_token = self.page_size
        peaks = {}

        # LM head peak: this is when we turn the hidden states into logits
        delta_m = self.config.hidden_size * self.activation_dtype.itemsize  # hidden_shape, shape [M, hidden_size]
        delta_m += self.config.vocab_size * torch.float32.itemsize  # logits, shape [M, V], always in fp32
        peaks["lm_head"] = (delta_m, 0, 0, 0)

        # Attention peak: this is when we read the key and value states from the cache
        delta_m = self.activation_dtype.itemsize * (
            self.config.hidden_size  # hidden state, shape [M, hidden_size]
            + mem_per_q_token  # q_projection, shape [M, mem_per_q_token]
            + 2 * mem_per_k_or_v_token  # new K and V, shape [M, page_size]
        )
        # old K and V, read from cache (worst case scenario: whole cache is read)
        delta_n = 2 * mem_per_k_or_v_token * self.activation_dtype.itemsize
        peaks["attention"] = (delta_m, delta_n, 0, 0)

        return peaks

    def get_available_memory(self) -> int:
        """Calculate available GPU memory for cache allocation in bytes, accounting for the maximum memory percent limit
        fixed by the continuous batching config."""
        _, total, reserved, allocated = get_device_and_memory_breakdown()
        available_memory = total - max(allocated, reserved)
        available_memory = int(available_memory * self.cb_config.max_memory_percent)
        logger.info(f"Memory available for cache allocation: {available_memory // 1024**2} MB")
        return available_memory

    def infer_max_batch_tokens_and_num_blocks(self) -> tuple[int, int]:
        """Infers max_batch_tokens and num_blocks based on the available memory and the size of the activation peaks.
        If neither value is provided, we use a default value of 8192 for max_batch_tokens, apply bounds depending on the
        available VRAM, and solve for num_blocks. If one value is provided, the other is found using a linear solve."""
        max_batch_tokens = self.cb_config.max_batch_tokens
        num_blocks = self.cb_config.num_blocks

        # If both values are provided, just make sure they make sense
        if max_batch_tokens is not None and num_blocks is not None:
            return self._check_footprint(max_batch_tokens, num_blocks)
        raise NotImplementedError("Not implemented")

        # If one or more value is provided, solve for the other
        if max_batch_tokens is not None or num_blocks is not None:
            max_batch_tokens, num_blocks = self._solve_for_peaks(
                max_batch_tokens, num_blocks, cache_fill_per_batch=None
            )
            return self._check_footprint(max_batch_tokens, num_blocks)

        # If no value is provided, use the default value for max_batch_tokens w/ VRAM-based upper bound
        upper_bound_vram, _ = self._solve_for_peaks(
            max_batch_tokens=None,
            num_blocks=None,
            cache_fill_per_batch=0.1,  # each cache must fill 10% of the cache at most
        )
        max_batch_tokens = min(self._default_max_batch_tokens, upper_bound_vram)
        max_batch_tokens = max(max_batch_tokens, self._min_max_batch_tokens)
        # Then solve with that value
        max_batch_tokens, num_blocks = self._solve_for_peaks(max_batch_tokens, num_blocks, cache_fill_per_batch=None)
        return self._check_footprint(max_batch_tokens, num_blocks)

    def _solve_for_peaks(
        self,
        max_batch_tokens: int | None,
        num_blocks: int | None,
        cache_fill_per_batch: float | None,
    ) -> tuple[int, int]:
        """Returns max_batch_tokens and num_blocks so that their memory footprint is within the available memory for all
        activation peaks. If neither value is given, a value must be provided for cache_fill_per_batch: this means we
        solve for both variables by saying each batch fill a certain percentage of the cache (eg, if cache_fill_per_batch
        is 0.01, each batch will fill 1% of the cache)."""
        solutions = []

        for peak_deltas in self.activation_peak.values():
            m, n = self._solve_for_peak(peak_deltas, max_batch_tokens, num_blocks, cache_fill_per_batch)
            solutions.append((m, n))

        final_m = min([solution[0] for solution in solutions])
        final_n = min([solution[1] for solution in solutions])
        return final_m, final_n

    def _solve_for_peak(
        self,
        peak: tuple[int, ...],
        max_batch_tokens: int | None,
        num_blocks: int | None,
        cache_fill_per_batch: float | None,
    ) -> tuple[int, int]:
        """Returns a couple of `(max_batch_tokens, num_blocks)` that satisfy the memory constraint for the given
        activation peak."""
        cm, cn, cmn, cmm = self._equation_coefficients(peak)

        # If neither variable is defined, use a quadratic solver
        if max_batch_tokens is None and num_blocks is None:
            # Substitute M = m·N → (coeff_nm·m + coeff_mm·m²)·N² + (coeff_n + coeff_m·m)·N − avail = 0
            if cache_fill_per_batch is None:
                raise ValueError("m must be provided if max_batch_tokens and num_blocks are None")
            m = cache_fill_per_batch  # as in, m is a substitute for big M, which is max_batch_tokens
            num_pages = self._solve_quadratic(cmn * m + cmm * m**2, cn + cm * m, -self.available_memory)
            max_batch_tokens = int(num_pages * m)
            num_blocks = int(num_pages) // self.block_size

        # Otherwise, use a linear solver
        elif num_blocks is None:
            # M given → linear in N: (coeff_n + coeff_nm·M)·N = avail − coeff_m·M − coeff_mm·M²
            M = max_batch_tokens
            num_pages = floor((self.available_memory - cm * M - cmm * M**2) / (cn + cmn * M))
            num_blocks = num_pages // self.block_size

        elif max_batch_tokens is None:
            # N given → quadratic in M: coeff_mm·M² + (coeff_m + coeff_nm·N)·M + (coeff_n·N − avail) = 0
            N = num_blocks * self.block_size
            max_batch_tokens = int(self._solve_quadratic(cmm, cm + cmn * N, cn * N - self.available_memory))

        return max_batch_tokens, num_blocks

    def _check_footprint(self, max_batch_tokens: int, num_blocks: int) -> tuple[int, int]:
        """Checks if the footprint of the cache is within the available memory."""
        memory_footprint = self.compute_memory_footprint(max_batch_tokens, num_blocks)
        if memory_footprint > self.available_memory:
            raise MemoryError(
                f"Memory footprint {memory_footprint} is more than available memory {self.available_memory}"
            )
        if max_batch_tokens <= 0 or num_blocks <= 0:
            raise ValueError(f"Invalid values: max_batch_tokens = {max_batch_tokens}, num_blocks = {num_blocks}")
        return max_batch_tokens, num_blocks

    def _solve_quadratic(self, a: float, b: float, c: float) -> int:
        """Largest positive root of a·x² + b·x + c = 0. Falls back to linear when a == 0. Rounded down."""
        if a == 0:
            return int(-c / b)
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            raise ValueError(f"No real solution (discriminant = {discriminant})")
        root = (-b + sqrt(discriminant)) / (2 * a)
        if root < 0:
            raise ValueError(f"No positive solution (root = {root})")
        return int(floor(root))

    # Formatting is disabled because of comment indentation, which improves readability.
    # fmt: off
    def _equation_coefficients(self, peak_deltas: tuple[int, ...]) -> tuple[int, ...]:
        raise NotImplementedError("Not implemented")
        """Given some deltas corresponding to an activation peak, returns the coefficients for the memory polynomial of
        that peak. The memory polynomial is described in that class docstring."""
        delta_m, delta_n, delta_mm, delta_mn = peak_deltas

        i = torch.int32.itemsize             # size of int32 in bytes, used for index, input_ids, ...
        a = self.activation_dtype.itemsize             # for now, the cache and the activation have the same dtype
        c = self.cache_dtype.itemsize
        k = self.io_multiplier               # 1 sync, 2 async (IO tensors only)

        # -- N terms: cost per cache page --------------------------------------------------
        coeff_n = (
            delta_n                                      # activation peak: N-proportional part
            + 2 * self.group_size * self.page_size * c   # kv_cache: 2 * group_size * [N, page_size] * cache_dtype
            + k * self.num_groups * 8                    # read_index: [num_groups, N + M]  (N part only, int64)
        )
        # -- M terms: cost per batch token -------------------------------------------------
        coeff_m = (
            delta_m                                    # activation peak: M-proportional part
            + k * 7 * i                                # bulk_input: [7, M] int32, packed as 7 rows
            + k * self.num_output_rows * i             # output_ids: [num_output_rows, M] int32
            + k * self.num_groups                      # block_table: [bt_groups, M, max_blocks_per_req] int32
            * self.max_blocks_per_request * i          #   (zero when fast-decode is off)
            + k * self.num_groups * 8                  # write_index: [num_groups, M] int64
            + k * self.num_groups * 8                  # read_index: [num_groups, N + M] (M part only, int64)
        )
        # TODO: the above could be refined by introducing the max_requests_per_batch, but then there is a min() and this
        # is no longer a simple polynomial. Could be worth checking into.
        # -- M·N terms: cost per (page × batch token) --------------------------------------
        coeff_mn = (
            delta_mn                             # activation peak: M·N-proportional part
            + k * self.num_attention_masks * a   # attention_mask: [1, 1, M, N + M] (N·M part only)
        )
        # -- M² terms: cost per (batch token squared) --------------------------------------
        coeff_mm = (
            delta_mm                            # activation peak: M²-proportional part
            + k * self.num_attention_masks * a  # attention_mask: [1, 1, M, N + M] (M² part only)
        )

        return coeff_m, coeff_n, coeff_mn, coeff_mm
    # fmt: on

    def compute_memory_footprint(self, max_batch_tokens: int, num_blocks: int) -> int:
        """Evaluate the memory polynomial at concrete (N, M) values, taking the max across activation peaks."""
        M = max_batch_tokens
        N = num_blocks * self.block_size

        max_memory_footprint = 0
        for peak in self.activation_peak.values():
            cm, cn, cmn, cmm = self._equation_coefficients(peak)
            memory_footprint = cn * N + cm * M + cmn * N * M + cmm * M * M
            max_memory_footprint = max(max_memory_footprint, memory_footprint)
        return max_memory_footprint
