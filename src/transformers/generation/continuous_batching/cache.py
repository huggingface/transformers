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
from math import floor, gcd, sqrt
from typing import Any

import torch

from ...configuration_utils import PreTrainedConfig
from ...generation.configuration_utils import ContinuousBatchingConfig
from ...utils.generic import is_flash_attention_requested
from ...utils.metrics import attach_tracer, traced
from .cache_manager import BlockManager, CacheAllocator, FullAttentionCacheAllocator, SlidingAttentionCacheAllocator
from .requests import RequestState, RequestStatus, get_device_and_memory_breakdown, logger


def group_layers_by_attn_type(config: PreTrainedConfig) -> tuple[list[list[int]], list[str]]:
    """
    Group layers depending on the attention mix, according to VLLM's hybrid allocator rules:
        - Layers in each group need to have the same type of attention
        - All groups have the same number of layers

    For a model with the following layer types: ["sliding", "full", "full", "sliding", "full", "full", "full", "full"]
    We would get four groups: [0, 3], [1, 2], [4,5] and [6,7].
    """
    # If the config has no layer_type attribute, it means all layers are the same attention type
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        attn_type = "sliding_attention" if getattr(config, "sliding_window", None) is not None else "full_attention"
        layer_types = [attn_type for _ in range(config.num_hidden_layers)]

    # We then count the number of layers of each type
    layer_counts = {}
    for i, layer_type in enumerate(layer_types):
        layer_counts[layer_type] = layer_counts.get(layer_type, []) + [i]

    # The size of all groups is the greatest common divisor of the number of layers of each type
    group_size = gcd(*[len(indices) for indices in layer_counts.values()])

    # We then group the layers by type
    layer_groups = []
    for layer_type, indices in layer_counts.items():
        for i in range(0, len(indices), group_size):
            layer_groups.append(indices[i : i + group_size])
    # And note the layer types
    group_types = [layer_types[lg[0]] for lg in layer_groups]
    return layer_groups, group_types


@attach_tracer()
class PagedAttentionCache:
    """
    Manages the cache for a paged attention mechanism, inspired by VLLM's hybrid allocator. The cache relies on making
    groups of layers to reduce the complexity of cache management and fragmentation.

    The cache uses a three-level hierarchy:
    - Pages: The smallest unit of cache, a page has a size of [num_heads, head_size], which is the space needed to
        store the key or value states for one token and one layer. For a model with only full-attention layers, to store
        the KV cache of one token, we need `2 * num_layers` pages: key and values each take `num_layers` pages.
        Pages are grouped into blocks:
    - Blocks: A block is a collection of `block_size` pages, serving as the allocation unit to reduce management
        complexity and fragmentation. Cache is allocated and freed block by block, not page by page. One block is
        allocated to one layer group, which only has one attention type, like full-attention or sliding-attention.
        If all layers in the model have the same attention type, then all layers will be in the same group. There is
        more than one group if and only if the model has a mixed attention types, like layers with full-attention and
        layers with sliding-attention.
    - Cache tensors: The physical supports for the cache. There are as many cache tensors as there are layer in a
        layer group, and the shape of the cache tensor is `[num_blocks * block_size, num_heads, head_size]`.

    Grouping layers into groups is useful because when we allocate one block to a group N, the block allocated is the
        same for all layers in group N, equivalently it is allocated across all cache tensors. This allows us to
        efficiently allocate and free blocks, and to efficiently read and write key and value states.

    For instance, imagine we have 8 blocks of cache and a model with two layer groups: a full-attention group with 3
    layers and a sliding-attention group with 3 layers. At creation time, the physical cache tensors look like this:

    cache_tensor_0: □ □ □ □ □ □ □ □
    cache_tensor_1: □ □ □ □ □ □ □ □
    cache_tensor_2: □ □ □ □ □ □ □ □

    where □ means the blocks is not allocated to any layer group yet. We have 3 cache tensors because there are
    3 layers per group.
    We allocate 1 block to each group, after allocation, the cache tensors look like this:

    cache_tensor_0: ✖ ◉ □ □ □ □ □ □
    cache_tensor_1: ✖ ◉ □ □ □ □ □ □
    cache_tensor_2: ✖ ◉ □ □ □ □ □ □

    where ✖ means the block is allocated to the full-attention group, and ◉ means the block is allocated to the
    sliding-attention group.
    Now, if we continue to generate, and the sliding window has been reached, we only need to allocate a new block
    for the full-attention group, and the cache tensors look like this:

    cache_tensor_0: ✖ ◉ ✖ □ □ □ □ □
    cache_tensor_1: ✖ ◉ ✖ □ □ □ □ □
    cache_tensor_2: ✖ ◉ ✖ □ □ □ □ □

    And after further generation, when we need a new block allocated:

    cache_tensor_0: ✖ ◉ ✖ ✖ □ □ □ □
    cache_tensor_1: ✖ ◉ ✖ ✖ □ □ □ □
    cache_tensor_2: ✖ ◉ ✖ ✖ □ □ □ □

    This would not have been possible if all layers were in the same group: we would have had to allocate a new block
    for the sliding-attention group, although it is not needed.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
        tp_size: int | None = None,
    ) -> None:
        """Initialize a paged attention cache for efficient memory usage. Also turns in prefix sharing if the model has
        only full attention layers.

        Args:
            config: Model configuration
            continuous_batching_config: Continuous batching configuration containing cache parameters
            device: Device for the cache tensors
            dtype: Data type of the cache
            tp_size: Tensor parallelism size
        """
        self.config = config
        self.dtype = dtype
        self.device = device

        # Extract model dimensions
        kv_heads = getattr(config, "num_key_value_heads", None)
        self.num_key_value_heads: int = kv_heads if kv_heads is not None else config.num_attention_heads
        head_dim = getattr(config, "head_dim", None)
        self.head_dim: int = head_dim if head_dim is not None else config.hidden_size // config.num_attention_heads

        # Extract cache dimensions. Default used to be 32, now it's 256 to be compatible with flash_with_kvcache.
        self.block_size = continuous_batching_config.block_size
        if self.block_size <= 0:
            raise ValueError(f"Block size must be positive, but got {self.block_size}")

        # Group layers depending on the attention mix
        layer_groups, group_types = group_layers_by_attn_type(config)
        group_size = len(layer_groups[0])
        self.num_groups = len(layer_groups)

        self.sliding_windows = {}
        self.layer_index_to_group_indices = {}
        for i, group in enumerate(layer_groups):
            sliding_window = config.sliding_window if group_types[i] == "sliding_attention" else 1
            for j, layer in enumerate(group):
                self.layer_index_to_group_indices[layer] = (i, j)
                self.sliding_windows[layer] = sliding_window

        # Handle TP (or dont)
        if tp_size is not None and tp_size > 1:
            if self.num_key_value_heads % tp_size != 0:
                raise ValueError(
                    f"Number of key value heads {self.num_key_value_heads} must be divisible by tensor parallel size {tp_size}."
                )
            # If the model is using tensor parallelism, we need to adjust the number of heads accordingly.
            # self.num_key_value_heads //= tp_size # TODO: why is this commented out?

        # Infer number of blocks and max batch tokens
        page_size = self.head_dim * self.num_key_value_heads

        if is_flash_attention_requested(self.config):
            num_attention_masks = 0  # only used to compute the default memory footprint args
        elif "sliding_attention" in group_types:
            # TODO: when we generalize to allow for block-attn, we can use `num_attention_masks=sum(set(group_types))`
            num_attention_masks = 2
        else:
            num_attention_masks = 1

        memory_handler = PagedAttentionMemoryHandler(
            block_size=self.block_size,
            page_size=page_size,
            num_groups=self.num_groups,
            group_size=group_size,
            peak_activation_per_token=(config.hidden_size + config.vocab_size),
            num_attention_masks=num_attention_masks,
            continuous_batching_config=continuous_batching_config,
        )
        num_blocks, max_batch_tokens = memory_handler.infer_num_blocks_and_max_batch_tokens(
            num_blocks=continuous_batching_config.num_blocks,
            max_batch_tokens=continuous_batching_config.max_batch_tokens,
            max_memory_percent=continuous_batching_config.max_memory_percent,
            cache_dtype=self.dtype,
        )

        # Add the inferred attributes to the class
        self.num_blocks = num_blocks
        self.max_batch_tokens = max_batch_tokens
        self.num_pages = self.num_blocks * self.block_size
        logger.info(
            f"PagedAttentionCache initialized with {self.num_blocks = }, {self.block_size = }, {page_size = }, "
            f"{self.max_batch_tokens = } {num_attention_masks = }"
        )

        # If max_blocks_per_request is not set, the default value is 16 max blocks. With default block size of 256, this
        # means a max sequence length of 4096 tokens for the fast decode path.
        max_blocks_per_request = continuous_batching_config.max_blocks_per_request
        if max_blocks_per_request is None:
            max_blocks_per_request = 0
            # logger.info( TODO: uncomment when we have good defaults
            #     f"max_blocks_per_request was not set, using {max_blocks_per_request}. This means max sequence "
            #     f"length for the decode fast path is {max_blocks_per_request * self.block_size}."
            # )
        self.max_blocks_per_request = max_blocks_per_request

        # Initialize the cache
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        # We add two extra blocks to the cache as a padding zone that no BlockManager ever allocates from: one for the
        # sentinel index (marks the spot of a new token in the read indices) and one for the trash index (for padding,
        # block is never used so writes are silently discarded)
        self.cache_shape = ((num_blocks + 2) * self.block_size, self.num_key_value_heads, self.head_dim)
        self.sentinel_index = self.cache_shape[0] - 1
        self.trash_index = self.sentinel_index - 1
        for _ in range(group_size):
            new_layer_key_cache = torch.empty(self.cache_shape, dtype=self.dtype, device=self.device)
            new_layer_value_cache = torch.empty(self.cache_shape, dtype=self.dtype, device=self.device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
        logger.info(f"{self.cache_shape = } {self.key_cache[0].shape = } {self.key_cache[0].numel() = }")

        # Block management data structures
        self.allow_block_sharing = continuous_batching_config.allow_block_sharing
        self.group_cache_managers: list[CacheAllocator] = []
        self.num_full_attention_groups = 0
        self.num_sliding_attention_groups = 0
        self.max_sliding_window_blocks_per_request = 0

        for i, group_type in enumerate(group_types):
            if group_type == "full_attention":
                cm = FullAttentionCacheAllocator(i, self.block_size, allow_block_sharing=self.allow_block_sharing)
                self.num_full_attention_groups += 1
            elif group_type == "sliding_attention":
                cm = SlidingAttentionCacheAllocator(
                    i, self.block_size, config.sliding_window, self.sentinel_index, self.trash_index
                )
                self.num_sliding_attention_groups += 1
                self.max_sliding_window_blocks_per_request = cm._max_blocks_per_request
            else:
                raise ValueError(f"Invalid group type: {group_type}")
            self.group_cache_managers.append(cm)

        # We only use prefix sharing if the whole model has only full attention layers and block sharing is allowed
        self.use_prefix_sharing = self.allow_block_sharing and group_types == ["full_attention"]
        self._block_manager = BlockManager(num_blocks, self.block_size)
        self._total_prefix_length: int = 0  # a counter to measure the impact of prefix sharing, also used in tests

        # For block table support, we lazy init the name of the block table key
        self._block_table_key = None

    def will_allocation_be_successful(self, num_requested_blocks: int, allocated_blocks: int) -> bool:
        """Returns a boolean indicating if the allocation of (num_requested_blocks) blocks will be successful. The
        number of newly allocated blocks needed is predicted by the following rules:
        - for full attention groups: since there is no sliding window for full attention layers, one requested block is
            always equivalent to one newly allocated block for EACH full attention group
        - for sliding window groups: because of the sliding window, the number of blocks allocated to a request is
            capped. Using the number of already (allocated_blocks) we can compute the number of new blocks to actually
            allocate to the request, which can be lower than the number of requested blocks. That number is the same for
            all sliding window groups, as only one sliding window size is supported.
        """
        # This is not in a branch, because it is very rare to have zero full attention layer
        needed_blocks = num_requested_blocks * self.num_full_attention_groups
        # Only take this branch if the model has sliding window attention layers
        if self.num_sliding_attention_groups:
            blocks_left = max(self.max_sliding_window_blocks_per_request - allocated_blocks, 0)
            needed_blocks += min(blocks_left, num_requested_blocks) * self.num_sliding_attention_groups
        return needed_blocks <= self.get_num_free_blocks()

    @traced
    def allocate_blocks(self, n_blocks: int, request_id: str, allocated_blocks: int) -> int | None:
        """Allocate cache blocks across all layer groups for a given request. Actual allocation is done by the cache
        managers, and this method only returns the maximum number of blocks actually allocated across all managers."""
        # First check allocation will be successful before starting, to avoid partial allocations
        if not self.will_allocation_be_successful(n_blocks, allocated_blocks):
            return None
        # Allocate blocks across all cache managers
        max_allocated = 0
        for cm in self.group_cache_managers:
            num_allocated_blocks = cm.allocate_blocks(n_blocks, request_id, self._block_manager)
            if num_allocated_blocks is None:
                raise ValueError(f"Failed to allocate {n_blocks} blocks for request {request_id}")
            max_allocated = max(max_allocated, num_allocated_blocks)
        return max_allocated

    @traced
    def free_blocks(self, request_id: str) -> None:
        """Free all allocated cache blocks for a given request across all layer groups. Actual deallocation is done
        by the cache managers."""
        for cm in self.group_cache_managers:
            cm.free_blocks(request_id, self._block_manager)

    def get_num_free_blocks(self) -> int:
        """Get the current number of unallocated blocks available for new requests."""
        return self._block_manager.num_free_blocks

    @traced
    def extend_read_and_write_indices(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        read_index: list[list[int]],
        write_index: list[list[int]],
    ) -> None:
        """Retrieve physical cache indices for reading KV states in the cache across all layer groups. This method
        coordinates with all cache managers to build the complete set of read indices needed for attention computation.
        """
        for cm, read_indices, write_indices in zip(self.group_cache_managers, read_index, write_index):
            indices = cm.get_read_indices(request_id, past_length, query_length)
            read_indices.extend(indices)
            indices = cm.get_write_indices(request_id, past_length, query_length)
            write_indices.extend(indices)

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        for i, cm in enumerate(self.group_cache_managers):
            cm.fill_block_table(request_id, past_length, query_length, block_table[i])

    @traced
    def get_seqlens_k(self, past_length: int, query_length: int) -> dict[str, int]:
        """Retrieve the key sequence length for the given request_id across all layer types. Returns a dictionary of
        layer types to their corresponding key sequence lengths."""
        seqlens_k = {}
        if self.num_full_attention_groups > 0:
            seqlens_k["full_attention"] = past_length + query_length
        if self.num_sliding_attention_groups > 0:
            seqlens_k["sliding_attention"] = query_length + min(past_length, self.config.sliding_window - 1)
        # NOTE: when we add more attention types / different sliding windows, we can go back to looping over CMs
        return seqlens_k

    @traced
    def update(
        self,
        key_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_kv, head_dim]
        value_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_kv, head_dim]
        layer_idx: int,
        read_index: list[torch.Tensor],  # shape [num_layer_groups, seqlen_kv + past_length]
        write_index: list[torch.Tensor],  # shape [num_layer_groups, seqlen_q]
    ) -> tuple[torch.Tensor, torch.Tensor]:  # shape [seqlen_kv + past_length, num_kv_heads, head_dim]
        """Update the cache with new key-value states for a specific layer. This method writes new KV states to the
        appropriate cache locations. The behavior differs based on the layer's attention type:

        - Full attention: New KV states are written to cache, then complete sequence is read from cache
        - Sliding window: Old KV is read from cache along with extra spaces for the new KV, then new KV is written to
            cache. This is because new KV might overwrite the old KV, so we need to read the old KV first.

        Returns the complete KV states (cached + new) for attention computation.
        """
        # Retrieve the layer read and write indices
        group_idx, layer_idx_in_group = self.layer_index_to_group_indices[layer_idx]
        layer_read_index = read_index[group_idx]
        layer_write_index = write_index[group_idx]
        # Select the correct cache
        k_cache = self.key_cache[layer_idx_in_group]
        v_cache = self.value_cache[layer_idx_in_group]
        # Transpose the key and value states to match the cache shape, after which shape is [seqlen_kv, num_kv_heads, head_dim]
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)

        # Case: full attention
        sliding_window = self.sliding_windows[layer_idx]
        if sliding_window == 1:
            k_cache.index_copy_(0, layer_write_index, key_states)
            v_cache.index_copy_(0, layer_write_index, value_states)
            key_states_with_cache = torch.index_select(k_cache, 0, layer_read_index)
            value_states_with_cache = torch.index_select(v_cache, 0, layer_read_index)

        # Case: sliding window -- we  need to be careful of read/write order because of chunked prefill, because it's
        # the only case where you may write over cache you need to use
        else:
            # Sentinel positions in read_index mark new-token slots; index_select reads garbage there,
            # then masked_scatter_ overwrites them with the actual new key/value states.
            mask = (layer_read_index == self.sentinel_index).unsqueeze(-1).unsqueeze(-1)
            key_states_with_cache = torch.index_select(k_cache, 0, layer_read_index)
            key_states_with_cache.masked_scatter_(mask, key_states)
            value_states_with_cache = torch.index_select(v_cache, 0, layer_read_index)
            value_states_with_cache.masked_scatter_(mask, value_states)
            # Write new KV values to the cache (padding slots in write_index point to the trash position)
            k_cache.index_copy_(0, layer_write_index, key_states)
            v_cache.index_copy_(0, layer_write_index, value_states)

        # Return the new KV values
        return key_states_with_cache, value_states_with_cache

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
                    f"flash_attn_with_kvcache_fn does not have a block_table or page_table argument: {inspect.signature(flash_attn_with_kvcache_fn)}"
                )
        return self._block_table_key

    def search_prefix_match(self, request_id: str, prompt_ids: list[int]) -> int:
        """Searches for a prefix match in the cache for the given (prompts_ids). If one is found, we reference the
        matching blocks in the (request_id), increase the reference count of the blocks and return the number of blocks
        that match. If no prefix match is found, we return 0."""
        current_hash = None
        allocated_blocks = []
        for b in range(len(prompt_ids) // self.block_size):
            tokens = prompt_ids[b * self.block_size : (b + 1) * self.block_size]
            # Prefix sharing is only supported when there is only one full attention layer group, so group_id=0.
            current_hash = self._block_manager.compute_hash(current_hash, tokens, group_id=0)
            block_id = self._block_manager._hash_to_id.get(current_hash)
            if block_id is not None:
                allocated_blocks.append(block_id)
                self._block_manager.increase_ref_count(block_id)
            else:
                break
        # If we found a matching prefix, we reference the blocks in the request
        if allocated_blocks:
            logger.debug(f"Found prefix match for request {request_id} with {len(allocated_blocks)} blocks")
            cm = self.group_cache_managers[0]
            cm.block_table[request_id] = allocated_blocks

        prefix_length = len(allocated_blocks) * self.block_size
        self._total_prefix_length += prefix_length
        return prefix_length

    def mark_shareable_blocks_as_complete(self, state: RequestState, num_complete_blocks: int) -> None:
        """Marks the blocks allocated to a request (state) as complete if they are shareable and they have been computed
        in the forward pass. A complete block is a block where the KV cache has been fully computed: if the block has
        enough space to hold the cache for N tokens, the block is marked as complete when the cache data is present for
        the N tokens. If block sharing is off, this is a no-op."""
        # The status can be FINISHED in async mode, because batch N+1 offloaded the request before batch N was over. So
        # we need to check for this case to avoid looking in the block table for blocks that no longer exist.
        if num_complete_blocks == 0 or state.status == RequestStatus.FINISHED:
            return None
        for cm in self.group_cache_managers:
            if cm.uses_block_sharing:
                self._block_manager.mark_shareable_blocks_as_complete(
                    num_complete_blocks=num_complete_blocks,
                    allocated_blocks=cm.block_table[state.request_id],
                    prompt_ids=(state.initial_tokens + state.generated_tokens),
                )

    def copy_cache(self, list_source_blocks: list[int], list_forked_blocks: list[int]) -> None:
        """Copy the cache from the source blocks to the forked blocks."""
        source_blocks = torch.tensor(list_source_blocks, device=self.device, dtype=torch.int32)
        forked_blocks = torch.tensor(list_forked_blocks, device=self.device, dtype=torch.int32)
        for key_cache, value_cache in zip(self.key_cache, self.value_cache):
            key_cache = key_cache.view(-1, self.block_size, self.num_key_value_heads, self.head_dim)
            value_cache = value_cache.view(-1, self.block_size, self.num_key_value_heads, self.head_dim)
            key_cache[forked_blocks] = key_cache[source_blocks]
            value_cache[forked_blocks] = value_cache[source_blocks]
        # FIXME: consolidate the cache into a single tensor of shape (group_size, 2, *self.k_or_v_cache_shape)
        # This will allow for  better .update and a single copy instead of one per cache tensor

    def fork_request(self, source_request_id: str, destination_request_ids: list[str]) -> tuple[list[int], list[int]]:
        """Fork the cache of a request (state) into the one of a list of requests with the given (dst_request_ids)."""
        # These lists will be the accumulators for the source and destination blocks for the cache copy
        source_blocks, destination_blocks = [], []
        # Main fork loop
        for cm in self.group_cache_managers:
            src_blocks, dst_blocks = cm.fork_blocks(source_request_id, destination_request_ids, self._block_manager)
            source_blocks.extend(src_blocks)
            destination_blocks.extend(dst_blocks)
        return source_blocks, destination_blocks

    def free_all_requests(self) -> None:
        """Free all blocks allocated to requests across all cache managers. This preserves prefix hashes in the block
        manager (blocks become initialized rather than uninitialized if they were complete), allowing prefix sharing
        to work across generation sessions."""
        all_request_ids = set()
        for cm in self.group_cache_managers:
            all_request_ids.update(cm.block_table.keys())
        for request_id in all_request_ids:
            self.free_blocks(request_id)


# TODO: rework computation with the groups and their sizes
class PagedAttentionMemoryHandler:
    """Determines the optimal number of pages (N) and max batch tokens (M) for the paged attention cache, given
    available GPU memory. The relation between N and number of blocks is: num_blocks = N // block_size.

    The memory footprint is a polynomial in N and M, where each term maps to a tensor allocated in
    ``ContinuousBatchingIOs._setup_static_tensors`` or ``PagedAttentionCache.__init__``:

        memory(N, M)  =  coeff_n · N  +  coeff_m · M  +  coeff_nm · N·M  +  coeff_mm · M²

    See ``_equation_coefficients`` for the breakdown.  All three solving modes (auto, fixed-N, fixed-M) reduce to
    solving this equation, which is at most quadratic in one variable.
    """

    _activation_dtype = torch.bfloat16
    _input_dtype = torch.int32
    _upper_bound_max_batch_tokens = 256
    _upper_bound_num_blocks = 4096

    def __init__(
        self,
        block_size: int,
        page_size: int,
        num_groups: int,
        group_size: int,
        peak_activation_per_token: int,
        num_attention_masks: int,
        continuous_batching_config: ContinuousBatchingConfig,
    ) -> None:
        """Initialize the memory handler."""
        self.block_size = block_size
        self.page_size = page_size
        self.num_groups = num_groups
        self.group_size = group_size
        self.peak_activation_per_token = peak_activation_per_token
        self.num_attention_masks = num_attention_masks
        self.max_blocks_per_request = continuous_batching_config.max_blocks_per_request or 0
        # This is the number of output rows for the output_ids tensor
        self.num_output_rows = 2 if continuous_batching_config.return_logprobs else 1
        # This account for the set of 2 IOs if async batching is used
        self.io_multiplier = 2 if continuous_batching_config.use_async_batching else 1

    @staticmethod
    def get_available_memory(max_memory_percent: float = 1.0) -> int:
        """Calculate available GPU memory for cache allocation, accounting for already allocated tensors."""
        _, total, reserved, allocated = get_device_and_memory_breakdown()
        available_memory = total - max(allocated, reserved)
        available_memory = int(available_memory * max_memory_percent)
        return available_memory

    # Formatting is disabled because of comment indentation, which improves readability.
    # fmt: off
    def _equation_coefficients(self, cache_dtype: torch.dtype) -> tuple[int, int, int, int]:
        """Returns (coeff_n, coeff_m, coeff_nm, coeff_mm) for the memory polynomial. Each addend is annotated with
        the tensor it corresponds to in `ContinuousBatchingIOs._setup_static_tensors`.
        """
        i = self._input_dtype.itemsize       # int32
        a = self._activation_dtype.itemsize  # bfloat16
        c = cache_dtype.itemsize
        k = self.io_multiplier               # 1 sync, 2 async (IO tensors only)

        # -- N terms: cost per cache page --------------------------------------------------
        coeff_n = (
            2 * self.group_size * self.page_size * c   # kv_cache: 2 * group_size * [N, page_size] * cache_dtype
            + k * self.num_groups * 8                  # read_index: [num_groups, N + M]  (N part only, int64)
        )
        # -- M terms: cost per batch token -------------------------------------------------
        coeff_m = (
            self.peak_activation_per_token * a         # activation peak (largest hidden state per token)
            + k * 7 * i                                # bulk_input: [7, M] int32, packed as 7 rows
            + k * self.num_output_rows * i             # output_ids: [num_output_rows, M] int32
            + k * self.num_groups                      # block_table: [bt_groups, M, max_blocks_per_req] int32
            * self.max_blocks_per_request * i          #   (zero when fast-decode is off)
            + k * self.num_groups * 8                  # write_index: [num_groups, M] int64
            + k * self.num_groups * 8                  # read_index: [num_groups, N + M] (M part only, int64)
        )
        # -- N·M terms: cost per (page × batch token) -------------------------------------
        coeff_nm = k * self.num_attention_masks * a    # attention_mask: [1, 1, M, N + M] (N·M part only)
        # -- M² terms: cost per (batch token squared) -------------------------------------
        coeff_mm = k * self.num_attention_masks * a    # attention_mask: [1, 1, M, N + M] (M² part only)

        return coeff_n, coeff_m, coeff_nm, coeff_mm
    # fmt: on

    @staticmethod
    def _solve_quadratic(a: float, b: float, c: float) -> float:
        """Largest positive root of a·x² + b·x + c = 0. Falls back to linear when a == 0."""
        if a == 0:
            return -c / b
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            raise ValueError(f"No real solution (discriminant = {discriminant})")
        root = (-b + sqrt(discriminant)) / (2 * a)
        if root < 0:
            raise ValueError(f"No positive solution (root = {root})")
        return root

    def infer_num_blocks_and_max_batch_tokens(
        self,
        num_blocks: int | None = None,
        max_batch_tokens: int | None = None,
        max_memory_percent: float = 0.8,  # FIXME: it seems we overcommit memory, was changed from 0.9 which caused OOMs in our benchmarking CI
        cache_dtype: torch.dtype = torch.float16,
    ) -> tuple[int, int]:
        """Solve for the missing variable(s) in the memory polynomial (see ``_equation_coefficients``). When both
        are unknown, assumes M = m·N (m = 0.01, i.e. one batch fills ~1 % of the cache) and solves the resulting
        quadratic in N.
        """
        available = self.get_available_memory(max_memory_percent)
        coeff_n, coeff_m, coeff_nm, coeff_mm = self._equation_coefficients(cache_dtype)
        logger.info(f"Cache memory: {available}")

        if num_blocks is None and max_batch_tokens is None:
            # Substitute M = m·N → (coeff_nm·m + coeff_mm·m²)·N² + (coeff_n + coeff_m·m)·N − avail = 0
            m = 0.01
            num_pages = self._solve_quadratic(
                coeff_nm * m + coeff_mm * m**2,
                coeff_n + coeff_m * m,
                -available,
            )
            num_blocks = min(floor(num_pages) // self.block_size, self._upper_bound_num_blocks)
            max_batch_tokens = min(int(num_pages * m), self._upper_bound_max_batch_tokens)

        elif num_blocks is None:
            # M given → linear in N: (coeff_n + coeff_nm·M)·N = avail − coeff_m·M − coeff_mm·M²
            M = max_batch_tokens
            num_pages = floor((available - coeff_m * M - coeff_mm * M**2) / (coeff_n + coeff_nm * M))
            num_blocks = min(num_pages // self.block_size, self._upper_bound_num_blocks)

        elif max_batch_tokens is None:
            # N given → quadratic in M: coeff_mm·M² + (coeff_m + coeff_nm·N)·M + (coeff_n·N − avail) = 0
            N = num_blocks * self.block_size
            M = self._solve_quadratic(coeff_mm, coeff_m + coeff_nm * N, coeff_n * N - available)
            max_batch_tokens = min(floor(M), self._upper_bound_max_batch_tokens)

        # Validate
        memory_footprint = self.compute_memory_footprint(
            max_batch_tokens=max_batch_tokens, num_blocks=num_blocks, cache_dtype=cache_dtype
        )
        if memory_footprint > available:
            raise MemoryError(f"Memory footprint {memory_footprint} is more than available memory {available}")
        return num_blocks, max_batch_tokens

    def compute_memory_footprint(self, num_blocks: int, max_batch_tokens: int, cache_dtype: torch.dtype) -> int:
        """Evaluate the memory polynomial at concrete (N, M) values."""
        N = num_blocks * self.block_size
        M = max_batch_tokens
        cn, cm, cnm, cmm = self._equation_coefficients(cache_dtype)
        return cn * N + cm * M + cnm * N * M + cmm * M * M
