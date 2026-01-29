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
from math import floor, gcd, sqrt

import torch

from ...configuration_utils import PreTrainedConfig
from ...generation.configuration_utils import GenerationConfig
from ...utils.generic import is_flash_attention_requested
from ...utils.metrics import attach_tracer, traced
from .cache_manager import BlockManager, CacheAllocator, FullAttentionCacheAllocator, SlidingAttentionCacheAllocator
from .requests import RequestState, get_device_and_memory_breakdown, logger


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
        generation_config: GenerationConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        tp_size: int | None = None,
        allow_block_sharing: bool = True,
    ) -> None:
        """Initialize a paged attention cache for efficient memory usage. Also turns in prefix sharing if the model has
        only full attention layers.

        Args:
            config: Model configuration
            generation_config: Generation configuration containing cache parameters
            device: Device for the cache tensors
            dtype: Data type of the cache
            tp_size: Tensor parallelism size
            allow_block_sharing: A flag to allow block sharing. If the model has some full attention layers, then prefix
                sharing is enabled as well.
        """
        self.config = config
        self.dtype = dtype
        self.device = device

        # Extract model dimensions
        kv_heads = getattr(config, "num_key_value_heads", None)
        self.num_key_value_heads: int = kv_heads if kv_heads is not None else config.num_attention_heads
        head_dim = getattr(config, "head_dim", None)
        self.head_dim: int = head_dim if head_dim is not None else config.hidden_size // config.num_attention_heads

        # Extract cache dimensions
        self.block_size = getattr(generation_config, "block_size", 32)

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
        )
        num_blocks, max_batch_tokens = memory_handler.infer_num_blocks_and_max_batch_tokens(
            num_blocks=getattr(generation_config, "num_blocks", None),
            max_batch_tokens=getattr(generation_config, "max_batch_tokens", None),
            max_memory_percent=getattr(
                generation_config, "max_memory", 0.8
            ),  # FIXME: it seems we overcommit memory, was changed from 0.9 which caused OOMs in our benchmarking CI
            cache_dtype=self.dtype,
        )

        # Add the inferred attributes to the class
        self.num_blocks = num_blocks
        self.max_batch_tokens = max_batch_tokens
        logger.info(
            f"PagedAttentionCache initialized with {self.num_blocks = }, {self.block_size = }, {page_size = }, "
            f"{self.max_batch_tokens = } {num_attention_masks = }"
        )

        # Initialize the cache
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        # We add two extra tokens to the cache to handle padding and generally discard unwanted tokens
        self.cache_shape = ((num_blocks + 2) * self.block_size, self.num_key_value_heads, self.head_dim)
        for _ in range(group_size):
            new_layer_key_cache = torch.empty(self.cache_shape, dtype=self.dtype, device=self.device)
            new_layer_value_cache = torch.empty(self.cache_shape, dtype=self.dtype, device=self.device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
        logger.info(f"{self.cache_shape = } {self.key_cache[0].shape = } {self.key_cache[0].numel() = }")

        # Block management data structures
        self.allow_block_sharing = allow_block_sharing
        self.group_cache_managers: list[CacheAllocator] = []
        self.num_full_attention_groups = 0
        self.num_sliding_attention_groups = 0
        self.max_sliding_window_blocks_per_request = 0

        for i, group_type in enumerate(group_types):
            if group_type == "full_attention":
                cm = FullAttentionCacheAllocator(i, self.block_size, allow_block_sharing=allow_block_sharing)
                self.num_full_attention_groups += 1
            elif group_type == "sliding_attention":
                cm = SlidingAttentionCacheAllocator(i, self.block_size, config.sliding_window)
                self.num_sliding_attention_groups += 1
                self.max_sliding_window_blocks_per_request = cm._max_blocks_per_request
            else:
                raise ValueError(f"Invalid group type: {group_type}")
            self.group_cache_managers.append(cm)

        # We only use prefix sharing if the whole model has only full attention layers and block sharing is allowed
        self.use_prefix_sharing = allow_block_sharing and group_types == ["full_attention"]
        self._block_manager = BlockManager(num_blocks, self.block_size)
        self.blocks_to_complete: dict[str, int] = {}
        self._total_prefix_length: int = 0  # a counter to measure the impact of prefix sharing, also used in tests

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
        # Retrieve the layer read and write indices, and if there is a sliding window
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
            k_cache[layer_write_index, :, :] = key_states
            v_cache[layer_write_index, :, :] = value_states
            key_states_with_cache = k_cache[layer_read_index, :, :]
            value_states_with_cache = v_cache[layer_read_index, :, :]

        # Case: sliding window -- we  need to be careful of read/write order because of chunked prefill, because it's
        # the only case where you may write over cache you need to use
        else:
            # Add the cache to the key and value states
            mask = (layer_read_index == -1).unsqueeze(-1).unsqueeze(-1)  # TODO: should this be precomputed?
            key_states_with_cache = k_cache[layer_read_index, :, :]
            key_states_with_cache.masked_scatter_(mask, key_states)
            value_states_with_cache = v_cache[layer_read_index, :, :]
            value_states_with_cache.masked_scatter_(mask, value_states)
            # Write new KV values to the cache
            k_cache[layer_write_index, :, :] = key_states
            v_cache[layer_write_index, :, :] = value_states

        # Return the new KV values
        return key_states_with_cache, value_states_with_cache

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

    def mark_shareable_blocks_as_complete(self, state: RequestState) -> None:
        """Marks the blocks allocated to a request (state) as complete if they are shareable and they have been computed
        in the forward pass. A complete block is a block where the KV cache has been fully computed: if the block has
        enough space to hold the cache for N tokens, the block is marked as complete when the cache data is present for
        the N tokens. If block sharing is off, this is a no-op."""
        num_complete_blocks = 0 if not self.allow_block_sharing else self.blocks_to_complete.pop(state.request_id)
        if num_complete_blocks == 0:
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


# TODO: rework computation with the groups and their sizes
class PagedAttentionMemoryHandler:
    """A helper class to determine the best number of pages and maximum number of tokens per batch for the paged
    attention cache, providing automatic sizing based on available GPU memory.
    The helper works using the number of pages, which is tied to the number of blocks by:
        num_blocks = num_pages // block_size

    The memory footprint consists of three main components:
    - Cache memory: the space needed to store the cache tensors:
        2 * layer_group_size * [num_pages, page_size] * cache_dtype
    - Activation memory: the space temporarily taken by the largest activation during the model forward pass:
        peak_activation_per_token * max_tokens_per_batch * activation_dtype_size
    - Static tensors: the space taken by the input/output buffers and metadata tensors for batch processing, sum of:
        - inputs_ids + outputs_ids + position_ids + logits_indices: 4 * max_tokens_per_batch * int32_size
        - attention_mask: num_attention_masks * num_pages * max_tokens_per_batch * activation_dtype_size
        - cumulative_seqlens_q + cumulative_seqlens_k: (1 + 2) * max_tokens_per_batch * int32_size
        - write_index_tensor: num_groups * max_tokens_per_batch * int32_size
        - read_index_tensor: num_groups * (num_pages + max_tokens_per_batch) * int32_size

    The handler can operate in three modes:
    1. Auto-sizing: Determines both number of pages and maximum number of tokens per batch using quadratic optimization
    2. Fixed cache: Calculates max batch tokens given a fixed number of pages
    3. Fixed batch: Calculates number of pages given a fixed maximum batch size

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
    ) -> None:
        """Initialize the memory handler with the parameters that cannot be automatically inferred.

        Args:
            block_size: Size of the cache blocks
            page_size: Size of the cache pages
            num_groups: Number of layer groups
            group_size: Number of layers per layer group
            peak_activation_per_token: Maximum size of activation tensor per token, = hidden_size + vocab_size
            num_attention_masks: Number of attention masks, 0 if no attention mask is used, 2 if hybrid model, else 1
        """
        self.block_size = block_size
        self.page_size = page_size
        self.num_groups = num_groups
        self.group_size = group_size
        self.peak_activation_per_token = peak_activation_per_token
        self.num_attention_masks = num_attention_masks

    @staticmethod
    def get_available_memory(max_memory_percent: float = 1.0) -> int:
        """Calculate available GPU memory for cache allocation, accounting for already allocated tensors.
        This method queries the current memory state and applies the specified percentage limit to determine
        how much memory can be safely used for the paged attention cache.

        Args:
            max_memory_percent: Fraction of available memory to use (0.0-1.0). 1.0 means use all available memory.

        Returns:
            int: Available memory in bytes for cache allocation
        """
        _, total, reserved, allocated = get_device_and_memory_breakdown()
        available_memory = total - max(allocated, reserved)
        available_memory = int(available_memory * max_memory_percent)
        return available_memory

    def infer_num_blocks_and_max_batch_tokens(
        self,
        num_blocks: int | None = None,
        max_batch_tokens: int | None = None,
        max_memory_percent: float = 0.8,  # FIXME: it seems we overcommit memory, was changed from 0.9 which caused OOMs in our benchmarking CI
        cache_dtype: torch.dtype = torch.float16,
    ) -> tuple[int, int]:
        """Determine optimal number of blocks and maximum number of tokens per batch based on available memory and
        constraints. Check the class docstring for more details. Naming the number of pages as N and the maximum number
        of tokens per batch as M, the equation solved is:

        available_memory = sum([
            MN * num_attention_masks * activation_dtype_size,
            2N * (layer_group_size * page_size * cache_dtype + 2 * num_group),
            M * (peak_activation_per_token * activation_dtype + 28 + 4 * num_group),
        ])

        where we already simplified int32_size = 4.
        """
        if num_blocks is None:
            if max_batch_tokens is None:
                # If neither num_blocks nor max_batch_tokens are provided, we use a second-order polynomial
                num_blocks, max_batch_tokens = self.compute_num_blocks_and_max_batch_tokens(
                    max_memory_percent, cache_dtype
                )
            else:
                # If only max_batch_tokens is provided, we infer the num_blocks
                num_blocks = self.compute_num_blocks(max_batch_tokens, max_memory_percent, cache_dtype)
        elif max_batch_tokens is None:
            # If only num_blocks is provided, we infer the max_batch_tokens
            max_batch_tokens = self.compute_max_batch_tokens(num_blocks, max_memory_percent, cache_dtype)
        else:
            # If both num_blocks and max_batch_tokens are provided, we use them (useless, but helps with typing)
            max_batch_tokens = max_batch_tokens

        # We check if the memory footprint is too large in all cases
        available_memory = self.get_available_memory(max_memory_percent)
        memory_footprint = self.compute_memory_footprint(
            max_batch_tokens=max_batch_tokens, num_blocks=num_blocks, cache_dtype=cache_dtype
        )
        if memory_footprint > available_memory:
            raise MemoryError(f"Memory footprint {memory_footprint} is more than available memory {available_memory}")
        return num_blocks, max_batch_tokens

    def compute_num_blocks_and_max_batch_tokens(
        self,
        max_memory_percent: float,
        cache_dtype: torch.dtype = torch.float16,
        m: float = 0.01,
    ) -> tuple[int, int]:
        """Calculate optimal number of blocks and maximum number of tokens per batch using quadratic optimization when
        neither is fixed. This method assumes a relationship M = m * N where m is a small ratio below 1 and solves the
        resulting quadratic equation to find the optimal N that maximizes utilization within memory constraints. m is
        the amount of cache we can fill with one batch: m=0.01 means a batch fills at most 1% of the cache. The equation
        to solve is:

        available_memory = sum([
            m * N^2 * num_attention_masks * activation_dtype_size,
            2N * (layer_group_size * page_size * cache_dtype + 2 * num_group),
            m * N * (peak_activation_per_token * activation_dtype + 28 + 4 * num_group),
        ])

        If num_attention_masks is 0, the equation simplifies to a 1st degree polynomial.
        """
        cache_memory = self.get_available_memory(max_memory_percent)
        logger.info(f"Cache memory: {cache_memory}")

        # Compute second-degree polynomial coefficients
        a = m * self.num_attention_masks * self._activation_dtype.itemsize
        b = 2 * (self.group_size * self.page_size * cache_dtype.itemsize + 2 * self.num_groups)
        b += m * (self.peak_activation_per_token * self._activation_dtype.itemsize + 28 + 4 * self.num_groups)
        c = -cache_memory
        logger.debug(f"Coefficients of 2nd degree polynomial: {a = }, {b = }, {c = }")

        # If num_attention_masks is 0, the equation simplifies to a 1st degree polynomial
        if self.num_attention_masks == 0:
            greatest_solution = -c / b
        # Otherwise, we solve the quadratic equation
        else:
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                raise ValueError(f"Discriminant is negative: {discriminant = }")
            greatest_solution = (-b + sqrt(discriminant)) / (2 * a)

        if greatest_solution < 0:
            raise ValueError(f"Greatest solution is negative: {greatest_solution = }")

        # Infer number of blocks and max batch tokens
        num_pages = floor(greatest_solution)
        num_blocks = num_pages // self.block_size
        if num_blocks > self._upper_bound_num_blocks:
            logger.info(f"{num_blocks = } is too large, setting to {self._upper_bound_num_blocks = }")
            num_blocks = self._upper_bound_num_blocks
        max_batch_tokens = int(greatest_solution * m)
        if max_batch_tokens > self._upper_bound_max_batch_tokens:
            logger.info(f"{max_batch_tokens = } is too large, setting to {self._upper_bound_max_batch_tokens = }")
            max_batch_tokens = self._upper_bound_max_batch_tokens
        return num_blocks, max_batch_tokens

    def compute_max_batch_tokens(
        self,
        num_blocks: int,
        max_memory_percent: float,
        cache_dtype: torch.dtype = torch.float16,
    ) -> int:
        """Calculate maximum batch tokens M given a fixed number of cache blocks. The formula for M is given by:

        M = (available_memory - 2N * (layer_group_size * page_size * cache_dtype + 2 * num_group))
            / (activation_dtype_size * (N * num_attention_masks + peak_activation_per_token) + 28 + 4 * num_group)
        """
        cache_memory = self.get_available_memory(max_memory_percent)
        num_pages = num_blocks * self.block_size
        # Compute numerator
        num = cache_memory
        num -= 2 * num_pages * (self.group_size * self.page_size * cache_dtype.itemsize + 2 * self.num_groups)
        # Compute denominator
        denum = self._activation_dtype.itemsize * (
            num_pages * self.num_attention_masks + self.peak_activation_per_token
        )
        denum += 28 + 4 * self.num_groups
        # Compute max batch tokens and return
        max_batch_tokens = floor(num / denum)
        if max_batch_tokens > self._upper_bound_max_batch_tokens:
            logger.info(f"{max_batch_tokens = } is too large, setting to {self._upper_bound_max_batch_tokens = }")
            max_batch_tokens = self._upper_bound_max_batch_tokens
        return max_batch_tokens

    def compute_num_blocks(
        self,
        max_batch_tokens: int,
        max_memory_percent: float,
        cache_dtype: torch.dtype = torch.float16,
    ) -> int:
        """Calculate number of cache blocks N given a fixed maximum token per token M. The formula for N is given by:

        N = (available_memory - M * (peak_activation_per_token * activation_dtype + 28 + 4 * num_group))
          / (2 * (layer_group_size * page_size * cache_dtype + 2 * num_group) + M * (num_attention_masks * activation_dtype_size))
        """
        cache_memory = self.get_available_memory(max_memory_percent)
        # Compute numerator
        num = cache_memory
        num -= max_batch_tokens * self.peak_activation_per_token * self._activation_dtype.itemsize
        num -= max_batch_tokens * (28 + 4 * self.num_groups)
        # Compute denominator
        denum = 2 * (self.group_size * self.page_size * cache_dtype.itemsize + 2 * self.num_groups)
        denum += max_batch_tokens * (self.num_attention_masks * self._activation_dtype.itemsize)
        denum += max_batch_tokens * self._activation_dtype.itemsize
        # Compute cache size and return number of blocks
        num_pages = floor(num / denum)
        num_blocks = num_pages // self.block_size
        if num_blocks > self._upper_bound_num_blocks:
            logger.info(f"{num_blocks = } is too large, setting to {self._upper_bound_num_blocks = }")
            num_blocks = self._upper_bound_num_blocks
        return num_blocks

    def compute_memory_footprint(
        self,
        num_blocks: int,
        max_batch_tokens: int,
        cache_dtype: torch.dtype,
    ) -> int:
        """Calculate the memory footprint breakdown for a given number of blocks and maximum batch tokens. The memory
        footprint is given by:

        available_memory = sum([
            MN * num_attention_masks * activation_dtype_size,
            2N * (layer_group_size * page_size * cache_dtype + 2 * num_group),
            M * (peak_activation_per_token * activation_dtype + 28 + 4 * num_group),
        ])
        but is broken down below.
        """
        num_pages = num_blocks * self.block_size

        cache_memory_footprint = 2 * self.group_size * num_pages * self.page_size * cache_dtype.itemsize

        activation_memory_footprint = self.peak_activation_per_token * self._activation_dtype.itemsize
        activation_memory_footprint *= max_batch_tokens

        inputs_outputs_positions_and_logits_memory_footprint = 4 * max_batch_tokens * 4  # second 4 is for int32 size

        attention_memory_footprint = self.num_attention_masks * self._activation_dtype.itemsize
        attention_memory_footprint *= num_pages * max_batch_tokens

        cumulative_seqlens_memory_footprint = 3 * max_batch_tokens * 4  # 4 is for int32 size

        write_index_memory_footprint = self.num_groups * max_batch_tokens * 4  # 4 is for int32 size
        read_index_memory_footprint = self.num_groups * (num_pages + max_batch_tokens) * 4  # 4 is for int32 size

        total_memory_footprint = sum(
            [
                cache_memory_footprint,
                activation_memory_footprint,
                inputs_outputs_positions_and_logits_memory_footprint,
                attention_memory_footprint,
                cumulative_seqlens_memory_footprint,
                write_index_memory_footprint,
                read_index_memory_footprint,
            ]
        )
        return total_memory_footprint
