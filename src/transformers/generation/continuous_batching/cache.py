# coding=utf-8
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
from collections import deque
from math import floor, gcd, sqrt
from typing import Optional, Union

import torch

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import GenerationConfig
from ...utils.metrics import attach_tracer, traced
from .cache_manager import CacheManager, FullAttentionCacheManager, SlidingAttentionCacheManager
from .requests import get_device_and_memory_breakdown, logger


NO_SLIDING_WINDOW = 1



def group_layers_by_attn_type(config: PretrainedConfig) -> tuple[list[list[int]], list[str]]:
    """
    Group layers depending on the attention mix, according to VLLM's hybrid allocator rules:
        - Layers in each group need to have the same type of attention
        - All groups have the same number of layers
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
    An object to manage the cache for a paged attention mechanism.
    At the core of this is the `cache` attribute, which is a tensor of shape `[num_blocks, block_size, page_size]`.
    A page is the smallest unit of cache, of size [num_layers_per_group, num_heads, head_size]. For a mode with only
    full-attention layers, num_layers_per_group == num_hidden_layers so page size is the space to store one token in the
    KV cache.
    A block contains `block_size` pages, and is the unit of allocation for the cache. The reason we group pages into
    blocks is to reduce the complexity of cache management and fragmentation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        num_requests: int = 100,
        layer_device_map: Optional[dict[int, Union[str, torch.device, int]]] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        """Initialize a paged attention cache for efficient memory usage.

        Args:
            config: Model configuration
            generation_config: Generation configuration containing cache parameters
            device: Device for the cache tensors
            dtype: Data type for the cache tensors
            layer_device_map: Optional mapping of layer indices to devices
            initial_prompt_shapes: Optional sample prompts to help calculate optimal cache size
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
        self.group_size = len(layer_groups[0])
        self.num_groups = len(layer_groups)

        self.layer_index_to_group_indices = {}
        for i, group in enumerate(layer_groups):
            for j, layer in enumerate(group):
                self.layer_index_to_group_indices[layer] = (i, j)

        # Handle TP (or dont)
        if tp_size is not None and tp_size > 1:
            raise NotImplementedError("Tensor parallelism is not supported yet")
            # if self.num_key_value_heads % tp_size != 0:
            #     raise ValueError(
            #         f"Number of key value heads {self.num_key_value_heads} must be divisible by tensor parallel size {tp_size}."
            #     )
            # If the model is using tensor parallelism, we need to adjust the number of heads accordingly.
            # self.num_key_value_heads //= tp_size # TODO: why is this commented out?

        # Infer number of blocks and max batch tokens
        self.page_size = self.group_size * self.head_dim * self.num_key_value_heads
        memory_handler = PagedAttentionMemoryHandler(
            block_size=self.block_size,
            page_size=self.page_size,
            num_groups=self.num_groups,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )
        num_blocks, max_batch_tokens = memory_handler.infer_num_blocks_and_max_batch_tokens(
            num_blocks=getattr(generation_config, "num_blocks", None),
            max_batch_tokens=getattr(generation_config, "max_batch_tokens", None),
            max_memory_percent=getattr(generation_config, "max_memory", 0.9),
            cache_dtype=self.dtype,
        )

        # Add the inferred attributes to the class
        self.num_blocks = num_blocks
        self.max_batch_tokens = max_batch_tokens
        logger.warning(
            f"PagedAttentionCache initialized with {self.num_blocks = }, {self.block_size = }, {self.page_size = }, "
            f"{self.max_batch_tokens = }"
        )

        # Initialize the cache
        self.cache_shape = (num_blocks, self.block_size, self.page_size)
        self.key_cache: torch.Tensor = torch.empty(self.cache_shape, dtype=self.dtype, device=device)
        self.value_cache: torch.Tensor = torch.empty(self.cache_shape, dtype=self.dtype, device=device)
        torch._dynamo.mark_static_address(self.key_cache)
        torch._dynamo.mark_static_address(self.value_cache)

        # Block management data structures
        self._free_blocks = deque(range(num_blocks))
        self.group_cache_managers: list[CacheManager] = []
        for i, group_type in enumerate(group_types):
            if group_type == "full_attention":
                cm = FullAttentionCacheManager(i, self.block_size)
            elif group_type == "sliding_attention":
                cm = SlidingAttentionCacheManager(i, self.block_size, config.sliding_window)
            else:
                raise ValueError(f"Invalid group type: {group_type}")
            self.group_cache_managers.append(cm)

        # Add the sliding windows to the class
        self.sliding_windows = {
            layer: getattr(self.group_cache_managers[i], "sliding_window", NO_SLIDING_WINDOW)
            for layer, (i, _) in self.layer_index_to_group_indices.items()
        }

    @traced
    def allocate_blocks(self, n_blocks: int, request_id: str) -> int:
        """Allocates n_blocks for a given request_id. Returns the number of blocks allocated if allocation was
        successful and None otherwise."""
        max_allocated = 0
        for cm in self.group_cache_managers:
            allocated = cm.allocate_blocks(n_blocks, request_id, self._free_blocks)
            if allocated is None:
                return None
            max_allocated = max(max_allocated, allocated)
        return max_allocated

    @traced
    def free_blocks(self, request_id: str) -> None:
        """Frees all blocks associated with a request_id."""
        for cm in self.group_cache_managers:
            cm.free_blocks(request_id, self._free_blocks)

    def get_num_free_blocks(self) -> int:
        """Returns the number of free blocks available."""
        return len(self._free_blocks)

    @traced
    def get_read_indices(
        self, request_id: str, past_length: int, query_length: int, read_index: list[list[int]]
    ) -> None:
        """Maps logical sequence indices to thephysical cache indices."""
        for cm, read_indices in zip(self.group_cache_managers, read_index):
            indices = cm.get_read_indices(request_id, past_length, query_length)
            read_indices.extend(indices)

    @traced
    def get_write_indices(
        self, request_id: str, past_length: int, query_length: int, write_index: list[list[int]]
    ) -> None:
        """Maps logical sequence indices to physical cache indices."""
        for cm, write_indices in zip(self.group_cache_managers, write_index):
            indices = cm.get_write_indices(request_id, past_length, query_length)
            write_indices.extend(indices)

    @traced
    def update(
        self,
        key_states: torch.Tensor, # shape [1, num_kv_heads, seqlen_kv, head_dim]
        value_states: torch.Tensor, # shape [1, num_kv_heads, seqlen_kv, head_dim]
        layer_idx: int,
        read_index: torch.Tensor, # shape [num_layer_groups, seqlen_kv + past_length]
        write_index: torch.Tensor, # shape [num_layer_groups, seqlen_q]
        group_read_write_length: list[tuple[int, int]], # shape [num_layer_groups]
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]: # shape [seqlen_kv + past_length, num_kv_heads, head_dim]
        """
        Write new KV values to the cache. Cache has shape [num_blocks, block_size, page_size] but because
        `num_blocks * block_size = num_pages` and `page_size = num_heads * num_layers_per_group * head_size`,
        we can view the cache as a tensor of shape [num_layers_per_group, num_pages, num_heads, head_size]
        """
        # Retrieve the layer read and write indices
        group_idx, layer_idx_in_group = self.layer_index_to_group_indices[layer_idx]
        group_read_length, group_write_length = group_read_write_length[group_idx]
        layer_read_index = read_index[group_idx, :group_read_length]
        layer_write_index = write_index[group_idx, :group_write_length]
        # Reshape cache for easier indexing
        num_pages = self.num_blocks * self.block_size
        k_cache_flat = self.key_cache.view(self.group_size, num_pages, self.num_key_value_heads, self.head_dim)
        v_cache_flat = self.value_cache.view(self.group_size, num_pages, self.num_key_value_heads, self.head_dim)
        # Transpose the key and value states to match the cache shape, after which shape is [seqlen_kv, num_kv_heads, head_dim]
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)
        # Add the cache to the key and value states
        mask = (layer_read_index == -1) # TODO: check if this can be efficiently precomputed / if we can pass a cutoff for each group
        key_states_with_cache = k_cache_flat[layer_idx_in_group, layer_read_index, :, :]
        key_states_with_cache[mask] = key_states
        value_states_with_cache = v_cache_flat[layer_idx_in_group, layer_read_index, :, :]
        value_states_with_cache[mask] = value_states
        # Write new KV values to the cache
        k_cache_flat[layer_idx_in_group, layer_write_index, :, :] = key_states
        v_cache_flat[layer_idx_in_group, layer_write_index, :, :] = value_states
        # Return the new KV values
        return key_states_with_cache, value_states_with_cache


class PagedAttentionMemoryHandler:
    _activation_dtype = torch.bfloat16
    _input_dtype = torch.int32
    _upper_bound_max_batch_tokens = 256
    _upper_bound_num_blocks = 4096 * 32

    def __init__(self, block_size: int, page_size: int, num_groups: int, hidden_size: int, vocab_size: int) -> None:
        self.block_size = block_size
        self.page_size = page_size
        self.num_groups = num_groups
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    @staticmethod
    def get_available_memory(max_memory_percent: float = 1.0) -> int:
        _, total, reserved, allocated = get_device_and_memory_breakdown()
        available_memory = total - max(allocated, reserved)
        available_memory = int(available_memory * max_memory_percent)
        return available_memory

    def infer_num_blocks_and_max_batch_tokens(
        self,
        num_blocks: Optional[int] = None,
        max_batch_tokens: Optional[int] = None,
        max_memory_percent: float = 0.9,
        cache_dtype: torch.dtype = torch.float16,
    ) -> tuple[int, int]:
        """
        The memory footprint depends on the cache size C and the max batch tokens M in the following way:
            Mem = Mem(cache) + Mem(activation) + Mem(static_tensors)
        where:
            Mem(cache) = 2 * page_size * num_groups * cache_dtype.itemsize * C
            Mem(activation) = M * (hidden_size + vocab_size) * activation_dtype.itemsize
            Mem(static_tensors) ~= 8M * input_dtype.itemsize + M * C * activation_dtype.itemsize

        Depending on if C or M is given, we use different methods to infer the values (C = num_blocks * block_size) and
        since block_size is fixed, num_blocks is the true variable to find.
        """
        # If neither num_blocks nor max_batch_tokens are provided, we use a second-order polynomial
        if num_blocks is None and max_batch_tokens is None:
            num_blocks, max_batch_tokens = self.compute_num_blocks_and_max_batch_tokens(
                max_memory_percent, cache_dtype
            )
        # If only num_blocks is provided, we infer the max_batch_tokens
        elif num_blocks is not None and max_batch_tokens is None:
            max_batch_tokens = self.compute_max_batch_tokens(num_blocks, max_memory_percent, cache_dtype)
        # If only max_batch_tokens is provided, we infer the num_blocks
        elif max_batch_tokens is not None and num_blocks is None:
            num_blocks = self.compute_num_blocks(max_batch_tokens, max_memory_percent, cache_dtype)

        # We check if the memory footprint is too large in all cases
        available_memory = self.get_available_memory(max_memory_percent)
        memory_footprint = self.compute_memory_footprint(
            max_batch_tokens=max_batch_tokens,
            num_blocks=num_blocks,
            cache_dtype=cache_dtype,
        )
        if sum(memory_footprint) > available_memory:
            raise MemoryError(f"Memory footprint {memory_footprint} is more than available memory {available_memory}")
        return num_blocks, max_batch_tokens

    def compute_num_blocks_and_max_batch_tokens(
        self,
        max_memory_percent: float = 0.9,
        cache_dtype: torch.dtype = torch.float16,
        m: float = 0.01,
    ) -> tuple[int, int]:
        """
        If neither M nor C is given, we assume M = m*C so we have to solve a second-order polynomial in C:
            Mem = C * 2 * self.num_heads * self.head_dim * self.num_layers * cache_dtype.itemsize
                + C * m * (hidden_size + vocab_size) * activation_dtype.itemsize
                + C * m * 8 * input_dtype.itemsize + C^2 * m * activation_dtype.itemsize

        We solve for C and then M = m*C.
        """
        cache_memory = self.get_available_memory(max_memory_percent)
        logger.info(f"Cache memory: {cache_memory}")

        # Compute memory footprints
        mem_per_activation_token = m * self._activation_dtype.itemsize * (self.hidden_size + self.vocab_size)
        mem_per_cache_token = 2 * self.page_size * self.num_groups * cache_dtype.itemsize
        mem_per_input_token = 8 * m * self._input_dtype.itemsize
        logger.info(f"Memory per activation token: {mem_per_activation_token}")
        logger.info(f"Memory per cache token: {mem_per_cache_token}")
        logger.info(f"Memory per input token: {mem_per_input_token}")

        # Compute second-degree polynomial coefficients
        a = m * self._activation_dtype.itemsize
        b = mem_per_input_token + mem_per_cache_token + mem_per_activation_token
        c = -cache_memory

        # Compute discriminant and greatest solution
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            raise ValueError(f"Discriminant is negative: {discriminant = }")
        greatest_solution = (-b + sqrt(discriminant)) / (2 * a)
        if greatest_solution < 0:
            raise ValueError(f"Greatest solution is negative: {greatest_solution = }")

        # Infer number of blocks and max batch tokens
        num_blocks = int(greatest_solution) // self.block_size
        if num_blocks > self._upper_bound_num_blocks:
            logger.warning(f"{num_blocks = } is too large, setting to {self._upper_bound_num_blocks = }")
            num_blocks = self._upper_bound_num_blocks
        max_batch_tokens = int(greatest_solution * m)
        if max_batch_tokens > self._upper_bound_max_batch_tokens:
            logger.warning(f"{max_batch_tokens = } is too large, setting to {self._upper_bound_max_batch_tokens = }")
            max_batch_tokens = self._upper_bound_max_batch_tokens
        return num_blocks, max_batch_tokens

    def compute_max_batch_tokens(
        self,
        num_blocks: int,
        max_memory_percent: float = 0.9,
        cache_dtype: torch.dtype = torch.float16,
    ) -> int:
        """
        If C is given, we have a formula for M:
            num = (Mem - C * 2 * page_size * num_groups * cache_dtype.itemsize)
            denum = (8 * input_dtype.itemsize + C * activation_dtype.itemsize + (hidden_size + vocab_size) * activation_dtype.itemsize)
        M = num / denum
        """
        cache_memory = self.get_available_memory(max_memory_percent)
        cache_size = num_blocks * self.block_size
        # Compute numerator
        num = cache_memory
        num -= cache_size * 2 * self.page_size * self.num_groups * cache_dtype.itemsize
        # Compute denominator
        denum = 8 * self._input_dtype.itemsize + cache_size * self._activation_dtype.itemsize
        denum += (self.hidden_size + self.vocab_size) * self._activation_dtype.itemsize
        # Compute max batch tokens and return
        return int(num / denum)

    def compute_num_blocks(
        self,
        max_batch_tokens: int,
        max_memory_percent: float = 0.9,
        cache_dtype: torch.dtype = torch.float16,
    ) -> int:
        """
        If M is given, we have a formula for C:
            num = Mem - M * (hidden_size + vocab_size) * activation_dtype.itemsize - 8 * M * input_dtype.itemsize
            denum = 2 * page_size * num_groups * cache_dtype.itemsize + M * activation_dtype.itemsize
        C = num / denum
        """
        cache_memory = self.get_available_memory(max_memory_percent)
        # Compute numerator
        num = cache_memory
        num -= self._activation_dtype.itemsize * (self.hidden_size + self.vocab_size) * max_batch_tokens
        num -= 8 * max_batch_tokens * self._input_dtype.itemsize
        # Compute denominator
        denum = 2 * self.page_size * self.num_groups * cache_dtype.itemsize
        denum += max_batch_tokens * self._activation_dtype.itemsize
        # Compute cache size and return number of blocks
        cache_size = int(num / denum)
        return floor(cache_size / self.block_size)

    def compute_memory_footprint(
        self,
        num_blocks: Optional[int] = None,
        max_batch_tokens: Optional[int] = None,
        cache_dtype: torch.dtype = torch.float16,
    ) -> tuple[int, int, int]:
        # Compute activation memory footprint
        activation_memory_footprint = self._activation_dtype.itemsize * (self.hidden_size + self.vocab_size)
        activation_memory_footprint *= max_batch_tokens
        # Compute cache memory footprint if num_blocks is provided
        if num_blocks is not None:
            cache_size = num_blocks * self.block_size
            bytes_per_token = 2 * self.page_size * self.num_groups * cache_dtype.itemsize
            cache_memory_footprint = cache_size * bytes_per_token
        else:
            cache_memory_footprint = -1
        # Compute static tensors memory footprint if num_blocks and max_batch_tokens is provided
        if num_blocks is not None and max_batch_tokens is not None:
            static_memory_footprint = sum(
                [
                    3 * max_batch_tokens * self._input_dtype.itemsize,  # input_ids, position_ids, output_ids
                    max_batch_tokens * cache_size * self._activation_dtype.itemsize,  # attention_mask
                    2 * max_batch_tokens * self._input_dtype.itemsize,  # cumulative_seqlens_qk (we remove the +1 to M)
                    3 * max_batch_tokens * self._input_dtype.itemsize,  # write_index, read_index, logits_indices
                ]
            )
        else:
            static_memory_footprint = -1
        return activation_memory_footprint, cache_memory_footprint, static_memory_footprint



# TODO: test the impact of this
# def get_read_indices(self, request_id: str, past_length: int) -> list[int]:
#     # Retrieve the block table for the request and raise an error if it doesn't exist
#     block_table = self._block_table.get(request_id)
#     if block_table is None:
#         raise ValueError(f"No block table found for request {request_id}")
#     # Compute the physical indices
#     physical_indices = []
#     n_left = past_length
#     for block_idx in block_table:
#         block_physical_index = block_idx * self.block_size
#         pages_used = min(self.block_size, n_left)
#         physical_indices.extend(block_physical_index + i for i in range(pages_used))
#         n_left -= pages_used
#         if n_left == 0:
#             return physical_indices
#     raise ValueError(f"Request {request_id} required too many indices: {past_length = } and {len(block_table) = }")
