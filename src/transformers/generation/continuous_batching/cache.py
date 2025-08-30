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
from math import floor, sqrt
from typing import Optional, Union

import torch

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import GenerationConfig
from ...utils.metrics import attach_tracer, traced
from .classes import RequestState, get_device_and_memory_breakdown, logger


@attach_tracer()
class PagedAttentionCache:
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
        self.dtype = dtype
        self.device = device

        # Extract model dimensions
        kv_heads = getattr(config, "num_key_value_heads", None)
        self.num_key_value_heads: int = kv_heads if kv_heads is not None else config.num_attention_heads
        head_dim = getattr(config, "head_dim", None)
        self.head_dim: int = head_dim if head_dim is not None else config.hidden_size // config.num_attention_heads

        self.num_hidden_layers = config.num_hidden_layers
        self.block_size = getattr(generation_config, "block_size", 32)

        # Handle TP
        if tp_size is not None and tp_size > 1:
            if self.num_key_value_heads % tp_size != 0:
                raise ValueError(
                    f"Number of key value heads {self.num_key_value_heads} must be divisible by tensor parallel size {tp_size}."
                )
            # If the model is using tensor parallelism, we need to adjust the number of heads accordingly.
            # self.num_key_value_heads //= tp_size # TODO: why is this commented out?

        # Infer number of blocks and max batch tokens
        memory_handler = PagedAttentionMemoryHandler(
            block_size=self.block_size,
            head_dim=self.head_dim,
            num_heads=self.num_key_value_heads,
            num_layers=self.num_hidden_layers,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
        )
        num_blocks, max_batch_tokens = memory_handler.infer_num_blocks_and_max_batch_tokens(
            num_blocks=getattr(generation_config, "num_blocks", None),
            max_batch_tokens=getattr(generation_config, "max_batch_tokens", None),
            max_memory_percent=getattr(generation_config, "max_memory", 0.9),
            cache_dtype=self.dtype,
        )

        # Add the infered attributes to the class
        self.num_blocks = num_blocks
        self.max_batch_tokens = max_batch_tokens
        logger.warning(f"PagedAttentionCache initialized with {self.num_blocks = } and {self.max_batch_tokens = } ")

        # Initialize the cache
        self.cache_shape = (self.num_key_value_heads, num_blocks, self.block_size, self.head_dim)
        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        for idx in range(config.num_hidden_layers):
            layer_device = layer_device_map[idx] if layer_device_map is not None else device
            new_layer_key_cache = torch.zeros(self.cache_shape, dtype=self.dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(self.cache_shape, dtype=self.dtype, device=layer_device)
            # Note: `mark_static_address` is used to tag the cache as a fixed data pointer,
            # preventing compiled graph breaks when updating the cache.
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

        # Block management data structures
        self._free_blocks = deque(range(num_blocks))
        self._block_tables: dict[str, list[int]] = {}

    @traced
    def allocate_blocks(self, n_blocks: int, request_id: str) -> list[int]:
        """Allocates n_blocks for a given request_id."""
        if len(self._free_blocks) < n_blocks:
            return False

        allocated = []
        for _ in range(n_blocks):
            allocated.append(self._free_blocks.popleft())

        if request_id not in self._block_tables:
            self._block_tables[request_id] = []
        self._block_tables[request_id].extend(allocated)
        return allocated

    @traced
    def free_blocks(self, request_id: str) -> None:
        """Frees all blocks associated with a request_id."""
        if request_id in self._block_tables:
            blocks_to_free = self._block_tables.pop(request_id)
            self._free_blocks.extend(blocks_to_free)
        else:
            logger.info(f"Attempted to free blocks for non-existent request_id: {request_id}")

    def get_num_free_blocks(self) -> int:
        """Returns the number of free blocks available."""
        return len(self._free_blocks)

    def get_block_table(self, request_id: str) -> list[int]:
        """Returns the block table for a request."""
        return self._block_tables.get(request_id, [])

    @traced
    def _get_physical_indices(self, state: RequestState, logical_indices: list[int]) -> list[int]:
        """
        Maps logical sequence indices to physical cache indices using the block table, using PyTorch.

        Args:
            request_id: The request ID.
            logical_indices: A list of logical indices.

        Returns:
            A list of physical indices.

        Raises:
            ValueError: If no block table is found for the request ID.
            IndexError: If a logical index maps to a block index that is out of bounds.
        """
        request_id = state.request_id
        block_table = self._block_tables.get(request_id)
        if not block_table:
            raise ValueError(f"No block table found for request {request_id}")

        block_size = self.block_size
        physical_indices = []

        for idx in logical_indices:
            block_idx = idx // block_size
            block_offset = idx % block_size

            if block_idx >= len(block_table):
                raise IndexError(
                    f"Logical index {idx} maps to block index {block_idx} which is out of bounds "
                    f"for request {request_id}"
                )

            physical_block_num = block_table[block_idx]
            physical_index = physical_block_num * block_size + block_offset
            physical_indices.append(physical_index)

        return physical_indices

    @traced
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        read_index,
        write_index,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Reshape cache for easier indexing
        total_slots = self.num_blocks * self.block_size
        k_cache_flat = self.key_cache[layer_idx].view(self.num_key_value_heads, total_slots, self.head_dim)
        v_cache_flat = self.value_cache[layer_idx].view(self.num_key_value_heads, total_slots, self.head_dim)
        k_cache_flat[:, write_index, :] = key_states[0]
        v_cache_flat[:, write_index, :] = value_states[0]
        return k_cache_flat[None, :, read_index, :], v_cache_flat[None, :, read_index, :]


class PagedAttentionMemoryHandler:
    _activation_dtype = torch.bfloat16
    _activation_safety_factor = 2
    _input_dtype = torch.int32
    _upper_bound_max_batch_tokens = 256
    _upper_bound_num_blocks = 4096

    def __init__(
        self,
        block_size: int,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_size: int,
        vocab_size: int,
    ) -> None:
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
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
            Mem(cache) = 2 * num_heads * head_dim * num_layers * cache_dtype.itemsize * C
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
        mem_per_cache_token = 2 * self.num_heads * self.head_dim * self.num_layers * cache_dtype.itemsize
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
            num = (Mem - C * 2 * num_heads * head_dim * num_layers * cache_dtype.itemsize)
            denum = (8 * input_dtype.itemsize + C * activation_dtype.itemsize + (hidden_size + vocab_size) * activation_dtype.itemsize)
        M = num / denum
        """
        cache_memory = self.get_available_memory(max_memory_percent)
        cache_size = num_blocks * self.block_size
        # Compute numerator
        num = cache_memory
        num -= cache_size * 2 * self.num_heads * self.head_dim * self.num_layers * cache_dtype.itemsize
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
            denum = 2 * num_heads * head_dim * num_layers * cache_dtype.itemsize + M * activation_dtype.itemsize
        C = num / denum
        """
        cache_memory = self.get_available_memory(max_memory_percent)
        # Compute numerator
        num = cache_memory
        num -= self._activation_dtype.itemsize * (self.hidden_size + self.vocab_size) * max_batch_tokens
        num -= 8 * max_batch_tokens * self._input_dtype.itemsize
        # Compute denominator
        denum = 2 * self.num_heads * self.head_dim * self.num_layers * cache_dtype.itemsize
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
            bytes_per_token = 2 * self.num_heads * self.head_dim * self.num_layers * cache_dtype.itemsize
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
