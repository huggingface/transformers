from typing import Optional
from math import sqrt, floor
import torch
from ...utils.logging import logging
from ...utils.metrics import traced


logger = logging.getLogger(__name__)


class PagedAttentionMemoryHandler:

    _activation_dtype = torch.bfloat16
    _activation_safety_factor = 2
    _input_dtype = torch.int32
    _upper_bound_max_batch_tokens = 1024
    _upper_bound_num_blocks = 1024

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
    def get_device_and_memory_breakdown() -> tuple[torch.device, int, int, int]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            reserved_memory = torch.cuda.memory_reserved(device)
            allocated_memory = torch.cuda.memory_allocated(device)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            # MPS memory reporting (PyTorch 2.0+)
            total_memory = torch.mps.driver_allocated_memory()
            allocated_memory = total_memory - torch.mps.recommended_max_memory()
            reserved_memory = 0  # MPS does not track reserved separately
        else:
            device = torch.device("cpu")
            total_memory = None
            reserved_memory = 0
            allocated_memory = 0
        return device, total_memory, reserved_memory, allocated_memory

    @staticmethod
    def get_available_memory(max_memory_percent: float = 1.0) -> int:
        _, total, reserved, allocated = PagedAttentionMemoryHandler.get_device_and_memory_breakdown()
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
        # If neither num_blocks nor max_batch_tokens are provided, we use a second-order polynomial
        if num_blocks is None and max_batch_tokens is None:
            num_blocks, max_batch_tokens = self.compute_num_blocks_and_max_batch_tokens(max_memory_percent, cache_dtype)
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
        logger.warning(f"{available_memory = }, {memory_footprint = }, {num_blocks = }, {max_batch_tokens = }")
        if sum(memory_footprint) > available_memory:
            raise MemoryError(f"Memory footprint {memory_footprint} is more than available memory {available_memory}")
        return num_blocks, max_batch_tokens


    def compute_num_blocks_and_max_batch_tokens(
        self,
        max_memory_percent: float = 0.9,
        cache_dtype: torch.dtype = torch.float16,
        m: float = 0.1,
    ) -> tuple[int, int]:
        cache_memory = self.get_available_memory(max_memory_percent)

        # Compute second-degree polynomial coefficients
        a = m * self._activation_dtype.itemsize
        b = 8 * m * self._input_dtype.itemsize
        b += 2 * self.num_heads * self.head_dim * self.num_layers * cache_dtype.itemsize
        c = self._activation_dtype.itemsize * (self.hidden_size + self.vocab_size) * self._activation_safety_factor
        c += 2 * self._input_dtype.itemsize
        c -= cache_memory

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
        cache_memory = self.get_available_memory(max_memory_percent)
        cache_size = num_blocks * self.block_size
        # Compute numerator
        num = cache_memory
        num -= self._activation_dtype.itemsize * (self.hidden_size + self.vocab_size) * self._activation_safety_factor
        num -= 2 * self._input_dtype.itemsize
        num -= cache_size * 2 * self.num_heads * self.head_dim * self.num_layers * cache_dtype.itemsize
        # Compute denominator
        denum = 8 * self._input_dtype.itemsize + cache_size * self._activation_dtype.itemsize
        # Compute max batch tokens and return
        return int(num / denum)

    def compute_num_blocks(
        self,
        max_batch_tokens: int,
        max_memory_percent: float = 0.9,
        cache_dtype: torch.dtype = torch.float16,
    ) -> int:
        cache_memory = self.get_available_memory(max_memory_percent)
        # Compute numerator
        num = cache_memory
        num -= self._activation_dtype.itemsize * (self.hidden_size + self.vocab_size) * self._activation_safety_factor
        num -= 8 * max_batch_tokens * self._input_dtype.itemsize
        num -= 2 * self._input_dtype.itemsize
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
        activation_memory_footprint *= self._activation_safety_factor
        # Compute cache memory footprint if num_blocks is provided
        if num_blocks is not None:
            cache_size = num_blocks * self.block_size
            bytes_per_token = 2 * self.num_heads * self.head_dim * self.num_layers * cache_dtype.itemsize
            cache_memory_footprint = cache_size * bytes_per_token
        else:
            cache_memory_footprint = -1
        # Compute static tensors memory footprint if num_blocks and max_batch_tokens is provided
        if num_blocks is not None and max_batch_tokens is not None:
            static_memory_footprint = sum([
                3 * max_batch_tokens * self._input_dtype.itemsize, # input_ids, position_ids, output_ids
                max_batch_tokens * cache_size * self._activation_dtype.itemsize, # attention_mask
                2 * (max_batch_tokens + 1) * self._input_dtype.itemsize, # cumulative_seqlens_qk
                3 * max_batch_tokens * self._input_dtype.itemsize, # write_index, read_index, logits_indices
            ])
        else:
            static_memory_footprint = -1
        return activation_memory_footprint, cache_memory_footprint, static_memory_footprint
