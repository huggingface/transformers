# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import queue
import statistics
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.profiler import profile, schedule, tensorboard_trace_handler
from tqdm import tqdm

from ..cache_utils import Cache
from ..configuration_utils import PretrainedConfig
from ..generation.configuration_utils import GenerationConfig
from ..utils.metrics import ContinuousBatchProcessorMetrics, attach_tracer, traced


class RequestStatus(Enum):
    """Status of a generation request through its lifecycle."""

    PENDING = "pending"
    PREFILLING = "prefilling"
    PREFILLING_SPLIT = "prefilling_split"
    SPLIT_PENDING_REMAINDER = "split_pending_remainder"
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"


# Setup your logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class GenerationOutput:
    """Tracks the output of a generation request.

    Attributes:
        request_id (str): The ID of the generation request.
        prompt_ids (list[int]): The IDs of the prompt tokens.
        generated_tokens (list[int]): The generated tokens.
        logprobs (list[float]): The log probabilities of the generated tokens.
        error (Optional[str]): Any error message associated with the request. When None, the request was successful.
    """

    request_id: str
    prompt_ids: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    error: Optional[str] = None
    status: RequestStatus = RequestStatus.PENDING
    created_time: float = field(default_factory=time.time)


@dataclass
class RequestState:
    """Tracks the state of a generation request through its lifecycle.

    Attributes:
        status (RequestStatus): can be one of PENDING, PREFILLING, PREFILLING_SPLIT,
                                SPLIT_PENDING_REMAINDER, DECODING, FINISHED, FAILED
    """

    # Required fields
    request_id: str
    prompt_ids: Optional[list[int]] = None  # the one being processed
    full_prompt_ids: Optional[list[int]] = None  # the full prompt
    remaining_prompt_ids: list[int] = field(default_factory=list)  # For split requests
    static_outputs: list[int] = field(default_factory=list)
    allocated_blocks: list[int] = field(default_factory=list)
    position_offset: int = 0  # Current position in the sequence for position_ids
    status: RequestStatus = RequestStatus.PENDING
    max_new_tokens: int = 20
    eos_token_id: int = -1
    created_time: float = field(default_factory=time.time)
    error: Optional[str] = None

    def current_len(self) -> int:
        """Get the current length of the sequence (prompt + generated tokens)."""
        return self.position_offset

    def generated_len(self) -> int:
        """Get the number of tokens generated so far."""
        return len(self.static_outputs)

    @traced
    def update_with_token(self, token_id: int) -> bool:
        """Update the request with a newly generated token and check for completion.

        Args:
            token_id: The token ID to add to the output sequence

        Returns:
            bool: True if the request is now complete, False otherwise
        """
        # Only update if we're in decoding state
        if self.status != RequestStatus.DECODING:
            return False

        is_eos = token_id == self.eos_token_id and self.eos_token_id != -1
        is_max_len = self.generated_len() >= self.max_new_tokens

        if is_eos or is_max_len:
            self.status = RequestStatus.FINISHED
            return True
        return False

    def __repr__(self):
        return f"RequestState(\n\trequest_id={self.request_id},\n\tstatus={self.status},\n\tout_tokens={self.generated_len()},\n\tquery_length={len(self.prompt_ids)}, \n\tremaining_tokens={len(self.remaining_prompt_ids)}, \n\tkv_length={self.position_offset}\n\tfull_prompt_lenght={len(self.full_prompt_ids)},\n\tallocated_blocks={self.allocated_blocks},\n\tgenerated_tokens={self.static_outputs}\n)"

    def to_generation_output(self):
        """Convert the request state to a GenerationOutput object."""
        return GenerationOutput(
            request_id=self.request_id,
            prompt_ids=self.full_prompt_ids,
            status=self.status,
            generated_tokens=self.static_outputs,
            logprobs=[],
            error=self.error,
        )


@attach_tracer()
class PagedAttentionCache(Cache):
    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        layer_device_map: Optional[dict[int, Union[str, torch.device, int]]] = None,
        initial_prompt_shapes: Optional[list[list[int]]] = None,
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
        # Extract model dimensions
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )
        self.num_hidden_layers = config.num_hidden_layers

        # Calculate optimal block size and number if not provided
        num_blocks = getattr(generation_config, "num_blocks", None)
        block_size = getattr(generation_config, "block_size", None)
        if num_blocks is None or block_size is None:
            logger.info("Calculating optimal block size and number...")
            num_blocks, block_size = compute_optimal_blocks(
                device, config, generation_config, initial_prompt_shapes or [], dtype, median_prefill_length=200
            )
            logger.info(f"Using calculated num_blocks={num_blocks}, block_size={block_size}")

        self.block_size = block_size
        self.num_blocks = num_blocks
        self.cache_shape = (self.num_key_value_heads, num_blocks, self.block_size, self.head_dim)

        self.dtype = dtype
        self.device = device

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
            logger.warning(f"Attempted to free blocks for non-existent request_id: {request_id}")

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


class Scheduler(ABC):
    """
    Abstract base class for scheduling requests in the continuous batch processor.
    It is expected that cache allocation and scheduling logic will be implemented in subclasses.
    """

    def __init__(self, cache: PagedAttentionCache, retain_cache_on_finish: bool = False):
        self.active_requests: dict[str, RequestState] = {}
        self.waiting_requests: dict[str, RequestState] = {}
        self.waiting_requests_order: deque[str] = deque()
        self.cache = cache
        self.retain_cache_on_finish = retain_cache_on_finish

    @abstractmethod
    def add_waiting_request(self, state: RequestState):
        """Add a request to the waiting list."""
        pass

    @abstractmethod
    def schedule_batch(self, token_budget: int) -> list[RequestState]:
        pass

    @traced
    def has_pending_requests(self) -> bool:
        """Check if there are requests ready to be processed."""
        return self.active_requests or self.waiting_requests

    @abstractmethod
    def finish_request(self, request_id: str, evict_from_cache: bool = True):
        """Finish processing a request and free its allocated blocks."""
        pass

    @traced
    def get_active_request_static_outputs(self, request_id: str) -> list[int]:
        if request_id in self.active_requests:
            return self.active_requests[request_id].static_outputs
        return []


@attach_tracer()
class FIFOScheduler(Scheduler):
    @traced
    def _allocate_blocks_if_needed(self, state: RequestState, len_next_tokens: int):
        # 1. we check that the occupancy is less than the requested length
        # 2. we allocate enough blocks to cover the requested length
        current_len = state.current_len()
        occupancy = len(state.allocated_blocks) * self.cache.block_size - current_len
        if occupancy < len_next_tokens or (len(state.allocated_blocks) == 0):
            blocks_needed = ((len_next_tokens - occupancy + 1) // self.cache.block_size) + 1
            allocated = self.cache.allocate_blocks(blocks_needed, state.request_id)
            if not allocated:
                return False
            state.allocated_blocks.extend(allocated)
        return True

    @traced(span_name="prepare_request")
    def _prepare_request_for_processing(
        self, state: RequestState, token_budget: int, request_ids_to_remove_from_waiting: set[str]
    ):
        """Prepare a request for processing in the current batch."""
        request_tokens = (
            state.remaining_prompt_ids if state.status == RequestStatus.SPLIT_PENDING_REMAINDER else state.prompt_ids
        )
        if len(request_tokens) < token_budget:
            # Can process the entire prompt/remainder
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING
                state.prompt_ids = state.remaining_prompt_ids
                state.remaining_prompt_ids = []
        else:
            # Need to split the request
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING_SPLIT
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING_SPLIT
            state.remaining_prompt_ids = request_tokens[token_budget:]
            state.prompt_ids = request_tokens[:token_budget]

    @traced
    def add_waiting_request(self, state: RequestState):
        """Add a request to the waiting list."""
        if self.retain_cache_on_finish and state.request_id in self.active_requests:
            old_state = self.active_requests.pop(state.request_id)
            state.prompt_ids = state.prompt_ids[len(old_state.full_prompt_ids) :]
            state.allocated_blocks = old_state.allocated_blocks
            state.position_offset = old_state.position_offset
        self.waiting_requests[state.request_id] = state
        self.waiting_requests_order.append(state.request_id)

    @traced
    def schedule_batch(self, token_budget: int) -> list[RequestState]:
        priority_states: list[RequestState] = []
        second_priority_states: list[RequestState] = []
        scheduled_requests = []

        for state in self.active_requests.values():
            if state.status == RequestStatus.DECODING:
                priority_states.append(state)
            if state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                second_priority_states.append(state)

        # Add waiting requests to second priority
        for req_id in self.waiting_requests_order:
            second_priority_states.append(self.waiting_requests[req_id])

        candidates = priority_states + second_priority_states
        request_ids_to_remove_from_waiting = set()

        for state in candidates:
            self._prepare_request_for_processing(state, token_budget, request_ids_to_remove_from_waiting)
            request_len = len(state.prompt_ids)
            if not self._allocate_blocks_if_needed(
                state, len(state.prompt_ids)
            ):  # don't schedule if we can't allocate blocks
                if len(self.cache._free_blocks) == 0:
                    break
                continue

            @traced
            def _add_to_scheduled_requests(state: RequestState):
                scheduled_requests.append(state)

            _add_to_scheduled_requests(state)

            token_budget -= request_len

            @traced
            def _remove_from_waiting_requests(state: RequestState):
                req_id = state.request_id
                if req_id in self.waiting_requests:
                    del self.waiting_requests[req_id]
                    request_ids_to_remove_from_waiting.add(req_id)

            _remove_from_waiting_requests(state)

            if token_budget == 0:
                break

        self.waiting_requests_order = deque(
            [req_id for req_id in self.waiting_requests_order if req_id not in request_ids_to_remove_from_waiting]
        )

        return scheduled_requests

    @traced
    def finish_request(self, request_id: str, evict_from_cache: bool = True):
        if evict_from_cache:
            self.cache.free_blocks(request_id)
            if request_id in self.active_requests:
                del self.active_requests[request_id]


@attach_tracer()
class PrefillFirstScheduler(Scheduler):
    @traced
    def _allocate_blocks_if_needed(self, state: RequestState, len_next_tokens: int):
        # 1. we check that the occupancy is less than the requested length
        # 2. we allocate enough blocks to cover the requested length
        current_len = state.current_len()
        occupancy = len(state.allocated_blocks) * self.cache.block_size - current_len
        if occupancy < len_next_tokens or (len(state.allocated_blocks) == 0):
            blocks_needed = ((len_next_tokens - occupancy + 1) // self.cache.block_size) + 1
            allocated = self.cache.allocate_blocks(blocks_needed, state.request_id)
            if not allocated:
                return False
            state.allocated_blocks.extend(allocated)
        return True

    @traced(span_name="prepare_request")
    def _prepare_request_for_processing(
        self, state: RequestState, token_budget: int, request_ids_to_remove_from_waiting: set[str]
    ):
        """Prepare a request for processing in the current batch."""
        request_tokens = (
            state.remaining_prompt_ids if state.status == RequestStatus.SPLIT_PENDING_REMAINDER else state.prompt_ids
        )
        if len(request_tokens) < token_budget:
            # Can process the entire prompt/remainder
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING
                state.prompt_ids = state.remaining_prompt_ids
                state.remaining_prompt_ids = []
        else:
            # Need to split the request
            if state.status == RequestStatus.PENDING:
                self.active_requests[state.request_id] = state
                state.status = RequestStatus.PREFILLING_SPLIT
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                state.status = RequestStatus.PREFILLING_SPLIT
            state.remaining_prompt_ids = request_tokens[token_budget:]
            state.prompt_ids = request_tokens[:token_budget]

    @traced
    def add_waiting_request(self, state: RequestState):
        """Add a request to the waiting list."""
        if self.retain_cache_on_finish and state.request_id in self.active_requests:
            old_state = self.active_requests.pop(state.request_id)
            state.prompt_ids = state.prompt_ids[len(old_state.full_prompt_ids) :]  # XXX: check for indexing error?
            state.allocated_blocks = old_state.allocated_blocks
            state.position_offset = old_state.position_offset
        self.waiting_requests[state.request_id] = state
        self.waiting_requests_order.append(state.request_id)

    @traced
    def schedule_batch(self, token_budget: int) -> list[RequestState]:
        priority_states: list[RequestState] = []
        second_priority_states: list[RequestState] = []
        scheduled_requests = []

        for state in self.active_requests.values():
            if state.status == RequestStatus.SPLIT_PENDING_REMAINDER:
                priority_states.append(state)
            elif state.status == RequestStatus.DECODING:
                second_priority_states.append(state)

        for req_id in self.waiting_requests_order:
            second_priority_states.append(self.waiting_requests[req_id])

        candidates = priority_states + second_priority_states

        request_ids_to_remove_from_waiting = set()

        for state in candidates:
            self._prepare_request_for_processing(state, token_budget, request_ids_to_remove_from_waiting)
            request_len = len(state.prompt_ids)
            if not self._allocate_blocks_if_needed(
                state, len(state.prompt_ids)
            ):  # don't schedule if we can't allocate blocks
                if len(self.cache._free_blocks) == 0:
                    break
                continue

            @traced
            def _add_to_scheduled_requests(state: RequestState):
                scheduled_requests.append(state)

            _add_to_scheduled_requests(state)

            token_budget -= request_len

            @traced
            def _remove_from_waiting_requests(state: RequestState):
                req_id = state.request_id
                if req_id in self.waiting_requests:
                    del self.waiting_requests[req_id]
                    request_ids_to_remove_from_waiting.add(req_id)

            _remove_from_waiting_requests(state)

            if token_budget == 0:
                break

        self.waiting_requests_order = deque(
            [req_id for req_id in self.waiting_requests_order if req_id not in request_ids_to_remove_from_waiting]
        )

        return scheduled_requests

    @traced
    def finish_request(self, request_id: str, evict_from_cache: bool = True):
        if evict_from_cache:
            self.cache.free_blocks(request_id)
            if request_id in self.active_requests:
                del self.active_requests[request_id]


@traced(standalone=True)
def compute_optimal_blocks(
    device: torch.device,
    config: PretrainedConfig,
    generation_config: GenerationConfig,
    inputs: list[list[int]],
    dtype: torch.dtype = torch.bfloat16,
    safety_margin: float = 0.9,
    median_prefill_length: Optional[int] = None,
):
    """Calculate optimal number and size of blocks for the KV cache.

    Args:
        device: The device where the model runs
        config: The model configuration
        generation_config: The generation configuration
        inputs: Sample input sequences to estimate memory requirements
        dtype: Data type for cache tensors
        safety_margin: Fraction of available memory to use
        median_prefill_length: Override for median prefill length calculation

    Returns:
        Tuple of (num_blocks, block_size)
    """
    # Extract model dimensions
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    num_hidden_layers = getattr(config, "num_hidden_layers", 40)

    # Get available device memory
    if device.type == "cuda":
        device_properties = torch.cuda.get_device_properties(device)
        total_memory = device_properties.total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - max(allocated_memory, reserved_memory)
    elif device.type == "mps":
        logger.warning("MPS memory estimation is approximate. Using conservative defaults.")
        return 2048, 256
    else:
        logger.warning(f"Unsupported device type {device.type} for optimal block calculation. Using defaults.")
        return 32, 128

    # Apply safety margin
    available_memory = int(available_memory * safety_margin)
    if available_memory <= 0:
        logger.warning("Not enough available memory. Using minimum configuration.")
        return 8, 128  # Minimum viable configuration

    # Calculate memory per token
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    memory_per_token = 2 * num_kv_heads * head_dim * dtype_size * num_hidden_layers  # For K and V caches

    # Estimate sequence length requirements
    tokens_to_generate = getattr(generation_config, "max_new_tokens", 20)

    if median_prefill_length is None and inputs:
        non_empty_inputs = [len(seq) for seq in inputs if seq]
        median_prefill_length = int(statistics.median(non_empty_inputs)) if non_empty_inputs else 64
    elif median_prefill_length is None:
        median_prefill_length = 64  # Reasonable default if no inputs provided

    # Total sequence length including generated tokens
    seq_length = median_prefill_length + tokens_to_generate

    # Calculate block parameters
    MIN_BLOCK_SIZE = 16

    # Estimate number of concurrent sequences
    per_sequence_memory = seq_length * memory_per_token
    max_concurrent_sequences = max(1, int(available_memory // per_sequence_memory))

    # Total tokens that can fit in memory
    total_tokens = available_memory // memory_per_token

    # Calculate block size (rounded to power of 2)
    initial_block_size = max(MIN_BLOCK_SIZE, total_tokens // (max_concurrent_sequences * 2))
    block_size = 1 << (initial_block_size - 1).bit_length()  # Round to power of 2

    # Calculate number of blocks
    num_blocks = max(1, total_tokens // block_size)

    logger.info(
        f"Optimal cache: {num_blocks} blocks of size {block_size} "
        f"(can handle ~{num_blocks * block_size // seq_length} sequences of length {seq_length})"
    )

    return int(num_blocks), int(block_size)


@dataclass
class PagedAttentionArgs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    cumulative_seqlens_q: torch.Tensor
    cumulative_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    write_index: torch.Tensor
    read_index: torch.Tensor
    logits_indices: torch.Tensor
    block_tables: dict[str, list[int]]
    cache: PagedAttentionCache
    use_cache: bool = False


@traced
def create_document_mask(cumulative_seqlens_q, cumulative_seqlens_k):
    # Number of documents
    valid_docs_q = cumulative_seqlens_q[1:] > cumulative_seqlens_q[:-1]
    valid_docs_k = cumulative_seqlens_k[1:] > cumulative_seqlens_k[:-1]
    num_valid_docs = min(valid_docs_q.sum(), valid_docs_k.sum())

    # Trim to valid docs
    cumulative_seqlens_q = cumulative_seqlens_q[: num_valid_docs + 1]
    cumulative_seqlens_k = cumulative_seqlens_k[: num_valid_docs + 1]

    total_q = cumulative_seqlens_q[-1]
    total_k = cumulative_seqlens_k[-1]

    q_indices = torch.arange(total_q, device=cumulative_seqlens_q.device)
    k_indices = torch.arange(total_k, device=cumulative_seqlens_k.device)

    q_doc_ids = torch.bucketize(q_indices, cumulative_seqlens_q[1:], right=True)
    k_doc_ids = torch.bucketize(k_indices, cumulative_seqlens_k[1:], right=False)
    doc_mask = q_doc_ids[:, None] == k_doc_ids[None, :]
    # apply causal mask where no decoding (same nb of q than k)

    is_causal = ~(cumulative_seqlens_q[1:] - cumulative_seqlens_q[:-1] == 1) * cumulative_seqlens_q[1:]
    apply_causal = torch.bucketize(q_indices, is_causal, right=True)[:, None] == k_doc_ids
    # TODO don't apply on prefill splitting
    causal_mask = torch.triu(torch.ones(total_q, total_k, device=q_doc_ids.device), diagonal=1).bool()
    doc_mask.masked_fill_((apply_causal & causal_mask), False)
    return doc_mask


# Continuous Batch Processor (Internal Logic)
@attach_tracer()
class ContinuousBatchProcessor:
    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        model_device: torch.device,
        model_dtype: torch.dtype,
        scheduler: Scheduler,
        streaming: bool = False,
        manual_eviction: bool = False,
    ):
        """Initialize the continuous batch processor.

        Args:
            cache: The paged attention cache to use
            generation_config: The generation configuration
            input_queue: Queue for incoming requests
            output_queue: Queue for outgoing results
            stop_event: Event to signal processing should stop
            model_device: Device for model inputs/outputs
            model_dtype: Data type for model inputs/outputs
            streaming: Whether to stream tokens as they're generated
        """
        self.cache = cache
        self.config = config
        self.generation_config = generation_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.model_device = model_device
        self.model_dtype = model_dtype
        self.scheduler = scheduler
        self.streaming = streaming
        self.manual_eviction = manual_eviction

        self.requests_in_batch: list[RequestState] = []

        # Get batch size parameters from generation config
        self._configure_batch_parameters()

        # Set up metrics collector
        self.metrics = ContinuousBatchProcessorMetrics(self.max_batch_tokens)

        self.setup_static_tensors()

    @traced(standalone=True)
    def setup_static_tensors(self):
        T = self.max_batch_tokens
        max_token_budget = self.cache.num_blocks * self.cache.block_size
        tensor_metadata = {"dtype": torch.int32, "device": self.model_device}
        self.tensor_metadata = tensor_metadata
        self.input_ids = torch.zeros((1, T), **tensor_metadata)
        self.position_ids = torch.zeros((1, T), **tensor_metadata)
        self.attention_mask = torch.zeros(
            (1, 1, T, max_token_budget), dtype=self.model_dtype, device=self.model_device
        )
        self.cumulative_seqlens_q = torch.zeros((T + 1,), **tensor_metadata)
        self.cumulative_seqlens_k = torch.zeros((T + 1,), **tensor_metadata)
        self.write_index = torch.zeros((T,), **tensor_metadata)
        self.read_index = torch.zeros((max_token_budget,), **tensor_metadata)
        self.logits_indices = torch.full((T,), -1, **tensor_metadata)
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0
        self.output_ids = torch.full((1, T), -1, **tensor_metadata)

    @traced
    @torch.no_grad()
    def reset_static_tensors(self):
        """Reset static tensors for the next batch."""
        self.input_ids.zero_()
        self.position_ids.zero_()
        self.attention_mask.fill_(torch.finfo(self.model_dtype).min)
        self.cumulative_seqlens_q.zero_()
        self.cumulative_seqlens_k.zero_()
        self.write_index.fill_(-1)
        self.read_index.fill_(-1)
        self.logits_indices.fill_(-1)
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0
        self.output_ids.zero_()

    def get_model_kwargs(self) -> PagedAttentionArgs:
        """Get model keyword arguments for the current batch."""
        # torch.set_printoptions(threshold=100000,linewidth=10000)
        return {
            "input_ids": self.input_ids,
            "position_ids": self.position_ids,
            "attention_mask": self.attention_mask,
            "cumulative_seqlens_q": self.cumulative_seqlens_q,
            "cumulative_seqlens_k": self.cumulative_seqlens_k,
            "write_index": self.write_index,
            "read_index": self.read_index,
            "logits_indices": self.logits_indices,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_k": self.max_seqlen_k,
            "block_tables": self.cache._block_tables,
            "cache": self.cache,
            "use_cache": False,
        }

    def __repr__(self):
        return (
            f"ContinuousBatchProcessor(input_queue={self.input_queue}, output_queue={self.output_queue}, active_requests={self.scheduler.active_requests}, waiting_requests={self.scheduler.waiting_requests})"
            + self.get_model_kwargs().__repr__()
        )

    @traced(standalone=True)
    def _configure_batch_parameters(self):
        """Set up batch processing parameters based on generation config."""
        # Calculate total cache capacity
        total_cache_tokens = self.cache.num_blocks * self.cache.block_size

        # Get or calculate max tokens per batch
        user_batch_tokens = getattr(self.generation_config, "max_batch_tokens", None)
        if user_batch_tokens is not None:
            self.max_batch_tokens = user_batch_tokens
        else:
            # Default to 1/8 of total cache capacity, adjusted for context
            self.max_context_len = getattr(self.generation_config, "max_position_embeddings", 2048)
            recommended_batch_size = min(total_cache_tokens // 8, self.max_context_len)
            self.max_batch_tokens = max(64, recommended_batch_size)

        # Context length and EOS token
        self.max_context_len = getattr(self.generation_config, "max_position_embeddings", 2048)

    @traced
    def _get_new_requests(self):
        """Pull new requests from the input queue and add to waiting list."""
        while not self.input_queue.empty():
            try:
                state = self.input_queue.get_nowait()
                if state is None:  # Sentinel value
                    continue
                self.scheduler.add_waiting_request(state)

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing new request: {e}", exc_info=True)
                state: RequestState = locals().get("state")
                if state is not None:
                    self._handle_request_error(e, state)

    @traced
    def _handle_request_error(self, error, state: RequestState):
        """Handle general request processing error."""
        state.status = RequestStatus.FAILED
        state.error = str(error)

        # Include any generated tokens if this is an active request
        if isinstance(state.request_id, str):
            state.static_outputs = self.scheduler.get_active_request_static_outputs(state.request_id)
        else:
            state.static_outputs = []

        self.metrics.record_request_completion(state.created_time, state.request_id)
        self.output_queue.put(state.to_generation_output())

    @traced
    def prepare_next_batch(self):
        """Prepare tensors and metadata for the next model forward pass."""
        # Get new requests from the queue
        self._get_new_requests()
        if not self.scheduler.has_pending_requests():
            return None

        self.metrics.record_queue_metrics(len(self.scheduler.active_requests), len(self.scheduler.waiting_requests))

        self.requests_in_batch = self.scheduler.schedule_batch(self.max_batch_tokens)
        if not self.requests_in_batch:
            return None

        # Get the request objects for this batch
        self.reset_static_tensors()
        position_ids = []
        input_ids = []
        read_index = []
        write_index = []
        cumulative_seqlens_q = [0]
        cumulative_seqlens_k = [0]
        logits_indices = []
        self.metrics.record_batch_metrics(self.requests_in_batch)

        for state in self.requests_in_batch:
            next_input_ids = state.prompt_ids
            input_ids.extend(next_input_ids)
            past_length = state.position_offset
            query_length = len(next_input_ids)
            key_length = query_length + past_length
            cache_index = list(range(key_length))

            positions_to_add = cache_index[past_length:]
            read_indices = self.cache._get_physical_indices(state, cache_index)
            write_indices = read_indices[-query_length:]

            position_ids.extend(positions_to_add)
            read_index.extend(read_indices)
            write_index.extend(write_indices)
            cumulative_seqlens_q.append(cumulative_seqlens_q[-1] + query_length)
            cumulative_seqlens_k.append(cumulative_seqlens_k[-1] + key_length)
            if len(state.remaining_prompt_ids) == 0:
                logits_indices.append(cumulative_seqlens_q[-1] - 1)
            self.max_seqlen_q = max(self.max_seqlen_q, query_length)
            self.max_seqlen_k = max(self.max_seqlen_k, key_length)
            state.position_offset += query_length

        logger.warning(
            f"Scheduled: {len(self.requests_in_batch)}, Waiting: {len(self.scheduler.waiting_requests)}, Active: {len(self.scheduler.active_requests)}. cum Q: {cumulative_seqlens_q[-1]}. cum KV: {cumulative_seqlens_k[-1]}, free blocks: {self.cache.get_num_free_blocks()}"
        )
        self._build_tensors(
            input_ids,
            position_ids,
            read_index,
            write_index,
            cumulative_seqlens_q,
            cumulative_seqlens_k,
            logits_indices,
        )

        self.metrics.record_kv_cache_memory_metrics(self.cache)

    @traced
    def _build_tensors(
        self,
        input_ids,
        position_ids,
        read_index,
        write_index,
        cumulative_seqlens_q,
        cumulative_seqlens_k,
        logits_indices,
    ):
        to_tensor = partial(torch.tensor, **self.tensor_metadata)
        self.input_ids[:, : len(input_ids)] = to_tensor(input_ids)
        self.position_ids[:, : len(position_ids)] = to_tensor(position_ids)
        self.write_index[: len(write_index)] = to_tensor(write_index)
        self.read_index[: len(read_index)] = to_tensor(read_index)
        self.cumulative_seqlens_q[: len(cumulative_seqlens_q)] = to_tensor(cumulative_seqlens_q)
        self.cumulative_seqlens_k[: len(cumulative_seqlens_k)] = to_tensor(cumulative_seqlens_k)
        self.logits_indices[: len(logits_indices)] = to_tensor(logits_indices)
        min_value = torch.finfo(self.model_dtype).min
        if self.config._attn_implementation != "paged_attention":  # we set `is_causal` to True in paged call`
            for i in range(len(cumulative_seqlens_q) - 1):
                if (
                    cumulative_seqlens_q[i + 1] - cumulative_seqlens_q[i]
                    < cumulative_seqlens_k[i + 1] - cumulative_seqlens_k[i]
                    and cumulative_seqlens_q[i + 1] - cumulative_seqlens_q[i] >= 1
                ):
                    diagonal = (
                        cumulative_seqlens_k[i + 1] - (cumulative_seqlens_q[i + 1] - cumulative_seqlens_q[i]) + 1
                    )
                    diagonal = diagonal - cumulative_seqlens_k[i]
                else:
                    diagonal = 1
                query_range = slice(cumulative_seqlens_q[i], cumulative_seqlens_q[i + 1])
                key_range = slice(cumulative_seqlens_k[i], cumulative_seqlens_k[i + 1])

                mask = torch.triu(
                    torch.full(
                        self.attention_mask[..., query_range, key_range].shape,
                        min_value,
                        dtype=self.model_dtype,
                        device=self.model_device,
                    ),
                    diagonal=diagonal,
                )
                self.attention_mask[..., query_range, key_range] = mask

    @traced
    def _sync(self):
        return self.output_ids.tolist()[0]  # should be the only synch we do

    @traced
    def _maybe_send_output(self, state: RequestState, token: int):
        """Send output to the queue based on streaming mode and request state."""
        if self.streaming:
            state.next_token = token
            self.output_queue.put(state.to_generation_output())
        elif state.status == RequestStatus.FINISHED:
            self.output_queue.put(state.to_generation_output())

    @traced
    def update_batch(self):
        """Update request states based on generated tokens."""
        out_tokens = self._sync()
        finished_request_ids = []
        for i, state in enumerate(self.requests_in_batch):
            req_id = state.request_id
            if len(state.remaining_prompt_ids) == 0:
                self.metrics.record_ttft_metric(state.created_time, state.request_id)
                state.status = RequestStatus.DECODING
                token = out_tokens[self.logits_indices[i]]
                state.static_outputs.extend([token])
                state.prompt_ids = [token]
                if state.update_with_token(token):
                    self.metrics.record_request_completion(state.created_time, state.request_id)
                    self.scheduler.finish_request(state.request_id, evict_from_cache=(not self.manual_eviction))
                    finished_request_ids.append(req_id)
                self._maybe_send_output(state, token)
            elif state.status == RequestStatus.PREFILLING_SPLIT:
                state.status = RequestStatus.SPLIT_PENDING_REMAINDER

    @traced
    def has_pending_requests(self) -> bool:
        """Check if there are any active or waiting requests."""
        return self.scheduler.has_pending_requests()

    @traced
    def handle_batch_error(self, error):
        """Handle errors during batch processing."""
        failed_reqs = self.requests_in_batch
        for req in failed_reqs:
            self._handle_request_error(error, req)
            self.scheduler.finish_request(req.request_id)

    @traced
    def fail_all_requests(self, error):
        """Fail all active requests with the given error.

        Args:
            error: The error to report in the failure message
        """
        for state in self.scheduler.active_requests.values():
            self._handle_request_error(error, state)
            self.scheduler.finish_request(state.request_id)

        # Also fail any requests in the waiting queue
        for req_id in list(self.scheduler.waiting_requests.keys()):
            state = self.scheduler.waiting_requests.pop(req_id)
            self._handle_request_error(error, state)

        # Clear the ordering queue
        self.scheduler.waiting_requests_order.clear()


SCHEDULER_MAPPING = {
    "fifo": FIFOScheduler,
    "prefill_first": PrefillFirstScheduler,
}


# Manager Class (User Interface)
@attach_tracer()
class ContinuousBatchingManager:
    """Manager for handling continuous batching of generation requests.

    This class provides the user interface for submitting generation requests,
    retrieving results, and managing the background generation thread.
    """

    def __init__(
        self,
        model,
        generation_config: GenerationConfig,
        manual_eviction: bool = False,
        max_queue_size=0,
        streaming: bool = True,
    ):
        """Initialize the continuous batching manager.

        Args:
            model: The language model for generation
            generation_config: Configuration for generation parameters
            max_queue_size: Maximum size of the request queue (0 = unlimited)
            streaming: Whether to stream tokens as they are generated
        """
        self.model = model
        self.generation_config = generation_config
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.streaming = streaming
        self.log_prob_generation = getattr(generation_config, "log_prob_generation", False)
        self._generation_thread = None
        self._request_counter = 0
        self._request_lock = threading.Lock()
        self.model.generation_config.top_p = None
        self.do_sample = getattr(generation_config, "do_sample", True)
        self.logit_processor = self.model._get_logits_processor(self.model.generation_config)
        self.use_cuda_graph = getattr(generation_config, "use_cuda_graph", True)
        self.profile = getattr(generation_config, "profile", False)
        self.manual_eviction = manual_eviction
        self.batch_processor: Optional[ContinuousBatchProcessor] = None

    @traced
    def start(self):
        """Start the background generation thread."""
        if self._generation_thread is not None and self._generation_thread.is_alive():
            logger.warning("Manager thread is already running.")
            return

        self._result_queue = queue.Queue()
        self._generation_thread = threading.Thread(target=self._run_generation_loop)
        self._generation_thread.start()
        logger.info("Continuous batching manager started.")

    def is_running(self):
        """Check if the background generation thread is running."""
        return self._generation_thread is not None and self._generation_thread.is_alive()

    def stop(self, block: bool = False, timeout: Optional[float] = None):
        """Signal the background thread to stop.

        Args:
            block: Whether to wait for the thread to stop
            timeout: Maximum time to wait for the thread to stop
        """
        if self._generation_thread is None:
            logger.warning("Manager not started.")
            return

        if not self.stop_event.is_set():
            self.stop_event.set()
            logger.info("Stopping continuous batching manager...")

        if block:
            self.join(timeout)

    def join(self, timeout: Optional[float] = None):
        """Wait for the background thread to finish.

        Args:
            timeout: Maximum time to wait for the thread to stop
        """
        if self._generation_thread is not None:
            self._generation_thread.join(timeout=timeout)
            if self._generation_thread.is_alive():
                logger.warning("Generation thread did not exit after join timeout.")
            else:
                logger.info("Continuous Batching Manager stopped.")
                self._generation_thread = None

    def add_request(
        self, input_ids: list[int], request_id: Optional[str] = None, max_new_tokens: Optional[int] = None
    ) -> str:
        """Add a new generation request to the queue.

        Args:
            input_ids: Input token IDs to use as prompt
            request_id: Optional custom request ID (auto-generated if None)
            **kwargs: Additional generation parameters

        Returns:
            str: The request ID
        """
        if request_id is None:
            with self._request_lock:
                request_id = f"req_{self._request_counter}"
                self._request_counter += 1

        max_new_tokens = self.generation_config.max_new_tokens if max_new_tokens is None else max_new_tokens

        state = RequestState(
            request_id=request_id,
            prompt_ids=list(input_ids),
            full_prompt_ids=list(input_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=self.generation_config.eos_token_id,
        )

        # Use block=True with timeout to handle backpressure if queue is full
        self.input_queue.put(state, block=True, timeout=10)  # XXX: pass timeout as fn arg?
        logger.debug(f"Added request {request_id} to queue.")
        return request_id

    def add_requests(self, inputs: list[list[int]], **kwargs):
        for i, input_ids in enumerate(inputs):
            # Assign a predictable request ID for ordering results later
            req_id = f"batch_req_{i}"
            self.add_request(input_ids, request_id=req_id, **kwargs)

    def get_result(self, timeout=None) -> Optional[GenerationOutput]:
        """Retrieve one result from the output queue.

        Args:
            timeout: Maximum time to wait for a result

        Returns:
            Optional[Dict]: The result data or None if timeout
        """
        if self._generation_thread is None and self.output_queue.empty():
            return None
        try:
            result = self.output_queue.get(block=True, timeout=timeout)
            logger.debug(f"Retrieved result for request {result.request_id}")
            return result
        except queue.Empty:
            return None

    def __iter__(self):
        """Iterate over results as they become available."""
        while (
            self._generation_thread is not None and self._generation_thread.is_alive() or not self.output_queue.empty()
        ):
            result = self.get_result(timeout=0.1)  # allow the model to run for 10 seconds
            if result is not None:
                yield result

    @traced
    def warmup(self, batch_processor):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Warmup the model with a dummy forward pass
            self._generation_step(batch_processor)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._generation_step(batch_processor)

    @traced
    # @torch.compile
    def _generation_step(self, batch_processor: ContinuousBatchProcessor):
        """Perform a single generation step. This is cuda graphed"""
        batch_data = batch_processor.get_model_kwargs()
        with torch.no_grad():
            logits = self._model_forward(batch_data)
            if self.log_prob_generation:
                batch_processor.output_probs.copy_(logits)  # TODO
            probs = self._process_logit(batch_data, logits)
            self._sample(batch_processor, probs)

    @traced(span_name="model_forward")
    def _model_forward(self, batch_data):
        return self.model(**batch_data).logits

    @traced(span_name="logit_processing")
    def _process_logit(self, batch_data, logits):
        return self.logit_processor(batch_data["input_ids"], logits)

    @traced(span_name="sampling")
    def _sample(self, batch_processor: ContinuousBatchProcessor, probs):
        if self.do_sample:  # sample
            probs = nn.functional.softmax(probs, dim=-1)
            next_tokens = torch.multinomial(probs[0], num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)
        batch_processor.output_ids.copy_(next_tokens)

    def _run_generation_loop(self):
        """Main processing loop running in the background thread."""
        batch_processor = None
        try:
            paged_attention_cache = PagedAttentionCache(
                self.model.config,
                self.generation_config,
                self.model.device,
                self.model.dtype,
            )

            scheduler = None
            if hasattr(self.generation_config, "scheduler"):
                scheduler = SCHEDULER_MAPPING.get(self.generation_config.scheduler)
                if scheduler is None:
                    logger.warning(f"Scheduler '{scheduler}' not found. Defaulting to FIFO.")
                    scheduler = FIFOScheduler
            else:
                # Default to fifo
                scheduler = FIFOScheduler

            batch_processor = ContinuousBatchProcessor(
                paged_attention_cache,
                self.model.config,
                self.generation_config,
                self.input_queue,
                self.output_queue,
                self.stop_event,
                self.model.device,
                self.model.dtype,
                scheduler(paged_attention_cache, self.manual_eviction),
                self.streaming,
                self.manual_eviction,
            )
            self.batch_processor = batch_processor
            is_first = True

            if self.profile:
                tracing_schedule = schedule(skip_first=2, warmup=3, active=200, repeat=100, wait=1)
                trace_handler = tensorboard_trace_handler(
                    dir_name="/fsx/arthur/transformers", use_gzip=True, worker_name="paged_compile"
                )
                activities = [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                with profile(
                    activities=activities,
                    schedule=tracing_schedule,
                    on_trace_ready=trace_handler,
                    record_shapes=False,
                    with_stack=True,
                ) as prof:
                    while not self.stop_event.is_set() or batch_processor.has_pending_requests():
                        self._inner_generation_loop(batch_processor, is_first)
                        if is_first:
                            is_first = False
                        prof.step()
            else:
                while not self.stop_event.is_set() or batch_processor.has_pending_requests():
                    self._inner_generation_loop(batch_processor, is_first)
                    if is_first:
                        is_first = False

        except Exception as e:
            logger.error(f"Error in generation loop: {e}", exc_info=True)
            self._handle_critical_error(e, batch_processor)
        finally:
            logger.info("Generation loop finished.")

    @traced(span_name="generation_loop")
    def _inner_generation_loop(self, batch_processor: ContinuousBatchProcessor, is_first: bool = False):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_processor.prepare_next_batch()
        if torch.cuda.is_available() and self.use_cuda_graph:
            if is_first:
                self.warmup(batch_processor)
            elif hasattr(self, "graph"):
                try:
                    self._graph_replay()
                except Exception as e:
                    logger.error(f"Model forward pass failed: {e}", exc_info=True)
                    batch_processor.handle_batch_error(e)
                    return
            else:
                self._generation_step(batch_processor)
        else:
            self._generation_step(batch_processor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_processor.update_batch()

    @traced(span_name="graph_replay")
    def _graph_replay(self):
        self.graph.replay()

    @traced
    def _handle_critical_error(self, error, batch_processor: Optional[ContinuousBatchProcessor]):
        """Handle critical errors that terminate the generation loop."""
        # Signal stop
        self.stop_event.set()

        # Fail pending requests in input queue
        try:
            while True:
                req_data = self.input_queue.get_nowait()
                if batch_processor is not None:
                    batch_processor._handle_request_error(error, req_data)
        except queue.Empty:
            pass

        # Fail active requests
        if batch_processor is not None:
            batch_processor.fail_all_requests(error)

    @traced
    def evict_request_from_cache(self, request_id: str):
        """Evict a request from the cache. It is assumed that the request is already finished."""
        if not self.manual_eviction:
            raise RuntimeError("Manual eviction is not enabled for this manager.")
        if self.batch_processor is not None:
            self.batch_processor.scheduler.finish_request(request_id)


class ContinuousMixin:
    """Mixin class for models to add continuous batching capabilities."""

    def init_continuous_batching(
        self,
        generation_config: Optional[GenerationConfig] = None,
        manual_eviction: bool = False,
        max_queue_size: int = 0,
        streaming: bool = False,
    ) -> ContinuousBatchingManager:
        """Initialize a manager for continuous batching inference.

        Args:
            generation_config: Custom generation configuration
            max_queue_size: Maximum size of the input request queue
            streaming: Whether to stream tokens as they are generated

        Returns:
            `ContinuousBatchingManager`: The manager instance to add requests and retrieve results.
        """
        if not hasattr(self, "config") or not hasattr(self, "device") or not hasattr(self, "dtype"):
            raise AttributeError("Model must have 'config', 'device', and 'dtype' attributes.")

        gen_config = generation_config if generation_config is not None else self.generation_config
        if gen_config is None:
            raise ValueError("A GenerationConfig must be provided or set in the model.")

        if gen_config.eos_token_id is None:
            logger.warning("`eos_token_id` not set in GenerationConfig. Setting to -1 (disabled).")
            gen_config.eos_token_id = -1

        # Create and return the manager
        return ContinuousBatchingManager(
            model=self,
            generation_config=gen_config,
            manual_eviction=manual_eviction,
            max_queue_size=max_queue_size,
            streaming=streaming,
        )

    @traced
    @torch.inference_mode()
    def generate_batch(
        self,
        inputs: list[list[int]],
        generation_config: Optional[GenerationConfig] = None,
        progress_bar: bool = True,
        **kwargs,
    ) -> list[list[int]]:
        """Generate sequences for a batch of prompts using continuous batching.

        Args:
            inputs: List of input token sequences (prompts)
            generation_config: Optional generation configuration
            **kwargs: Additional generation parameters

        Returns:
            `list[list[int]]`: A list containing the generated sequences (including prompt tokens
                                if not handled otherwise) for each input prompt, in the same order.
                                Returns an empty list `[]` for requests that failed.
        """
        if not inputs:
            return []

        # Initialize manager with the batch inputs
        manager = self.init_continuous_batching(generation_config=generation_config)
        manager.start()
        results = {}
        num_requests = len(inputs)
        try:
            from tqdm.contrib.logging import logging_redirect_tqdm

            with logging_redirect_tqdm([logger]):
                with tqdm(
                    total=num_requests,
                    disable=(not progress_bar),
                    desc=f"Solving {num_requests} requests",
                    unit="request",
                ) as pbar:
                    manager.add_requests(inputs, **kwargs)
                    finished_count = 0
                    while finished_count < num_requests:
                        result = manager.get_result(timeout=1)
                        if result:
                            req_id = result.request_id
                            if result.status == RequestStatus.FINISHED:
                                results[req_id] = result
                                finished_count += 1
                                pbar.update(1)
                        else:
                            if not manager.is_running():
                                logger.error("Generation thread terminated unexpectedly.")
                                break

        except Exception as e:
            logger.error(f"Error during batch generation: {e}", exc_info=True)
        finally:
            manager.stop(block=True, timeout=5.0)
        return results
