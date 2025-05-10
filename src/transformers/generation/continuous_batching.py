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
import math
import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Union

import torch
from tqdm import tqdm

from ..cache_utils import Cache
from ..configuration_utils import PretrainedConfig
from ..generation.configuration_utils import GenerationConfig
from ..generation.utils import GenerationMixin
from ..utils import (
    logging,
)


logger = logging.get_logger(__name__)


@dataclass
class RequestState:
    """Tracks the state of a generation request through its lifecycle.

    Attributes:
        status (str): can be one of 'pending', 'prefilling', 'prefilling_split', 'split_pending_remainder', 'decoding', 'finished', 'failed'
    """

    # Required fields
    request_id: str
    prompt_ids: List[int]

    # Optional/generated fields
    output_ids: List[int] = field(default_factory=list)
    remaining_prompt_ids: List[int] = field(default_factory=list)  # For split requests
    allocated_blocks: List[int] = field(default_factory=list)
    position_offset: int = 0  # Current position in the sequence for position_ids
    status: str = "pending"
    max_new_tokens: int = 20
    eos_token_id: int = -1
    created_time: float = field(default_factory=time.time)

    def current_len(self) -> int:
        """Get the current length of the sequence (prompt + generated tokens)."""
        return self.position_offset

    def generated_len(self) -> int:
        """Get the number of tokens generated so far."""
        return len(self.output_ids)

    def update_with_token(self, token_id: int) -> bool:
        """Update the request with a newly generated token and check for completion.

        Args:
            token_id: The token ID to add to the output sequence

        Returns:
            bool: True if the request is now complete, False otherwise
        """
        # Only update if we're in decoding state
        if self.status != "decoding":
            return False

        # Add the token to output
        self.output_ids.append(token_id)

        is_eos = token_id == self.eos_token_id and self.eos_token_id != -1
        is_max_len = self.generated_len() >= self.max_new_tokens

        if is_eos or is_max_len:
            self.status = "finished"
            return True

        return False

    def __repr__(self):
        return f"RequestState(request_id={self.request_id}, status={self.status}, generated_len={self.generated_len()}, current_len={self.current_len()})"


class PagedAttentionCache(Cache):
    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
        initial_prompt_shapes: Optional[List[List[int]]] = None,
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
        self.cache_shape = (num_blocks, self.num_key_value_heads, self.block_size, self.head_dim)

        self.dtype = dtype
        self.device = device

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        for idx in range(config.num_hidden_layers):
            layer_device = layer_device_map[idx] if layer_device_map is not None else device
            self.key_cache.append(torch.zeros(self.cache_shape, dtype=self.dtype, device=layer_device))
            self.value_cache.append(torch.zeros(self.cache_shape, dtype=self.dtype, device=layer_device))

        # Block management data structures
        self._free_blocks = deque(range(num_blocks))
        self._block_tables: Dict[str, List[int]] = {}

    def allocate_blocks(self, n_blocks: int, request_id: str) -> List[int]:
        """Allocates n_blocks for a given request_id."""
        if len(self._free_blocks) < n_blocks:
            logger.warning(
                f"Not enough free blocks for {request_id}. Requested: {n_blocks}, Available: {len(self._free_blocks)}"
            )
            return []

        allocated = []
        for _ in range(n_blocks):
            allocated.append(self._free_blocks.popleft())

        if request_id not in self._block_tables:
            self._block_tables[request_id] = []
        self._block_tables[request_id].extend(allocated)

        return allocated

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

    def get_block_table(self, request_id: str) -> List[int]:
        """Returns the block table for a request."""
        return self._block_tables.get(request_id, [])


    def _get_physical_indices(self, request_id: str, logical_indices: List[int]) -> torch.Tensor:
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
        block_table = self._block_tables.get(request_id)
        if not block_table:
            raise ValueError(f"No block table found for request {request_id}")

        # Convert to PyTorch tensor for vectorized operations
        logical_indices_tensor = torch.tensor(logical_indices)
        block_size = self.block_size

        # Vectorized calculations
        block_indices = logical_indices_tensor // block_size
        block_offsets = logical_indices_tensor % block_size

        if torch.any(block_indices >= len(block_table)):
            raise IndexError(
                f"Logical indices map to block indices out of bounds for request {request_id}"
            )

        block_table_tensor = torch.tensor(block_table)
        physical_block_nums = block_table_tensor[block_indices]  # Use PyTorch indexing
        physical_indices = physical_block_nums * block_size + block_offsets
        return physical_indices

    def _reshape_cache_for_update(self, layer_idx: int):
        """Reshapes K/V cache for easier indexing during updates."""
        total_slots = self.num_blocks * self.block_size
        k_cache = self.key_cache[layer_idx].view(self.num_key_value_heads, total_slots, self.head_dim)
        v_cache = self.value_cache[layer_idx].view(self.num_key_value_heads, total_slots, self.head_dim)
        return k_cache, v_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        read_index,
        **kwargs,
    ) -> (torch.Tensor, torch.Tensor):
        """Updates the key and value states in the cache at the specified indices."""
        write_index = kwargs.get("write_index")

        if write_index is None or write_index.numel() == 0:
            # Nothing to write, return the current cache state
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        # Validate input shape
        if key_states.shape[2] != write_index.numel() or value_states.shape[2] != write_index.numel():
            raise ValueError(
                f"Mismatch between number of tokens to write ({key_states.shape[0]}) and number of fill indices ({write_index.numel()})"
            )

        # Reshape cache for easier indexing
        k_cache_flat, v_cache_flat = self._reshape_cache_for_update(layer_idx)

        # Ensure indices are on the same device as the cache
        indices_device = write_index.to(k_cache_flat.device)

        try:
            # Update cache with new key and value states
            k_cache_flat[:, indices_device, :] = key_states.to(k_cache_flat.device, k_cache_flat.dtype)[0]
            v_cache_flat[:, indices_device, :] = value_states.to(v_cache_flat.device, v_cache_flat.dtype)[0]
        except IndexError as e:
            logger.error(
                f"IndexError during cache update. Fill indices shape: {indices_device.shape}, "
                f"Max index: {indices_device.max() if indices_device.numel() > 0 else 'N/A'}, "
                f"Cache shape: {k_cache_flat.shape}"
            )
            raise e

        # Return the updated cache slices needed for attention
        k_cache_flat, v_cache_flat = self._reshape_cache_for_update(layer_idx)
        return k_cache_flat[:, read_index, :][None, ...], v_cache_flat[:, read_index, :][None, ...]

    def write_to_cache(
        self, request_id: str, key_states: torch.Tensor, value_states: torch.Tensor, logical_indices: List[int]
    ):
        """Writes key/value states to the cache at specified logical indices for a request."""
        if not logical_indices:
            return  # Nothing to write

        physical_indices: torch.Tensor = self._get_physical_indices(request_id, logical_indices)

        # Validate input shapes
        if key_states.shape[0] != len(logical_indices) or value_states.shape[0] != len(logical_indices):
            raise ValueError(
                f"Mismatch between number of tokens to write ({key_states.shape[0]}) and number of indices ({len(logical_indices)})"
            )

        # Update each layer's cache
        for layer_idx in range(self.num_hidden_layers):
            k_cache_flat, v_cache_flat = self._reshape_cache_for_update(layer_idx)

            try:
                # Write states to cache
                k_cache_flat[physical_indices] = key_states.to(k_cache_flat.device, k_cache_flat.dtype)
                v_cache_flat[physical_indices] = value_states.to(v_cache_flat.device, v_cache_flat.dtype)
            except IndexError as e:
                logger.error(
                    f"IndexError during cache write for request {request_id}. Physical indices: {physical_indices.tolist()}, Max index: {k_cache_flat.shape[0] - 1}"
                )
                raise e


def compute_optimal_blocks(
    device: torch.device,
    config: PretrainedConfig,
    generation_config: GenerationConfig,
    inputs: List[List[int]],
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
    innput_ids: torch.Tensor
    position_ids: torch.Tensor
    cumulative_seqlens_q: torch.Tensor
    cumulative_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    write_index: torch.Tensor
    read_index: torch.Tensor
    logits_indices: torch.Tensor
    block_tables: Dict[str, List[int]]
    cache: PagedAttentionCache
    use_cache: bool = False


# Continuous Batch Processor (Internal Logic)
class ContinuousBatchProcessor:
    def __init__(
        self,
        cache: PagedAttentionCache,
        generation_config: GenerationConfig,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        model_device: torch.device,
        model_dtype: torch.dtype,
        streaming: bool = False,
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
        self.generation_config = generation_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.model_device = model_device
        self.model_dtype = model_dtype
        self.streaming = streaming

        self.active_requests: Dict[str, RequestState] = {}
        self.waiting_requests: Deque[RequestState] = deque()
        self.requests_to_process_next: List[str] = []

        # Get batch size parameters from generation config
        self._configure_batch_parameters()
        self.setup_static_tensors()

    def setup_static_tensors(self):
        T = self.max_batch_tokens
        max_token_budget = self.cache.num_blocks * self.cache.block_size
        tensor_metadata = {"dtype": torch.long, "device": self.model_device}
        self.tensor_metadata = tensor_metadata
        self.input_ids              = torch.zeros((1, T), **tensor_metadata)
        self.position_ids           = torch.zeros((1, T), **tensor_metadata)
        self.cumulative_seqlens_q   = torch.zeros((T+1,),  **tensor_metadata)
        self.cumulative_seqlens_k   = torch.zeros((T+1,),  **tensor_metadata)
        self.write_index            = torch.zeros((T,),  **tensor_metadata)
        self.read_index             = torch.zeros((max_token_budget,), **tensor_metadata)
        self.logits_indices         = torch.zeros((T,),  **tensor_metadata)
        self.max_seqlen_q           = 0
        self.max_seqlen_k           = 0

    def reset_static_tensors(self):
        """Reset static tensors for the next batch."""
        self.input_ids.zero_()
        self.position_ids.zero_()
        self.cumulative_seqlens_q.zero_()
        self.cumulative_seqlens_k.zero_()
        self.write_index.zero_()
        self.read_index.zero_()
        self.logits_indices.zero_()
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0

    def get_model_kwargs(self) -> PagedAttentionArgs:
        """Get model keyword arguments for the current batch."""
        return PagedAttentionArgs(
            input_ids=self.input_ids,
            position_ids=self.position_ids,
            cumulative_seqlens_q=self.cumulative_seqlens_q,
            cumulative_seqlens_k=self.cumulative_seqlens_k,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            write_index=self.write_index,
            read_index=self.read_index,
            logits_indices=self.logits_indices,
            block_tables=self.cache._block_tables,
            cache=self.cache,
            use_cache=True,
        )

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
        self.eos_token_id = self.generation_config.eos_token_id

    def _get_new_requests(self):
        """Pull new requests from the input queue and add to waiting list."""
        while not self.input_queue.empty():
            try:
                req_data = self.input_queue.get_nowait()
                if req_data is None:  # Sentinel value
                    continue

                if not isinstance(req_data, dict) or "request_id" not in req_data or "input_ids" not in req_data:
                    logger.error(f"Invalid request format: {req_data}")
                    if "request_id" in req_data:
                        self._handle_request_error("Invalid request format", req_data["request_id"])
                    continue

                request_id = req_data["request_id"]
                input_ids = req_data["input_ids"]
                max_new_tokens = req_data.get("max_new_tokens", self.generation_config.max_new_tokens or 20)

                if not input_ids:
                    logger.warning(f"Request {request_id} received with empty input_ids. Ignoring.")
                    self._handle_request_error("Empty input_ids provided", request_id)
                    continue

                state = RequestState(
                    request_id=request_id,
                    prompt_ids=list(input_ids),
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.eos_token_id,
                )
                self.waiting_requests.append(state)

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing new request: {e}", exc_info=True)
                self._handle_request_error(e, locals().get("req_data", {}))

    def _handle_request_error(self, error, request_id):
        """Handle general request processing error."""
        error_response = {
            "request_id": request_id,
            "status": "failed",
            "error": f"Error processing request: {str(error)}",
        }

        # Include any generated tokens if this is an active request
        if isinstance(request_id, str) and request_id in self.active_requests:
            error_response["output_ids"] = self.active_requests[request_id].output_ids
        else:
            error_response["output_ids"] = []

        self.output_queue.put(error_response)

    def _schedule_batch(self) -> List[str]:
        """Select requests for the next processing batch."""
        selected_requests = []
        num_free_blocks = self.cache.get_num_free_blocks()
        batch_token_count = 0

        # First prioritize running decoding requests (need just 1 token each)
        decoding_requests = [req_id for req_id, state in self.active_requests.items() if state.status == "decoding"]

        if decoding_requests:
            # Start with the max batch tokens
            available_slots = self.max_batch_tokens
            selected_requests.extend(decoding_requests[:available_slots])
            if len(selected_requests) == available_slots:
                return selected_requests
            batch_token_count += len(decoding_requests)  # 1 token per request

        # priortise requests for which we already started prefilling
        candidates: List[RequestState] = [
            state for state in self.active_requests.values() if state.status == "split_pending_remainder"
        ]
        candidates.extend(list(self.waiting_requests))

        request_ids_to_remove_from_waiting = set()

        for i, state in enumerate(candidates):
            if state.request_id in selected_requests:
                continue

            # Get tokens to process
            tokens_to_process = (
                len(state.remaining_prompt_ids) if state.status == "split_pending_remainder" else len(state.prompt_ids)
            )
            if tokens_to_process == 0:
                if i < len(self.waiting_requests):
                    request_ids_to_remove_from_waiting.add(state.request_id)
                continue

            # Check if we need to split the request
            if batch_token_count + tokens_to_process > self.max_batch_tokens:
                remaining_space = self.max_batch_tokens - batch_token_count
                # Only split if we have enough space for a meaningful chunk
                if remaining_space >= 16:
                    tokens_to_process = remaining_space
                else:
                    continue  # Not enough space to split

            # Check if we have enough blocks
            needed_blocks = math.ceil(tokens_to_process / self.cache.block_size)
            if needed_blocks > num_free_blocks:
                logger.debug(
                    f"Request {state.request_id} needs {needed_blocks} blocks but only {num_free_blocks} available. Skipping."
                )
                continue

            # We can process this request
            num_free_blocks -= needed_blocks
            batch_token_count += tokens_to_process
            selected_requests.append(state.request_id)

            # Set up the request based on whether we're splitting
            self._prepare_request_for_processing(state, tokens_to_process, request_ids_to_remove_from_waiting)

        # Remove processed requests from waiting queue
        self.waiting_requests = deque(
            [req for req in self.waiting_requests if req.request_id not in request_ids_to_remove_from_waiting]
        )

        logger.debug(f"Scheduled batch with {len(selected_requests)} requests and {batch_token_count} tokens")
        return selected_requests

    def _prepare_request_for_processing(
        self, state: RequestState, tokens_to_process, request_ids_to_remove_from_waiting
    ):
        """Prepare a request for processing in the current batch."""
        source_tokens = state.remaining_prompt_ids if state.status == "split_pending_remainder" else state.prompt_ids

        if tokens_to_process >= len(source_tokens):
            # Can process the entire prompt/remainder
            if state.status == "pending":
                self.active_requests[state.request_id] = state
                state.status = "prefilling"
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == "split_pending_remainder":
                state.status = "prefilling"
        else:
            # Need to split the request
            if state.status == "pending":
                self.active_requests[state.request_id] = state
                state.remaining_prompt_ids = source_tokens[tokens_to_process:]
                state.prompt_ids = source_tokens[:tokens_to_process]
                state.status = "prefilling_split"
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == "split_pending_remainder":
                state.prompt_ids = source_tokens[:tokens_to_process]
                state.remaining_prompt_ids = source_tokens[tokens_to_process:]
                state.status = "prefilling_split"

    def prepare_next_batch(self):
        """Prepare tensors and metadata for the next model forward pass."""
        # Get new requests from the queue
        self._get_new_requests()

        if not self.active_requests and not self.waiting_requests:
            return None

        # Schedule requests for this batch
        self.requests_to_process_next = self._schedule_batch()
        if not self.requests_to_process_next:
            return None

        # Get the request objects for this batch
        requests_in_batch = [self.active_requests[req_id] for req_id in self.requests_to_process_next]
        self.reset_static_tensors()

        # pointers into our static flat tensors
        token_position = 0
        read_position  = 0
        write_position = 0
        cumq_ptr        = 1
        cumk_ptr        = 1
        for state in requests_in_batch:
            if state.status == "decoding":
                next_input_ids = [state.output_ids[-1]]
                positions_to_add = [state.current_len()]

                if not self._allocate_blocks_if_needed(state, state.current_len() + 1):
                    continue

                # Map logical indices to physical block indices for this request
                read_indices = self.cache._get_physical_indices(state.request_id, state.current_len())
                write_indices = self.cache._get_physical_indices(state.request_id, list(range(state.current_len())))[0]

                seq_len_q = 1  # Query length is 1 for generation
                seq_len_k = state.current_len() + 1  # Key length includes context
            elif state.status.startswith("prefilling"):
                next_input_ids = state.prompt_ids
                start_pos = state.current_len()
                positions_to_add = list(range(start_pos, start_pos + len(next_input_ids)))

                if not self._allocate_blocks_if_needed(state, len(next_input_ids)):
                    continue

                # Get physical indices
                read_indices = positions_to_add
                write_indices = self.cache._get_physical_indices(state.request_id, positions_to_add)

                seq_len_q = seq_len_k = len(next_input_ids)
            else:
                logger.warning(f"Request {state.request_id} in unexpected state '{state.status}'. Skipping.")
                continue

            self.input_ids[:, token_position : token_position + len(next_input_ids)].copy_(
                torch.tensor(next_input_ids, **self.tensor_metadata)
            )
            self.position_ids[:, token_position : token_position + len(positions_to_add)].copy_(
                torch.tensor(positions_to_add, **self.tensor_metadata)
            )
            self.read_index[read_position : read_position + len(read_indices)].copy_(
                torch.tensor(read_indices, **self.tensor_metadata)
            )
            self.write_index[write_position : write_position + len(write_indices)].copy_(
                torch.tensor(write_indices, **self.tensor_metadata)
            )
            self.cumulative_seqlens_q[cumq_ptr].copy_(
                torch.tensor(self.cumulative_seqlens_q[-1] + seq_len_q, **self.tensor_metadata)
            )
            self.cumulative_seqlens_k[cumk_ptr].copy_(
                torch.tensor(self.cumulative_seqlens_k[-1] + seq_len_k, **self.tensor_metadata)
            )
            token_position += len(next_input_ids)
            read_position += len(read_indices)
            write_position += len(write_indices)
            cumq_ptr += 1
            cumk_ptr += 1

            self.max_seqlen_q = max(max_seqlen_q, seq_len_q)
            self.max_seqlen_k = max(max_seqlen_k, seq_len_k)
            self.logits_indices[cumk_ptr].copy_(torch.tensor(self.cumulative_seqlens_q[-1] - 1, **self.tensor_metadata))
            state.position_offset += len(token_position[-len(next_input_ids) :])

        if not token_position:
            return None

        # Calculate max total sequence length in the batch
        self.max_seqlen_k = max(state.current_len() for state in requests_in_batch)
        return self.get_model_kwargs().to_dict()

    def _allocate_blocks_if_needed(self, state: RequestState, needed_slots: int):
        """Helper function to allocate blocks for a request."""
        current_blocks = len(state.allocated_blocks)
        needed_blocks = math.ceil(needed_slots / self.cache.block_size)

        if needed_blocks > current_blocks:
            blocks_needed = needed_blocks - current_blocks
            allocated = self.cache.allocate_blocks(blocks_needed, state.request_id)
            if not allocated:
                return False
            state.allocated_blocks.extend(allocated)
        return True

    def update_batch(self, generated_ids: torch.Tensor):
        """Update request states based on generated tokens."""
        # TODO I think many slowdowns can come from here
        # this should be probably vectorized
        # we need to batch adding the tokens
        # and potentially have a tensor with values containing the status

        token_idx = 0
        finished_request_ids = []

        for req_id in self.requests_to_process_next:
            if req_id not in self.active_requests:
                logger.warning(f"Request {req_id} not found in active requests during update.")
                continue

            state = self.active_requests[req_id]

            if state.status == "prefilling":
                state.status = "decoding"
                # state.prompt_ids = []  # Clear prompt as it's now in cache

                token = generated_ids[token_idx].item()
                token_idx += 1

                if state.update_with_token(token):
                    finished_request_ids.append(req_id)

                self._send_output(state, token)

            elif state.status == "prefilling_split":
                token_idx += 1  # Skip the token

                if state.remaining_prompt_ids:
                    state.status = "split_pending_remainder"
                    # state.prompt_ids = []
                else:
                    state.status = "decoding"
                    # state.prompt_ids = []

                    token = generated_ids[token_idx].item()
                    token_idx += 1

                    if state.update_with_token(token):
                        finished_request_ids.append(req_id)

                    self._send_output(state, token)

            elif state.status == "decoding":
                if token_idx >= len(generated_ids):
                    error_msg = f"Token index {token_idx} out of bounds for generated_ids (len {len(generated_ids)}) for request {req_id}."
                    logger.error(error_msg)
                    raise IndexError(error_msg)
                else:
                    token = generated_ids[token_idx].item()
                    token_idx += 1

                    if state.update_with_token(token):
                        finished_request_ids.append(req_id)

                    self._send_output(state, token)

            elif state.status == "split_pending_remainder":
                logger.warning(f"Request {req_id} in 'split_pending_remainder' state during update.")

        for req_id in finished_request_ids:
            if req_id in self.active_requests:
                self.cache.free_blocks(req_id)
                del self.active_requests[req_id]

    def _send_output(self, state: RequestState, token: int):
        """Send output to the queue based on streaming mode and request state."""
        output = {
            "request_id": state.request_id,
            "status": state.status,
            "prompt_token_ids": state.prompt_ids,
        }

        if self.streaming:
            output["next_token"] = token
            self.output_queue.put(output)

        elif state.status == "finished":
            output["output_ids"] = state.output_ids
            self.output_queue.put(output)

    def has_pending_requests(self) -> bool:
        """Check if there are any active or waiting requests."""
        return bool(self.active_requests or self.waiting_requests)

    def handle_batch_error(self, error):
        """Handle errors during batch processing."""
        failed_ids = self.requests_to_process_next
        for req_id in failed_ids:
            if req_id in self.active_requests:
                self._handle_request_error(error, req_id)
                self.cache.free_blocks(req_id)
                del self.active_requests[req_id]

    def fail_all_requests(self, error):
        """Fail all active requests with the given error.

        Args:
            error: The error to report in the failure message
        """
        for req_id, state in list(self.active_requests.items()):
            self._handle_request_error(error, req_id)
            self.cache.free_blocks(req_id)
            del self.active_requests[req_id]

        # Also fail any requests in the waiting queue
        while self.waiting_requests:
            state = self.waiting_requests.popleft()
            self._handle_request_error(error, state.request_id)


# Manager Class (User Interface)
class ContinuousBatchingManager:
    """Manager for handling continuous batching of generation requests.

    This class provides the user interface for submitting generation requests,
    retrieving results, and managing the background generation thread.
    """

    def __init__(
        self, model: GenerationMixin, generation_config: GenerationConfig, max_queue_size=0, streaming: bool = False
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

        self._generation_thread = None
        self._request_counter = 0
        self._request_lock = threading.Lock()

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

    def add_request(self, input_ids: List[int], request_id: Optional[str] = None, **kwargs) -> str:
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

        req_data = {"request_id": request_id, "input_ids": input_ids, **kwargs}
        self.input_queue.put(req_data, block=True, timeout=10)  # XXX: pass timeout as fn arg?
        logger.debug(f"Added request {request_id} to queue.")
        return request_id

    def add_requests(self, inputs: List[List[int]], **kwargs):
        for i, input_ids in enumerate(inputs):
            # Assign a predictable request ID for ordering results later
            req_id = f"batch_req_{i}"
            self.add_request(input_ids, request_id=req_id, **kwargs)

    def get_result(self, timeout=None) -> Optional[Dict]:
        """Retrieve one result from the output queue.

        Args:
            timeout: Maximum time to wait for a result

        Returns:
            Optional[Dict]: The result data or None if timeout
        """
        if self._generation_thread is None and self.output_queue.empty():
            return None

        try:
            result = self.output_queue.get(block=True, timeout=timeout) # TODO Pop here rather?
            logger.debug(f"Retrieved result for request {result.get('request_id')}")
            return result
        except queue.Empty:
            return None

    def __iter__(self):
        """Iterate over results as they become available."""
        while (
            self._generation_thread is not None and self._generation_thread.is_alive() or not self.output_queue.empty()
        ):
            result = self.get_result(timeout=0.1)
            if result is not None:
                yield result

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

            batch_processor = ContinuousBatchProcessor(
                paged_attention_cache,
                self.generation_config,
                self.input_queue,
                self.output_queue,
                self.stop_event,
                self.model.device,
                self.model.dtype,
                self.streaming,
            )
            first = True
            while not self.stop_event.is_set() or batch_processor.has_pending_requests():
                batch_data: PagedAttentionArgs = batch_processor.prepare_next_batch()
                if batch_data is None:
                    if self.stop_event.is_set() and not batch_processor.has_pending_requests():
                        break  # No more work to do
                    time.sleep(0.005)  # Prevent CPU spinning
                    continue

                if first:
                    self.graph = self.warmup(batch_data)
                    first = False
                try:
                    self.graph.replay()
                    outputs = self.static_outputs
                except Exception as e:
                    logger.error(f"Model forward pass failed: {e}", exc_info=True)
                    batch_processor.handle_batch_error(e)
                    continue

                # TODO this should also be part of the graph
                # TODO we can leverage logits processors
                # Get next token logits and sample next tokens
                next_token_logits = outputs.logits[:, batch_data.logits_indices, :]
                generated_ids = torch.argmax(next_token_logits, dim=-1).squeeze(0)
                batch_processor.update_batch(generated_ids)

        except Exception as e:
            logger.error(f"Error in generation loop: {e}", exc_info=True)
            self._handle_critical_error(e, batch_processor)
        finally:
            logger.info("Generation loop finished.")

    def _handle_critical_error(self, error, batch_processor: Optional[ContinuousBatchProcessor]):
        """Handle critical errors that terminate the generation loop."""
        # Signal stop
        self.stop_event.set()

        # Fail pending requests in input queue
        try:
            while True:
                req_data = self.input_queue.get_nowait()
                self._handle_request_error(error, getattr(req_data, "request_id", None))
        except queue.Empty:
            pass

        # Fail active requests
        if batch_processor is not None:
            batch_processor.fail_all_requests(error)


class ContinuousMixin:
    """Mixin class for models to add continuous batching capabilities."""

    def init_continuous_batching(
        self, generation_config: Optional[GenerationConfig] = None, max_queue_size: int = 0, streaming: bool = False
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
            model=self, generation_config=gen_config, max_queue_size=max_queue_size, streaming=streaming
        )

    @torch.no_grad()
    def generate_batch(
        self,
        inputs: List[List[int]],
        generation_config: Optional[GenerationConfig] = None,
        progress_bar: bool = False,
        **kwargs,
    ) -> List[List[int]]:
        """Generate sequences for a batch of prompts using continuous batching.

        Args:
            inputs: List of input token sequences (prompts)
            generation_config: Optional generation configuration
            **kwargs: Additional generation parameters

        Returns:
            `List[List[int]]`: A list containing the generated sequences (including prompt tokens
                                if not handled otherwise) for each input prompt, in the same order.
                                Returns an empty list `[]` for requests that failed.
        """
        if not inputs:
            return []

        # Initialize manager with the batch inputs
        manager = self.init_continuous_batching(generation_config=generation_config)
        results = {}
        num_requests = len(inputs)
        try:
            with tqdm(total=num_requests, disable=(not progress_bar), desc=f"Generating {num_requests} requests") as pbar:
                manager.add_requests(inputs, **kwargs)
                manager.start() # we don't want to start before adding all requests
                finished_count = 0
                while finished_count < num_requests:
                    result = manager.get_result(timeout=1.0)
                    if result:
                        req_id = result["request_id"]
                        if result["status"] == "finished":
                            results[req_id] = result
                        else:  # Failed
                            logger.warning(f"Request {req_id} failed: {result.get('error', 'Unknown error')}")
                            results[req_id] = []
                        finished_count += 1
                        pbar.update(1)
                    else:
                        if not manager.is_running():
                            logger.error("Generation thread terminated unexpectedly.")
                            break

        except Exception as e:
            logger.error(f"Error during batch generation: {e}", exc_info=True)
            for i in range(num_requests):
                if i not in results:
                    results[i] = []
        finally:
            manager.stop(block=True, timeout=5.0)

        # Return results in the original order
        return results