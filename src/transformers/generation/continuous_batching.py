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
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch


if False:
    from flash_attn import flash_attn_varlen_func

from ..cache_utils import Cache
from ..utils import (
    is_accelerate_available,
    logging,
)
from ..generation.utils import GenerationMixin
from ..generation.configuration_utils import GenerationConfig
from ..configuration_utils import PretrainedConfig


if TYPE_CHECKING:
    from .streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    pass


from transformers import GenerationMixin


# Request State Tracking
@dataclass
class RequestState:
    request_id: str
    prompt_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    remaining_prompt_ids: List[int] = field(default_factory=list)  # For split requests
    allocated_blocks: List[int] = field(default_factory=list)
    cache_indices: List[int] = field(default_factory=list)  # Physical indices in the flat cache tensor
    position_offset: int = 0  # Current position in the sequence for position_ids
    status: str = "pending"  # pending, prefilling, decoding, finished, failed
    max_new_tokens: int = 20
    eos_token_id: int = -1
    created_time: float = field(default_factory=time.time)
    # Add other generation parameters if needed (temperature, top_k, etc.)

    def is_done(self) -> bool:
        return self.status in ["finished", "failed"]

    def current_len(self) -> int:
        return self.position_offset

    def generated_len(self) -> int:
        return len(self.output_ids)


class PagedAttentionCache(Cache):
    def __init__(
        self,
        config: PretrainedConfig,
        generation_config: GenerationConfig,  # Pass the whole config
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
        initial_prompt_shapes: Optional[List[List[int]]] = None,  # Optional shapes for initial optimal calculation
    ) -> None:
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )
        self.num_hidden_layers = config.num_hidden_layers

        num_blocks = getattr(generation_config, "num_blocks", None)
        block_size = getattr(generation_config, "block_size", None)
        if num_blocks is None or block_size is None:
            # We determine the best size, provide initial shapes if available
            logger.info("Calculating optimal block size and number...")
            num_blocks, block_size = compute_optimal_blocks(
                device, config, generation_config, initial_prompt_shapes or [], dtype, median_prefill_length=50
            )
            logger.info(f"Using calculated num_blocks={num_blocks}, block_size={block_size}")

        self.block_size = block_size
        self.num_blocks = num_blocks
        cache_shape = (num_blocks, self.num_key_value_heads, self.block_size, self.head_dim)

        self.dtype = dtype
        self.device = device  # Store main device

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = device
            self.key_cache.append(torch.zeros(cache_shape, dtype=self.dtype, device=layer_device))
            self.value_cache.append(torch.zeros(cache_shape, dtype=self.dtype, device=layer_device))

        self._lock = threading.Lock()
        self._free_blocks = deque(range(num_blocks))
        # Maps request_id to list of block numbers used by that request
        self._block_tables: Dict[str, List[int]] = {}

    def allocate_blocks(self, n_blocks: int, request_id: str) -> List[int]:
        """Allocates n_blocks for a given request_id. Returns the list of allocated block numbers."""
        if len(self._free_blocks) < n_blocks:
            logger.warning(f"Not enough free blocks. Requested: {n_blocks}, Available: {len(self._free_blocks)}")
            # Decide on behavior: raise error, return partial, return empty? Returning empty for now.
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
        with self._lock:
            if request_id in self._block_tables:
                blocks_to_free = self._block_tables.pop(request_id)
                self._free_blocks.extend(blocks_to_free)  # Add back to the deque
            else:
                logger.warning(f"Attempted to free blocks for non-existent request_id: {request_id}")

    def get_num_free_blocks(self) -> int:
        with self._lock:
            return len(self._free_blocks)

    def get_block_table(self, request_id: str) -> List[int]:
        with self._lock:
            return self._block_tables.get(request_id, [])

    def _get_physical_indices(self, request_id: str, logical_indices: List[int]) -> List[int]:
        """Maps logical sequence indices to physical cache indices using the block table."""
        with self._lock:
            block_table = self._block_tables.get(request_id)
            if not block_table:
                raise ValueError(f"No block table found for request {request_id}")

        physical_indices = []
        for idx in logical_indices:
            block_idx = idx // self.block_size
            block_offset = idx % self.block_size
            if block_idx >= len(block_table):
                raise IndexError(
                    f"Logical index {idx} maps to block index {block_idx}, but only {len(block_table)} blocks allocated for request {request_id}"
                )
            physical_block_num = block_table[block_idx]
            physical_idx = physical_block_num * self.block_size + block_offset
            physical_indices.append(physical_idx)
        return physical_indices

    def _reshape_cache_for_update(self, layer_idx: int):
        """Reshapes K/V cache for easier indexing during updates."""
        # Shape: (num_blocks * block_size, num_heads, head_dim)
        total_slots = self.num_blocks * self.block_size
        k_cache = self.key_cache[layer_idx].view(self.num_key_value_heads, total_slots, self.head_dim)
        v_cache = self.value_cache[layer_idx].view(self.num_key_value_heads, total_slots, self.head_dim)
        return k_cache, v_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cumulative_seqlens_k: torch.Tensor,
        cache_index,
        **kwargs,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Updates the key and value states in the cache at the specified indices.

        Args:
            key_states (`torch.Tensor`): New key states to be added to the cache. Shape: (num_tokens_to_write, num_heads, head_dim).
            value_states (`torch.Tensor`): New value states to be added to the cache. Shape: (num_tokens_to_write, num_heads, head_dim).
            layer_idx (`int`): The index of the layer to update.
            fill_index (`torch.Tensor`): Tensor containing the physical indices in the flat cache where the new states should be written. Shape: (num_tokens_to_write,).
            kwargs: Additional arguments (not used here but kept for compatibility).

        Returns:
            Tuple[`torch.Tensor`, `torch.Tensor`]: A tuple containing the *entire* key and value cache tensors for the specified layer, possibly after reshaping for attention calculation.
                                                  Note: The reshaping might depend on the specific attention implementation. Returning the full cache for now.
        """
        fill_index = kwargs["fill_index"]
        if fill_index.numel() == 0:
            # Nothing to write, return the current cache state
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        if key_states.shape[2] != fill_index.numel() or value_states.shape[2] != fill_index.numel():
            raise ValueError(
                f"Mismatch between number of tokens to write ({key_states.shape[0]}) and number of fill indices ({fill_index.numel()})"
            )

        # Reshape cache for easier indexing
        k_cache_flat, v_cache_flat = self._reshape_cache_for_update(layer_idx)

        # Ensure indices are on the same device as the cache
        indices_device = fill_index.to(k_cache_flat.device)

        try:
            k_cache_flat[:, indices_device, :] = key_states.to(k_cache_flat.device, k_cache_flat.dtype)[0]
            v_cache_flat[:, indices_device, :] = value_states.to(v_cache_flat.device, v_cache_flat.dtype)[0]
        except IndexError as e:
            logger.error(
                f"IndexError during cache update. Fill indices shape: {indices_device.shape}, "
                f"Max index: {indices_device.max() if indices_device.numel() > 0 else 'N/A'}, "
                f"Cache shape: {k_cache_flat.shape}"
            )
            raise e

        k_cache_flat, v_cache_flat = self._reshape_cache_for_update(layer_idx)
        return k_cache_flat[:,cache_index,:][None,...], v_cache_flat[:,cache_index,:][None,...]

    def write_to_cache(
        self, request_id: str, key_states: torch.Tensor, value_states: torch.Tensor, logical_indices: List[int]
    ):
        """Writes key/value states to the cache at specified logical indices for a request."""
        if not logical_indices:
            return  # Nothing to write

        physical_indices = self._get_physical_indices(request_id, logical_indices)
        physical_indices_tensor = torch.tensor(physical_indices, device=self.device, dtype=torch.long)

        # key_states/value_states shape: (num_tokens_to_write, num_heads, head_dim)
        # Ensure num_tokens_to_write matches len(logical_indices)
        if key_states.shape[0] != len(logical_indices) or value_states.shape[0] != len(logical_indices):
            raise ValueError(
                f"Mismatch between number of tokens to write ({key_states.shape[0]}) and number of indices ({len(logical_indices)})"
            )

        for layer_idx in range(self.num_hidden_layers):
            # TODO: Handle layer device mapping if needed
            k_cache_flat, v_cache_flat = self._reshape_cache_for_update(layer_idx)

            # Use index_copy_ or index_put_ for potentially better performance?
            # index_select might be slow if physical_indices_tensor is large.
            # Ensure shapes match:
            # k_cache_flat[physical_indices_tensor] shape: (num_tokens, num_heads, head_dim)
            # key_states shape: (num_tokens, num_heads, head_dim) - should match
            try:
                k_cache_flat[physical_indices_tensor] = key_states.to(k_cache_flat.device, k_cache_flat.dtype)
                v_cache_flat[physical_indices_tensor] = value_states.to(v_cache_flat.device, v_cache_flat.dtype)
            except IndexError as e:
                logger.error(
                    f"IndexError during cache write for request {request_id}. Physical indices: {physical_indices_tensor.tolist()}, Max index: {k_cache_flat.shape[0] - 1}"
                )
                raise e

    def get_kv_for_attention(self, layer_idx: int, request_id: str, sequence_len: int) -> (torch.Tensor, torch.Tensor):
        """Retrieves the K/V cache for a given request up to sequence_len."""
        # This needs to return K/V tensors compatible with the attention function (e.g., flash_attn_varlen_func)
        # It likely requires knowing the block table for the *entire batch* if using varlen func.
        # This part is complex and depends heavily on the attention implementation details.

        # Simplification: Assume attention expects K/V for the *single* sequence.
        # This won't work directly with flash_attn_varlen_func which expects concatenated K/V for the batch.
        # We might need to gather K/V based on the batch's block tables *outside* the cache object.

        # Placeholder: Return slices corresponding to the logical indices 0..sequence_len-1
        logical_indices = list(range(sequence_len))
        physical_indices = self._get_physical_indices(request_id, logical_indices)
        physical_indices_tensor = torch.tensor(physical_indices, device=self.device, dtype=torch.long)

        k_cache_flat, v_cache_flat = self._reshape_cache_for_update(layer_idx)

        # Gather K/V states using the physical indices
        # Shape: (sequence_len, num_heads, head_dim)
        k_states = k_cache_flat[physical_indices_tensor]
        v_states = v_cache_flat[physical_indices_tensor]

        return k_states, v_states


def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_seqlens_q=None,
    cumulative_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    block_table: Optional[torch.Tensor] = None,
    cache: Optional[PagedAttentionCache] = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Args:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full k
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.  but if there is a block table it can be the full v
        cumulative_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cumulative_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        block_table [optional]: (num_blocks, max_num_blocks_per_seq), dtype torch.int32. This array should be used to index into
            the cache. Here, you already pass concatenated K and V. k[block_table] gives the cache the positions in cache that we need to fill?
            If we use with_kv_cache as it supports paged attention, it means it supports writing in a paged cache. But it does not support computing with
            ragged input.
            Whiile flash_attn_varlen_func, supports ragged inputs, but it does not write into the kv_cache.
            Paged <==> fragmented cache, helpful for very long sequences.
            continuous <==> ragged inputs -> no padding
    """
    k, v = cache.update(k, v, module.layer_idx, cumulative_seqlens_q, cumulative_seqlens_k)

    attn_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cumulative_seqlens_q,
        cumulative_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        block_table=block_table,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        rotary_interleaved=True,
        **kwargs,
    )

    return attn_output


# FIXME: overshoots widely
def compute_optimal_blocks(
    device: torch.device,
    config: PretrainedConfig,
    generation_config: GenerationConfig,
    inputs: List[List[int]],
    dtype: torch.dtype = torch.bfloat16,
    safety_margin: float = 0.9,  # Safety margin for memory usage
    median_prefill_length: Optional[int] = None,
):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    num_hidden_layers = getattr(config, "num_hidden_layers", 40)

    # Get device memory properties
    if device.type == "cuda":
        device_properties = torch.cuda.get_device_properties(device)
        total_memory = device_properties.total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - max(allocated_memory, reserved_memory)
    elif device.type == "mps":
        # Placeholder for MPS device, might need adjustments
        available_memory = torch.mps.current_allocated_memory()  # This might not reflect total available
        logger.warning("MPS memory estimation is approximate. Optimal blocks calculation might be inaccurate.")
        # Defaulting block size for MPS might be safer until better estimation is available
        return 32, 256  # Example default
    else:
        logger.warning(f"Unsupported device type {device.type} for optimal block calculation. Using defaults.")
        # Default values if device type is unknown
        return 32, 256  # Default num_blocks, block_size

    # Apply safety margin
    available_memory *= safety_margin
    if available_memory <= 0:
        raise MemoryError("Not enough available memory after applying safety margin to calculate optimal blocks.")

    # Memory per tensor element (for K and V caches)
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    memory_per_token_slot = 2 * num_kv_heads * head_dim * dtype_size * num_hidden_layers  # Factor of 2 for K and V

    if memory_per_token_slot == 0:
        logger.warning("Calculated memory_per_token_slot is zero. Using default block size.")
        return 32, 256  # Default values

    tokens_to_generate = getattr(generation_config, "max_new_tokens", 20)  # Default if not set
    if median_prefill_length is None:
        # Use median prompt length as estimate if not provided
        if inputs:
            # Filter out empty inputs before calculating median
            non_empty_inputs = [len(elem) for elem in inputs if elem]
            if non_empty_inputs:
                median_prefill_length = int(statistics.median(non_empty_inputs))
            else:
                median_prefill_length = 0  # No valid inputs to calculate median
        else:
            median_prefill_length = 0  # Default if no inputs

    # Estimate memory usage PER SEQUENCE (for its entire lifetime in cache)
    # Based on median prompt length and max generation length
    # Add 1 to tokens_to_generate for potential off-by-one in allocation? Consider sequence length.
    estimated_sequence_len = median_prefill_length + tokens_to_generate
    per_sequence_memory_estimate = (
        estimated_sequence_len * memory_per_token_slot
    )  # For the layers of the model as the cache is per layer

    # --- Block Size Calculation ---
    MIN_BLOCK_SIZE = 16  # Minimum allowed block size (power of 2)

    if per_sequence_memory_estimate <= 0:
        # If sequences take no memory (e.g., generate 0 tokens), allocate max blocks with min size
        logger.warning("per_sequence_memory_estimate is zero or negative. Using minimum block size.")
        total_token_slots = available_memory // memory_per_token_slot
        final_block_size = MIN_BLOCK_SIZE
        final_num_blocks = total_token_slots // final_block_size

    else:
        # Estimate number of concurrent sequences that can fit
        num_concurrent_sequences_estimate = int(available_memory // per_sequence_memory_estimate)
        if num_concurrent_sequences_estimate <= 0:
            num_concurrent_sequences_estimate = 1  # Assume at least one sequence can run

        # Calculate total token slots available in memory
        total_token_slots = int(available_memory // memory_per_token_slot)

        # Calculate initial block size by distributing slots among estimated sequences
        # Ensure num_concurrent_sequences_estimate is at least 1
        # Cast intermediate results to int to ensure block_size_initial is int
        block_size_initial = total_token_slots // max(1, num_concurrent_sequences_estimate)

        # Round block_size up to the nearest power of 2
        if block_size_initial <= 0:
            block_size_rounded = MIN_BLOCK_SIZE
        else:
            # Efficient power-of-2 ceiling: 1 << (x - 1).bit_length() for x > 0
            block_size_rounded = 1 << (block_size_initial - 1).bit_length()

        # Ensure minimum block size
        final_block_size = max(block_size_rounded, MIN_BLOCK_SIZE)

        # Recalculate the number of blocks based on the final block size
        final_num_blocks = total_token_slots // final_block_size

    # Ensure at least one block is allocated if possible
    if final_num_blocks <= 0:
        logger.warning(f"Calculated final_num_blocks is {final_num_blocks}. Setting to 1.")
        final_num_blocks = 1
        # Check if even one block of minimum size fits
        if MIN_BLOCK_SIZE * memory_per_token_slot > available_memory:
            raise MemoryError(
                f"Cannot fit even one block of size {MIN_BLOCK_SIZE} in available memory ({available_memory} bytes)."
            )

    logger.info(
        f"Optimal blocks calculated: num_blocks={int(final_num_blocks)}, block_size={int(final_block_size)} "
        f"(available_memory={available_memory / (1024**3):.2f} GB, "
        f"mem_per_token={memory_per_token_slot} bytes, "
        f"est_seq_len={estimated_sequence_len})"
    )
    return int(final_num_blocks), int(final_block_size)


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
    ):
        self.cache = cache
        self.generation_config = generation_config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.model_device = model_device
        self.model_dtype = model_dtype

        self.active_requests: Dict[str, RequestState] = {}
        self.waiting_requests: Deque[RequestState] = deque()  # Requests waiting for cache space
        self.requests_to_process_next: List[str] = []  # IDs of requests selected for the next batch

        self.max_batch_size = getattr(generation_config, "batch_size", 8)  # Or determine dynamically?
        self.max_context_len = getattr(
            generation_config, "max_position_embeddings", 2048
        )  # Or model.config.max_position_embeddings
        self.eos_token_id = generation_config.eos_token_id

    def _get_new_requests(self):
        """Pull new requests from the input queue and add to waiting list."""
        while not self.input_queue.empty():
            try:
                req_data = self.input_queue.get_nowait()
                if req_data is None:  # Sentinel value for stopping
                    continue

                request_id = req_data["request_id"]
                input_ids = req_data["input_ids"]
                # TODO: Get other params like max_new_tokens from req_data if needed
                max_new_tokens = req_data.get("max_new_tokens", self.generation_config.max_new_tokens or 20)

                if not input_ids:
                    logger.warning(f"Request {request_id} received with empty input_ids. Ignoring.")
                    # Optionally put a "failed" status on the output queue
                    continue

                if len(input_ids) > self.max_context_len:
                    logger.warning(
                        f"Request {request_id} prompt length ({len(input_ids)}) exceeds max context length ({self.max_context_len}). Truncating."
                    )
                    input_ids = input_ids[-self.max_context_len :]  # Keep the end

                state = RequestState(
                    request_id=request_id,
                    prompt_ids=list(input_ids),  # Ensure it's a list
                    max_new_tokens=max_new_tokens,
                    eos_token_id=self.eos_token_id,
                )
                self.waiting_requests.append(state)

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing new request: {e}", exc_info=True)

    def _schedule_batch(self) -> List[str]:
        """Selects requests for the next processing batch based on state and available cache."""
        selected_requests = []
        num_free_blocks = self.cache.get_num_free_blocks()
        potential_batch_tokens = 0

        # 1. Prioritize requests already running (in active_requests and status 'decoding')
        # These only need 1 new token slot, which usually fits.
        running_requests = [req_id for req_id, state in self.active_requests.items() if state.status == "decoding"]
        selected_requests.extend(running_requests)
        potential_batch_tokens += len(running_requests)  # 1 token per running request

        # 2. Add requests that finished prefilling ('prefilling' -> 'decoding')
        # These also only need 1 new token slot for the first generation step.
        prefilled_requests = [req_id for req_id, state in self.active_requests.items() if state.status == "prefilling"]
        # Check if adding these exceeds max batch size
        can_add_count = self.max_batch_size - len(selected_requests)
        add_now = prefilled_requests[:can_add_count]
        selected_requests.extend(add_now)
        potential_batch_tokens += len(add_now)

        # 3. Try to add new requests from the waiting queue or remaining parts of split requests
        candidates = list(self.waiting_requests)  # Check waiting list first
        # Also consider active requests that are split and waiting for continuation
        split_waiting = [
            req_id for req_id, state in self.active_requests.items() if state.status == "split_pending_remainder"
        ]
        # Prioritize split continuations? Or FIFO with waiting_requests? Let's try FIFO for now.
        candidates.extend([self.active_requests[req_id] for req_id in split_waiting])

        processed_candidate_indices = set()

        for i, state in enumerate(candidates):
            if len(selected_requests) >= self.max_batch_size:
                break

            if state.request_id in selected_requests:  # Already added (e.g., was prefilling)
                continue

            tokens_to_process = (
                len(state.remaining_prompt_ids) if state.status == "split_pending_remainder" else len(state.prompt_ids)
            )
            if tokens_to_process == 0:  # Should not happen, but safety check
                logger.warning(f"Request {state.request_id} has 0 tokens to process in scheduling.")
                if i < len(self.waiting_requests):  # Remove from waiting if it's bad
                    processed_candidate_indices.add(i)
                continue

            needed_blocks = math.ceil(tokens_to_process / self.cache.block_size)
            can_allocate_all = needed_blocks <= num_free_blocks

            if can_allocate_all:
                num_free_blocks -= needed_blocks
                potential_batch_tokens += tokens_to_process
                selected_requests.append(state.request_id)
                # Update state: move from waiting to active, or set status for split
                if state.status == "pending":  # New request from waiting queue
                    self.active_requests[state.request_id] = state
                    state.status = "prefilling"
                    processed_candidate_indices.add(i)
                elif state.status == "split_pending_remainder":
                    # Already in active_requests, just update status
                    state.status = "prefilling"  # Prefill the remainder

            elif num_free_blocks > 0:  # Try splitting
                # Only split new requests or remainder requests, not already decoding ones
                if state.status in ["pending", "split_pending_remainder"]:
                    allocatable_blocks = num_free_blocks
                    num_free_blocks = 0  # Use all remaining blocks

                    # Calculate how many tokens fit
                    num_tokens_to_prefill = allocatable_blocks * self.cache.block_size

                    source_tokens = (
                        state.remaining_prompt_ids if state.status == "split_pending_remainder" else state.prompt_ids
                    )

                    if num_tokens_to_prefill >= len(source_tokens):
                        # This should have been caught by can_allocate_all, but safety check
                        logger.warning(f"Logic error in splitting: {num_tokens_to_prefill=} >= {len(source_tokens)=}")
                        num_tokens_to_prefill = len(source_tokens)

                    if num_tokens_to_prefill == 0 and allocatable_blocks > 0:
                        # Can fit at least one block, but calculation is zero? Should fit block_size tokens.
                        num_tokens_to_prefill = min(self.cache.block_size, len(source_tokens))

                    if num_tokens_to_prefill > 0:
                        potential_batch_tokens += num_tokens_to_prefill
                        selected_requests.append(state.request_id)

                        if state.status == "pending":  # New request being split
                            self.active_requests[state.request_id] = state
                            state.remaining_prompt_ids = state.prompt_ids[num_tokens_to_prefill:]
                            state.prompt_ids = state.prompt_ids[
                                :num_tokens_to_prefill
                            ]  # Set prompt_ids to the part being prefilled now
                            state.status = "prefilling_split"
                            processed_candidate_indices.add(i)
                        elif state.status == "split_pending_remainder":
                            # Prefilling a remainder, but it needs further splitting
                            current_remainder = state.remaining_prompt_ids
                            state.prompt_ids = current_remainder[:num_tokens_to_prefill]  # Part to prefill now
                            state.remaining_prompt_ids = current_remainder[num_tokens_to_prefill:]
                            state.status = "prefilling_split"  # Still prefilling a split part
                    else:
                        # Cannot even fit a single token from the split part
                        logger.debug(f"Cannot split request {state.request_id}, not enough space for minimal prefill.")

            else:  # No free blocks left
                break  # Stop trying to add new/split requests

        # Update waiting_requests deque by removing processed ones
        new_waiting = deque()
        for i, state in enumerate(self.waiting_requests):
            if i not in processed_candidate_indices:
                new_waiting.append(state)
        self.waiting_requests = new_waiting

        return selected_requests

    def prepare_next_batch(self):
        """Prepares tensors and metadata for the next model forward pass."""
        self._get_new_requests()

        if not self.active_requests and not self.waiting_requests:
            return None

        self.requests_to_process_next = self._schedule_batch()

        if not self.requests_to_process_next:
            return None

        batch_input_ids = []
        batch_position_ids = []
        batch_cache_indices = []  # Physical indices for K/V cache *read*
        batch_fill_indices = []  # Physical indices for K/V cache *write*
        cumulative_seqlens_q = [0]
        cumulative_seqlens_k = [0]  # K sequence length includes context
        max_seqlen_q = 0
        max_seqlen_k = 0
        logits_indices = []  # Indices within the batch's concatenated output tokens to get the next token logits

        requests_in_batch: List[RequestState] = [
            self.active_requests[req_id] for req_id in self.requests_to_process_next
        ]

        # Sort requests? Maybe by length for prefill, doesn't matter much for generation
        # requests_in_batch.sort(key=lambda r: len(r.prompt_ids) if r.status.startswith("prefill") else 1)

        for state in requests_in_batch:
            if state.status == "decoding":
                # Add the single next token ID (using last generated or last prompt token)
                last_token = state.output_ids[-1]
                tokens_to_add = [last_token]
                start_pos = state.current_len()  # Position of the token we are decoding NEXT
                positions_to_add = [start_pos]

                # Allocate blocks if needed (should only need 1 block extra occasionally)
                current_num_blocks = len(state.allocated_blocks)
                needed_total_slots = state.current_len() + 1  # Slot needed for the token we are about to generate
                needed_total_blocks = math.ceil(needed_total_slots / self.cache.block_size)

                if needed_total_blocks > current_num_blocks:
                    new_blocks_needed = needed_total_blocks - current_num_blocks
                    allocated = self.cache.allocate_blocks(new_blocks_needed, state.request_id)
                    if len(allocated) < new_blocks_needed:
                        # This should be rare if scheduling logic is correct
                        logger.error(
                            f"Failed to allocate block for decoding request {state.request_id}. This might lead to errors."
                        )
                        # How to handle this? Skip request? Mark as failed?
                        # For now, continue, but cache write might fail.
                    state.allocated_blocks.extend(allocated)

                # Calculate cache indices for read (all previous tokens) and write (the new token position)
                read_logical_indices = list(range(state.current_len()))
                write_logical_index = state.current_len()

                # Map logical indices to physical block indices for this request
                physical_read_indices = self.cache._get_physical_indices(state.request_id, read_logical_indices)
                physical_write_index = self.cache._get_physical_indices(state.request_id, [write_logical_index])[0]

                batch_cache_indices.extend(physical_read_indices)
                batch_cache_indices.append(physical_write_index)
                batch_fill_indices.append(physical_write_index)  # Only one write position per generation request

                seq_len_q = 1  # Query length is 1 for generation
                seq_len_k = state.current_len() + 1  # Key length includes the token being generated

            elif state.status.startswith("prefilling"):  # prefilling or prefilling_split
                tokens_to_add = state.prompt_ids  # The prompt part being prefilled now
                start_pos = (
                    state.position_offset
                )  # Should be 0 for new requests, or previous offset for split remainders
                positions_to_add = list(range(start_pos, start_pos + len(tokens_to_add)))

                # Allocate blocks for these tokens
                needed_blocks = math.ceil(len(tokens_to_add) / self.cache.block_size)
                # Check if we already allocated some blocks (e.g., for split prefill)
                new_blocks_needed = needed_blocks - len(
                    state.allocated_blocks
                )  # Can be negative if blocks already allocated
                if new_blocks_needed > 0:
                    allocated = self.cache.allocate_blocks(new_blocks_needed, state.request_id)
                    if len(allocated) < new_blocks_needed:
                        logger.error(
                            f"Failed to allocate {new_blocks_needed} blocks for prefilling request {state.request_id}. Scheduling logic failed."
                        )
                        # Mark request as failed? Skip?
                        state.status = "failed"
                        # Need to remove from batch? Or handle failure later?
                        continue  # Skip adding this request to the batch tensors
                    state.allocated_blocks.extend(allocated)

                write_logical_indices = list(range(start_pos, start_pos + len(tokens_to_add)))
                physical_write_indices = self.cache._get_physical_indices(state.request_id, write_logical_indices)

                batch_fill_indices.extend(physical_write_indices)
                batch_cache_indices.extend(physical_write_indices)

                seq_len_q = len(tokens_to_add)  # Query length is the number of prefill tokens
                seq_len_k = len(tokens_to_add)  # Key length is also the number of prefill tokens (causal attention)

            else:  # Should not happen if scheduling is correct
                logger.warning(
                    f"Request {state.request_id} in unexpected state '{state.status}' during batch preparation. Skipping."
                )
                continue

            batch_input_ids.extend(tokens_to_add)
            batch_position_ids.extend(positions_to_add)

            # Update cumulative lengths
            cumulative_seqlens_q.append(cumulative_seqlens_q[-1] + seq_len_q)
            cumulative_seqlens_k.append(cumulative_seqlens_k[-1] + seq_len_k)  # Adjust if K includes context

            # Update max lengths
            max_seqlen_q = max(max_seqlen_q, seq_len_q)
            max_seqlen_k = max(max_seqlen_k, seq_len_k)  # This K length might be local to the request

            # Index for fetching the NEXT token's logits: it's the last token added for this request
            logits_indices.append(cumulative_seqlens_q[-1] - 1)

            # Update state's internal position offset for next iteration
            state.position_offset += len(tokens_to_add)

        if not batch_input_ids:
            # No valid requests could be processed in the batch
            return None

        # Convert lists to tensors
        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=self.model_device).unsqueeze(
            0
        )  # Shape: (1, total_tokens)
        position_ids_tensor = torch.tensor(batch_position_ids, dtype=torch.long, device=self.model_device).unsqueeze(
            0
        )  # Shape: (1, total_tokens)
        cumulative_seqlens_q_tensor = torch.tensor(cumulative_seqlens_q, dtype=torch.int32, device=self.model_device)
        # TODO: Verify cumulative_seqlens_k logic - does it need full context length or just batch length?
        # Assuming flash_attn_varlen_func needs cumulative lengths of K *within the batch*:
        cumulative_seqlens_k_tensor = torch.tensor(cumulative_seqlens_k, dtype=torch.int32, device=self.model_device)

        # Placeholder kwargs - these need careful construction based on assumed PagedAttention API used by model
        model_kwargs = {
            "cumulative_seqlens_q": cumulative_seqlens_q_tensor,
            "cumulative_seqlens_k": cumulative_seqlens_k_tensor,  # K includes context length? If so, need state.current_len()
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,  # Needs to be max *total* K length in batch?
            "fill_index": torch.tensor(
                batch_fill_indices, dtype=torch.long, device=self.model_device
            ),  # Indices to WRITE to KV cache
            "cache_index": torch.tensor(
                batch_cache_indices, dtype=torch.long, device=self.model_device
            ),  # Indices to READ from KV cache
            "logits_indices": logits_indices,  # Indices used *after* forward pass to get next token logits
            "block_tables": {
                req_id: self.cache.get_block_table(req_id) for req_id in self.requests_to_process_next
            },  # Pass block tables if needed by attention mechanism
            "cache": self.cache,
            # XXX: this to disable the automatic declaration of `DynamicCache`, thus saving memory
            "use_cache": False,
        }
        # Recalculate max_seqlen_k to be the maximum *total* sequence length in the batch
        max_total_len_k = 0
        for state in requests_in_batch:
            max_total_len_k = max(
                max_total_len_k, state.current_len()
            )  # current_len reflects length *after* this batch
        model_kwargs["max_seqlen_k"] = max_total_len_k

        return input_ids_tensor, position_ids_tensor, model_kwargs

    def update_batch(self, generated_ids: torch.Tensor):
        """Updates request states based on the generated tokens."""
        token_idx_in_generation = 0
        finished_request_ids = []

        for req_id in self.requests_to_process_next:
            if req_id not in self.active_requests:
                logger.warning(
                    f"Request {req_id} not found in active requests during update. Might have finished or failed."
                )
                continue

            state = self.active_requests[req_id]

            if state.status == "prefilling":  # Just finished prefilling the whole prompt
                # No token generated yet, just move to 'decoding' state
                state.status = "decoding"
                state.prompt_ids = []  # Clear prompt_ids as they are now in cache
                token = generated_ids[token_idx_in_generation]
                token_idx_in_generation += 1

                state.output_ids.append(token)

                is_eos = token == state.eos_token_id
                is_max_len = state.generated_len() >= state.max_new_tokens

                if is_eos or is_max_len:
                    state.status = "finished"
                    logger.debug(f"Request {req_id} finished. Reason: {'EOS' if is_eos else 'Max Length'}")
                    finished_request_ids.append(req_id)
                    self.output_queue.put(
                        {
                            "request_id": state.request_id,
                            "output_ids": state.output_ids,
                            "status": "finished",
                        }
                    )

            elif state.status == "prefilling_split":  # Finished prefilling a *part* of the prompt
                # Ignore the generated token for this step.
                token_idx_in_generation += 1  # Consume the token from generated_ids
                # Check if there's a remainder
                if state.remaining_prompt_ids:
                    state.status = "split_pending_remainder"  # Will be scheduled for prefilling later
                    state.prompt_ids = []  # Clear the processed part
                else:
                    # No remainder, move to generation
                    state.status = "decoding"
                    state.prompt_ids = []
                logger.debug(f"Request {req_id} finished prefill split. Status: {state.status}")

            elif state.status == "decoding":
                # This request generated a token
                if token_idx_in_generation >= len(generated_ids):
                    logger.error(
                        f"Token index {token_idx_in_generation} out of bounds for generated_ids (len {len(generated_ids)}) for request {req_id}. Batching logic error."
                    )
                    state.status = "failed"
                else:
                    token = generated_ids[token_idx_in_generation].item()
                    token_idx_in_generation += 1

                    state.output_ids.append(token)

                    is_eos = token == state.eos_token_id
                    is_max_len = state.generated_len() >= state.max_new_tokens

                    if is_eos or is_max_len:
                        state.status = "finished"
                        logger.debug(f"Request {req_id} finished. Reason: {'EOS' if is_eos else 'Max Length'}")
                        finished_request_ids.append(req_id)
                        self.output_queue.put(
                            {
                                "request_id": state.request_id,
                                "output_ids": state.output_ids,
                                "status": "finished",
                            }
                        )

            elif state.status == "split_pending_remainder":
                # This state should only exist *between* batches. If it's in requests_to_process_next,
                # it should have been transitioned to 'prefilling' or 'prefilling_split'.
                logger.warning(
                    f"Request {req_id} encountered in 'split_pending_remainder' state during update. Logic error."
                )
                # It didn't generate a token in this round. No update needed, but state is wrong.
                pass  # Keep it active, hope scheduling fixes it next round

        # Clean up finished requests
        for req_id in finished_request_ids:
            if req_id in self.active_requests:
                self.cache.free_blocks(req_id)
                del self.active_requests[req_id]

    def has_pending_requests(self) -> bool:
        """Check if there are any active or waiting requests."""
        return bool(self.active_requests or self.waiting_requests)


# Manager Class (User Interface)
class ContinuousBatchingManager:
    def __init__(self, model: GenerationMixin, generation_config: GenerationConfig, max_queue_size=0):
        self.model = model
        self.generation_config = generation_config
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self._generation_thread = None
        self._request_counter = 0
        self._request_lock = threading.Lock()  # For request counter

        # Defer cache initialization until thread start, get initial shapes if possible
        self._initial_prompt_shapes = []  # Can be populated before starting

    def add_initial_prompts(self, prompts: List[List[int]]):
        """Optional: Provide initial prompts to help determine optimal cache size before starting."""
        if self._generation_thread is not None:
            raise RuntimeError("Cannot add initial prompts after the manager has started.")
        self._initial_prompt_shapes = prompts

    def start(self):
        """Starts the background generation thread."""
        if self._generation_thread is not None and self._generation_thread.is_alive():
            logger.warning("Manager thread is already running.")
            return

        # Clear any stale state before starting
        self._result_queue = queue.Queue()
        self._generation_thread = threading.Thread(target=self._run_generation_loop)
        self._generation_thread.start()
        logger.info("Continuous batching manager started.")

    def is_running(self):
        """Checks if the background generation thread is currently running."""
        return self._generation_thread is not None and self._generation_thread.is_alive()

    def stop(self, block: bool = True, timeout: Optional[float] = None):
        """Signals the background generation thread to stop.

        Args:
            block (bool): Whether to block until the thread stops.
            timeout (Optional[float]): Maximum time to wait for the thread to stop.
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
        """Waits for the background generation thread to finish.

        Args:
            timeout (Optional[float]): Maximum time to wait for the thread to stop.
        """
        if self._generation_thread is not None:
            self._generation_thread.join(timeout=timeout)
            if self._generation_thread.is_alive():
                logger.warning("Generation thread did not exit after join timeout.")
            else:
                logger.info("Continuous Batching Manager stopped.")
                self._generation_thread = None  # Mark as stopped

    def add_request(self, input_ids: List[int], request_id: Optional[str] = None, **kwargs) -> str:
        """Adds a new generation request to the queue."""
        if self.stop_event.is_set():
            raise RuntimeError("Manager is stopped or stopping.")
        if self._generation_thread is None:
            raise RuntimeError("Manager has not been started yet. Call start().")

        if request_id is None:
            with self._request_lock:
                request_id = f"req_{self._request_counter}"
                self._request_counter += 1
            # Include generation params from kwargs if needed
        req_data = {"request_id": request_id, "input_ids": input_ids, **kwargs}

        # Use block=True with timeout to handle backpressure if queue is full
        self.input_queue.put(req_data, block=True, timeout=10)  # Adjust timeout as needed
        logger.debug(f"Added request {request_id} to queue.")
        return request_id

    def get_result(self, timeout=None) -> Optional[Dict]:
        """Retrieves one finished result from the output queue."""
        if self._generation_thread is None and self.output_queue.empty():
            # Avoid blocking indefinitely if manager never started or already stopped and emptied
            return None

        result = self.output_queue.get(block=True, timeout=timeout)
        logger.debug(f"Retrieved result for request {result.get('request_id')}")
        return result  # Expected format: {"request_id": ..., "output_ids": ..., "status": ...}

    def __iter__(self):
        """Allows iterating over results as they become available."""
        while (
            self._generation_thread is not None and self._generation_thread.is_alive() or not self.output_queue.empty()
        ):
            try:
                yield self.get_result(timeout=0.1)  # Short timeout to allow checking thread status
            except queue.Empty:
                if self._generation_thread is None or not self._generation_thread.is_alive():
                    # Thread stopped and queue is empty, break the iterator
                    break
                continue  # Continue waiting if thread is alive

    def _run_generation_loop(self):
        """The main loop running in the background thread."""
        try:
            # 1. Initialize Cache
            # TODO: Handle potential errors during initialization
            paged_attention_cache = PagedAttentionCache(
                self.model.config,
                self.generation_config,
                self.model.device,
                self.model.dtype,
                initial_prompt_shapes=self._initial_prompt_shapes,
            )
            self._initial_prompt_shapes = None  # Clear after use

            # 2. Initialize Batch Processor
            batch_processor = ContinuousBatchProcessor(
                paged_attention_cache,
                self.generation_config,
                self.input_queue,
                self.output_queue,
                self.stop_event,
                self.model.device,
                self.model.dtype,
            )

            # 3. Generation Loop
            while not self.stop_event.is_set() or batch_processor.has_pending_requests():
                # Prepare Batch
                batch_data = batch_processor.prepare_next_batch()

                if batch_data is None:
                    if self.stop_event.is_set() and not batch_processor.has_pending_requests():
                        break  # Stop condition met
                    # Wait briefly if no work, prevents high CPU usage
                    time.sleep(0.005)  # 5ms wait
                    continue

                input_ids, position_ids, model_kwargs = batch_data

                # Run Model Forward
                # Ensure the model's forward pass uses the 'past_key_values' or 'cache' object correctly
                # and potentially calls its update/write methods.
                # The `model_kwargs` should contain everything needed.
                try:
                    with torch.no_grad():  # Ensure inference mode
                        outputs = self.model.forward(
                            input_ids=input_ids,
                            position_ids=position_ids,
                            **model_kwargs,
                        )
                except Exception as e:
                    logger.error(f"Model forward pass failed: {e}", exc_info=True)
                    # How to handle? Mark involved requests as failed? Stop manager?
                    # Mark requests in the failed batch as failed
                    failed_ids = batch_processor.requests_to_process_next
                    for req_id in failed_ids:
                        if req_id in batch_processor.active_requests:
                            state = batch_processor.active_requests[req_id]
                            state.status = "failed"
                            self.output_queue.put(
                                {"request_id": req_id, "output_ids": state.output_ids, "status": "failed"}
                            )
                            batch_processor.cache.free_blocks(req_id)
                            del batch_processor.active_requests[req_id]
                    continue  # Skip logits processing and update for this batch

                # Get Logits for the last token of each sequence in the batch
                # Use the logits_indices prepared earlier
                next_token_logits = outputs.logits[:, model_kwargs["logits_indices"], :]

                # TODO: Implement proper sampling (top-k, top-p, temperature) based on generation_config or request params
                # Using simple argmax for now:
                generated_ids = torch.argmax(next_token_logits, dim=-1).squeeze(0)  # Squeeze batch dim if necessary

                batch_processor.update_batch(generated_ids)

        except Exception as e:
            logger.error(f"Error in generation loop: {e}", exc_info=True)
            # Signal stop on critical error
            self.stop_event.set()
            # Potentially try to fail pending requests gracefully
            try:
                while True:
                    req_data = self.input_queue.get_nowait()
                    if req_data and "request_id" in req_data:
                        self.output_queue.put(
                            {
                                "request_id": req_data["request_id"],
                                "output_ids": [],
                                "status": "failed",
                                "error": str(e),
                            }
                        )
            except queue.Empty:
                pass
            if "batch_processor" in locals():
                for req_id, state in list(batch_processor.active_requests.items()):
                    self.output_queue.put(
                        {"request_id": req_id, "output_ids": state.output_ids, "status": "failed", "error": str(e)}
                    )
                    batch_processor.cache.free_blocks(req_id)
                    del batch_processor.active_requests[req_id]

        finally:
            logger.info("Generation loop finished.")


class ContinuousMixin:
    def init_continuous_batching(
        self, generation_config: Optional[GenerationConfig] = None, max_queue_size: int = 0
    ) -> ContinuousBatchingManager:
        """
        Initializes and returns a manager for continuous batching inference.

        Args:
            generation_config (`GenerationConfig`, *optional*):
                Custom generation configuration. If None, `self.generation_config` is used.
            max_queue_size (`int`, *optional*, defaults to 0):
                Maximum size of the input request queue. 0 means infinite.

        Returns:
            `ContinuousBatchingManager`: The manager instance to add requests and retrieve results.
        """
        if not hasattr(self, "config") or not hasattr(self, "device") or not hasattr(self, "dtype"):
            raise AttributeError("Model must have 'config', 'device', and 'dtype' attributes for continuous batching.")

        gen_config = generation_config if generation_config is not None else self.generation_config
        if gen_config is None:
            raise ValueError("A GenerationConfig must be provided or set in the model.")

        # Ensure necessary generation config parameters are present (or provide defaults)
        if gen_config.eos_token_id is None:
            logger.warning("`eos_token_id` not set in GenerationConfig. Setting to -1 (disabled).")
            gen_config.eos_token_id = -1  # Or get from model.config?

        manager = ContinuousBatchingManager(model=self, generation_config=gen_config, max_queue_size=max_queue_size)
        return manager

    @torch.no_grad()
    def generate_batch(
        self,
        inputs: List[List[int]],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> List[List[int]]:
        """
        Generates sequences for a batch of prompts using continuous batching.

        This method provides a simpler interface compared to manually managing the
        `ContinuousBatchingManager`. It initializes, runs, and shuts down the
        manager internally for the given batch of inputs.

        Args:
            inputs (`List[List[int]]`):
                A list of prompts, where each prompt is represented as a list of token IDs.
            generation_config (`GenerationConfig`, *optional*):
                Custom generation configuration. If None, `self.generation_config` is used.
            **kwargs:
                Additional keyword arguments to pass to the generation process for each request
                (e.g., `max_new_tokens`). These will override defaults in the `generation_config`.

        Returns:
            `List[List[int]]`: A list containing the generated sequences (including prompt tokens
                                if not handled otherwise) for each input prompt, in the same order.
                                Returns an empty list `[]` for requests that failed.
        """
        if not inputs:
            return []

        manager = self.init_continuous_batching(generation_config=generation_config)
        manager.add_initial_prompts(inputs)  # Help with cache sizing
        manager.start()

        results = {}
        request_ids = {}
        num_requests = len(inputs)

        try:
            # Add all requests
            for i, input_ids in enumerate(inputs):
                # Assign a predictable request ID for ordering results later
                req_id = f"batch_req_{i}"
                # Pass request-specific kwargs (like max_new_tokens) if provided
                req_kwargs = kwargs.copy()
                manager.add_request(input_ids=input_ids, request_id=req_id, **req_kwargs)
                request_ids[req_id] = i  # Store original index

            # Collect results until all requests are done
            finished_requests = 0
            while finished_requests < num_requests:
                try:
                    result = manager.get_result(timeout=1.0)  # Use timeout to avoid potential deadlocks
                    if result:
                        req_id = result["request_id"]
                        if req_id in request_ids:
                            original_index = request_ids[req_id]
                            if result["status"] == "finished":
                                # Combine prompt and output? The manager currently only returns output_ids.
                                # Let's return only output_ids for now, consistent with manager.
                                results[original_index] = result["output_ids"]
                            else:  # Failed request
                                logger.warning(f"Request {req_id} failed: {result.get('error', 'Unknown error')}")
                                results[original_index] = []  # Indicate failure with empty list
                        else:
                            logger.warning(f"Received result for unknown request ID: {req_id}")

                    finished_requests += 1

                except queue.Empty:
                    # Timeout occurred, check if the manager is still alive
                    if not manager._generation_thread.is_alive():
                        logger.error("Generation thread terminated unexpectedly.")
                        break
                    continue

        except Exception as e:
            logger.error(f"Error during batch generation: {e}", exc_info=True)
            # Mark all requests as failed if a general error occurs
            for i in range(num_requests):
                if i not in results:
                    results[i] = []
        finally:
            # Ensure manager is stopped and joined regardless of errors
            manager.stop()
            manager.join(timeout=10.0)  # Add timeout to join

        # Return results in the original order
        ordered_results = [results.get(i, []) for i in range(num_requests)]
        return ordered_results
