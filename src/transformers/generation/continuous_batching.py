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
from typing import Deque, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm


# TODO: move to its own file for usage in other modules
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode, get_tracer

    resource = Resource.create({"service.name": "transformers"})

    metrics_exporter = PeriodicExportingMetricReader(OTLPMetricExporter())
    meter_provider = MeterProvider(resource=resource, metric_readers=[metrics_exporter])
    metrics.set_meter_provider(meter_provider)

    trace_exporter = OTLPSpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(tracer_provider)

    _has_opentelemetry = True
except ImportError:
    _has_opentelemetry = False

from ..cache_utils import Cache
from ..configuration_utils import PretrainedConfig
from ..generation.configuration_utils import GenerationConfig
from ..generation.utils import GenerationMixin
from ..utils import (
    logging,
)


def traced(func=None, *, span_name=None):
    """
    Decorator to trace function calls with OpenTelemetry.

    Can be used as @traced or @traced(span_name="custom_name")

    Args:
        func: The function to trace
        span_name: Optional custom name for the span (defaults to function name)

    Returns:
        Decorated function with tracing
    """

    def decorator(func):
        if not _has_opentelemetry:
            return func

        import functools

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, "tracer"):
                return func(self, *args, **kwargs)

            name = span_name or func.__name__
            with self.tracer.start_as_current_span(name) as span:
                # Add function signature details to span
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Add args and kwargs as attributes where possible
                if args:
                    for i, arg in enumerate(args):
                        if isinstance(arg, (str, int, float, bool)) or arg is None:
                            span.set_attribute(f"args.{i}", str(arg))

                # Add request_id if it's a common parametbatch_processoeer
                if "request_id" in kwargs and isinstance(kwargs["request_id"], str):
                    span.set_attribute("request_id", kwargs["request_id"])

                # Add important batch information
                if func.__name__ == "prepare_next_batch" and hasattr(self, "requests_to_process_next"):
                    span.set_attribute("batch.size", len(getattr(self, "requests_to_process_next", [])))

                try:
                    result = func(self, *args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    raise

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


logger = logging.get_logger(__name__)


@dataclass
class RequestState:
    """Tracks the state of a generation request through its lifecycle.

    Attributes:
        status (str): can be one of 'pending', 'prefilling', 'prefilling_split', 'split_pending_remainder', 'decoding', 'finished', 'failed'
    """

    # Required fields
    request_id: str
    prompt_ids: List[int] = None            # the one being processed
    full_prompt_ids: List[int] = None       # the full prompt
    remaining_prompt_ids: List[int] = field(default_factory=list)  # For split requests
    static_outputs: List[int] = field(default_factory=list)
    allocated_blocks: List[int] = field(default_factory=list)
    position_offset: int = 0  # Current position in the sequence for position_ids
    status: str = "pending"
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
        self.static_outputs.append(token_id)
        self.prompt_ids = [token_id]

        is_eos = token_id == self.eos_token_id and self.eos_token_id != -1
        is_max_len = self.generated_len() >= self.max_new_tokens

        if is_eos or is_max_len:
            self.status = "finished"
            return True

        return False

    def __repr__(self):
        return f"RequestState(\nrequest_id={self.request_id},\nstatus={self.status}, \nout_tokens={self.generated_len()}, \ncurrent_prompt={len(self.prompt_ids)}, \nprefilled_tokens={self.current_len()}, \nremaining_length={len(self.remaining_prompt_ids)}, \nfull_lenght={len(self.full_prompt_ids)},\nallocated_blocks={self.allocated_blocks}\n)"


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
        self._setup_tracer()
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

    def _setup_tracer(self):
        """Initialize OpenTelemetry tracing if available."""
        if not _has_opentelemetry:
            return

        # Create a tracer for our functions
        self.tracer = get_tracer("transformers.generation.paged_attention_cache")

    @traced
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
        logical_indices_tensor = torch.tensor(logical_indices, dtype=torch.long, device=self.device)
        block_size = self.block_size

        # Vectorized calculations
        block_indices = logical_indices_tensor // block_size
        block_offsets = logical_indices_tensor % block_size

        if torch.any(block_indices > len(block_table)):
            raise IndexError(f"Logical indices map to block indices out of bounds for request {request_id}")

        block_table_tensor = torch.tensor(block_table, dtype=torch.long, device=self.device)
        physical_block_nums = block_table_tensor[block_indices]
        physical_indices = physical_block_nums * block_size + block_offsets
        return physical_indices

    # @traced
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        read_index,
        write_index,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Reshape cache for easier indexing
        total_slots = self.num_blocks * self.block_size
        k_cache_flat = self.key_cache[layer_idx].view(self.num_key_value_heads, total_slots, self.head_dim)
        v_cache_flat = self.value_cache[layer_idx].view(self.num_key_value_heads, total_slots, self.head_dim)

        write_index = write_index.to(k_cache_flat.device)
        k_cache_flat[:, write_index, :] = key_states[0]
        v_cache_flat[:, write_index, :] = value_states[0]
        return k_cache_flat[None, :, read_index, :], v_cache_flat[None, :, read_index, :]


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
    block_tables: Dict[str, List[int]]
    cache: PagedAttentionCache
    use_cache: bool = False


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

        # Set up OpenTelemetry metrics if available
        self._setup_metrics()

        # Get batch size parameters from generation config
        self._configure_batch_parameters()
        self.setup_static_tensors()

    def setup_static_tensors(self):
        T = self.max_batch_tokens
        max_token_budget = self.cache.num_blocks * self.cache.block_size
        tensor_metadata = {"dtype": torch.long, "device": self.model_device}
        self.tensor_metadata = tensor_metadata
        self.input_ids = torch.zeros((1, T), **tensor_metadata)
        self.output_ids = torch.ones((1, T), **tensor_metadata) * -1
        self.attention_mask = torch.zeros(
            (1, 1, T, max_token_budget), dtype=self.model_dtype, device=self.model_device
        )
        self.position_ids = torch.zeros((1, T), **tensor_metadata)
        self.cumulative_seqlens_q = torch.zeros((T + 1,), **tensor_metadata)
        self.cumulative_seqlens_k = torch.zeros((T + 1,), **tensor_metadata)
        self.write_index = torch.zeros((T,), **tensor_metadata)
        self.read_index = torch.zeros((max_token_budget,), **tensor_metadata)
        self.logits_indices = torch.ones((T,), **tensor_metadata) * -1
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0

    def reset_static_tensors(self):
        """Reset static tensors for the next batch."""
        self.input_ids.zero_()
        self.attention_mask.fill_(torch.finfo(self.model_dtype).min)
        self.position_ids.zero_()
        self.cumulative_seqlens_q.zero_()
        self.cumulative_seqlens_k.zero_()
        self.write_index.fill_(-1)
        self.read_index.fill_(-1)
        # self.logits_indices.zero_()
        self.max_seqlen_q = 0
        self.max_seqlen_k = 0

    def get_model_kwargs(self) -> PagedAttentionArgs:
        """Get model keyword arguments for the current batch."""
        return PagedAttentionArgs(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
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

    def __repr__(self):
        return f"ContinuousBatchProcessor(input_queue={self.input_queue}, output_queue={self.output_queue}, active_requests={self.active_requests}, waiting_requests={self.waiting_requests})" + self.get_model_kwargs().__repr__() 

    def _setup_metrics(self):
        """Initialize OpenTelemetry metrics and tracing if the library is available."""

        if not _has_opentelemetry:
            logger.info("OpenTelemetry is not installed. Metrics and tracing will not be recorded.")
            return

        # Create a resource to identify this component in the metrics system
        resource = Resource.create({"service.name": "transformers.generation.continuous_batching_processor"})

        metrics_exporter = OTLPMetricExporter()
        meter_provider = MeterProvider(resource=resource, metric_readers=[metrics_exporter])
        metrics.set_meter_provider(meter_provider)

        # Set up the tracer provider with an OTLP exporter for tracing
        trace_exporter = OTLPSpanExporter()
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        trace.set_tracer_provider(tracer_provider)

        # Create a tracer for our functions
        self.tracer = get_tracer("transformers.generation")

        # Create a meter for our metrics
        self.meter = metrics.get_meter("transformers.generation")

        # Create histogram for time to first token
        self.ttft_histogram = self.meter.create_histogram(
            name="ttft_milliseconds",
            description="Time to first token in milliseconds",
            unit="ms",
        )

        # Create histogram for decode/prefill ratio
        self.decode_prefill_ratio_gauge = self.meter.create_gauge(
            name="decode_prefill_ratio",
            description="Ratio of decode tokens to prefill tokens in a batch",
            unit="ratio",
        )

        # Create counters for decode and prefill tokens
        self.prefill_tokens_counter = self.meter.create_counter(
            name="prefill_tokens_processed",
            description="Number of prefill tokens processed",
            unit="tokens",
        )

        self.decode_tokens_counter = self.meter.create_counter(
            name="decode_tokens_processed",
            description="Number of decode tokens processed",
            unit="tokens",
        )

        # Create histogram for batch fill percentage
        self.batch_fill_percentage_histogram = self.meter.create_histogram(
            name="batch_fill_percentage",
            description="Percentage of max_batch_tokens utilized in each batch",
            unit="percent",
        )

    def _setup_metrics(self):
        """Initialize OpenTelemetry metrics and tracing if the library is available."""

        if not _has_opentelemetry:
            logger.info("OpenTelemetry is not installed. Metrics and tracing will not be recorded.")
            return

        self.tracer = get_tracer("transformers.generation.continuous_batch_processor")

        self.meter = metrics.get_meter("transformers.generation.continuous_batch_processor")

        # Create histogram for time to first token
        self.ttft_histogram = self.meter.create_histogram(
            name="ttft_milliseconds",
            description="Time to first token in milliseconds",
            unit="ms",
        )

        # Create histogram for decode/prefill ratio
        self.decode_prefill_ratio_gauge = self.meter.create_gauge(
            name="decode_prefill_ratio",
            description="Ratio of decode tokens to prefill tokens in a batch",
            unit="ratio",
        )

        # Create counters for decode and prefill tokens
        self.prefill_tokens_counter = self.meter.create_counter(
            name="prefill_tokens_processed",
            description="Number of prefill tokens processed",
            unit="tokens",
        )

        self.decode_tokens_counter = self.meter.create_counter(
            name="decode_tokens_processed",
            description="Number of decode tokens processed",
            unit="tokens",
        )

        # Create histogram for batch fill percentage
        self.batch_fill_percentage_histogram = self.meter.create_histogram(
            name="batch_fill_percentage",
            description="Percentage of max_batch_tokens utilized in each batch",
            unit="percent",
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

    @traced
    def _get_new_requests(self):
        """Pull new requests from the input queue and add to waiting list."""
        while not self.input_queue.empty():
            try:
                state = self.input_queue.get_nowait()
                if state is None:  # Sentinel value
                    continue

                self.waiting_requests.append(state)

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
        state.status="failed"
        state.error=str(error)

        # Include any generated tokens if this is an active request
        if isinstance(state.request_id, str) and state.request_id in self.active_requests:
            state.static_outputs = self.active_requests[state.request_id].static_outputs
        else:
            state.static_outputs = []

        self.output_queue.put(state)

    @traced
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
            token_budget = (
                len(state.remaining_prompt_ids) if state.status == "split_pending_remainder" else len(state.prompt_ids)
            )
            if token_budget == 0:
                if i < len(self.waiting_requests):
                    request_ids_to_remove_from_waiting.add(state.request_id)
                continue

            # Check if we need to split the request
            if batch_token_count + token_budget > self.max_batch_tokens:
                remaining_space = self.max_batch_tokens - batch_token_count
                # Only split if we have enough space for a meaningful chunk
                if remaining_space >= 1:
                    token_budget = remaining_space
                else:
                    continue  # Not enough space to split

            # Check if we have enough blocks
            needed_blocks = math.ceil(token_budget / self.cache.block_size)
            if needed_blocks > num_free_blocks:
                logger.debug(
                    f"Request {state.request_id} needs {needed_blocks} blocks but only {num_free_blocks} available. Skipping."
                )
                continue

            # We can process this request
            num_free_blocks -= needed_blocks
            batch_token_count += token_budget
            selected_requests.append(state.request_id)

            # Set up the request based on whether we're splitting
            self._prepare_request_for_processing(state, token_budget, request_ids_to_remove_from_waiting)

        # Remove processed requests from waiting queue
        self.waiting_requests = deque(
            [req for req in self.waiting_requests if req.request_id not in request_ids_to_remove_from_waiting]
        )

        logger.debug(f"Scheduled batch with {len(selected_requests)} requests and {batch_token_count} tokens")
        return selected_requests

    @traced(span_name="prepare_request")
    def _prepare_request_for_processing(
        self, state: RequestState, token_budget, request_ids_to_remove_from_waiting
    ):
        """Prepare a request for processing in the current batch."""
        request_tokens = state.remaining_prompt_ids if state.status == "split_pending_remainder" else state.prompt_ids
        if len(request_tokens) < token_budget:
            # Can process the entire prompt/remainder
            if state.status == "pending":
                self.active_requests[state.request_id] = state
                state.status = "prefilling"
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == "split_pending_remainder":
                state.status = "prefilling"
                state.prompt_ids = state.remaining_prompt_ids
                state.remaining_prompt_ids = []
        else:
            # Need to split the request
            if state.status == "pending":
                self.active_requests[state.request_id] = state
                state.status = "prefilling_split"
                request_ids_to_remove_from_waiting.add(state.request_id)
            elif state.status == "split_pending_remainder":
                state.status = "prefilling_split"
            state.remaining_prompt_ids = request_tokens[token_budget:]
            state.prompt_ids = request_tokens[:token_budget]

    @traced
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
        read_position = 0
        write_position = 0
        cumq_ptr = 1  # because at 0 we always have 0?
        cumk_ptr = 1
        generated_tokens = (self.output_ids >= 0).sum()

        position_ids = []
        input_ids = []
        read_index = []
        write_index = []
        cumulative_seqlens_q = [0]
        cumulative_seqlens_k = [0]
        logits_to_track = 0
        logits_indices = []
        self._record_batch_metrics(requests_in_batch)

        for state in requests_in_batch:
            if state.status == "decoding":
                next_input_ids = state.prompt_ids
                input_ids.extend(next_input_ids)
                positions_to_add = [state.current_len()]
                if not self._allocate_blocks_if_needed(state, state.current_len()+1):
                    continue

                # Map logical indices to physical block indices for this request
                write_indices = self.cache._get_physical_indices(state.request_id, [state.current_len()])
                read_indices = self.cache._get_physical_indices(state.request_id, list(range(state.current_len() + 1)))

                seq_len_q = 1  # Query length is 1 for generation
                seq_len_k = state.current_len()
                state.position_offset += 1
            else:
                next_input_ids = state.prompt_ids
                input_ids.extend(next_input_ids)
                start_pos = state.current_len()
                positions_to_add = list(range(start_pos, start_pos + len(next_input_ids)))
                state.position_offset += len(next_input_ids)

                if not self._allocate_blocks_if_needed(state, state.current_len()):
                    continue

                write_indices = self.cache._get_physical_indices(state.request_id, positions_to_add)
                read_indices = self.cache._get_physical_indices(state.request_id, list(range(state.current_len())))
                seq_len_q = len(next_input_ids)
                seq_len_k = state.current_len() # Key length includes previously split context #TODO not sure about +1

            position_ids.extend(positions_to_add)
            read_index.extend(read_indices)
            write_index.extend(write_indices)
            cumulative_seqlens_q.append(cumulative_seqlens_q[-1] + seq_len_q)
            cumulative_seqlens_k.append(cumulative_seqlens_k[-1] + seq_len_k)
            if len(state.remaining_prompt_ids) == 0:
                logits_indices.append(cumulative_seqlens_q[-1] - 1)
                logits_to_track += 1

            self.max_seqlen_q = max(self.max_seqlen_q, seq_len_q)
            self.max_seqlen_k = max(self.max_seqlen_k, seq_len_k)

            token_position += len(positions_to_add)
            read_position += len(read_indices)
            write_position += len(write_indices)
            cumq_ptr += 1
            cumk_ptr += 1

        if self.input_ids[:, generated_tokens : generated_tokens + token_position].shape[1] != torch.tensor(input_ids, **self.tensor_metadata).shape[0]:
            print("Scheduler did not do his job? token position - generated_tokens is off")
        # now if sdpa or eager, create the attention mask!
        self.input_ids[:, :token_position].copy_(
            torch.tensor(input_ids, **self.tensor_metadata)
        )
        self.position_ids[:, :token_position].copy_(torch.tensor(position_ids, **self.tensor_metadata))
        self.write_index[:write_position].copy_(torch.tensor(write_index, **self.tensor_metadata))
        self.read_index[:read_position].copy_(torch.tensor(read_index, **self.tensor_metadata))
        self.cumulative_seqlens_q[:cumq_ptr].copy_(torch.tensor(cumulative_seqlens_q, **self.tensor_metadata))
        self.cumulative_seqlens_k[:cumk_ptr].copy_(torch.tensor(cumulative_seqlens_k, **self.tensor_metadata))
        self.logits_indices[:logits_to_track].copy_(torch.tensor(logits_indices, **self.tensor_metadata))
        for i in range(len(cumulative_seqlens_q) - 1):
            if cumulative_seqlens_q[i + 1] - cumulative_seqlens_q[i] < cumulative_seqlens_k[i+1] - cumulative_seqlens_k[i] and cumulative_seqlens_q[i + 1] - cumulative_seqlens_q[i] >= 1:
                diagonal = cumulative_seqlens_k[i+1] - (cumulative_seqlens_q[i + 1] - cumulative_seqlens_q[i]) + 1
                diagonal = diagonal - cumulative_seqlens_k[i]
            else:
                diagonal = 1
            self.attention_mask[...,cumulative_seqlens_q[i] : cumulative_seqlens_q[i+1],cumulative_seqlens_k[i] : cumulative_seqlens_k[i+1]].copy_(
                torch.triu(torch.ones(self.attention_mask[...,cumulative_seqlens_q[i] : cumulative_seqlens_q[i+1],cumulative_seqlens_k[i] : cumulative_seqlens_k[i+1]].shape), 
                           diagonal=diagonal) * torch.finfo(self.model_dtype).min
            )
        print("")

    @traced
    def _allocate_blocks_if_needed(self, state: RequestState, needed_slots: int):
        """Helper function to allocate blocks for a request."""
        current_blocks = len(state.allocated_blocks)
        needed_blocks = (needed_slots // self.cache.block_size) + 1
        if needed_blocks > current_blocks:
            blocks_needed = needed_blocks - current_blocks
            allocated = self.cache.allocate_blocks(blocks_needed, state.request_id)
            if not allocated:
                return False
            state.allocated_blocks.extend(allocated)
        return True

    @traced
    def update_batch(self):
        """Update request states based on generated tokens."""
        # TODO I ANM HERE NEED TO FIGURE LOGIC OPTIAML HERE
        token_idx = 0
        has_eos = self.output_ids == self.generation_config.eos_token_id
        is_max_len = self.cumulative_seqlens_q[1:] + 1 >= self.max_context_len
        to_remove = has_eos | is_max_len
        tokens_to_keep = torch.where(~to_remove & self.output_ids >= 0)[1]  # can get request ids with this
        out_tokens = self.output_ids[:, tokens_to_keep][0].cpu().detach().tolist()  # should be the only synch we do
        finished_request_ids = []
        for req_id in self.requests_to_process_next:
            if req_id not in self.active_requests:
                logger.warning(f"Request {req_id} not found in active requests during update.")
                continue

            state = self.active_requests[req_id]
            if len(state.remaining_prompt_ids) == 0:
                self._record_ttft_metric(state)
                state.status = "decoding"
                token = out_tokens[token_idx]
                token_idx += 1
                if state.update_with_token(token):
                    finished_request_ids.append(req_id)
                self._maybe_send_output(state, token)
            elif state.status == "prefilling_split":
                state.status = "split_pending_remainder"
            elif state.status == "split_pending_remainder":
                logger.warning(f"Request {req_id} in 'split_pending_remainder' state during update.")

        for req_id in finished_request_ids:
            if req_id in self.active_requests:
                self.cache.free_blocks(req_id)
                del self.active_requests[req_id]

    @traced
    def _record_ttft_metric(self, state: RequestState) -> None:
        """Record Time to First Token (TTFT)"""
        if not _has_opentelemetry or not state.created_time:
            return

        ttft_ms = (time.time() - state.created_time) * 1000.0

        try:
            self.ttft_histogram.record(ttft_ms)
            logger.debug(f"Recorded TTFT for request {state.request_id}: {ttft_ms:.2f}ms")
        except Exception as e:
            logger.warning(f"Failed to record TTFT metric: {e}")

    @traced
    def _record_batch_metrics(self, requests_in_batch: List[RequestState]) -> None:
        """Record metrics about the batch composition including decode/prefill ratio and batch fill percentage."""
        if not _has_opentelemetry or not requests_in_batch:
            return

        decode_tokens = 0
        prefill_tokens = 0

        for state in requests_in_batch:
            if state.status == "decoding":
                decode_tokens += 1
            elif state.status.startswith("prefilling"):
                prefill_tokens += len(state.prompt_ids)

        total_batch_tokens = decode_tokens + prefill_tokens

        try:
            if prefill_tokens > 0:
                self.prefill_tokens_counter.add(prefill_tokens)

            if decode_tokens > 0:
                self.decode_tokens_counter.add(decode_tokens)

            if prefill_tokens > 0:
                ratio = decode_tokens / prefill_tokens
            elif decode_tokens > 0:
                ratio = float("inf")
                self.decode_prefill_ratio_gauge.set(ratio)

            fill_percentage = (total_batch_tokens / self.max_batch_tokens) * 100.0
            self.batch_fill_percentage_histogram.record(fill_percentage)
            logger.debug(
                f"Batch metrics: {decode_tokens} decode tokens, {prefill_tokens} prefill tokens, "
                f"batch fill: {fill_percentage:.2f}% ({total_batch_tokens}/{self.max_batch_tokens})"
            )
        except Exception as e:
            logger.warning(f"Failed to record batch metrics: {e}")

    @traced
    def _maybe_send_output(self, state: RequestState, token: int):
        """Send output to the queue based on streaming mode and request state."""
        if self.streaming:
            state.next_token = token
            self.output_queue.put(state)
        elif state.status == "finished":
            self.output_queue.put(state)

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
            self._handle_request_error(error, state)
            self.cache.free_blocks(req_id)
            del self.active_requests[req_id]

        # Also fail any requests in the waiting queue
        while self.waiting_requests:
            state = self.waiting_requests.popleft()
            self._handle_request_error(error, state)


# Manager Class (User Interface)
class ContinuousBatchingManager:
    """Manager for handling continuous batching of generation requests.

    This class provides the user interface for submitting generation requests,
    retrieving results, and managing the background generation thread.
    """

    def __init__(
        self, model: GenerationMixin, generation_config: GenerationConfig, max_queue_size=0, streaming: bool = True 
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
        self.logit_processor = self.model._get_logits_processor(self.model.generation_config)
        self.do_sample = getattr(generation_config, "do_sample", False)
        self._setup_tracer()

    def _setup_tracer(self):
        """Initialize OpenTelemetry tracing if available."""
        if not _has_opentelemetry:
            return

        self.tracer = get_tracer("transformers.generation.continuous_batching_manager")

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

    @traced
    def add_request(
        self, input_ids: List[int], request_id: Optional[str] = None, max_new_tokens: Optional[int] = None
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

    def add_requests(self, inputs: List[List[int]], **kwargs):
        for i, input_ids in enumerate(inputs):
            # Assign a predictable request ID for ordering results later
            req_id = f"batch_req_{i}"
            self.add_request(input_ids, request_id=req_id, **kwargs)

    @traced
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
            result = self.output_queue.get(block=True, timeout=timeout)  # TODO Pop here rather?
            logger.debug(f"Retrieved result for request {result.request_id}")
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

    def _generation_step(self, batch_processor: ContinuousBatchProcessor):
        """Perform a single generation step. This is cuda graphed"""
        batch_data = batch_processor.get_model_kwargs()
        with torch.no_grad():
            import time
            start = time.time()
            model_outputs = self.model(**batch_data.__dict__)
            print(f"Model forward pass took {time.time() - start:.2f} seconds")
            # Copy is needed to avoid keeping a hanging ref
            logits = model_outputs.logits[:, batch_data.logits_indices]
            if self.log_prob_generation:
                batch_processor.output_probs.copy_(logits)  # TODO

            # b. Compute log probs -- get log probabilities from logits, process logits with processors (*e.g.*
            # `temperature`, ...), and add new logprobs to existing running logprobs scores.

            # TODO re activate once the unpute prepartion is fixed. Does not seem to work btw
            # probs = self.logit_processor(batch_data.input_ids, logits)
            probs = logits
            if self.do_sample:  # sample
                probs = nn.functional.softmax(probs, dim=-1)
                next_tokens = torch.multinomial(probs[0], num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            next_tokens = next_tokens[:,batch_data.logits_indices!=-1]
            batch_processor.output_ids.fill_(-1)
            batch_processor.output_ids[:, :next_tokens.shape[1]].copy_(next_tokens)

    @traced(span_name="generation_loop")
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
                batch_processor.prepare_next_batch()
                if torch.cuda.is_available():
                    if first:
                        self.warmup(batch_processor)
                        first = False
                    try:
                        self.graph.replay()
                    except Exception as e:
                        logger.error(f"Model forward pass failed: {e}", exc_info=True)
                        batch_processor.handle_batch_error(e)
                        continue
                else:
                    self._generation_step(batch_processor)

                batch_processor.update_batch()

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
                if batch_processor is not None:
                    batch_processor._handle_request_error(error, getattr(req_data, "request_id", None))
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

    @torch.inference_mode()
    def generate_batch(
        self,
        inputs: List[List[int]],
        generation_config: Optional[GenerationConfig] = None,
        progress_bar: bool = True,
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
            with tqdm(
                total=num_requests, disable=(not progress_bar), desc=f"Generating {num_requests} requests"
            ) as pbar:
                manager.add_requests(inputs, **kwargs)
                manager.start()  # we don't want to start before adding all requests
                finished_count = 0
                while finished_count < num_requests:
                    result = manager.get_result(timeout=1.0)
                    if result:
                        req_id = result.request_id
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
