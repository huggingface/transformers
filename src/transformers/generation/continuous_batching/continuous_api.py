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
import asyncio
import gc
import queue
import threading
from abc import abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from math import ceil
from time import perf_counter
from typing import Any

import torch
from torch import nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import ContinuousBatchingConfig, GenerationConfig
from ...modeling_flash_attention_utils import lazy_import_paged_flash_attention
from ...utils.generic import is_flash_attention_requested
from ...utils.logging import logging
from ...utils.metrics import ContinuousBatchProcessorMetrics, attach_tracer, traced
from ..logits_process import LogitsProcessorList
from .cache import PagedAttentionCache
from .cb_logits_processors import ContinuousBatchingLogitsProcessorList
from .input_outputs import ContinuousBatchingAsyncIOs, ContinuousBatchingIOs
from .offloading_manager import OffloadingManager
from .requests import GenerationOutput, RequestState, RequestStatus, logger
from .scheduler import SCHEDULER_MAPPING, FIFOScheduler, Scheduler
from .utils import WorkloadHints, attn_mask_is_needed, create_warmup_future_states, pad_to_interval, pad_to_pow2


"""
To enable cuda graphs, we need the dimensions of all tensors to be static, which is counter-intuitive for CB. In CB, as
generation goes on, there are two dimensions that change:
- the number of queries tokens (Q), which can vary from batch to batch
- the number of keys/values tokens (KV), which grows as the cache does

To solve this, we slice along those dimensions to fixed lengths. The size of the slices is controlled by interval sizes:
- q_padding_interval_size: the padding granularity for queries (in tokens)
- kv_padding_interval_size: the padding granularity for KV cache (in tokens)

For example, with q_padding_interval_size=64 and an actual query length of 100, we pad to 128 tokens.

Smaller intervals mean finer granularity and thus less padding, but more unique graph signatures. Since graphs take
memory and time to create, we use an LRU cache with a fixed size to limit memory usage. Good defaults:
- Q: 64 tokens gives ~4 graphs for max_batch_tokens=256, which is a good balance
- KV: 8192 tokens (256 blocks at block_size=32) gives reasonable granularity for large caches

The maximum number of cached graphs is controlled by max_cached_graphs (default 32), which uses LRU eviction.
All defaults are stored in ContinuousBatchingConfig.resolve_sentinel_values().
"""


# We cannot use `PreTrainedModel` for circular import reasons, so this helps keep track of the basic types
class ProtoPretrainedModel(nn.Module):
    config: PretrainedConfig
    dtype: torch.dtype
    device: torch.device

    @abstractmethod
    def set_attn_implementation(self, attn_implementation: str) -> None:
        pass

    @abstractmethod
    def _get_logits_processor(self, generation_config: GenerationConfig) -> LogitsProcessorList:
        pass


class OutputRouter:
    """Dedicated object for routing generation outputs to the right destination.

    When an async handler is registered for a request, the output is forwarded
    to that handler via ``call_soon_threadsafe``. Otherwise the output is placed
    on the shared ``output_queue``.
    """

    def __init__(self) -> None:
        self.output_queue = queue.Queue()
        self.result_handlers: dict[str, tuple[Callable, asyncio.AbstractEventLoop]] = {}
        self._lock = threading.Lock()

    def deliver(self, output: GenerationOutput) -> None:
        """Route a single output to its registered handler or the output_queue."""
        with self._lock:
            entry = self.result_handlers.get(output.request_id)
        if entry is not None:
            callback, loop = entry
            loop.call_soon_threadsafe(callback, output)
        else:
            self.output_queue.put(output)

    def deliver_batch(self, outputs: list[GenerationOutput]) -> None:
        """Route a batch of outputs, using a single ``call_soon_threadsafe`` to minimize cross-thread overhead.

        Outputs without a registered handler fall back to the shared ``output_queue``.
        """
        callbacks: list[tuple[Callable, GenerationOutput]] = []
        loop = None
        with self._lock:
            for output in outputs:
                entry = self.result_handlers.get(output.request_id)
                if entry is not None:
                    callback, loop = entry
                    callbacks.append((callback, output))
                else:
                    self.output_queue.put(output)
        if callbacks and loop is not None:

            def _run_batch(batch=callbacks):
                for cb, out in batch:
                    cb(out)

            loop.call_soon_threadsafe(_run_batch)


# Continuous Batch Processor (Internal Logic)
@attach_tracer()
class ContinuousBatchProcessor:
    inputs_and_outputs: ContinuousBatchingIOs | ContinuousBatchingAsyncIOs
    scheduler: Scheduler

    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        logit_processor: ContinuousBatchingLogitsProcessorList,
        input_queue: queue.Queue,
        output_router: OutputRouter,
        stop_event: threading.Event,
        model_device: torch.device,
        model_dtype: torch.dtype,
        scheduler: Scheduler,
    ) -> None:
        """Initialize the continuous batch processor.

        Args:
            cache: A [`PagedAttentionCache`] object
            config: The model configuration
            generation_config: The generation configuration
            continuous_batching_config: The continuous batching configuration
            logit_processor: The [`ContinuousBatchingLogitsProcessorList`] object used to process the logits.
            input_queue: Queue for incoming requests
            output_router: An [`OutputRouter`] object that routes outputs to handlers or the output queue.
            stop_event: Event to signal processing should stop
            model_device: Device for model inputs/outputs
            model_dtype: Data type for model inputs/outputs
            scheduler: The [`Scheduler`] to use
        """
        self.cache = cache
        self.config = config
        self.cb_config = continuous_batching_config
        self.logit_processor = logit_processor
        self.input_queue = input_queue
        self.output_router = output_router
        self.stop_event = stop_event
        self.model_device = model_device
        self.model_dtype = model_dtype
        self.scheduler = scheduler

        # Generation-related attributes
        self.do_sample = getattr(generation_config, "do_sample", True)
        self.return_logprobs = continuous_batching_config.return_logprobs

        # Retrieve the size of the sliding window if there is one
        self.sliding_window = 1 if getattr(config, "sliding_window", None) is None else config.sliding_window
        # Cuda graphs for the generation step
        self.q_padding_interval_size = self.cb_config.q_padding_interval_size
        self.kv_padding_interval_size = self.cb_config.kv_padding_interval_size
        self.use_cuda_graph_varlen, self.use_cuda_graph_decode = self.cb_config.get_cuda_graph_booleans()

        # Set up metrics collector
        self.max_batch_tokens = cache.max_batch_tokens
        self.metrics = ContinuousBatchProcessorMetrics(cache.max_batch_tokens)

        # If the user turned on the decode fast path (ie. using a block table), check if it is available
        self._ensure_decode_fast_path_is_available()  # this needs to happen before self.inputs_and_outputs is created

        # Resolve compile behavior
        self.cb_config.resolve_compile_configs(
            fallback_compile_config=getattr(generation_config, "compile_config", None),
            is_flash_attn=is_flash_attention_requested(config=config),
            decode_fast_path_available=self.cache.max_blocks_per_request > 0,
        )
        varlen_config, decode_config = self.cb_config.varlen_compile_config, self.cb_config.decode_compile_config

        # Compile the varlen path if config provided
        self._compiled_varlen = None
        if varlen_config is not None:
            self._compiled_varlen = torch.compile(self._forward_process_and_sample, **varlen_config.to_dict())

        # Compile the decode path if config provided
        self._compiled_decode = None
        if decode_config is not None:
            self._compiled_decode = torch.compile(self._forward_process_and_sample, **decode_config.to_dict())

        # Padding is turned on when either cuda graphs or compile is used
        use_cuda_graphs = self.use_cuda_graph_varlen or self.use_cuda_graph_decode
        self._pad_inputs = use_cuda_graphs or (varlen_config is not None or decode_config is not None)
        # Set up the graph pool. This allows all graphs to share the same memory pool, greatly saving memory.
        self.graph_pool = torch.cuda.graph_pool_handle() if use_cuda_graphs else None

        # Setup inputs and outputs
        io_kwargs = {
            "cache": cache,
            "config": config,
            "device": model_device,
            "model_dtype": model_dtype,
            "return_logprobs": self.return_logprobs,
            "logit_processor": self.logit_processor,
            "use_cuda_graph_varlen": self.use_cuda_graph_varlen,
        }
        self.use_async_batching = self.cb_config.use_async_batching

        if self.use_async_batching:
            # Since in async there are 2 IO pairs, there are also 2 graph buffers: we divide the max_cached_graphs by 2
            io_kwargs["max_graphs"] = ceil(self.cb_config.max_cached_graphs / 2)
            self.inputs_and_outputs = ContinuousBatchingAsyncIOs(**io_kwargs)
        else:
            io_kwargs["max_graphs"] = self.cb_config.max_cached_graphs
            self.inputs_and_outputs = ContinuousBatchingIOs(**io_kwargs)

        # Offloading manager: handles CPU offloading, soft reset, and restoration
        self.offloading_manager = OffloadingManager(
            cache=cache,
            scheduler=scheduler,
            cpu_offload_space_gib=continuous_batching_config.cpu_offload_space,
            safety_threshold=continuous_batching_config.cpu_offload_space_safety_threshold,
            compute_stream=self.inputs_and_outputs.compute_stream,
        )

    def __repr__(self) -> str:
        return (
            f"ContinuousBatchProcessor(input_queue={self.input_queue}, "
            f"active_requests={self.scheduler.active_requests}, waiting_requests={self.scheduler.waiting_requests})"
            + self.inputs_and_outputs.get_model_kwargs().__repr__()
        )

    def __del__(self) -> None:
        self.inputs_and_outputs = None  # clean up CUDA graphs in priority
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_decode_fast_path_is_available(self) -> None:
        """Ensures the decode fast path is available. If it is not, set the max blocks per request to 0. If it is
        available, and no user-provided max blocks per request, set it to the fallback default."""
        # First, set max blocks per request to 32 if it needs to be auto-inferred
        user_requested = self.cb_config.max_blocks_per_request is not None
        if not user_requested:
            self.cache.max_blocks_per_request = self.cb_config.fallback_max_blocks_per_request
            logger_warning = lambda x: x  # silences warning for user_requested=False  # noqa: E731
        else:
            logger_warning = logger.warning

        # Then, if the decode fast path is not turned off, check if it is available
        if self.cache.max_blocks_per_request != 0:
            # NOTE: block table should be available with FA2 and FA3, but there seems to be an issue with FA2 atm
            if is_flash_attention_requested(self.config, version=3):
                flash_attn_with_kvcache = lazy_import_paged_flash_attention(self.config._attn_implementation)[1]
                conditions = [
                    self.cache.num_sliding_attention_groups == 0,  # TODO: add support for sliding window layers
                    torch.cuda.is_available(),  # Block table is only supported on CUDA
                    flash_attn_with_kvcache is not None,  # The `flash_attn_with_kvcache` fn is needed
                ]
                # Throw a warning only if the decode fast path was requested by the user
                if not all(conditions):
                    logger_warning(
                        f"Although {self.cache.max_blocks_per_request = }, the decode fast path is not available "
                        f"because the one condition is not met: {conditions}."
                    )
                    self.cache.max_blocks_per_request = 0
            # Specific warning for attn implementation other than FA3
            else:
                logger_warning(
                    f"Although {self.cache.max_blocks_per_request = }, the decode fast path is not available "
                    f"because the attention implementation is not FA3. Got {self.config._attn_implementation = }."
                )
                self.cache.max_blocks_per_request = 0

    def reset(self) -> None:
        """Reset the batch processor for a new generation loop."""
        self.offloading_manager.reset()
        self.scheduler.reset()
        self.inputs_and_outputs.reset()
        self.cache.free_all_requests()
        self.metrics = ContinuousBatchProcessorMetrics(self.cache.max_batch_tokens)

    @traced
    def _get_new_requests(self) -> None:
        """Pull new requests from the input queue and add to waiting list."""
        while not self.input_queue.empty():
            try:
                state = self.input_queue.get_nowait()
                if state is None:  # Sentinel value
                    continue
                self.logit_processor.check_kwargs(state.logit_processor_kwargs)
                self.scheduler.add_waiting_request(state)

            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"Error processing new request: {e}", exc_info=True)
                state: RequestState = locals().get("state")
                if state is not None:
                    self._handle_request_error(e, state)

    @traced
    def _handle_request_error(self, error: Exception, state: RequestState) -> None:
        """Handle general request processing error."""
        state.status = RequestStatus.FAILED
        state.error = str(error)

        # Include any generated tokens if this is an active request
        if isinstance(state.request_id, str):
            state.generated_tokens = self.scheduler.get_active_request_static_outputs(state.request_id)
        else:
            state.generated_tokens = []

        self.metrics.record_request_completion(state.created_time, state.request_id)
        self.output_router.deliver(state.to_generation_output())

    def maybe_pad_inputs(self, num_q_tokens: int, max_kv_read: int, use_decode_fast_path: bool) -> tuple[int, int]:
        """Pads the inputs sizes for the next batch if it is needed. Often it is, for max performance."""
        if self._pad_inputs:
            # For varlen batches, we pad using interval sizes
            if not use_decode_fast_path:
                num_q_tokens = pad_to_interval(num_q_tokens, self.q_padding_interval_size, self.max_batch_tokens)
                max_kv_read = pad_to_interval(max_kv_read, self.kv_padding_interval_size, self.cache.num_pages)
            # For decode fast path batches, we pad using powers of 2 and use no KV
            else:
                num_q_tokens = pad_to_pow2(num_q_tokens, self.max_batch_tokens)
                max_kv_read = 0
        return num_q_tokens, max_kv_read

    @traced
    def prepare_next_batch(self) -> bool:
        """Prepare tensors and metadata for the next model forward pass. Returns True if there are requests to process,
        False otherwise."""

        # Get new requests from the queue, stop if there are no pending requests
        self._get_new_requests()
        cancelled_states = self.scheduler.clear_cancelled_requests()
        # Also free CPU-offloaded cache for cancelled states. This is CPU-only, so it isn't batched like D2H transfers
        for state in cancelled_states:
            self.offloading_manager.free_request_cpu_cache(state)
        if not self.scheduler.has_pending_requests():
            return False
        self.metrics.record_queue_metrics(len(self.scheduler.active_requests), len(self.scheduler.waiting_requests))

        # Schedule the next batch of requests
        requests_in_batch, use_decode_fast_path, num_q_tokens, max_kv_read = self.scheduler.schedule_batch(
            self.max_batch_tokens, self.cache.num_pages
        )
        # If requests_in_batch is None, it means we need to offload some requests if possible
        if requests_in_batch is None:
            if len(self.scheduler.active_requests) > 1:
                self.offloading_manager.offload_one_request()
                return False
            else:
                raise RuntimeError("No requests can be scheduled and no request can be offloaded.")
        # If it's an empty list, it means we have no requests to process
        if not requests_in_batch:
            return False

        # Restore any CPU-offloaded requests that were just scheduled
        self.offloading_manager.restore_scheduled_requests(requests_in_batch)

        # Otherwise, we can continue with the non-empty batch and log in the dimensions before padding
        self.metrics.record_batch_metrics(requests_in_batch)
        logger.debug(
            f"Scheduled: {len(requests_in_batch)}, Waiting: {len(self.scheduler.waiting_requests)}, "
            f"Active: {len(self.scheduler.active_requests)}. cum Q: {num_q_tokens}. "
            f"cum KV: {max_kv_read}, free blocks: {self.cache.get_num_free_blocks()}"
        )

        # If inputs are static sized, eg. for compile, we find the padded sizes of the queries and keys/values
        num_q_tokens, max_kv_read = self.maybe_pad_inputs(num_q_tokens, max_kv_read, use_decode_fast_path)

        self.inputs_and_outputs.prepare_batch_tensors(
            requests_in_batch, self.logit_processor, use_decode_fast_path, num_q_tokens, max_kv_read
        )
        self.metrics.record_kv_cache_memory_metrics(self.cache)
        return True

    @traced
    def update_batch(self) -> None:
        """Update request states based on generated tokens."""
        requests_in_batch, new_tokens, logprobs = self.inputs_and_outputs.prepare_batch_update()
        current_logits_index = 0
        pending_outputs = []
        for future_state in requests_in_batch:
            state = future_state.state
            # Early return if the request was finished or offloaded between scheduling and update (async mode)
            if state.status in (RequestStatus.FINISHED, RequestStatus.PENDING):
                if self.use_async_batching:
                    # Skip this request, but still consume its token from new_tokens if it had one
                    if future_state.has_new_token:
                        current_logits_index += 1
                    continue
                raise RuntimeError(f"Tried to update {state.status.name} request {state.request_id} in sync mode.")
            # If the request has a new token, it means prefill has already ended or just finished
            if future_state.has_new_token:
                # If there is just one temporary token, it means prefill just ended
                if state.generated_len() == 0:
                    self.metrics.record_ttft_metric(state.created_time, state.request_id)
                    state.status = RequestStatus.DECODING

                token = new_tokens[current_logits_index]
                logprob = logprobs[current_logits_index] if logprobs is not None else None
                current_logits_index += 1

                # Update the request and stop if it is complete
                is_finished = state.update_and_check_completion(token, logprob)
                # We mark the completed blocks as such
                self.cache.mark_shareable_blocks_as_complete(state, future_state.complete_blocks)
                if is_finished:
                    self.metrics.record_request_completion(state.created_time, state.request_id)
                    self.scheduler.finish_request(state.request_id)
                    self.scheduler.block_new_requests = False
                if state.streaming or state.status == RequestStatus.FINISHED:
                    pending_outputs.append(state.to_generation_output())
            #  Otherwise, the request is still prefilling, but the prefill has been split
            elif state.status == RequestStatus.PREFILLING:
                self.cache.mark_shareable_blocks_as_complete(state, future_state.complete_blocks)

        if pending_outputs:
            self.output_router.deliver_batch(pending_outputs)

        # If some requests need to be forked, we do it now
        copy_source, copy_destination = [], []
        while self.scheduler._requests_to_fork:
            # Get the number of children and reset it so it's not forked again
            state_to_fork = self.scheduler._requests_to_fork.pop()
            num_children = state_to_fork.num_children
            state_to_fork.num_children = 0
            # Create the new request and add them to the scheduler
            new_request_ids = [f"{state_to_fork.request_id}__child#{i}" for i in range(num_children)]
            for new_request_id in new_request_ids:
                self.scheduler.active_requests[new_request_id] = state_to_fork.fork(new_request_id)
            # Fork the cache
            copy_src, copy_dst = self.cache.fork_request(state_to_fork.request_id, new_request_ids)
            copy_source.extend(copy_src)
            copy_destination.extend(copy_dst)
            # FIXME: if fork cant be done, create a new pending request without forking instead of crashing everything

        # The copy induced by the fork is done in one go (if it's even needed)
        if copy_source:
            # FIXME: this will avoid any race condition, but it can cause issue when using async batching with a sliding
            # window model. Fix will be fixed in a PR in the near future (tempfix, v5.3)
            compute_stream = self.inputs_and_outputs.compute_stream
            maybe_stream = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
            with maybe_stream:
                self.cache.copy_cache(copy_source, copy_destination)

    @traced
    def has_pending_requests(self) -> bool:
        """Check if there are any active or waiting requests."""
        return self.scheduler.has_pending_requests()

    @traced
    def handle_batch_error(self, error):
        """Handle errors during batch processing."""
        failed_future_states = self.inputs_and_outputs.prepare_batch_update()[0]
        for future_state in failed_future_states:
            self._handle_request_error(error, future_state.state)
            self.scheduler.finish_request(future_state.state.request_id)

    @traced
    def fail_all_requests(self, error: Exception) -> None:
        """Fail all active requests with the given error."""

        requests = list(self.scheduler.active_requests.values())
        for state in requests:
            self._handle_request_error(error, state)
            self.scheduler.finish_request(state.request_id)

        # Also fail any requests in the waiting queue
        self.offloading_manager.free_all_waiting_cpu_caches()
        for req_id in list(self.scheduler.waiting_requests.keys()):
            state = self.scheduler.waiting_requests.pop(req_id)
            self._handle_request_error(error, state)

        # Clear the ordering queue
        self.scheduler.waiting_requests_order.clear()

    @traced
    @torch.no_grad()
    def _generation_step(self, model: nn.Module) -> None:
        """Perform a single generation step."""

        # Retrieve the model kwargs with or without padding
        batch_data = self.inputs_and_outputs.get_model_kwargs(use_padding=self._pad_inputs)
        carry_over_ids, prev_output_ids, output_ids = self.inputs_and_outputs.get_cb_kwargs()
        compute_stream = self.inputs_and_outputs.compute_stream

        # Get the appropriate forward function (compiled or not, based on current path)
        if self.inputs_and_outputs.use_block_table:
            forward_fn = self._forward_process_and_sample if self._compiled_decode is None else self._compiled_decode
            use_cuda_graph = self.use_cuda_graph_decode
        else:
            forward_fn = self._forward_process_and_sample if self._compiled_varlen is None else self._compiled_varlen
            use_cuda_graph = self.use_cuda_graph_varlen

        # If we are not using cuda graphs, we perform the generation step and return
        if not use_cuda_graph:
            maybe_stream = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
            with maybe_stream:
                forward_fn(model, batch_data, carry_over_ids, prev_output_ids, output_ids)

        # Otherwise, we use create or replay the graph (cuda is available in this path)
        else:
            graph = self.inputs_and_outputs.get_graph()
            # Case: the graph already exists, so we replay it
            if graph is not None:
                with torch.cuda.stream(compute_stream):
                    graph.replay()
            # Otherwise, the graph does not exist, so we create it
            else:
                args = (model, batch_data, carry_over_ids, prev_output_ids, output_ids)
                self.capture_graph(forward_fn, compute_stream, *args)

        # In any case, we transfer the outputs to the host
        self.inputs_and_outputs.retrieve_device_outputs()

    def capture_graph(self, forward_fn: Any, compute_stream: torch.cuda.Stream, *args) -> None:
        # Warmup (ensures the right result is computed before capturing the graph)
        with torch.cuda.stream(compute_stream):
            forward_fn(*args)
        # Capture
        graph = torch.cuda.CUDAGraph()
        # Continuous batching can run multiple manager threads concurrently in one process, but PyTorch's
        # default capture mode ("global") errors on CUDA actions from other threads. This means capture can be
        # invalidated even when each manager uses a different device. To avoid this, we use "thread_local" mode.
        with torch.cuda.graph(graph, stream=compute_stream, pool=self.graph_pool, capture_error_mode="thread_local"):
            forward_fn(*args)
        # Store
        self.inputs_and_outputs.set_graph(graph)

    @traced
    def _forward_process_and_sample(
        self,
        model: nn.Module,
        batch_data: dict,
        carry_over_ids: torch.Tensor,
        prev_output_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> None:
        """This function performs the forward pass, logits processing, and sampling; which are broken down into smaller
        function to be easier to trace with OpenTelemetry."""
        self.inputs_and_outputs.carry_over_tokens(batch_data["input_ids"], carry_over_ids, prev_output_ids)
        logits = self._model_forward(model, batch_data).float()  # convert to fp32 to match generate
        scores = self._process_logit(batch_data, logits) if self.logit_processor.do_processing else logits
        self._sample(scores, batch_data["logits_indices"], output_ids)

    @traced(span_name="model_forward")
    def _model_forward(self, model: nn.Module, batch_data: dict) -> torch.Tensor:
        return model(**batch_data).logits

    @traced(span_name="logit_processing")
    def _process_logit(self, batch_data: dict, logits: torch.Tensor) -> torch.Tensor:
        # Handle shape compatibility: logit processors expect 2D tensors [batch_size, vocab_size]
        # but continuous batching always produces 3D tensors [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        logits_2d = logits.view(batch_size * seq_len, vocab_size)
        input_ids_2d = batch_data["input_ids"].view(batch_size * seq_len)
        # Process with 2D tensors
        processed_logits_2d = self.logit_processor(input_ids_2d, logits_2d, batch_data["logits_processor_args"])
        # Reshape back to 3D
        return processed_logits_2d.view(batch_size, seq_len, vocab_size)

    @traced(span_name="sampling")
    def _sample(self, scores: torch.Tensor, logits_indices: torch.Tensor, output_ids: torch.Tensor) -> None:
        # Apply softmax if we are sampling or if we are generating log probabilities
        if self.do_sample or self.return_logprobs:
            probs = nn.functional.softmax(scores[0], dim=-1)  # shape [seq_len, vocab_size]
        # Otherwise just remove the batch size dimension, which is always 1
        else:
            probs = scores.squeeze(0)  # shape [seq_len, vocab_size]

        # Retrieve next tokens through sampling or argmax
        if self.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1)  # shape [seq_len, 1]
        else:
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True)  # shape [seq_len, 1]

        # Maybe retrieve log probabilities
        if self.return_logprobs:
            per_token_probs = probs.gather(dim=1, index=next_tokens).squeeze(-1)
            logprobs = per_token_probs.log()  # shape [seq_len]
        # And always remove the extra dimension for the gather
        next_tokens = next_tokens.squeeze(-1)  # shape [seq_len]

        # Get seq_len dimension to slice the logits indices
        tokens = next_tokens.size(0)
        # Shuffle the next tokens to match the order of the batch's requests
        indices = logits_indices[:tokens]
        next_tokens = next_tokens[indices]
        # Copy the next tokens and maybe their logprobs to the static output tensor
        output_ids[0, :tokens].copy_(next_tokens)
        if self.return_logprobs:
            # Shuffle the logprobs the same way as the next tokens
            logprobs = logprobs[indices]
            # In order to match the dtype of output_ids, we cast the fp32 logprobs as int32 without changing the
            # underlying data. It's just a trick to use the same storage for both tensors.
            output_ids[1, :tokens].copy_(logprobs.view(dtype=torch.int32))

    @torch.inference_mode()
    def warmup(self, model: nn.Module) -> None:
        """Pre-capture CUDA graphs (or trigger compile warmup) for varlen and decode paths. In async mode, both IO
        pairs are warmed up since each has its own graph buffer and static tensors. The varlen path is warmed up at
        the largest possible `(q, kv)` sizes so subsequent captures fit inside it without growing the pool."""

        if not self._pad_inputs:
            logger.info("CUDA graphs and compile are disabled, skipping warmup.")
            return None

        num_query_tokens = self.max_batch_tokens
        num_pages = self.cache.num_blocks * self.cache.block_size
        num_cache_tokens = num_pages - num_query_tokens
        compute_stream = self.inputs_and_outputs.compute_stream

        # In async mode, each IO pair has its own graph buffer and static tensors, so we warm up both
        num_io_pairs = 2 if self.use_async_batching else 1

        for pair_idx in range(num_io_pairs):
            if self.use_async_batching:
                self.inputs_and_outputs.current_pair = pair_idx
                logger.info(f"Warming up IO pair {pair_idx + 1}/2...")

            # --- Varlen path ---
            padded_q, padded_kv = self.maybe_pad_inputs(
                num_q_tokens=num_query_tokens,
                max_kv_read=num_cache_tokens + num_query_tokens,
                use_decode_fast_path=False,
            )
            logger.info(f"Warming up varlen path ({padded_q} Q tokens, {padded_kv} KV tokens)...")

            future_states = create_warmup_future_states(
                1, RequestStatus.PREFILLING, num_query_tokens, num_cache_tokens, self.cache
            )
            try:
                start = perf_counter()
                self.inputs_and_outputs.prepare_batch_tensors(
                    future_states, self.logit_processor, False, padded_q, padded_kv - padded_q
                )
                batch_data = self.inputs_and_outputs.get_model_kwargs(use_padding=True)
                carry_over_ids, prev_output_ids, output_ids = self.inputs_and_outputs.get_cb_kwargs()
                forward_fn = self._compiled_varlen or self._forward_process_and_sample
                forward_fn_args = (model, batch_data, carry_over_ids, prev_output_ids, output_ids)
                if self.use_cuda_graph_varlen:
                    self.capture_graph(forward_fn, compute_stream, *forward_fn_args)
                else:
                    with torch.cuda.stream(compute_stream):
                        forward_fn(*forward_fn_args)
                logger.info(f"Varlen warmup completed in {perf_counter() - start:.2f}s")
            except Exception as e:
                logger.warning(f"Failed to warm up varlen path: {e}. Graph pool may fragment and OOM under load.")
            finally:
                for fs in future_states:
                    self.cache.free_blocks(fs.state.request_id)

            # Exit here if the decode fast path is not available
            if self.cache.max_blocks_per_request == 0:
                continue

            # --- Decode fast path ---
            logger.info("Warming up decode fast path...")
            decode_graphs = 0
            start = perf_counter()

            num_requests = 1
            while True:
                future_states = create_warmup_future_states(
                    num_requests, RequestStatus.DECODING, 1, self.cache.block_size, self.cache
                )
                if not future_states:
                    break
                try:
                    padded_q, _ = self.maybe_pad_inputs(
                        num_q_tokens=num_requests, max_kv_read=0, use_decode_fast_path=True
                    )
                    self.inputs_and_outputs.prepare_batch_tensors(
                        future_states, self.logit_processor, True, padded_q, 0
                    )
                    batch_data = self.inputs_and_outputs.get_model_kwargs(use_padding=True)
                    carry_over_ids, prev_output_ids, output_ids = self.inputs_and_outputs.get_cb_kwargs()
                    forward_fn = self._compiled_decode or self._forward_process_and_sample
                    forward_fn_args = (model, batch_data, carry_over_ids, prev_output_ids, output_ids)
                    if self.use_cuda_graph_decode:
                        self.capture_graph(forward_fn, compute_stream, *forward_fn_args)
                    else:
                        with torch.cuda.stream(compute_stream):
                            forward_fn(*forward_fn_args)
                    decode_graphs += 1
                except Exception as e:
                    logger.warning(f"Failed to warm up decode path for {num_requests} requests: {e}")
                finally:
                    for fs in future_states:
                        self.cache.free_blocks(fs.state.request_id)
                if num_requests >= self.max_batch_tokens:
                    break
                num_requests = min(2 * num_requests, self.max_batch_tokens)
            logger.info(f"Decode warmup completed ({decode_graphs} graphs) in {perf_counter() - start:.2f}s.")

        # If using async batching, reset to pair 0 for the generation loop
        if self.use_async_batching:
            self.inputs_and_outputs.current_pair = 0


# Manager Class (User Interface)
@attach_tracer()
class ContinuousBatchingManager:
    """Manager for handling continuous batching of generation requests. It provides a user interface for submitting
    generation requests, retrieving results, and managing the background generation thread. This class should not be
    created directly, but through one of the following entry points (all methods of the `ContinuousMixin` mixin):
    - `init_continuous_batching`
    - `continuous_batching_context_manager`
    - `generate_batch`
    """

    def __init__(
        self,
        model: ProtoPretrainedModel,
        generation_config: GenerationConfig,
        continuous_batching_config: ContinuousBatchingConfig,
    ) -> None:
        """Initialize the continuous batching manager.

        Args:
            model: The language model for generation
            generation_config: Configuration for generation parameters
            continuous_batching_config: Configuration for continuous batching parameters
        """
        # Reload paged version of the attention implementation if necessary
        if "paged|" not in model.config._attn_implementation:
            model.set_attn_implementation(f"paged|{model.config._attn_implementation}")

        # Internal arguments
        self.model = model.eval()
        self.generation_config = generation_config
        self.continuous_batching_config = continuous_batching_config
        self.warmed_up = False  # Set to True after warmup is completed. Useful for persistent managers.
        # This is an approximation until the cache is created: it will infer the correct value in cache.__init__
        self._use_prefix_sharing = self.continuous_batching_config.allow_block_sharing

        self.input_queue = queue.Queue(maxsize=self.continuous_batching_config.max_queue_size)
        self._has_new_requests = threading.Event()
        self.output_router = OutputRouter()
        self.stop_event = threading.Event()
        self.batch_processor: ContinuousBatchProcessor | None = None
        self._generation_thread = None
        self._request_counter = 0
        self._request_lock = threading.Lock()

        # Generation config related arguments
        num_return_sequences = getattr(generation_config, "num_return_sequences", None)
        self.num_return_sequences = num_return_sequences if num_return_sequences is not None else 1

        self.logit_processor = ContinuousBatchingLogitsProcessorList(
            logits_processor=self.model._get_logits_processor(generation_config),
            per_request_processors=self.continuous_batching_config.per_request_processors,
            drop_unsupported_processors=self.continuous_batching_config.drop_unsupported_processors,
        )

        # Cuda graph behavior is determined below using either user-specified arguments or heuristics
        is_attn_mask_needed = attn_mask_is_needed(self.model.config)
        self.continuous_batching_config.decide_use_cuda_graphs(
            compile_config=getattr(generation_config, "compile_config", None),
            is_attn_mask_needed=is_attn_mask_needed,
        )
        # Same for asynchronous batching behavior
        self.use_async_batching = self.continuous_batching_config.decide_use_async_batching(is_attn_mask_needed)

        # Resolve default parameters for Q and KV interval sizes, and max cached graphs. If one of those parameters is
        # not specified (set to 0) then we use the default value and change its value in the config.
        self.continuous_batching_config.resolve_sentinel_values()
        self.q_padding_interval_size = self.continuous_batching_config.q_padding_interval_size
        self.kv_padding_interval_size = self.continuous_batching_config.kv_padding_interval_size
        self.max_cached_graphs = self.continuous_batching_config.max_cached_graphs

    @traced
    def start(self) -> None:
        """Start the background generation thread."""
        if self._generation_thread is not None and self._generation_thread.is_alive():
            logger.warning("Manager thread is already running.")
            return
        self.stop_event.clear()
        self._generation_thread = threading.Thread(target=self._run_generation_loop)
        self._generation_thread.start()

    def is_running(self) -> bool:
        """Check if the background generation thread is running."""
        return self._generation_thread is not None and self._generation_thread.is_alive()

    def warmup(self) -> None:
        """Pre-capture CUDA graphs for varlen and decode paths by running dummy batches. Initializes the batch
        processor if not already done."""
        if self.batch_processor is None:
            self.batch_processor = self._create_batch_processor()
        self.batch_processor.warmup(self.model)
        self.warmed_up = True

    # NOTE: don't forget to update `continuous_batching_context_manager` when changing this method's definition
    def stop(self, block: bool = True, timeout: float | None = None, keep_for_next_session: bool = False) -> None:
        """Signal the background thread to stop.

        Args:
            block: Whether to wait for the thread to stop
            timeout: Maximum time to wait for the thread to stop
            keep_for_next_session: Whether to cache this on the model for future use
        """
        if self.batch_processor is None:
            logger.warning("\nBatch processor was not initialized.")
        elif self.batch_processor.cache.use_prefix_sharing:
            logger.info(
                f"\nPrefix sharing was on. Total prefix length: {self.batch_processor.cache._total_prefix_length}"
            )

        if self._generation_thread is None:
            suffix = " Hence the unstarted manager will not be kept for next session." if keep_for_next_session else ""
            logger.warning("Manager not started." + suffix)
            return

        stop_trigger_time = perf_counter()
        if not self.stop_event.is_set():
            self.stop_event.set()
            logger.info("Stopping continuous batching manager...")

        if block:
            self.join(stop_trigger_time, timeout)

        # If the manager is not being kept for next session, we clear the batch processor
        if not keep_for_next_session:
            self.batch_processor = None
        # Otherwise, we keep the batch processor and cache the manager as a model attribute
        else:
            logger.info("Continuous batching manager will be kept for next session.")
            self.model._cached_continuous_batching_manager = self
        # In all cases, a little cleanup is good
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def join(self, stop_trigger_time: float, timeout: float | None = None) -> None:
        """Wait for the background thread to finish.

        Args:
            timeout: Maximum time to wait for the thread to stop
        """
        if self._generation_thread is not None:
            self._generation_thread.join(timeout=timeout)
            if self._generation_thread.is_alive():
                logger.warning(f"Generation thread did not exit after join timeout ({timeout}).")
            else:
                end = perf_counter()
                logger.info(f"Continuous Batching Manager stopped after {end - stop_trigger_time:.2f}s.")
                self._generation_thread = None

    def add_request(
        self,
        input_ids: list[int],
        request_id: str | None = None,
        max_new_tokens: int | None = None,
        streaming: bool = False,
        record_timestamps: bool = False,
        eos_token_id: int | list[int] | None = None,
        **logit_processor_kwargs: Any,
    ) -> str:
        """Add a new generation request to the queue.

        Args:
            input_ids: Input token IDs to use as prompt
            request_id: Optional custom request ID (auto-generated if None)
            max_new_tokens: Maximum number of new tokens to generate
            streaming: Whether to stream tokens as they're generated
            record_timestamps: Whether to record timestamps for each generated token
            eos_token_id: End-of-sequence token ID(s)
            logit_processor_kwargs: Keyword arguments for the logits processor.

        Returns:
            str: The request ID
        """
        if request_id is None:
            with self._request_lock:
                request_id = f"req_{self._request_counter}"
                self._request_counter += 1

        max_new_tokens = self.generation_config.max_new_tokens if max_new_tokens is None else max_new_tokens
        eos_token_id = self.generation_config.eos_token_id if eos_token_id is None else eos_token_id

        # NOTE: do we want to handle a case when the user wants token ids returned instead of decoded text?
        state = RequestState(
            request_id=request_id,
            initial_tokens=list(input_ids),
            num_children=self.num_return_sequences - 1,
            record_timestamps=record_timestamps,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            streaming=streaming,
            logit_processor_kwargs=logit_processor_kwargs,
        )

        # Use block=True with timeout to handle backpressure if queue is full
        self.input_queue.put(state, block=True, timeout=10)
        self._has_new_requests.set()
        return request_id

    def add_requests(
        self,
        inputs: list[list[int]],
        max_new_tokens: int | None = None,
        streaming: bool = False,
        record_timestamps: bool = False,
        **logit_processor_kwargs: Any,
    ) -> None:
        # Infer the request ids of all incoming requests
        with self._request_lock:
            request_ids = [f"req_{i}" for i in range(self._request_counter, self._request_counter + len(inputs))]
            self._request_counter += len(inputs)
        # If there is prefix sharing, we sort the inputs to maximize cache hits but keep the order of the requests
        ids_and_inputs = list(zip(request_ids, inputs))
        if self._use_prefix_sharing:
            ids_and_inputs = sorted(ids_and_inputs, key=lambda x: x[1], reverse=True)
        # Look for an EOS token ID in the generation config and then in the model config. If no EOS is found, we set it
        # to -1 to avoid looking for it in each add_request call
        eos_token_id = self.generation_config.eos_token_id
        eos_token_id = self.model.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_token_id = -1 if eos_token_id is None else eos_token_id
        # Add requests in order
        for request_id, input_ids in ids_and_inputs:
            self.add_request(
                input_ids=input_ids,
                request_id=request_id,
                max_new_tokens=max_new_tokens,
                streaming=streaming,
                record_timestamps=record_timestamps,
                eos_token_id=eos_token_id,
                **logit_processor_kwargs,
            )

    def cancel_request(self, request_id: str) -> None:
        """Cancel a request by its ID.

        Args:
            request_id: The ID of the request to cancel
        """
        if self.batch_processor is not None:
            self.batch_processor.scheduler.set_request_cancellation(request_id)

    # TODO:handle benchmarking properly when updating / fixing the requeue logic
    def get_result(self, request_id: str | None = None, timeout: float | None = None) -> GenerationOutput | None:
        """Retrieve one result from the output queue.

        Args:
            request_id: If set, only return results matching this ID (others are requeued).
            timeout: Maximum time to wait for a result.

        Returns:
            Optional[GenerationOutput]: The result data or None if timeout.
        """
        if self._generation_thread is None and self.output_router.output_queue.empty():
            return None
        try:
            result = self.output_router.output_queue.get(block=True, timeout=timeout)
            if request_id is not None and result.request_id != request_id:
                self.output_router.output_queue.put(result)
                return None
            return result
        except queue.Empty:
            return None

    def __iter__(self):
        """Iterate over results as they become available."""
        while self._generation_thread is not None and self._generation_thread.is_alive():
            result = self.get_result(timeout=0.1)
            if result is not None:
                yield result

    def request_id_iter(self, request_id: str) -> Generator[GenerationOutput]:
        """Iterate over results matching a specific request id (blocking).

        Uses the shared output queue with requeue. For high-concurrency serving,
        use :meth:`register_result_handler` instead.
        """
        while self._generation_thread is not None and self._generation_thread.is_alive():
            result = self.get_result(request_id=request_id, timeout=0.1)
            if result is not None:
                yield result
                if result.is_finished():
                    return

    def register_result_handler(self, request_id: str, callback: Callable) -> None:
        """Register a callback for result delivery (streaming or non-streaming).

        The callback is invoked on the event loop via ``call_soon_threadsafe``
        each time a result is produced for this request. For streaming requests,
        this happens on every token; for non-streaming, only on completion.

        The handler is automatically cleaned up when the request finishes.

        Args:
            request_id (`str`): The request ID to receive outputs for.
            callback (`callable`): Called with a ``GenerationOutput`` for each result.
        """
        loop = asyncio.get_running_loop()

        def _auto_cleanup(result):
            callback(result)
            if result.is_finished():
                with self.output_router._lock:
                    self.output_router.result_handlers.pop(request_id, None)

        with self.output_router._lock:
            self.output_router.result_handlers[request_id] = (_auto_cleanup, loop)

    @traced
    def _generation_step(self) -> None:
        """Perform a single generation step. This is mostly cuda graphed"""
        if self.batch_processor is None:
            raise RuntimeError("Tried to perform a generation step before the batch processor was initialized.")
        self.batch_processor._generation_step(self.model)

    def _create_batch_processor(self) -> ContinuousBatchProcessor:
        # Resolve max_memory_percent now that we know whether any logit processors are active.
        self.continuous_batching_config.resolve_max_memory_percent(self.logit_processor.do_processing)
        # Create the PagedAttentionCache
        paged_attention_cache = PagedAttentionCache(
            self.model.config,
            self.continuous_batching_config,
            self.model.device,
            self.model.dtype,
            tp_size=getattr(self.model, "_tp_size", None),  # Use model's actual TP setting
        )
        self._use_prefix_sharing = paged_attention_cache.use_prefix_sharing  # update the approximation

        # Create the scheduler
        scheduler_type = self.continuous_batching_config.scheduler_type
        scheduler = SCHEDULER_MAPPING.get(scheduler_type, None)
        if scheduler is None:
            logger.warning(f"Scheduler '{scheduler_type}' not found. Defaulting to FIFO.")
            scheduler = FIFOScheduler

        # Create the batch processor
        batch_processor = ContinuousBatchProcessor(
            cache=paged_attention_cache,
            config=self.model.config,
            generation_config=self.generation_config,
            continuous_batching_config=self.continuous_batching_config,
            logit_processor=self.logit_processor,
            input_queue=self.input_queue,
            output_router=self.output_router,
            stop_event=self.stop_event,
            model_device=self.model.device,
            model_dtype=self.model.dtype,
            scheduler=scheduler(paged_attention_cache),
        )
        return batch_processor

    @torch.inference_mode()
    def _run_generation_loop(self) -> None:
        """Main processing loop running in the background thread."""
        try:
            # Try to retrieve an already initialized batch processor
            batch_processor = getattr(self, "batch_processor", None)
            # If the batch processor already exists, we just reset it for a new generation loop
            if isinstance(batch_processor, ContinuousBatchProcessor):
                batch_processor.reset()
            # Otherwise, we create a new batch processor
            else:
                batch_processor = self._create_batch_processor()

            # Start the generation loop
            self.batch_processor = batch_processor
            self.current_batch = 0

            # If using the async API, we bootstrap the first batch w/out update
            if batch_processor.use_async_batching:
                if not batch_processor.prepare_next_batch():
                    raise RuntimeError("Failed to bootstrap the first batch.")
                self._generation_step()
                self.current_batch += 1

            while (not self.stop_event.is_set()) or batch_processor.has_pending_requests():
                self._inner_generation_loop(batch_processor)
                self.current_batch += 1

            # In async mode, the last batch's results are still in flight - process them now
            # We need to switch back to the pair that has the last batch's D2H pending
            if isinstance(batch_processor.inputs_and_outputs, ContinuousBatchingAsyncIOs):
                batch_processor.inputs_and_outputs.current_pair = 1 - batch_processor.inputs_and_outputs.current_pair
                batch_processor.update_batch()

        except Exception as e:
            logger.error(f"Error in generation loop: {e}", exc_info=True)
            self._handle_critical_error(e, batch_processor)
        finally:
            logger.info("Generation loop finished.")

    @traced(span_name="generation_loop")
    def _inner_generation_loop(self, batch_processor: ContinuousBatchProcessor) -> None:
        # Loop body ends if there is no requests in the batch
        if not batch_processor.prepare_next_batch():
            # Wait for new requests instead of busy-spinning.
            self._has_new_requests.wait(timeout=0.1)
            self._has_new_requests.clear()
            return
        self._generation_step()
        batch_processor.update_batch()

    @traced
    def _handle_critical_error(self, error: Exception, batch_processor: ContinuousBatchProcessor | None) -> None:
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


class ContinuousMixin:
    """Mixin class for models to add continuous batching capabilities. Continuous batching has three entry points:
    - `init_continuous_batching`, which is the actual entry point for continuous batching
    - `continuous_batching_context_manager`, which itself is a wrapper around `init_continuous_batching`
    - `generate_batch`, which is really a wrapper around `continuous_batching_context_manager`

    They are defined in this order. Any change made to any of those three entry points should be reflected in the other
    two.
    """

    generation_config: GenerationConfig

    @torch.inference_mode()
    def init_continuous_batching(
        self,
        generation_config: GenerationConfig | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        workload_hints: WorkloadHints | None = None,
        **deprecated_kwargs,
    ) -> ContinuousBatchingManager:
        """Initialize a manager for continuous batching inference.

        Args:
            generation_config: An optional generation configuration, which may contain a CompileConfig object
            continuous_batching_config: An optional continuous batching configuration
            workload_hints: Optional WorkloadHints to help the continuous batching manager make better decisions for
                default values
            **deprecated_kwargs: Deprecated arguments that are now passed in the continuous_batching_config. Those are:
                max_queue_size, q_padding_interval_size, kv_padding_interval_size, allow_block_sharing,
                use_async_batching, max_cached_graphs
        Returns:
            `ContinuousBatchingManager`: The manager instance to add requests and retrieve results.
        """
        # Mandatory attributes
        if not hasattr(self, "config") or not hasattr(self, "device") or not hasattr(self, "dtype"):
            raise AttributeError("Model must have 'config', 'device', and 'dtype' attributes.")

        # If a persistent manager is found we return it
        cached_manager = getattr(self, "_cached_continuous_batching_manager", None)
        if isinstance(cached_manager, ContinuousBatchingManager):
            logger.info(
                "Cached continuous batching manager found: it will be re-used instead of creating a new one. If you"
                " want to create a new manager, you should call `destroy_cached_continuous_batching_manager` first."
            )
            return cached_manager

        # Retrieve generation config
        gen_config = generation_config if generation_config is not None else self.generation_config
        if gen_config is None:
            raise ValueError("A GenerationConfig must be provided or set in the model.")
        # Warn about EOS
        if gen_config.eos_token_id is None:
            logger.warning("`eos_token_id` not set in GenerationConfig. Setting to -1 (disabled).")
            gen_config.eos_token_id = -1

        # Retrieve continuous batching config, or create it if none is provided
        if continuous_batching_config is None:
            if isinstance(getattr(gen_config, "continuous_batching_config", None), ContinuousBatchingConfig):
                continuous_batching_config = gen_config.continuous_batching_config
            else:
                continuous_batching_config = ContinuousBatchingConfig()
        continuous_batching_config.account_for_cb_deprecated_arguments(**deprecated_kwargs)
        if workload_hints is not None:
            workload_hints.resolve_using_hints(continuous_batching_config)

        # Create and return the manager
        return ContinuousBatchingManager(
            model=self, generation_config=gen_config, continuous_batching_config=continuous_batching_config
        )

    def destroy_cached_continuous_batching_manager(self) -> None:
        """Destroy the cached continuous batching manager and free GPU resources."""
        cached_manager = getattr(self, "_cached_continuous_batching_manager", None)
        if isinstance(cached_manager, ContinuousBatchingManager):
            cached_manager.stop(block=True, timeout=None, keep_for_next_session=False)
            delattr(self, "_cached_continuous_batching_manager")

    @contextmanager
    @torch.inference_mode()
    def continuous_batching_context_manager(
        self,
        generation_config: GenerationConfig | None = None,
        block: bool = True,
        timeout: float | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        persistent_manager: bool = False,
        warmup: bool = True,
        workload_hints: WorkloadHints | None = None,
        **deprecated_kwargs,
    ) -> Generator[ContinuousBatchingManager]:
        """A context manager to safely use the continuous batching manager. Arguments are similar to the ones of
        `init_continuous_batching`, except for:
            - block: whether to block the thread when stopping the manager. Default is True.
            - timeout: maximum time to wait for the thread to stop. Default is None (no timeout).
            - warmup: whether to pre-capture CUDA graphs at the largest sizes before running. Default is True.
        """
        manager = self.init_continuous_batching(
            generation_config=generation_config,
            continuous_batching_config=continuous_batching_config,
            workload_hints=workload_hints,
            **deprecated_kwargs,
        )
        if warmup and not manager.warmed_up:
            # Warmup is long (~30 sec): best to signal the user it's happening than let them think the manager is stuck
            logger.warning("Warming up for continuous batching...")
            start = perf_counter()
            manager.warmup()
            logger.warning(f"Warming up completed in {perf_counter() - start:.2f}s.")
        manager.start()
        try:
            yield manager
        finally:
            # This is a dummy log needed for the logs of stop to show. It won't show.
            logger.debug("Continuous batching loop finished")
            manager.stop(block=block, timeout=timeout, keep_for_next_session=persistent_manager)

    # TODO: support streaming
    @traced
    @torch.inference_mode()
    def generate_batch(
        self,
        inputs: list[list[int]],
        generation_config: GenerationConfig | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        record_timestamps: bool = False,
        progress_bar: bool = True,
        persistent_manager: bool = False,
        warmup: bool = True,
        **kwargs,
    ) -> dict[str, GenerationOutput]:
        """Generate sequences for a batch of prompts using continuous batching.

        Args:
            inputs: List of input token sequences (prompts)
            generation_config: Optional generation configuration
            continuous_batching_config: Optional continuous batching configuration
            record_timestamps: If set to true, the requests will have a timestamp for each token generated
            progress_bar: If set to true, a progress bar will be displayed
            persistent_manager: whether to persist the manager after the generation is finished. Default is False.
            warmup: whether to pre-capture CUDA graphs before processing requests. Default is True.
            **kwargs: Additional generation parameters. Only max_new_tokens is used, but other deprecated arguments
                are extracted and passed to the continuous_batching_config object.
        Returns:
            `dict[str, GenerationOutput]`: a dictionary of request ids to GenerationOutput objects
        """
        # If no input are provided, return an empty dictionary
        if not inputs:
            return {}

        # If the logger level is less than DEBUG, disable the progress bar
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.warning("Progress bar is disabled when logger level is less than DEBUG")
            progress_bar = False

        # Extract deprecated arguments from regular kwargs (deprecated in v5.3). These args are now expected in the
        # continuous_batching_config object.
        deprecated_kwargs = {}
        deprecated_keys = [
            "q_padding_interval_size",
            "kv_padding_interval_size",
            "allow_block_sharing",
            "use_async_batching",
            "max_cached_graphs",
            "max_queue_size",
        ]
        for depr_key in deprecated_keys:
            if depr_key in kwargs:
                deprecated_kwargs[depr_key] = kwargs.pop(depr_key)

        # Compute the total number of requests
        gen_cfg = self.generation_config if generation_config is None else generation_config
        num_return_sequences = gen_cfg.num_return_sequences if gen_cfg.num_return_sequences is not None else 1
        num_requests = len(inputs) * num_return_sequences

        # Extract max_new_tokens from kwargs because it's the only expected kwarg
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        max_new_tokens = gen_cfg.max_new_tokens if max_new_tokens is None else max_new_tokens

        # Compute workload hints
        workload_hints = WorkloadHints(
            max_prompt_length=max(len(input_ids) for input_ids in inputs),
            max_generated_length=max_new_tokens if max_new_tokens is not None else 0,
        )

        # Prepare context managers for the main loop
        manager_cm = self.continuous_batching_context_manager(
            generation_config=generation_config,
            continuous_batching_config=continuous_batching_config,
            block=True,
            timeout=5,
            persistent_manager=persistent_manager,
            warmup=warmup,
            workload_hints=workload_hints,
            **deprecated_kwargs,
        )
        logging_cm = logging_redirect_tqdm([logger])
        pbar_cm = tqdm(
            total=num_requests,
            disable=(not progress_bar),
            desc=f"Solving {num_requests} requests",
            unit="request",
        )

        # Main loop
        results = {}
        finished_count = 0
        with manager_cm as manager, logging_cm, pbar_cm as pbar:
            try:
                manager.add_requests(inputs=inputs, max_new_tokens=max_new_tokens, record_timestamps=record_timestamps)
                while finished_count < num_requests:
                    result = manager.get_result(timeout=1)
                    if result:
                        req_id = result.request_id
                        if result.is_finished():
                            results[req_id] = result
                            finished_count += 1
                            pbar.update(1)
                    elif not manager.is_running():
                        logger.error("Generation thread terminated unexpectedly.")
                        # This helps get some information in stdout
                        print("Returning results of generate_batch despite unexpected termination.")
                        break

            except Exception as e:
                logger.error(f"Error during batch generation: {e}", exc_info=True)

        # Re-order requests to match the order of the inputs
        reordered_results = {}
        for i in range(len(inputs)):
            # We cannot guarantee generation success for all requests, so check if the request is in the results
            result = results.get(f"req_{i}")
            if result is not None:
                reordered_results[f"req_{i}"] = result
            else:
                logger.error(f"Request req_{i} not found in results.")
        return reordered_results
