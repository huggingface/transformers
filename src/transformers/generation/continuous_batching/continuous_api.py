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
import queue
import threading
from abc import abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from math import ceil
from time import perf_counter

import torch
from torch import nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import CompileConfig, GenerationConfig
from ...generation.logits_process import LogitsProcessorList
from ...utils.logging import logging
from ...utils.metrics import ContinuousBatchProcessorMetrics, attach_tracer, traced
from .cache import PagedAttentionCache
from .input_ouputs import ContinuousBatchingIOs
from .requests import GenerationOutput, RequestState, RequestStatus, logger
from .scheduler import SCHEDULER_MAPPING, FIFOScheduler, Scheduler


"""
To enable cuda graphs, we need the dimensions of all tensors to be static, which is counter-intuitive for CB. In CB, as
generation goes on, there are two dimensions that change:
- the number of queries tokens (Q), which can vary from batch to batch
- the number of keys/values tokens (KV), which grows as the cache does

To solve this, we slice along those dimensions to fixed lengths. The size of the slices is controlled by the variables
num_x_padding_intervals: NUM_X_PADDING_INTERVALS means that we create at most NUM_X_PADDING_INTERVALS graphs for the X
dimension. So if the maximum number of queries tokens is 1000, and NUM_Q_PADDING_INTERVALS is 4, we will slice the
number of queries token by intervals of 1000 / 4 = 250 tokens, ie. to 250, 500, 750 or 1000 queries tokens.

Smaller slices means more granularity and thus less padding. But since each graph takes up space on the GPU and time to
create, we don't want to many graphs. And since the size of the KV dimension is the number of queries tokens plus the
number of tokens cached, dimension of KV is usually much larger than the dimension of Q. So we have more granularity
for the KV dimension than the query dimension.

This variable used to be called NUM_X_CUDA_GRAPHS, but we renamed it to NUM_X_PADDING_INTERVALS because it is used for
padding in the case of cuda graphs AND torch.compile.
"""
NUM_Q_PADDING_INTERVALS = 4
NUM_KV_PADDING_INTERVALS = 8


def pad_by_intervals(size: int, max_value: int, nb_intervals: int) -> int:
    """Return the smallest multiple of (max_value) // (nb_intervals) greater than (size)."""
    interval_size = max_value // nb_intervals
    if interval_size == 0:
        return max_value
    padded = ceil(size / interval_size) * interval_size if size > 0 else interval_size
    return min(padded, max_value)


def attn_mask_is_needed(config: PretrainedConfig) -> bool:
    """Checks if attention mask is needed for the given (config)."""
    return config._attn_implementation in ["paged|eager", "paged|sdpa"]


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
        manual_eviction: bool,
        use_cuda_graph: bool,
        q_padding_intervals: int,
        kv_padding_intervals: int,
    ) -> None:
        """Initialize the continuous batch processor.

        Args:
            cache: A [`PagedAttentionCache`] object
            config: The model configuration
            generation_config: The generation configuration
            input_queue: Queue for incoming requests
            output_queue: Queue for outgoing results
            stop_event: Event to signal processing should stop
            model_device: Device for model inputs/outputs
            model_dtype: Data type for model inputs/outputs
            scheduler: The [`Scheduler`] to use
            manual_eviction: Whether to manually evict blocks from the cache
            use_cuda_graph: Whether to use cuda graphs or not during CB. Check the docstring at the top of the file for
                more details.
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
        self.manual_eviction = manual_eviction

        # Retrieve the size of the sliding window if there is one
        self.sliding_window = 1 if getattr(config, "sliding_window", None) is None else config.sliding_window
        # Accumulator for batch scheduling
        self.requests_in_batch: list[RequestState] = []
        # Cuda graphs for the generation step
        self.q_padding_intervals = q_padding_intervals
        self.kv_padding_intervals = kv_padding_intervals
        self._graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] | None = {} if use_cuda_graph else None
        # Compile-related arguments
        self.compile_config: CompileConfig | None = getattr(generation_config, "compile_config", None)
        self._forward_process_and_sample_is_compiled = False

        self._pad_inputs = use_cuda_graph or (self.compile_config is not None and not self.compile_config.dynamic)

        # Set up metrics collector
        self.max_batch_tokens = cache.max_batch_tokens
        self.metrics = ContinuousBatchProcessorMetrics(cache.max_batch_tokens)

        # Setup inputs and outputs
        self.inputs_and_outputs = ContinuousBatchingIOs(cache, config, model_device, model_dtype)

    def __repr__(self) -> str:
        return (
            f"ContinuousBatchProcessor(input_queue={self.input_queue}, output_queue={self.output_queue}, "
            f"active_requests={self.scheduler.active_requests}, waiting_requests={self.scheduler.waiting_requests})"
            + self.inputs_and_outputs.get_model_kwargs().__repr__()
        )

    @traced
    def _get_new_requests(self) -> None:
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
                state: RequestState = locals().get("state")  # type:ignore
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
        self.output_queue.put(state.to_generation_output())

    # TODO: there should be a way to choose the offloading policy: biggest request, oldest request, etc.
    # Including a policy to not allow offloading and crashing the generation
    def soft_reset_one_request(self) -> None:
        """Soft resets one active request by removing it from active requests and re-adding it to the waiting queue.

        The generated tokens are kept as part of the new request's initial prompt. When `block_new_requests` is False,
        the oldest request is offloaded; when True, the newest request is offloaded. This method also sets
        `block_new_requests` to True to prevent infinite loops of offloading and re-scheduling requests.
        """
        # The offloaded request is the newest (resp. oldest) if block_new_requests is True (resp. False)
        if self.scheduler.block_new_requests:
            request_id, state = self.scheduler.active_requests.popitem()
        else:
            request_id, state = next(iter(self.scheduler.active_requests.items()))
        logger.info(
            f"Soft resetting request {request_id} with {len(state.initial_tokens)} initial tokens and "
            f"{len(state.generated_tokens)} generated tokens"
        )
        # Create a copy of the offloaded request keeping the generated tokens as addition to the initial prompt
        new_state = state.create_equivalent_initial_request()
        # Actual offloading of the request
        self.scheduler.finish_request(request_id, evict_from_cache=True)
        self.scheduler.add_waiting_request(new_state)
        # This flag blocks any new requests from being scheduled until one request is finished. This ensures that we
        # don't enter an offload / schedule loop
        self.scheduler.block_new_requests = True

    @traced
    def prepare_next_batch(self) -> bool:
        """Prepare tensors and metadata for the next model forward pass. Returns True if there are requests to process,
        False otherwise."""

        # Get new requests from the queue, stop if there are no pending requests
        self._get_new_requests()
        self.scheduler.clear_cancelled_requests()
        if not self.scheduler.has_pending_requests():
            return False
        self.metrics.record_queue_metrics(len(self.scheduler.active_requests), len(self.scheduler.waiting_requests))

        # Schedule the next batch of requests, stop if there are no requests in the batch
        requests_in_batch = self.scheduler.schedule_batch(self.max_batch_tokens, self.cache.num_pages)

        # If requests_in_batch is None, it means we need to offload some requests if possible
        if requests_in_batch is None:
            if len(self.scheduler.active_requests) > 1:
                self.soft_reset_one_request()
                return False
            else:
                raise RuntimeError("No requests can be scheduled and no request can be offloaded.")
        # If it's an empty list, it means we have no requests to process
        self.requests_in_batch = requests_in_batch
        if not self.requests_in_batch:
            return False

        # Otherwise, we can continue with the non-empty batch
        self.metrics.record_batch_metrics(self.requests_in_batch)
        self.inputs_and_outputs.prepare_batch_tensors(requests_in_batch)

        # Record the memory metrics of the KV cache
        self.metrics.record_kv_cache_memory_metrics(self.cache)
        if logger.isEnabledFor(logging.DEBUG):
            cumulative_seqlens_q = self.inputs_and_outputs.cumulative_seqlens_q
            cumulative_seqlens_k = self.inputs_and_outputs.cumulative_seqlens_k
            ck = max(cumulative_seqlens_k[layer_type][-1] for layer_type in cumulative_seqlens_k)
            logger.debug(
                f"Scheduled: {len(self.requests_in_batch)}, Waiting: {len(self.scheduler.waiting_requests)}, "
                f"Active: {len(self.scheduler.active_requests)}. cum Q: {cumulative_seqlens_q[-1]}. "
                f"cum KV: {ck}, free blocks: {self.cache.get_num_free_blocks()}"
            )
        return True

    @traced
    def _maybe_send_output(self, state: RequestState) -> None:
        """Send output to the queue based on streaming mode and request state."""
        if state.streaming or state.status == RequestStatus.FINISHED:
            self.output_queue.put(state.to_generation_output())

    @traced
    def update_batch(self) -> None:
        """Update request states based on generated tokens."""
        new_tokens = self.inputs_and_outputs.output_ids[: len(self.requests_in_batch)].tolist()
        current_logits_index = 0
        for state in self.requests_in_batch:
            # If the request has no remaining prompt ids, it means prefill has already ended or just finished
            if len(state.remaining_prefill_tokens) == 0:
                # If there is just one temporary token, it means prefill just ended
                if state.generated_len() == 1:
                    self.metrics.record_ttft_metric(state.created_time, state.request_id)
                    state.status = RequestStatus.DECODING

                token = new_tokens[current_logits_index]
                state.tokens_to_process = [token]
                current_logits_index += 1

                # Update the request and stop if it is complete
                is_finished = state.update_and_check_completion(token)
                # We mark the completed blocks as such
                self.cache.mark_shareable_blocks_as_complete(state)
                if is_finished:
                    self.metrics.record_request_completion(state.created_time, state.request_id)
                    self.scheduler.finish_request(state.request_id, evict_from_cache=(not self.manual_eviction))
                    self.scheduler.block_new_requests = False
                self._maybe_send_output(state)
            #  Otherwise, the request is still prefilling, but the prefill has been split
            elif state.status == RequestStatus.PREFILLING_SPLIT:
                self.cache.mark_shareable_blocks_as_complete(state)
                state.status = RequestStatus.SPLIT_PENDING_REMAINDER
            else:
                raise ValueError(f"Request {state.request_id} is in an unexpected state: {state.status}")

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
            self.cache.copy_cache(copy_source, copy_destination)

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
    def fail_all_requests(self, error: Exception) -> None:
        """Fail all active requests with the given error.

        Args:
            error: The error to report in the failure message
        """

        requests = list(self.scheduler.active_requests.values())
        for state in requests:
            self._handle_request_error(error, state)
            self.scheduler.finish_request(state.request_id)

        # Also fail any requests in the waiting queue
        for req_id in list(self.scheduler.waiting_requests.keys()):
            state = self.scheduler.waiting_requests.pop(req_id)
            self._handle_request_error(error, state)

        # Clear the ordering queue
        self.scheduler.waiting_requests_order.clear()

    @traced
    @torch.no_grad()
    def _generation_step(self, model: nn.Module, logit_processor: LogitsProcessorList, do_sample: bool) -> None:
        """Perform a single generation step."""

        # If a compile config is specified, we compile the forward pass once in a wrapper
        if self.compile_config is not None and not self._forward_process_and_sample_is_compiled:
            self._forward_process_and_sample = torch.compile(
                self._forward_process_and_sample,
                fullgraph=self.compile_config.fullgraph,
                mode=self.compile_config.mode,
                dynamic=self.compile_config.dynamic,
                backend=self.compile_config.backend,
                options=self.compile_config.options,
            )
            self._forward_process_and_sample_is_compiled = True

        # If inputs are static sized, we find the padded sizes of the queries and keys/values
        if self._pad_inputs:
            actual_query_length = self.inputs_and_outputs.actual_query_length
            actual_index_sizes = self.inputs_and_outputs.actual_index_sizes
            padded_q = pad_by_intervals(actual_query_length, self.max_batch_tokens, self.q_padding_intervals)
            max_read_index_size = max(actual_index_sizes[i][0] for i in range(self.cache.num_groups))
            # The space planned for query tokens will be added later, so we remove it from the space planned for KV
            padded_read_index_size = pad_by_intervals(max_read_index_size, self.cache.num_pages, self.kv_padding_intervals)
        else:
            padded_q, padded_read_index_size = 0, 0
        # Retrieve the model kwargs with or without padding
        batch_data = self.inputs_and_outputs.get_model_kwargs(padded_q, padded_read_index_size)

        # If we are not using cuda graphs, we perform the generation step and return
        if self._graphs is None:
            self._forward_process_and_sample(model, batch_data, logit_processor, do_sample)
            return None

        # If we have a graph that fits, we replay it
        graph = self._graphs.get((padded_q, padded_read_index_size))
        if graph is not None:
            graph.replay()
            return None

        # Otherwise, we need to create it
        logger.info(f"Creating graph for {(padded_q, padded_read_index_size) = }")
        stream = torch.cuda.Stream(device=model.device)
        stream.wait_stream(torch.cuda.current_stream())
        # Warmup
        with torch.cuda.stream(stream):
            self._forward_process_and_sample(model, batch_data, logit_processor, do_sample)
        torch.cuda.current_stream().wait_stream(stream)
        # Catpure
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self._forward_process_and_sample(model, batch_data, logit_processor, do_sample)
        self._graphs[(padded_q, padded_read_index_size)] = graph

    @traced
    def _forward_process_and_sample(
        self, model: nn.Module, batch_data: dict, logit_processor: LogitsProcessorList, do_sample: bool
    ) -> None:
        """This function performs the forward pass, logits processing, and sampling; which are broken down into smaller
        function to be easier to trace with OpenTelemetry."""
        logits = self._model_forward(model, batch_data)
        # if self.log_prob_generation:    batch_processor.output_probs.copy_(logits)  # TODO
        probs = self._process_logit(batch_data, logits, logit_processor)
        self._sample(probs, batch_data, do_sample)

    @traced(span_name="model_forward")
    def _model_forward(self, model: nn.Module, batch_data: dict) -> torch.Tensor:
        return model(**batch_data).logits

    @traced(span_name="logit_processing")
    def _process_logit(
        self, batch_data: dict, logits: torch.Tensor, logit_processor: LogitsProcessorList
    ) -> torch.Tensor:
        # Pass continuous batching context to logits processor if it supports it.
        if hasattr(logit_processor, "set_continuous_batching_context"):
            logit_processor.set_continuous_batching_context(batch_data["logits_indices"], batch_data["cu_seq_lens_q"])
        # Handle shape compatibility: logit processors expect 2D tensors [batch_size, vocab_size]
        # but continuous batching always produces 3D tensors [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        # NOTE: to be an exact match with generate, we should also convert logits2d to float32 here, but it's not needed in practice
        logits_2d = logits.view(batch_size * seq_len, vocab_size)
        input_ids_2d = batch_data["input_ids"].view(batch_size * seq_len)
        # Process with 2D tensors#
        processed_logits_2d = logit_processor(input_ids_2d, logits_2d)  # type: ignore[arg-type]
        # Reshape back to 3D
        return processed_logits_2d.view(batch_size, seq_len, vocab_size)

    @traced(span_name="sampling")
    def _sample(self, probs: torch.Tensor, batch_data: dict, do_sample: bool) -> None:
        if do_sample:
            probs = nn.functional.softmax(probs, dim=-1)
            # probs[0] has shape [seq_len, vocab_size], multinomial returns [seq_len, 1]
            next_tokens = torch.multinomial(probs[0], num_samples=1).squeeze(-1)  # Now [seq_len]
        else:
            next_tokens = torch.argmax(probs, dim=-1)  # shape is [1, seq_len]
            next_tokens = next_tokens.squeeze(0)  # shape is [seq_len]
        tokens = next_tokens.size(0)  # Get seq_len dimension
        #
        indices = batch_data["logits_indices"][:tokens]
        next_tokens = next_tokens[indices]
        self.inputs_and_outputs.output_ids[:tokens].copy_(next_tokens)


# Manager Class (User Interface)
@attach_tracer()
class ContinuousBatchingManager:
    """Manager for handling continuous batching of generation requests.

    This class provides the user interface for submitting generation requests,
    retrieving results, and managing the background generation thread.
    """

    def __init__(
        self,
        model: ProtoPretrainedModel,
        generation_config: GenerationConfig,
        manual_eviction: bool = False,
        max_queue_size: int = 0,
        num_q_padding_intervals: int = 0,
        num_kv_padding_intervals: int = 0,
        allow_block_sharing: bool = True,
    ) -> None:
        """Initialize the continuous batching manager.

        Args:
            model: The language model for generation
            generation_config: Configuration for generation parameters
            max_queue_size: Maximum size of the request queue (0 = unlimited)
            num_q_padding_intervals: (optional) Number of intervals used to pad the query dimension
            num_kv_padding_intervals: (optional) Number of intervals used to pad the keys/values dimension
            allow_block_sharing: (optional) Whether to allow block sharing if the model has some full attention layers
        """
        # Reload paged version of the attention implementation if necessary
        if "paged|" not in model.config._attn_implementation:
            model.set_attn_implementation(f"paged|{model.config._attn_implementation}")

        # Internal arguments
        self.model = model.eval()
        self.manual_eviction = manual_eviction
        self._allow_block_sharing = allow_block_sharing
        self._use_prefix_sharing = allow_block_sharing  # approximation until the cache is created

        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.batch_processor: ContinuousBatchProcessor | None = None
        self._generation_thread = None
        self._request_counter = 0
        self._request_lock = threading.Lock()

        # Generation config related arguments
        generation_config = model.generation_config if generation_config is None else generation_config
        self.generation_config = generation_config
        self.log_prob_generation = getattr(generation_config, "log_prob_generation", False)
        self.do_sample = getattr(generation_config, "do_sample", True)
        self.logit_processor: LogitsProcessorList = self.model._get_logits_processor(generation_config)
        num_return_sequences = getattr(generation_config, "num_return_sequences", None)
        self.num_return_sequences = num_return_sequences if num_return_sequences is not None else 1

        # self.model.generation_config.top_p = None NOTE: figure out why this was here

        # Cuda graph behavior is determined below using either user-specified arguments or heuristics
        self.use_cuda_graph = self._decide_use_cuda_graphs(
            use_cuda_graph=getattr(generation_config, "use_cuda_graph", None),
            num_q_padding_intervals=num_q_padding_intervals,
            num_kv_padding_intervals=num_kv_padding_intervals,
            compile_config=getattr(generation_config, "compile_config", None),
        )

        # We set the number of padding intervals for Q and KV
        self.q_padding_intervals = num_q_padding_intervals if num_q_padding_intervals > 0 else NUM_Q_PADDING_INTERVALS
        self.kv_padding_intervals = (
            num_kv_padding_intervals if num_kv_padding_intervals > 0 else NUM_KV_PADDING_INTERVALS
        )

        # Log probability generation is not supported yet (TODO)
        if self.log_prob_generation:
            raise NotImplementedError("log_prob_generation is not supported yet")

    def _decide_use_cuda_graphs(
        self,
        use_cuda_graph: bool | None,
        num_q_padding_intervals: int,
        num_kv_padding_intervals: int,
        compile_config: CompileConfig | None,
    ) -> bool:
        """Returns whether or not to use cuda graphs for continuous batching, depending on the following criteria:
        - (use_cuda_graph) which is the user choice
        - (num_q_padding_intervals) or (num_kv_padding_intervals) which is used to pad inputs: if it was specified by
            the user, it's probable they want to use cuda graphs so inputs need to be padded
        - (compile_config): if compile is on, turn on cuda graphs unless the compile mode uses its own cudagraphs
        If none of the above criteria are met, we use a default heuristic based on the attention implementation: we turn
        on cuda graphs if and only if no attention mask is needed.
        """
        # If use_cuda_graph is specified, we follow the user's choice
        if use_cuda_graph is not None:
            return use_cuda_graph
        # If a number of padding intervals was specified for either Q or KV, we activate cuda graphs
        if num_q_padding_intervals > 0 or num_kv_padding_intervals > 0:
            return True
        # If a compile config was found, turn off cuda graphs if the compile config already uses them
        if compile_config is not None:
            options = torch._inductor.list_mode_options().get(compile_config.mode, compile_config.options)
            compile_uses_cudagraphs = options.get("triton.cudagraphs", False)
            if compile_uses_cudagraphs:
                logger.warning(
                    f"Compile config {compile_config.mode = } uses cudagraphs, which usually does not work well with "
                    "continuous batching. We recommend using mode 'default' or 'max-autotune-no-cudagraphs' instead."
                )
            return not compile_uses_cudagraphs  # TODO: should this also match the dynamic shapes?
        # Otherwise we have a default heuristic based on the attention implementation:
        # attention implementations where an attention mask is needed suffer a lot more from the padding associated
        # with cuda graphs, so default is to turn cuda graphs off for those implementations
        use_cuda_graph = not attn_mask_is_needed(self.model.config)
        logger.warning(
            f"No behavior specified for use_cuda_graph, defaulting to {use_cuda_graph = } because "
            f"{self.model.config._attn_implementation = }. If you want to save memory, turn off cuda graphs, but "
            "they tend to improve performances by a lot."
        )
        return use_cuda_graph

    @traced
    def start(self) -> None:
        """Start the background generation thread."""
        if self._generation_thread is not None and self._generation_thread.is_alive():
            logger.warning("Manager thread is already running.")
            return

        self._generation_thread = threading.Thread(target=self._run_generation_loop)
        self._generation_thread.start()

    def is_running(self) -> bool:
        """Check if the background generation thread is running."""
        return self._generation_thread is not None and self._generation_thread.is_alive()

    # NOTE: don't forget to update `continuous_batching_context_manager` when changing this method's definition
    def stop(self, block: bool = True, timeout: float | None = None) -> None:
        """Signal the background thread to stop.

        Args:
            block: Whether to wait for the thread to stop
            timeout: Maximum time to wait for the thread to stop
        """
        if self.batch_processor is None:
            logger.warning("\nBatch processor was not initialized.")
        else:
            if self.batch_processor.cache.use_prefix_sharing:
                logger.info(
                    f"\nPrefix sharing was on. Total prefix length: {self.batch_processor.cache._total_prefix_length}"
                )

        if self._generation_thread is None:
            logger.warning("Manager not started.")
            return

        stop_trigger_time = perf_counter()
        if not self.stop_event.is_set():
            self.stop_event.set()
            logger.info("Stopping continuous batching manager...")

        if block:
            self.join(stop_trigger_time, timeout)

        self.batch_processor = None

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

        # NOTE: do we want to handle a case when the user wants token ids returned instead of decoded text?
        state = RequestState(
            request_id=request_id,
            initial_tokens=list(input_ids),
            num_children=self.num_return_sequences - 1,
            record_timestamps=record_timestamps,
            tokens_to_process=list(input_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=self.generation_config.eos_token_id,
            streaming=streaming,
        )

        # Use block=True with timeout to handle backpressure if queue is full
        self.input_queue.put(state, block=True, timeout=10)  # XXX: pass timeout as fn arg?
        return request_id

    def add_requests(
        self,
        inputs: list[list[int]],
        max_new_tokens: int | None = None,
        streaming: bool = False,
        record_timestamps: bool = False,
    ) -> None:
        # If there is prefix sharing, we sort the inputs to maximize cache hits
        if self._use_prefix_sharing:
            inputs = sorted(inputs, reverse=True)
        # Add requests in order
        for input_ids in inputs:
            self.add_request(
                input_ids, max_new_tokens=max_new_tokens, streaming=streaming, record_timestamps=record_timestamps
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
            timeout: Maximum time to wait for a result

        Returns:
            Optional[GenerationOutput]: The result data or None if timeout
        """
        if self._generation_thread is None and self.output_queue.empty():
            return None
        try:
            result = self.output_queue.get(block=True, timeout=timeout)
            # NOTE: requeue logic here
            if request_id is not None and result.request_id != request_id:
                self.output_queue.put(result)
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

    # FIXME: stop iteration when request status is finished?
    def request_id_iter(self, request_id: str) -> Generator[GenerationOutput]:
        """Iterate over results matching a specific request id as they become available."""
        request_cancelled = False
        while self._generation_thread is not None and self._generation_thread.is_alive() and not request_cancelled:
            result = self.get_result(request_id=request_id, timeout=0.1)
            if result is not None:
                yield result
            if self.batch_processor is not None:
                request_cancelled = self.batch_processor.scheduler.request_is_cancelled(request_id)

    @traced
    def _generation_step(self) -> None:
        """Perform a single generation step. This is mostly cuda graphed"""
        if self.batch_processor is None:
            raise RuntimeError("Tried to perform a generation step before the batch processor was initialized.")
        self.batch_processor._generation_step(self.model, self.logit_processor, self.do_sample)

    def _run_generation_loop(self) -> None:
        """Main processing loop running in the background thread."""
        batch_processor: ContinuousBatchProcessor | None = None
        try:
            t0 = perf_counter()
            paged_attention_cache = PagedAttentionCache(
                self.model.config,
                self.generation_config,
                self.model.device,
                self.model.dtype,
                tp_size=getattr(self.model, "_tp_size", None),  # Use model's actual TP setting
                allow_block_sharing=self._allow_block_sharing,
            )
            self._use_prefix_sharing = paged_attention_cache.use_prefix_sharing  # update the approximation
            logger.debug(f"PagedAttentionCache created in {perf_counter() - t0} seconds")

            scheduler = None
            if hasattr(self.generation_config, "scheduler"):
                scheduler = SCHEDULER_MAPPING.get(self.generation_config.scheduler, None)
                if scheduler is None:
                    logger.warning(f"Scheduler '{scheduler}' not found. Defaulting to FIFO.")
                    scheduler = FIFOScheduler
            else:
                # Default to fifo
                scheduler = FIFOScheduler

            t1 = perf_counter()
            batch_processor = ContinuousBatchProcessor(
                cache=paged_attention_cache,
                config=self.model.config,
                generation_config=self.generation_config,
                input_queue=self.input_queue,
                output_queue=self.output_queue,
                stop_event=self.stop_event,
                model_device=self.model.device,
                model_dtype=self.model.dtype,
                scheduler=scheduler(paged_attention_cache, self.manual_eviction),
                manual_eviction=self.manual_eviction,
                use_cuda_graph=self.use_cuda_graph,
                q_padding_intervals=self.q_padding_intervals,
                kv_padding_intervals=self.kv_padding_intervals,
            )
            self.batch_processor = batch_processor
            self.current_batch = 0
            logger.debug(f"batch_processor created in {perf_counter() - t1} seconds")
            while (not self.stop_event.is_set()) or batch_processor.has_pending_requests():
                self._inner_generation_loop(batch_processor)
                self.current_batch += 1

        except Exception as e:
            logger.error(f"Error in generation loop: {e}", exc_info=True)
            self._handle_critical_error(e, batch_processor)
        finally:
            logger.info("Generation loop finished.")

    @traced(span_name="generation_loop")
    def _inner_generation_loop(self, batch_processor: ContinuousBatchProcessor) -> None:
        # Pre-loop synchronization
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Loop body ends if there is no requests in the batch
        if not batch_processor.prepare_next_batch():
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

    @traced
    def evict_request_from_cache(self, request_id: str) -> None:
        """Evict a request from the cache. It is assumed that the request is already finished."""
        if not self.manual_eviction:
            raise RuntimeError("Manual eviction is not enabled for this manager.")
        if self.batch_processor is not None:
            self.batch_processor.scheduler.finish_request(request_id)


class ContinuousMixin:
    """Mixin class for models to add continuous batching capabilities."""

    generation_config: GenerationConfig

    @contextmanager
    def continuous_batching_context_manager(
        self,
        generation_config: GenerationConfig | None = None,
        manual_eviction: bool = False,
        max_queue_size: int = 0,
        num_q_cuda_graphs: int = 0,
        num_kv_cuda_graphs: int = 0,
        allow_block_sharing: bool = True,
        block: bool = True,
        timeout: float | None = None,
    ) -> Generator[ContinuousBatchingManager]:
        manager = self.init_continuous_batching(
            generation_config,
            manual_eviction,
            max_queue_size,
            num_q_cuda_graphs,
            num_kv_cuda_graphs,
            allow_block_sharing,
        )
        manager.start()
        try:
            yield manager
        finally:
            logger.debug(
                "Continuous batching loop finished"
            )  # a dummy log needed for the logs of stop to show. Won't show
            manager.stop(block=block, timeout=timeout)

    # NOTE: don't forget to update `continuous_batching_context_manager` when changing this method's definition
    def init_continuous_batching(
        self,
        generation_config: GenerationConfig | None = None,
        manual_eviction: bool = False,
        max_queue_size: int = 0,
        num_q_padding_intervals: int = 0,
        num_kv_padding_intervals: int = 0,
        allow_block_sharing: bool = True,
    ) -> ContinuousBatchingManager:
        """Initialize a manager for continuous batching inference.

        Args:
            generation_config: An optional generation configuration, which may contain a CompileConfig object
            manual_eviction: Whether to manually evict requests from the cache
            max_queue_size: Maximum size of the input request queue
            num_q_padding_intervals: Number of intervals used to pad the query dimension
            num_kv_padding_intervals: Number of intervals used to pad the keys/values dimension
            allow_block_sharing: A flag to allow block sharing if the model has some full attention layers

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
            model=self,  # type: ignore
            generation_config=gen_config,
            manual_eviction=manual_eviction,
            max_queue_size=max_queue_size,
            num_q_padding_intervals=num_q_padding_intervals,
            num_kv_padding_intervals=num_kv_padding_intervals,
            allow_block_sharing=allow_block_sharing,
        )

    # TODO: support streaming
    @traced
    @torch.inference_mode()
    def generate_batch(
        self,
        inputs: list[list[int]],
        generation_config: GenerationConfig | None = None,
        num_q_padding_intervals: int = 0,
        num_kv_padding_intervals: int = 0,
        allow_block_sharing: bool = True,
        record_timestamps: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> dict[str, GenerationOutput]:
        """Generate sequences for a batch of prompts using continuous batching.

        Args:
            inputs: List of input token sequences (prompts)
            generation_config: Optional generation configuration
            num_q_padding_intervals: Number of intervals used to pad the query dimension
            num_kv_padding_intervals: Number of intervals used to pad the keys/values dimension
            allow_block_sharing: A flag to allow block sharing if the model has some full attention layers
            record_timestamps: If set to true, the requests will have a timestamp for each token generated
            progress_bar: If set to true, a progress bar will be displayed
            **kwargs: Additional generation parameters

        Returns:
            `dict[str, GenerationOutput]`: a dictionary of request ids to GenerationOutput objects
        """
        if not inputs:
            return {}
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.warning("Progress bar is disabled when logger level is less than DEBUG")
            progress_bar = False

        # Initialize manager with the batch inputs
        results = {}
        gen_cfg = self.generation_config if generation_config is None else generation_config
        num_requests = len(inputs) * (gen_cfg.num_return_sequences if gen_cfg.num_return_sequences is not None else 1)
        # Prepare context managers for the main loop
        manager_cm = self.continuous_batching_context_manager(
            generation_config=generation_config,
            num_q_cuda_graphs=num_q_padding_intervals,
            num_kv_cuda_graphs=num_kv_padding_intervals,
            allow_block_sharing=allow_block_sharing,
            block=True,
            timeout=5,
        )
        logging_cm = logging_redirect_tqdm([logger])
        pbar_cm = tqdm(
            total=num_requests,
            disable=(not progress_bar),
            desc=f"Solving {num_requests} requests",
            unit="request",
        )
        # Main loop
        with manager_cm as manager, logging_cm, pbar_cm as pbar:
            try:
                manager.add_requests(
                    inputs=inputs, max_new_tokens=kwargs.get("max_new_tokens"), record_timestamps=record_timestamps
                )
                finished_count = 0
                while finished_count < num_requests:
                    result = manager.get_result(timeout=1)
                    if result:
                        req_id = result.request_id
                        if result.is_finished():
                            results[req_id] = result
                            finished_count += 1
                            pbar.update(1)
                    else:
                        if not manager.is_running():
                            logger.error("Generation thread terminated unexpectedly.")
                            # This helps get some information in stdout
                            print("Returning results of generate_batch despite unexpected termination.")
                            break

            except Exception as e:
                logger.error(f"Error during batch generation: {e}", exc_info=True)
        return results
