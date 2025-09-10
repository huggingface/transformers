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
import queue
import threading
from dataclasses import dataclass
from functools import partial
from itertools import count
from time import perf_counter
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import GenerationConfig
from ...utils.logging import logging
from ...utils.metrics import ContinuousBatchProcessorMetrics, attach_tracer, traced
from .cache import PagedAttentionCache
from .requests import GenerationOutput, RequestState, RequestStatus, get_device_and_memory_breakdown, logger
from .scheduler import SCHEDULER_MAPPING, FIFOScheduler, Scheduler


def build_attention_mask(
    attention_mask: torch.Tensor,
    cumulative_seqlens_q: torch.Tensor,
    cumulative_seqlens_k: torch.Tensor,
    sliding_window: int = 1,
) -> None:
    """Builds an attention mask inplace using the cumulative seqlens of the query and key. If given a sliding window, it
    will also apply a sliding window mask on top. The attention mask is not boolean, it uses zeroes and -inf (or its
    equivalent) so it's more of an attention score bias tensor."""
    min_value = torch.finfo(attention_mask.dtype).min
    for i in range(len(cumulative_seqlens_q) - 1):
        seqlen_q = cumulative_seqlens_q[i + 1] - cumulative_seqlens_q[i]
        seqlen_k = cumulative_seqlens_k[i + 1] - cumulative_seqlens_k[i]
        if seqlen_q < seqlen_k and seqlen_q >= 1:
            causal_diagonal = seqlen_k - seqlen_q + 1
        else:
            causal_diagonal = 1
        query_range = slice(cumulative_seqlens_q[i], cumulative_seqlens_q[i + 1])
        key_range = slice(cumulative_seqlens_k[i], cumulative_seqlens_k[i + 1])
        # Apply causal mask
        minus_inf = torch.full(
            attention_mask[..., query_range, key_range].shape,
            min_value,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        masked = torch.triu(minus_inf, diagonal=causal_diagonal)
        # Apply sliding window mask if needed
        if sliding_window > 1:
            sliding_diagonal = seqlen_k - seqlen_q + sliding_window
            masked = torch.tril(masked, diagonal=sliding_diagonal)
        # Replace in attention mask
        attention_mask[..., query_range, key_range] = masked


@dataclass
class PagedAttentionArgs:
    input_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    position_ids: torch.Tensor
    cumulative_seqlens_q: torch.Tensor
    cumulative_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    write_index: list[torch.Tensor]
    read_index: list[torch.Tensor]
    logits_indices: torch.Tensor
    cache: PagedAttentionCache
    use_cache: bool = False


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
        slice_inputs: bool = True,  # TODO: There should be an heuristic to decide on slicing, compile, cuda graphs...
    ):
        """Initialize the continuous batch processor.

        Args:
            cache: The paged attention cache to use
            config: The model configuration
            generation_config: The generation configuration
            input_queue: Queue for incoming requests
            output_queue: Queue for outgoing results
            stop_event: Event to signal processing should stop
            model_device: Device for model inputs/outputs
            model_dtype: Data type for model inputs/outputs
            scheduler: The [`Scheduler`] to use
            streaming: Whether to stream tokens as they're generated
            manual_eviction: Whether to manually evict blocks from the cache
            slice_inputs: Whether to slice the inputs to the model
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
        self.slice_inputs = slice_inputs

        # Retrieve the size of the sliding window if there is one
        self.sliding_window = 1 if getattr(config, "sliding_window", None) is None else config.sliding_window

        self.requests_in_batch: list[RequestState] = []

        # Set up metrics collector
        self.max_batch_tokens = cache.max_batch_tokens
        self.metrics = ContinuousBatchProcessorMetrics(cache.max_batch_tokens)

        # Setup static tensors
        self.total_query_length = 0
        self.total_key_length = 0
        self.total_batch_size = 0
        self.setup_static_tensors(cache.num_groups)

    def return_attention_mask(self) -> bool:
        return self.config._attn_implementation != "paged_attention"  # we set `is_causal` to True in paged call

    @traced(standalone=True)
    def setup_static_tensors(self, num_groups: int):
        T = self.max_batch_tokens
        num_pages = self.cache.num_blocks * self.cache.block_size
        tensor_metadata = {"dtype": torch.int32, "device": self.model_device}
        self.tensor_metadata = tensor_metadata
        self.input_ids = torch.empty((1, T), **tensor_metadata)
        self.position_ids = torch.empty((1, T), **tensor_metadata)
        self.cumulative_seqlens_q = torch.empty((T + 1,), **tensor_metadata)
        self.cumulative_seqlens_k = {
            "full_attention": torch.empty((T + 1), **tensor_metadata),
            "sliding_attention": torch.empty((T + 1), **tensor_metadata),
            # TODO: can be generalized using layer types, for block-attn for instance
        }

        # There is one read and write index tensor per group
        self.write_index_tensors = [torch.empty((T,), **tensor_metadata) for _ in range(num_groups)]
        self.read_index_tensors = [torch.empty((num_pages + T), **tensor_metadata) for _ in range(num_groups)]
        # +T is because there are -1 for seqlen_q when model uses a sliding window

        self.logits_indices = torch.empty((T,), **tensor_metadata)
        self.max_seqlen_q = 0
        self.max_seqlen_k = {"full_attention": 0, "sliding_attention": 0}
        self.output_ids = torch.empty((1, T), **tensor_metadata)
        # Since attenention_mask is not always needed, we only allocate it if it is
        if self.return_attention_mask():
            # TODO: this could be 2 iff model is hybrid, and then we can also change memory handler to account for it
            size_0 = 1 if self.sliding_window == 1 else 2
            self.attention_mask = torch.empty(
                (size_0, 1, T, num_pages), dtype=self.model_dtype, device=self.model_device
            )
        else:
            logger.warning(f"Attention mask is not needed for {self.config._attn_implementation}")
            self.attention_mask = None
        self.reset_static_tensors(full_reset=True)

    @traced
    @torch.no_grad()
    def reset_static_tensors(self, full_reset: bool = False):
        """Reset static tensors for the next batch. In between batches, reset only the parts that were used in the last
        batch, but for initialisation, we can reset everything using the (full_reset) flag."""
        # Compute the slice to reset
        t = self.total_query_length if self.slice_inputs and not full_reset else self.write_index_tensors[0].size(-1)
        c = self.total_key_length if self.slice_inputs and not full_reset else self.read_index_tensors[0].size(-1)
        b = self.total_batch_size if self.slice_inputs and not full_reset else self.write_index_tensors[0].size(0)
        # Reset the tensors
        self.input_ids[:, :t].zero_()
        self.position_ids[:, :t].zero_()
        self.cumulative_seqlens_q[: b + 1].zero_()
        for layer_type in self.cumulative_seqlens_k:
            self.cumulative_seqlens_k[layer_type][: b + 1].zero_()
            self.max_seqlen_k[layer_type] = 0
        for i in range(self.cache.num_groups):
            self.write_index_tensors[i][:t].fill_(-1)
            self.read_index_tensors[i][: t + c].fill_(-1)
        self.logits_indices[:t].fill_(-1)
        self.max_seqlen_q = 0
        self.output_ids[:, :t].fill_(-1)
        if self.attention_mask is not None:
            self.attention_mask[:, :, :t, :c].fill_(torch.finfo(self.model_dtype).min)

    def get_model_kwargs(self) -> PagedAttentionArgs:
        """Get model keyword arguments for the current batch."""
        # Compute the slice to return
        t = self.total_query_length if self.slice_inputs else self.write_index.size(-1)
        b = self.total_batch_size
        # Prepare the kwargs
        kwargs = {
            "input_ids": self.input_ids[:, :t],
            "position_ids": self.position_ids[:, :t],
            "cu_seq_lens_q": self.cumulative_seqlens_q[: b + 1],
            "cu_seq_lens_k": {},
            "read_index": self.read_index,  # slicing is done during building
            "write_index": self.write_index,  # slicing is done during building
            "logits_indices": self.logits_indices[:t],
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_k": self.max_seqlen_k,
            "cache": self.cache,
            "use_cache": False,
        }
        for layer_type in self.cumulative_seqlens_k:
            kwargs["cu_seq_lens_k"][layer_type] = self.cumulative_seqlens_k[layer_type][: b + 1]
        # If the attention mask is not None, we slice it as the others
        if self.attention_mask is not None:
            kwargs["attention_mask"] = {}
            for layer_type, seqlens_k in kwargs["cu_seq_lens_k"].items():
                kwargs["attention_mask"][layer_type] = self.attention_mask[:1, :, :t, : seqlens_k[-1]]
        else:
            kwargs["attention_mask"] = None
        return kwargs

    def __repr__(self):
        return (
            f"ContinuousBatchProcessor(input_queue={self.input_queue}, output_queue={self.output_queue}, "
            f"active_requests={self.scheduler.active_requests}, waiting_requests={self.scheduler.waiting_requests})"
            + self.get_model_kwargs().__repr__()
        )

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
    def prepare_next_batch(self) -> bool:
        """Prepare tensors and metadata for the next model forward pass."""
        # Get new requests from the queue
        self._get_new_requests()
        self.scheduler.clear_cancelled_requests()
        if not self.scheduler.has_pending_requests():
            return False

        self.metrics.record_queue_metrics(len(self.scheduler.active_requests), len(self.scheduler.waiting_requests))

        self.requests_in_batch = self.scheduler.schedule_batch(self.max_batch_tokens)
        if not self.requests_in_batch:
            return False

        # Get the request objects for this batch
        self.reset_static_tensors()  # TOOD: with slice_inputs, this might be unnecessary
        position_ids = []
        input_ids = []
        read_index = [[] for _ in range(self.cache.num_groups)]
        write_index = [[] for _ in range(self.cache.num_groups)]
        cumulative_seqlens_q = [0]
        cumulative_seqlens_k = {"full_attention": [0], "sliding_attention": [0]}
        logits_indices = []
        self.metrics.record_batch_metrics(self.requests_in_batch)

        self.total_query_length = 0
        self.total_key_length = 0
        self.total_batch_size = 0

        for state in self.requests_in_batch:
            next_input_ids = state.prompt_ids
            input_ids.extend(next_input_ids)
            past_length = state.position_offset
            query_length = len(next_input_ids)
            key_length = query_length + past_length

            self.total_query_length += query_length
            self.total_key_length += key_length
            self.total_batch_size += 1

            positions_to_add = list(range(past_length, key_length))
            self.cache.get_read_indices(state.request_id, past_length, query_length, read_index)
            self.cache.get_write_indices(state.request_id, past_length, query_length, write_index)

            position_ids.extend(positions_to_add)
            cumulative_seqlens_q.append(cumulative_seqlens_q[-1] + query_length)

            cumulative_seqlens_k["full_attention"].append(
                cumulative_seqlens_k["full_attention"][-1] + query_length + past_length
            )
            cumulative_seqlens_k["sliding_attention"].append(
                cumulative_seqlens_k["sliding_attention"][-1]
                + query_length
                + min(past_length, self.sliding_window - 1)
            )

            if len(state.remaining_prompt_ids) == 0:
                logits_indices.append(cumulative_seqlens_q[-1] - 1)
            self.max_seqlen_q = max(self.max_seqlen_q, query_length)
            self.max_seqlen_k["full_attention"] = max(self.max_seqlen_k["full_attention"], query_length + past_length)
            self.max_seqlen_k["sliding_attention"] = max(
                self.max_seqlen_k["sliding_attention"], query_length + min(past_length, self.sliding_window - 1)
            )
            state.position_offset += query_length

        logger.debug(
            f"Scheduled: {len(self.requests_in_batch)}, Waiting: {len(self.scheduler.waiting_requests)}, "
            f"Active: {len(self.scheduler.active_requests)}. cum Q: {cumulative_seqlens_q[-1]}. "
            f"cum KV: {max(ck[-1] for ck in cumulative_seqlens_k)}, free blocks: {self.cache.get_num_free_blocks()}"
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

        return True

    @traced
    def _build_tensors(
        self,
        input_ids,
        position_ids,
        read_index: list[list[int]],
        write_index: list[list[int]],
        cumulative_seqlens_q,
        cumulative_seqlens_k,
        logits_indices,
    ):
        to_tensor = partial(torch.tensor, **self.tensor_metadata)
        self.input_ids[:, : len(input_ids)] = to_tensor(input_ids)
        self.position_ids[:, : len(position_ids)] = to_tensor(position_ids)

        self.read_index = []
        self.write_index = []
        for i, group_read_indices, group_write_indices in zip(count(), read_index, write_index):
            # Write in the actual tensors
            self.read_index_tensors[i][: len(group_read_indices)] = to_tensor(group_read_indices)
            self.write_index_tensors[i][: len(group_write_indices)] = to_tensor(group_write_indices)
            # Slice to the right size
            r = len(group_read_indices) if self.slice_inputs else self.read_index_tensors[i].size(-1)
            w = len(group_write_indices) if self.slice_inputs else self.write_index_tensors[i].size(-1)
            # Add to the index
            self.read_index.append(self.read_index_tensors[i][:r])
            self.write_index.append(self.write_index_tensors[i][:w])

        self.cumulative_seqlens_q[: len(cumulative_seqlens_q)] = to_tensor(cumulative_seqlens_q)
        for layer_type in self.cumulative_seqlens_k:
            l = len(cumulative_seqlens_k[layer_type])
            self.cumulative_seqlens_k[layer_type][:l] = to_tensor(cumulative_seqlens_k[layer_type])
        self.logits_indices[: len(logits_indices)] = to_tensor(logits_indices)

        if self.attention_mask is not None:
            build_attention_mask(self.attention_mask[0], cumulative_seqlens_q, cumulative_seqlens_k["full_attention"])
            if self.sliding_window != 1:
                build_attention_mask(
                    self.attention_mask[1],
                    cumulative_seqlens_q,
                    cumulative_seqlens_k["sliding_attention"],
                    self.sliding_window,
                )

    @traced
    def _sync(self):
        if self.output_ids is not None:
            try:
                out = self.output_ids.tolist()[0]  # should be the only sync we do
            except Exception:
                out = [0, 1]
        else:
            out = [0, 0]
        return out

    @traced
    def _maybe_send_output(self, state: RequestState, token: int):
        """Send output to the queue based on streaming mode and request state."""
        if self.streaming:
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
                state.prompt_ids = [token]
                if state.update_with_token(token):
                    self.metrics.record_request_completion(state.created_time, state.request_id)
                    self.scheduler.finish_request(state.request_id, evict_from_cache=(not self.manual_eviction))
                    finished_request_ids.append(req_id)
                self._maybe_send_output(state, token)
            elif state.status == RequestStatus.PREFILLING_SPLIT:
                state.status = RequestStatus.SPLIT_PENDING_REMAINDER
        if self.cache.get_num_free_blocks() == 0:
            raise ValueError("No more free blocks")

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
        slice_inputs: bool = True,
    ):
        """Initialize the continuous batching manager.

        Args:
            model: The language model for generation
            generation_config: Configuration for generation parameters
            max_queue_size: Maximum size of the request queue (0 = unlimited)
            streaming: Whether to stream tokens as they are generated
        """
        self.model = model.eval()
        generation_config = model.generation_config if generation_config is None else generation_config
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
        self.logit_processor = self.model._get_logits_processor(generation_config)
        self.use_cuda_graph = getattr(generation_config, "use_cuda_graph", True)
        self.profile = getattr(generation_config, "profile", False)
        self.manual_eviction = manual_eviction
        self.batch_processor: Optional[ContinuousBatchProcessor] = None
        self.slice_inputs = slice_inputs

    @traced
    def start(self):
        """Start the background generation thread."""
        if self._generation_thread is not None and self._generation_thread.is_alive():
            logger.warning("Manager thread is already running.")
            return

        self._result_queue = queue.Queue()
        self._generation_thread = threading.Thread(target=self._run_generation_loop)
        self._generation_thread.start()

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

        # NOTE: do we want to handle a case when the user wants token ids returned instead of decoded text?
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
        for input_ids in inputs:
            self.add_request(input_ids, **kwargs)

    def cancel_request(self, request_id: str):
        """Cancel a request by its ID.

        Args:
            request_id: The ID of the request to cancel
        """
        if self.batch_processor is not None:
            self.batch_processor.scheduler.set_request_cancellation(request_id)

    def get_result(self, request_id=None, timeout=None) -> Optional[GenerationOutput]:
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
            if request_id is not None and result.request_id != request_id:
                self.output_queue.put(result)
                return None
            logger.debug(f"Retrieved result for request {result.request_id}")
            return result
        except queue.Empty:
            return None

    def __iter__(self):
        """Iterate over results as they become available."""
        while self._generation_thread is not None and self._generation_thread.is_alive():
            result = self.get_result(timeout=0.1)
            if result is not None:
                yield result

    def request_id_iter(self, request_id):
        """Iterate over results matching a specific request id as they become available."""
        request_cancelled = False
        while self._generation_thread is not None and self._generation_thread.is_alive() and not request_cancelled:
            result = self.get_result(request_id=request_id, timeout=0.1)
            if result is not None:
                yield result
            if self.batch_processor is not None:
                request_cancelled = self.batch_processor.scheduler.request_is_cancelled(request_id)

    @staticmethod
    def supported_attention_implementations() -> set[str]:
        return {"eager_paged", "sdpa_paged", "flash_attention_2"}

    @staticmethod
    def default_attention_implementation() -> str:
        return "sdpa_paged"

    @traced
    def warmup(self, batch_processor):
        stream = torch.cuda.Stream(device=self.model.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Warmup the model with a dummy forward pass
            self._generation_step(batch_processor)
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
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
        # Pass continuous batching context to logits processor if it supports it. TODO we should find a way to make this a little bit cleaner!
        if hasattr(self.logit_processor, "set_continuous_batching_context"):
            self.logit_processor.set_continuous_batching_context(
                batch_data["logits_indices"], batch_data["cu_seq_lens_q"]
            )

        # Handle shape compatibility: logit processors expect 2D tensors [batch_size, vocab_size]
        # but continuous batching always produces 3D tensors [batch_size, seq_len, vocab_size]
        batch_size, seq_len, vocab_size = logits.shape
        logits_2d = logits.view(batch_size * seq_len, vocab_size)
        input_ids_2d = batch_data["input_ids"].view(batch_size * seq_len)

        # Process with 2D tensors
        processed_logits_2d = self.logit_processor(input_ids_2d, logits_2d)

        # Reshape back to 3D
        return processed_logits_2d.view(batch_size, seq_len, vocab_size)

    @traced(span_name="sampling")
    def _sample(self, batch_processor: ContinuousBatchProcessor, probs):
        if self.do_sample:  # sample
            probs = nn.functional.softmax(probs, dim=-1)
            # probs[0] has shape [seq_len, vocab_size], multinomial returns [seq_len, 1]
            next_tokens = torch.multinomial(probs[0], num_samples=1).squeeze(-1)  # Now [seq_len]
            # Add batch dimension back to match argmax output
            next_tokens = next_tokens.unsqueeze(0)  # Now [1, seq_len]
        else:
            next_tokens = torch.argmax(probs, dim=-1)  # Already [1, seq_len]

        tokens = next_tokens.size(1)  # Get seq_len dimension
        batch_processor.output_ids[:, :tokens].copy_(next_tokens)

    def _run_generation_loop(self):
        """Main processing loop running in the background thread."""
        batch_processor = None
        try:
            ref_time = perf_counter()
            paged_attention_cache = PagedAttentionCache(
                self.model.config,
                self.generation_config,
                self.model.device,
                self.model.dtype,
                tp_size=getattr(self.model, "_tp_size", None),  # Use model's actual TP setting
            )
            logger.debug(f"PagedAttentionCache created in {perf_counter() - ref_time} seconds")

            scheduler = None
            if hasattr(self.generation_config, "scheduler"):
                scheduler = SCHEDULER_MAPPING.get(self.generation_config.scheduler, None)
                if scheduler is None:
                    logger.warning(f"Scheduler '{scheduler}' not found. Defaulting to FIFO.")
                    scheduler = FIFOScheduler
            else:
                # Default to fifo
                scheduler = FIFOScheduler

            ref_time = perf_counter()
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
                slice_inputs=self.slice_inputs,
            )
            self.batch_processor = batch_processor
            self.current_batch = 0
            logger.debug(f"batch_processor created in {perf_counter() - ref_time} seconds")
            while (not self.stop_event.is_set()) or batch_processor.has_pending_requests():
                self._inner_generation_loop(batch_processor)
                self.current_batch += 1

        except Exception as e:
            logger.error(f"Error in generation loop: {e}", exc_info=True)
            self._handle_critical_error(e, batch_processor)
        finally:
            logger.info("Generation loop finished.")

    @traced(span_name="generation_loop")
    def _inner_generation_loop(self, batch_processor: ContinuousBatchProcessor):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not batch_processor.prepare_next_batch():
            return
        if logger.level <= logging.DEBUG:
            device, total, reserved, allocated = get_device_and_memory_breakdown()
            logger.debug(f"[Memory] Device: {device}, Total: {total}, Reserved: {reserved}, Allocated: {allocated}")
        if torch.cuda.is_available() and self.use_cuda_graph:
            if self.current_batch == 0:
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
        slice_inputs: bool = True,
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
            slice_inputs=slice_inputs,
        )

    @traced
    @torch.inference_mode()
    def generate_batch(
        self,
        inputs: list[list[int]],
        generation_config: Optional[GenerationConfig] = None,
        progress_bar: bool = True,
        slice_inputs: bool = True,
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
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.warning("Progress bar is disabled when logger level is less than DEBUG")
            progress_bar = False

        # Initialize manager with the batch inputs
        manager = self.init_continuous_batching(generation_config=generation_config, slice_inputs=slice_inputs)
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
