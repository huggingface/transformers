# Copyright 2026 The HuggingFace Inc. team
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
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from math import ceil
from typing import Any

import torch
from torch import nn

from ...generation.configuration_utils import ContinuousBatchingConfig
from .cache import PagedAttentionCache
from .cb_logits_processors import ContinuousBatchingLogitsProcessorList
from .input_outputs import ContinuousBatchingAsyncIOs, ContinuousBatchingIOs
from .requests import FutureRequestState, RequestState, RequestStatus, create_warmup_future_states
from .utils import (
    accelerator_graph_capture_context,
    create_accelerator_graph,
    get_memory_pools,
    logger,
    memory_pool_context,
    pad_to_interval,
    pad_to_pow2,
    pop_peak_memory,
    stream_context,
)


def read_memory(device: torch.device) -> tuple[int, int]:
    """Stricter version of `pop_peak_memory` that raises an error if the peak is not tracked."""
    memory = pop_peak_memory(device)
    if memory is None:
        raise RuntimeError(f"The peak of allocated memory is not tracked on {device}.")
    return memory


def infer_max_single_request_tokens(cache: PagedAttentionCache) -> int:
    """Returns the largest total length a single request can have on an empty cache, i.e. the largest length for
    which all allocators can be served enough sectors at the same time."""
    dummy_state = RequestState(request_id="", initial_tokens=[])
    # Solve using dichotomy
    low, high = 0, cache.max_tokens_read
    while low < high:
        mid = (low + high + 1) // 2
        if cache.can_store_request_tokens(dummy_state, mid, dry_run=True):
            low = mid
        else:
            high = mid - 1
    return low


class ModelRunner:
    """This class is the continuous batching entry point for running the model. As a rule of thumb, anything running on
    the device should happen from this class. The class is also used to size the cache through a few probe forwards."""

    # Parameters for the cache sizing probe
    _probe_max_rows = 32
    _probe_query_tokens = (16, 64)
    _probe_read_tokens = 16384

    def __init__(
        self,
        logit_processor: ContinuousBatchingLogitsProcessorList,
        cb_config: ContinuousBatchingConfig,
        inputs_and_outputs: ContinuousBatchingIOs | ContinuousBatchingAsyncIOs,
        cache: PagedAttentionCache,
        do_sample: bool,
        return_logprobs: bool,
    ) -> None:
        # Main attributes
        self.logit_processor = logit_processor
        self.cb_config = cb_config
        self.inputs_and_outputs = inputs_and_outputs
        # Helper attributes
        self.do_sample = do_sample
        self.return_logprobs = return_logprobs
        self.use_cuda_graph_varlen, self.use_cuda_graph_decode = self.cb_config.cuda_graph_booleans
        self.cache = cache
        self._model_supports_logits_to_keep: bool | None = None  # resolved on first forward

        # Padding only happen when CUDA graphs or compile is used
        cuda_graph = self.use_cuda_graph_varlen or self.use_cuda_graph_decode
        compile = self.cb_config.varlen_compile_config is not None or self.cb_config.decode_compile_config is not None
        self.pad_inputs = cuda_graph or compile

        # Set up the graph pool. This allows all graphs to share the same memory pool, greatly saving memory.
        if self.use_cuda_graph_varlen or self.use_cuda_graph_decode:
            self.mem_pool, self.graph_pool_id = get_memory_pools()
        else:
            self.mem_pool, self.graph_pool_id = None, None

        # Set up compiled version of the forward pass for the varlen path
        self._compiled_varlen = None
        if self.cb_config.varlen_compile_config is not None:
            self._compiled_varlen = torch.compile(
                self._forward_process_and_sample, **self.cb_config.varlen_compile_config.to_dict()
            )

        # Set up compiled version of the forward pass for the decode path
        self._compiled_decode = None
        if self.cb_config.decode_compile_config is not None:
            self._compiled_decode = torch.compile(
                self._forward_process_and_sample, **self.cb_config.decode_compile_config.to_dict()
            )

    def maybe_pad_inputs(self, num_q_tokens: int, max_kv_read: int, use_decode_fast_path: bool) -> tuple[int, int]:
        """Pads the input sizes for the next batch if it is needed. Often it is, for max performance."""
        if not self.pad_inputs:
            return num_q_tokens, max_kv_read
        max_batch_tokens = self.cache.max_batch_tokens
        # For varlen batches, we pad using interval sizes
        if not use_decode_fast_path:
            num_q_tokens = pad_to_interval(num_q_tokens, self.cb_config.q_padding_interval_size, max_batch_tokens)
            max_kv_read = pad_to_interval(
                max_kv_read, self.cb_config.kv_padding_interval_size, self.cache.max_tokens_read
            )
        # For decode fast path batches, we pad using powers of 2 and use no KV
        else:
            num_q_tokens = pad_to_pow2(num_q_tokens, self.cb_config.max_requests_per_batch)
            max_kv_read = 0
        return num_q_tokens, max_kv_read

    def supports_logits_to_keep(self, model: nn.Module) -> bool:
        """Returns True if the model accepts the logits_to_keep kwarg in its forward."""
        if self._model_supports_logits_to_keep is None:
            self._model_supports_logits_to_keep = (
                hasattr(model, "_supports_logits_to_keep") and model._supports_logits_to_keep()
            )
        return self._model_supports_logits_to_keep

    def compute_batch(self, model: nn.Module, batch_data: dict) -> None:
        """Runs the forward pass, processes the logits and samples the next tokens. It also handles which version of
        the forward pass to use (varlen or decode), whether to use CUDA graphs (with the eventual capture of the graph)
        and torch compile."""
        # These tensors are device-resident, this is just pointer retrieval
        carry_over_ids, prev_output_ids, output_ids = self.inputs_and_outputs.get_cb_kwargs()
        # This is the stream on which the compute happens
        compute_stream = self.inputs_and_outputs.compute_stream

        # If supported, the model slices hidden states at logits_indices before lm_head, instead of us slicing logits
        if self.supports_logits_to_keep(model):
            batch_data["logits_to_keep"] = batch_data["logits_indices"]

        # Get the appropriate forward function (compiled or not, based on current path)
        forward_fn, use_cuda_graph = self._get_forward_fn(use_block_table=self.inputs_and_outputs.use_block_table)

        # If we are not using CUDA graphs, we perform the generation step and return
        if not use_cuda_graph:
            with stream_context(compute_stream):
                forward_fn(model, batch_data, carry_over_ids, prev_output_ids, output_ids)

        # Otherwise, we either create or replay the graph (an accelerator is available in this path)
        else:
            graph = self.inputs_and_outputs.get_graph()
            # Case: the graph already exists, so we replay it
            if graph is not None:
                with stream_context(compute_stream):
                    graph.replay()
            # Otherwise, the graph does not exist, so we create it
            else:
                args = (model, batch_data, carry_over_ids, prev_output_ids, output_ids)
                self._capture_graph(forward_fn, compute_stream, *args)

    def _get_forward_fn(self, use_block_table: bool) -> tuple[Callable, bool]:
        """Helper function to get the appropriate forward function based on the block table and compile behavior."""
        if use_block_table:
            forward_fn = self._forward_process_and_sample if self._compiled_decode is None else self._compiled_decode
            use_cuda_graph = self.use_cuda_graph_decode
        else:
            forward_fn = self._forward_process_and_sample if self._compiled_varlen is None else self._compiled_varlen
            use_cuda_graph = self.use_cuda_graph_varlen
        return forward_fn, use_cuda_graph

    def _capture_graph(self, forward_fn: Callable, compute_stream: torch.cuda.Stream, *args) -> None:
        """Helper function to capture and store a graph for a given forward function."""
        # Warmup (ensures the right result is computed before capturing the graph)
        with stream_context(compute_stream), memory_pool_context(self.mem_pool):
            forward_fn(*args)
        # Capture using a thread-local capture mode to avoid capturing GPU operations from outside the model forward
        graph = create_accelerator_graph()
        with accelerator_graph_capture_context(graph, stream=compute_stream, pool=self.graph_pool_id):
            forward_fn(*args)
        # Store
        self.inputs_and_outputs.set_graph(graph)

    def _forward_process_and_sample(
        self,
        model: nn.Module,
        batch_data: dict,
        carry_over_ids: torch.Tensor,
        prev_output_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> None:
        """This function performs the forward pass, logits processing, and sampling. This is what is either captured
        and/or compiled."""
        # Perform carry-over (no-op for synchronous batching)
        self.inputs_and_outputs.carry_over_tokens(batch_data["input_ids"], carry_over_ids, prev_output_ids)

        # Run model forward pass
        logits = model(**batch_data).logits  # shape [1, seq_len OR num_logits, vocab_size]

        # If it has not been done by the model, extract only the logits that are used to predict new tokens
        logits_indices = batch_data["logits_indices"]  # shape [num_logits]
        if "logits_to_keep" not in batch_data:
            logits = logits[:, logits_indices, :]  # shape [1, num_logits, vocab_size]
        # Convert to fp32 to match generate
        logits = logits.float()  # shape [1, num_logits, vocab_size]

        # Process logits if there are any logit processors
        if self.logit_processor.do_processing:
            # Handle shape inconsistency between generate and continuous batching (dummy_dim is always 1)
            dummy_dim, num_logits, vocab_size = logits.shape
            logits_2d = logits.view(dummy_dim * num_logits, vocab_size)
            sliced_input_ids_2d = batch_data["input_ids"][0, logits_indices]  # shape [num_logits]
            # Process with 2D tensors
            logits_2d = self.logit_processor(sliced_input_ids_2d, logits_2d, batch_data["logits_processor_args"])
            # Reshape back to 3D
            scores = logits_2d.view(dummy_dim, num_logits, vocab_size)
        else:
            scores = logits

        # Sample next tokens
        self._sample(scores, output_ids)

    def _sample(self, scores: torch.Tensor, output_ids: torch.Tensor) -> None:
        """Private method to sample next tokens from the scores."""
        # Apply softmax if we are sampling or if we are generating log probabilities
        if self.do_sample or self.return_logprobs:
            probs = nn.functional.softmax(scores[0], dim=-1)  # shape [num_logits, vocab_size]
        else:
            probs = scores.squeeze(0)  # shape [num_logits, vocab_size]

        # Retrieve next tokens through sampling or argmax
        if self.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1)  # shape [num_logits, 1]
        else:
            next_tokens = torch.argmax(probs, dim=-1, keepdim=True)  # shape [num_logits, 1]

        # Maybe retrieve log probabilities
        if self.return_logprobs:
            per_token_probs = probs.gather(dim=1, index=next_tokens).squeeze(-1)
            logprobs = per_token_probs.log()  # shape [num_logits]

        # Always remove the extra dimension for the gather
        next_tokens = next_tokens.squeeze(-1)  # shape [num_logits]

        # Copy the next tokens and maybe their logprobs to the static output tensor
        tokens = next_tokens.size(0)
        output_ids[0, :tokens].copy_(next_tokens)
        if self.return_logprobs:
            # In order to match the dtype of output_ids, we cast the fp32 logprobs as int32 without changing the
            # underlying data. It's just a trick to use the same storage for both tensors.
            output_ids[1, :tokens].copy_(logprobs.view(dtype=torch.int32))

    @torch.no_grad()
    def warmup(self, model: nn.Module) -> None:
        """Pre-capture CUDA graphs and/or trigger compile warmup for varlen and decode paths (if available). Unless the
        force_warmup flag is set, the warmup is only performed if the CUDA graphs or compile are enabled."""
        # Early return if the warmup is not needed
        if not self.pad_inputs:
            return None

        # In async mode, each IO pair has its own graph buffer and static tensors, so we warm up both
        total_duration = 0
        iterations = 2 if isinstance(self.inputs_and_outputs, ContinuousBatchingAsyncIOs) else 1
        for _ in range(iterations):
            # Warm up the varlen path, w/ the largest possible sizes to get the biggest pool and avoid fragmentation
            num_q_tokens = self.cache.max_batch_tokens
            max_kv_read = max(0, infer_max_single_request_tokens(self.cache) - num_q_tokens)
            total_duration += self.run_one_warmup(model=model, num_q_tokens=num_q_tokens, max_kv_read=max_kv_read)

            # Exit here if the decode fast path is not available
            if self.cache.max_decode_fast_path_length <= 0:
                continue

            # Warm up the decode path
            num_requests = 1
            while True:
                total_duration += self.run_one_warmup(model=model, num_q_tokens=num_requests, max_kv_read=None)
                if num_requests >= self.cb_config.max_requests_per_batch:
                    break
                num_requests = min(2 * num_requests, self.cb_config.max_requests_per_batch)

            # Switch to the other IO pair if this is async
            if isinstance(self.inputs_and_outputs, ContinuousBatchingAsyncIOs):
                self.inputs_and_outputs.swap_io_pairs()
        # Warmup requests pull all sectors into their allocators, so reset the cache toreturn them to the global pool
        self.cache.reset()
        logger.info(f"Warmup completed in {total_duration:.2f}s")

    @contextmanager
    def _fake_requests(
        self, num_requests: int, status: RequestStatus, num_q_tokens: int, past_length: int
    ) -> Iterator[list[FutureRequestState]]:
        """Makes up fake requests with their cache blocks allocated, as many as the cache can hold, and frees their
        blocks on exit."""
        future_states = create_warmup_future_states(num_requests, status, num_q_tokens, past_length, self.cache)
        try:
            yield future_states
        finally:
            for fs in future_states:
                self.cache.free_blocks(fs.state.request_id)

    def run_one_warmup(self, model: nn.Module, num_q_tokens: int, max_kv_read: int | None) -> float:
        """Warms up the decode fast path (if max_kv_read is None) or varlen path (if max_kv_read is an int) for a
        specific number of query and cache-resident tokens. `max_kv_read` is the number of tokens already in cache,
        matching the terminology used by `prepare_batch_tensors` and the scheduler."""
        # Make up fake request states according to the chosen path
        use_decode_fast_path = max_kv_read is None
        if use_decode_fast_path:
            num_requests = num_q_tokens
            status = RequestStatus.DECODING
            num_q_tokens = 1
            max_kv_read = self.cb_config.page_size
            logger.debug(f"Warming up decode fast path for {num_requests = }.")
        else:
            num_requests = 1
            status = RequestStatus.PREFILLING
            logger.debug(f"Warming up varlen path for {num_q_tokens = }, {max_kv_read = }.")
        # To limit the compute cost of prefill warmup, we cap the past length to the page size
        past_length = min(max_kv_read, self.cb_config.page_size)
        with self._fake_requests(num_requests, status, num_q_tokens, past_length) as future_states:
            if not future_states:
                logger.warning(
                    f"Failed to warm up: no blocks allocated for {num_requests = }, {num_q_tokens = }, {past_length = }."
                )
                return 0.0

            # Pad the inputs to the appropriate size
            padded_q, padded_kv = self.maybe_pad_inputs(
                num_q_tokens=num_q_tokens * num_requests,
                max_kv_read=max_kv_read,
                use_decode_fast_path=use_decode_fast_path,
            )

            # Actual warmup, in a try-except block since a failed warmup only degrades performance
            start = time.perf_counter()
            try:
                self.inputs_and_outputs.prepare_batch_tensors(
                    future_states, self.logit_processor, use_decode_fast_path, padded_q, padded_kv, use_padding=True
                )
                batch_data = self.inputs_and_outputs.get_model_kwargs(use_padding=True)
                # Flash sizes the workspace according to max_length_k, so we set it to the maximum value for one request
                if not use_decode_fast_path:
                    # The graph key is read from the IO holding the graphs, so we update it there as well
                    io = self.inputs_and_outputs
                    io = io.io_pairs[io.current_pair].device_io if isinstance(io, ContinuousBatchingAsyncIOs) else io
                    for allocator in self.cache.cache_allocators.values():
                        max_length_k = allocator.get_seqlen_k(max_kv_read, num_q_tokens)
                        if self.use_cuda_graph_varlen:
                            max_length_k = pad_to_pow2(max_length_k, self.cache.max_tokens_read, 1024)
                        batch_data["max_length_k"][allocator.layer_type] = max_length_k
                        io.max_length_k[allocator.layer_type] = max_length_k
                self.compute_batch(model, batch_data)
                duration = time.perf_counter() - start
                logger.debug(f"Warmup completed in {duration:.2f}s")
            except Exception as e:
                duration = 0.0
                logger.warning(f"Failed to warm up: {e}.\nGraph pool may fragment and OOM under load.")
        return duration

    @torch.no_grad()
    def measure_peak_memory(
        self, model: nn.Module, num_requests: int, num_q_tokens: int, past_length: int
    ) -> tuple[int, int, int]:
        """Runs one eager varlen step on fake requests (`num_requests` requests of `num_q_tokens` query tokens, each with
        `past_length` tokens already in cache) and measures the memory it allocates, relative to what was allocated
        before the step. Returns:
            - the peak in the decoder, ie. until the LM head starts
            - the peak from the LM head onwards (LM head, logits processing and sampling)
            - the memory still allocated when the LM head starts
        If the model has no output embeddings to split the step on, both peaks are the peak of the whole step and the LM
        head is considered to start with the step. Errors are not caught."""
        device = torch.device(self.cache.device)
        device_module = torch.get_device_module(device)

        # Right before the LM head, the decoder peak is read and the peak tracking restarts for the rest of the step
        lm_head_memory = []  # (allocated, peak) when the LM head starts

        def split_at_lm_head(module: nn.Module, args: Any) -> None:
            """Hook to read the memory when the LM head starts."""
            if not lm_head_memory:
                lm_head_memory.append(read_memory(device))

        output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        hook = None if output_embeddings is None else output_embeddings.register_forward_pre_hook(split_at_lm_head)

        try:
            with self._fake_requests(num_requests, RequestStatus.PREFILLING, num_q_tokens, past_length) as states:
                if len(states) < num_requests:
                    raise RuntimeError(
                        f"The cache cannot hold {num_requests} requests of {num_q_tokens + past_length} tokens to "
                        "measure the memory of a batch."
                    )

                # Wait for the previous step to be done with the inputs before overwriting them
                device_module.synchronize(device)

                # Measure the memory allocated before the batch is prepared, as it can create large temporary tensors
                allocated_before, _ = read_memory(device)
                self.inputs_and_outputs.prepare_batch_tensors(
                    states,
                    self.logit_processor,
                    use_decode_fast_path=False,
                    num_q_tokens=num_requests * num_q_tokens,
                    max_kv_read=num_requests * past_length,
                    use_padding=False,
                )
                batch_data = self.inputs_and_outputs.get_model_kwargs(use_padding=False)
                if self.supports_logits_to_keep(model):
                    batch_data["logits_to_keep"] = batch_data["logits_indices"]

                # The inputs are written on the default stream, so the forward on the compute stream must wait for them
                device_module.synchronize(device)
                with stream_context(self.inputs_and_outputs.compute_stream):
                    self._forward_process_and_sample(model, batch_data, *self.inputs_and_outputs.get_cb_kwargs())
                device_module.synchronize(device)
                _, last_peak = read_memory(device)

        # In all cases, un-hook the model if it was hooked
        finally:
            if hook is not None:
                hook.remove()

        # If the model can't be split at the LM head, use the whole step peak as peak of the decoder and of the LM head
        if not lm_head_memory:
            step_peak = last_peak - allocated_before
            return step_peak, step_peak, 0
        # Otherwise, break down the values and offfset them by the memory allocated before the step
        lm_head_allocated, decoder_peak = lm_head_memory[0]
        return decoder_peak - allocated_before, last_peak - allocated_before, lm_head_allocated - allocated_before

    @classmethod
    def infer_probe_num_blocks(cls, cb_config: ContinuousBatchingConfig) -> int:
        """Infers the number of cache blocks the probe needs to run `measure_batch_activations`. Since the probe
        forwards run one by one and free their blocks, the probe cache only needs to fit the largest forward."""
        max_batch_tokens, page_size = cb_config.max_batch_tokens, cb_config.page_size
        probed_rows = min(cb_config.max_requests_per_batch, cls._probe_max_rows)
        read_tokens = max(max_batch_tokens, cls._probe_read_tokens)
        return max(
            ceil(max_batch_tokens / page_size),  # one request with all the batch tokens
            probed_rows,  # one-token requests, each holding one block
            ceil((read_tokens + max(cls._probe_query_tokens)) / page_size),  # few query tokens reading a long context
        )

    def measure_batch_activations(self, model: nn.Module) -> tuple[int, float, float]:
        """Measures the activations of the worst-case batch with eager steps on fake requests. Returns the decoder peak
        of a batch reading no cache, the growth of that peak per cache token read, and the peak of the LM head onwards
        (LM head, logits processing and sampling) for `max_requests_per_batch` rows of logits. The cache must have at
        least `probe_num_blocks` blocks."""
        max_batch_tokens, num_rows = self.cache.max_batch_tokens, self.cb_config.max_requests_per_batch
        probed_rows = min(num_rows, self._probe_max_rows)
        read_tokens = max(max_batch_tokens, self._probe_read_tokens)

        # Peaks of a batch made of a single request with all the batch tokens
        decoder_peak, post_peak, _ = self.measure_peak_memory(model, 1, max_batch_tokens, 0)
        # The peak of the LM head onwards grows with the number of logit rows: we extrapolate it linearly. The cost of a
        # row is measured on one-token requests, so the decoder leaves almost nothing allocated when the LM head starts:
        # in a large batch, the hidden states still allocated then can hide that cost behind them.
        if probed_rows > 1:
            half_rows = probed_rows // 2
            _, half_peak, half_start = self.measure_peak_memory(model, half_rows, 1, 0)
            _, full_peak, full_start = self.measure_peak_memory(model, probed_rows, 1, 0)
            row_cost = max(0, (full_peak - full_start) - (half_peak - half_start)) / (probed_rows - half_rows)
            post_peak += row_cost * (num_rows - 1)

        # The decoder peak grows with the cache tokens read, but it is a max over layers: in a large batch, that growth
        # can hide behind MLP activations or token-independent buffers. So it is measured on a few query tokens, where
        # attention dominates, for two query lengths, since some attention implementations materialize scores of shape
        # [num_heads, query, key]. The growth is affine in the query length, so we extrapolate it to the batch tokens.
        query_lengths = sorted({min(q, max_batch_tokens) for q in self._probe_query_tokens})
        read_costs = []
        for num_q_tokens in query_lengths:
            no_read_peak, _, _ = self.measure_peak_memory(model, 1, num_q_tokens, 0)
            read_peak, _, _ = self.measure_peak_memory(model, 1, num_q_tokens, read_tokens)
            read_costs.append(max(0, read_peak - no_read_peak) / read_tokens)
        read_cost = read_costs[0]
        if len(read_costs) > 1:
            per_query_token = max(0.0, (read_costs[1] - read_costs[0]) / (query_lengths[1] - query_lengths[0]))
            read_cost += per_query_token * (max_batch_tokens - query_lengths[0])
        return decoder_peak, read_cost, post_peak
