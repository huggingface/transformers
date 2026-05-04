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
from collections.abc import Callable
from contextlib import nullcontext

import torch
from torch import nn

from ...generation.configuration_utils import ContinuousBatchingConfig
from .cache import PagedAttentionCache
from .cb_logits_processors import ContinuousBatchingLogitsProcessorList
from .input_outputs import ContinuousBatchingAsyncIOs, ContinuousBatchingIOs
from .requests import RequestStatus, logger
from .utils import create_warmup_future_states, pad_to_interval, pad_to_pow2


class ModelRunner:
    """This class is the continuous batching entry point for running the model. As a rule of thumb, anything running on
    the device should happen from this class."""

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

        # Padding only happen when CUDA graphs or compile is used
        cuda_graph = self.use_cuda_graph_varlen or self.use_cuda_graph_decode
        compile = self.cb_config.varlen_compile_config is not None or self.cb_config.decode_compile_config is not None
        self.pad_inputs = cuda_graph or compile

        # Set up the graph pool. This allows all graphs to share the same memory pool, greatly saving memory.
        if self.use_cuda_graph_varlen or self.use_cuda_graph_decode:
            self.graph_pool = torch.cuda.graph_pool_handle()
        else:
            self.graph_pool = None

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
            max_kv_read = pad_to_interval(max_kv_read, self.cb_config.kv_padding_interval_size, self.cache.num_pages)
        # For decode fast path batches, we pad using powers of 2 and use no KV
        else:
            num_q_tokens = pad_to_pow2(num_q_tokens, max_batch_tokens)
            max_kv_read = 0
        return num_q_tokens, max_kv_read

    def compute_batch(self, model: nn.Module, batch_data: dict) -> None:
        """Runs the forward pass, processes the logits and samples the next tokens. It also handles which version of
        the forward pass to use (varlen or decode), whether to use CUDA graphs (with the eventual capture of the graph)
        and torch compile."""
        # These tensors are device-resident, this is just pointer retrieval
        carry_over_ids, prev_output_ids, output_ids = self.inputs_and_outputs.get_cb_kwargs()
        # This is the stream on which the compute happens
        compute_stream = self.inputs_and_outputs.compute_stream

        # Get the appropriate forward function (compiled or not, based on current path)
        forward_fn, use_cuda_graph = self._get_forward_fn(use_block_table=self.inputs_and_outputs.use_block_table)

        # If we are not using CUDA graphs, we perform the generation step and return
        if not use_cuda_graph:
            maybe_stream = torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext()
            with maybe_stream:
                forward_fn(model, batch_data, carry_over_ids, prev_output_ids, output_ids)

        # Otherwise, we either create or replay the graph (CUDA is available in this path)
        else:
            graph = self.inputs_and_outputs.get_graph()
            # Case: the graph already exists, so we replay it
            if graph is not None:
                with torch.cuda.stream(compute_stream):
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
        with torch.cuda.stream(compute_stream):
            forward_fn(*args)
        # Capture using a thread-local capture mode to avoid capturing GPU operations from outside the model forward
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=compute_stream, pool=self.graph_pool, capture_error_mode="thread_local"):
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

        # Run model forward pass and convert to fp32 to match generate
        logits = model(**batch_data).logits.float()

        # Process logits if there are any logit processors
        if self.logit_processor.do_processing:
            # Handle shape inconsistency between generate and continuous batching (dummy_dim is always 1)
            dummy_dim, num_tokens, vocab_size = logits.shape
            logits_2d = logits.view(dummy_dim * num_tokens, vocab_size)
            input_ids_2d = batch_data["input_ids"].view(dummy_dim * num_tokens)
            # Process with 2D tensors
            logits_2d = self.logit_processor(input_ids_2d, logits_2d, batch_data["logits_processor_args"])
            # Reshape back to 3D
            scores = logits_2d.view(dummy_dim, num_tokens, vocab_size)
        else:
            scores = logits

        # Sample next tokens
        self._sample(scores, batch_data["logits_indices"], output_ids)

    def _sample(self, scores: torch.Tensor, logits_indices: torch.Tensor, output_ids: torch.Tensor) -> None:
        """Private method to sample next tokens from the scores."""
        # Apply softmax if we are sampling or if we are generating log probabilities
        if self.do_sample or self.return_logprobs:
            probs = nn.functional.softmax(scores[0], dim=-1)  # shape [seq_len, vocab_size]
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

        # Always remove the extra dimension for the gather
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
        """Pre-capture CUDA graphs and/or trigger compile warmup for varlen and decode paths (if available). Unless the
        force_warmup flag is set, the warmup is only performed if the CUDA graphs or compile are enabled."""
        # Early return if the warmup is not needed
        if not self.pad_inputs:
            return None

        # In async mode, each IO pair has its own graph buffer and static tensors, so we warm up both
        total_duration = 0
        iterations = 2 if isinstance(self.inputs_and_outputs, ContinuousBatchingAsyncIOs) else 1
        for _ in range(iterations):
            # Warm up the varlen path, with the largest possible dimensions to get the biggest pool and avoid fragmentation
            num_q_tokens = self.cache.max_batch_tokens
            max_kv_read = self.cache.num_blocks * self.cache.block_size
            max_kv_read -= num_q_tokens  # make room for the new tokens
            total_duration += self.run_one_warmup(model=model, num_q_tokens=num_q_tokens, max_kv_read=max_kv_read)

            # Exit here if the decode fast path is not available
            if self.cache.max_blocks_per_request == 0:
                continue

            # Warm up the decode path
            num_requests = 1
            while True:
                total_duration += self.run_one_warmup(model=model, num_q_tokens=num_requests, max_kv_read=None)
                if num_requests >= self.cache.max_batch_tokens:
                    break
                num_requests = min(2 * num_requests, self.cache.max_batch_tokens)

            # Switch to the other IO pair if this is async
            if isinstance(self.inputs_and_outputs, ContinuousBatchingAsyncIOs):
                self.inputs_and_outputs.swap_io_pairs()
        logger.info(f"Warmup completed in {total_duration:.2f}s")

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
            max_kv_read = self.cache.block_size
            logger.debug(f"Warming up decode fast path for {num_requests = }.")
        else:
            num_requests = 1
            status = RequestStatus.PREFILLING
            logger.debug(f"Warming up varlen path for {num_q_tokens = }, {max_kv_read = }.")
        future_states = create_warmup_future_states(num_requests, status, num_q_tokens, max_kv_read, self.cache)
        if not future_states:
            logger.warning(
                f"Failed to warm up: no blocks allocated for {num_requests = }, {num_q_tokens = }, {max_kv_read = }."
            )
            return 0.0

        # Pad the inputs to the appropriate size
        padded_q, padded_kv = self.maybe_pad_inputs(
            num_q_tokens=num_q_tokens * num_requests,
            max_kv_read=max_kv_read,
            use_decode_fast_path=use_decode_fast_path,
        )

        # Actual warmup, which happens in a try-finally block to ensure the blocks are freed even if the warmup fails
        start = time.perf_counter()
        try:
            self.inputs_and_outputs.prepare_batch_tensors(
                future_states, self.logit_processor, use_decode_fast_path, padded_q, padded_kv
            )
            batch_data = self.inputs_and_outputs.get_model_kwargs(use_padding=True)
            carry_over_ids, prev_output_ids, output_ids = self.inputs_and_outputs.get_cb_kwargs()
            forward_fn, use_cuda_graph = self._get_forward_fn(use_block_table=self.inputs_and_outputs.use_block_table)
            forward_fn_args = (model, batch_data, carry_over_ids, prev_output_ids, output_ids)
            if use_cuda_graph:
                self._capture_graph(forward_fn, self.inputs_and_outputs.compute_stream, *forward_fn_args)
            else:
                with torch.cuda.stream(self.inputs_and_outputs.compute_stream):
                    forward_fn(*forward_fn_args)
            duration = time.perf_counter() - start
            logger.debug(f"Warmup completed in {duration:.2f}s")

        # Exception handling
        except Exception as e:
            duration = 0.0
            logger.warning(f"Failed to warm up: {e}.\nGraph pool may fragment and OOM under load.")

        # In any case, free the blocks allocated for the fake warmup requests
        finally:
            for fs in future_states:
                self.cache.free_blocks(fs.state.request_id)
        return duration
