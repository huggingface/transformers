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
import os
import time
from collections.abc import Callable

import torch
from torch import nn

from ...distributed.parallel_utils import with_fast_tp_all_reduce
from ...generation.configuration_utils import ContinuousBatchingConfig
from .cache import PagedAttentionCache
from .cb_logits_processors import ContinuousBatchingLogitsProcessorList
from .input_outputs import _CB_MTP_BT, _CB_MTP_QUERY, ContinuousBatchingAsyncIOs, ContinuousBatchingIOs
from .requests import RequestState, RequestStatus, create_warmup_future_states
from .utils import (
    accelerator_graph_capture_context,
    create_accelerator_graph,
    get_memory_pools,
    logger,
    memory_pool_context,
    pad_to_interval,
    pad_to_pow2,
    stream_context,
)


import os as _glue_os

_GLUE_SAMPLE = _glue_os.environ.get("GLUE_SAMPLE", "0") == "1"
# PROTOTYPE of the engine work package (bench only, off by default). CB_PROTO_CAPTURE_THEN_REPLAY=1: a decode batch whose
# padded request count already ran once eagerly (every kernel it uses is compiled and tuned) captures its graph first
# and replays the capture to produce the step's result, instead of an eager forward followed by a capture pass.
_PROTO_CAPTURE_THEN_REPLAY = os.environ.get("CB_PROTO_CAPTURE_THEN_REPLAY", "0") == "1"
# [scratch, bench only] With synchronous IOs the inputs of a batch are written with device copies queued on the default
# stream (`storage[i, :n] = torch.tensor(..., device=...)`), and the forward runs on `compute_stream` without waiting
# for them: a forward that reads the read index in its first kernels sometimes sees the previous batch's. On when
# FUSE_COMPILE is (its per-layer path maps the read index to cache rows once, at the start of the step).
_FUSE_WAIT_DEFAULT_STREAM = _glue_os.environ.get(
    "CB_WAIT_DEFAULT_STREAM", "0" if _glue_os.environ.get("FUSE_COMPILE", "0") == "0" else "1"
) == "1"
_CB_MTP = _glue_os.environ.get("CB_MTP", "")  # gap2-mtp prototype, see input_outputs.py
# "shadow" measurement: draft sampled from the MTP distribution instead of its argmax (the acceptance of a sampled
# draft under the probability ratio test is sum(min(p, q)), estimated on the host from log p(draft) and log q(draft))
_CB_MTP_DRAFT_SAMPLE = _glue_os.environ.get("CB_MTP_DRAFT", "greedy") == "sample"


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
        self._model_supports_logits_to_keep: bool | None = None  # resolved on first forward
        self._proto_eager_seen: set[tuple[bool, int]] = set()  # (decode path, padded query tokens) that ran eagerly

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
        # gap3-mtp prototype: every request of a decode batch brings `_CB_MTP_QUERY` tokens, the requests are padded
        elif _CB_MTP_BT:
            num_requests = num_q_tokens // _CB_MTP_QUERY
            num_q_tokens = _CB_MTP_QUERY * pad_to_pow2(num_requests, self.cb_config.max_requests_per_batch)
            max_kv_read = 0
            if num_q_tokens > max_batch_tokens:  # host-side bound check: the static inputs hold max_batch_tokens
                raise RuntimeError(f"CB_MTP block table path: {num_q_tokens} query tokens > {max_batch_tokens}")
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
        if self.supports_logits_to_keep(model) and not _CB_MTP:
            batch_data["logits_to_keep"] = batch_data["logits_indices"]

        # Get the appropriate forward function (compiled or not, based on current path)
        forward_fn, use_cuda_graph = self._get_forward_fn(use_block_table=self.inputs_and_outputs.use_block_table)
        if _CB_MTP:
            forward_fn = self._forward_verify_sample_and_draft

        if _FUSE_WAIT_DEFAULT_STREAM and compute_stream is not None and compute_stream.device.type == "cuda":
            compute_stream.wait_stream(torch.cuda.default_stream(compute_stream.device))

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
        io = self.inputs_and_outputs
        device_io = io.io_pairs[io.current_pair].device_io if isinstance(io, ContinuousBatchingAsyncIOs) else io
        seen_key = (device_io.use_block_table, device_io.num_q_tokens)
        if _PROTO_CAPTURE_THEN_REPLAY and device_io.use_block_table and seen_key in self._proto_eager_seen:
            graph = create_accelerator_graph()
            with accelerator_graph_capture_context(graph, stream=compute_stream, pool=self.graph_pool_id):
                forward_fn(*args)
            self.inputs_and_outputs.set_graph(graph)
            with stream_context(compute_stream):
                graph.replay()
            return
        self._proto_eager_seen.add(seen_key)
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

    def _forward_verify_sample_and_draft(
        self,
        model: nn.Module,
        batch_data: dict,
        carry_over_ids: torch.Tensor,
        prev_output_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> None:
        """gap2-mtp prototype: the forward pass, the verification of the drafted tokens fed as second query token,
        the sampling, and the next draft from the model's MTP module. Static shapes: every logits row computes a
        first token (row `a`: the token verified or sampled after the request's last trusted token) and a bonus token
        (row `b`: the token after the draft, only meaningful when the draft is accepted).

        Sampling stays exact: the draft is a point mass, so it is accepted with probability p(draft) under the target
        distribution and a rejected draft is replaced by a sample from the target distribution without that token.
        The logprobs written are the target model's logprobs of the emitted tokens."""
        if self.logit_processor.do_processing:
            raise NotImplementedError("CB_MTP prototype: logits processors are not supported")
        mtp = batch_data["mtp"]
        model_kwargs = {k: v for k, v in batch_data.items() if k != "mtp"}
        hidden_states = model.model(**model_kwargs).last_hidden_state  # [1, num_q_tokens, hidden], after the final norm

        row_a = batch_data["logits_indices"].long()  # [num_rows]
        has_draft = mtp["has_draft"].bool()
        row_b = row_a + mtp["has_draft"]
        num_rows = row_a.shape[0]
        logits = model.lm_head(hidden_states[:, torch.cat([row_a, row_b])])[0].float()  # [2 * num_rows, vocab]
        logits_a, logits_b = logits[:num_rows], logits[num_rows:]
        draft = batch_data["input_ids"][0, row_b].long()  # only a draft where has_draft

        logprobs_first = logprobs_bonus = None
        if self.do_sample:
            logprobs_a = nn.functional.log_softmax(logits_a, dim=-1)
            draft_logprob = logprobs_a.gather(1, draft[:, None]).squeeze(1)
            uniform = torch.empty_like(draft_logprob).uniform_()
            accepted = has_draft & (uniform < draft_logprob.exp())
            # A rejected draft is resampled from the target distribution without the draft token (the residual
            # distribution of a point-mass proposal); rows without a draft sample from the full distribution
            draft_score = torch.where(has_draft, float("-inf"), logits_a.gather(1, draft[:, None]).squeeze(1))
            noise = torch.empty_like(logits).exponential_().log_()
            resampled = torch.argmax(logits_a.scatter(1, draft[:, None], draft_score[:, None]) - noise[:num_rows], dim=-1)
            first_tokens = torch.where(accepted, draft, resampled)
            bonus_tokens = torch.argmax(logits_b - noise[num_rows:], dim=-1)
            if self.return_logprobs:
                logprobs_first = logprobs_a.gather(1, first_tokens[:, None]).squeeze(1)
                logprobs_bonus = nn.functional.log_softmax(logits_b, dim=-1).gather(1, bonus_tokens[:, None]).squeeze(1)
            # measurement: the target's logprob of the draft the previous step made for this position
            prev_draft_logprob = logprobs_a.gather(1, mtp["prev_draft"].long()[:, None]).squeeze(1)
            output_ids[6, :num_rows].copy_(prev_draft_logprob.view(dtype=torch.int32))
        elif self.return_logprobs:
            # same operations as `_sample`, so the tokens and logprobs compare bit for bit with the non-speculative path
            probs = nn.functional.softmax(logits, dim=-1)
            tokens = torch.argmax(probs, dim=-1, keepdim=True)
            logprobs = probs.gather(dim=1, index=tokens).squeeze(-1).log()
            first_tokens, bonus_tokens = tokens[:num_rows, 0], tokens[num_rows:, 0]
            logprobs_first, logprobs_bonus = logprobs[:num_rows], logprobs[num_rows:]
            accepted = has_draft & (first_tokens == draft)
        else:
            tokens = torch.argmax(logits, dim=-1)
            first_tokens, bonus_tokens = tokens[:num_rows], tokens[num_rows:]
            accepted = has_draft & (first_tokens == draft)

        # Draft the next token of every request: the MTP module runs over every query token of the step, so its own
        # cache holds the same positions as the main layers'. Position p pairs with the token at p + 1
        sampled = torch.cat([first_tokens, bonus_tokens])
        src = mtp["src"].long()
        mtp_input_ids = torch.where(src < 0, mtp["shift_ids"].long(), sampled[src.clamp(min=0)])
        mtp_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
        mtp_hidden_states = model.mtp_forward(hidden_states, mtp_input_ids[None], **mtp_kwargs)
        draft_rows = torch.where(accepted, row_b, row_a)
        draft_logits = model.lm_head(mtp_hidden_states[:, draft_rows])[0]
        if _CB_MTP_DRAFT_SAMPLE and self.do_sample:
            draft_logprobs = nn.functional.log_softmax(draft_logits.float(), dim=-1)
            next_draft = torch.argmax(draft_logprobs - torch.empty_like(draft_logprobs).exponential_().log_(), dim=-1)
            next_draft_logprob = draft_logprobs.gather(1, next_draft[:, None]).squeeze(1)
            output_ids[7, :num_rows].copy_(next_draft_logprob.view(dtype=torch.int32))
        else:
            next_draft = torch.argmax(draft_logits, dim=-1)

        output_ids[0, :num_rows].copy_(first_tokens)
        output_ids[2, :num_rows].copy_(bonus_tokens)
        output_ids[4, :num_rows].copy_(accepted)
        output_ids[5, :num_rows].copy_(next_draft)
        if logprobs_first is not None:
            output_ids[1, :num_rows].copy_(logprobs_first.view(dtype=torch.int32))
            output_ids[3, :num_rows].copy_(logprobs_bonus.view(dtype=torch.int32))

    def _sample(self, scores: torch.Tensor, output_ids: torch.Tensor) -> None:
        """Private method to sample next tokens from the scores."""
        # WP glue (bench only), off by default: Gumbel-max sampling straight from the logits (4 kernels, no softmax and
        # no multinomial) when no logprobs are requested. Same distribution, different random stream
        if _GLUE_SAMPLE and self.do_sample and not self.return_logprobs:
            noise = torch.empty_like(scores[0]).exponential_().log_()
            next_tokens = torch.argmax(scores[0] - noise, dim=-1)  # shape [num_logits]
            output_ids[0, : next_tokens.size(0)].copy_(next_tokens)
            return
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

    @with_fast_tp_all_reduce  # [gap2 prototype] warm-up captures take the engine's fast TP all-reduce
    def run_one_warmup(self, model: nn.Module, num_q_tokens: int, max_kv_read: int | None) -> float:
        """Warms up the decode fast path (if max_kv_read is None) or varlen path (if max_kv_read is an int) for a
        specific number of query and cache-resident tokens. `max_kv_read` is the number of tokens already in cache,
        matching the terminology used by `prepare_batch_tensors` and the scheduler."""
        # Make up fake request states according to the chosen path
        use_decode_fast_path = max_kv_read is None
        if use_decode_fast_path:
            num_requests = num_q_tokens
            status = RequestStatus.DECODING
            num_q_tokens = _CB_MTP_QUERY if _CB_MTP_BT else 1  # gap3-mtp prototype: the last token and the draft
            max_kv_read = self.cb_config.page_size
            logger.debug(f"Warming up decode fast path for {num_requests = }.")
        else:
            num_requests = 1
            status = RequestStatus.PREFILLING
            logger.debug(f"Warming up varlen path for {num_q_tokens = }, {max_kv_read = }.")
        # To limit the compute cost of prefill warmup, we cap the past length to the page size
        past_length = min(max_kv_read, self.cb_config.page_size)
        future_states = create_warmup_future_states(num_requests, status, num_q_tokens, past_length, self.cache)
        if use_decode_fast_path and _CB_MTP_BT:
            for fs in future_states:
                fs.state.mtp_has_draft = True  # gap3-mtp prototype: the decode graphs verify a draft
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

        # Actual warmup, which happens in a try-finally block to ensure the blocks are freed even if the warmup fails
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

        # Exception handling
        except Exception as e:
            duration = 0.0
            logger.warning(f"Failed to warm up: {e}.\nGraph pool may fragment and OOM under load.")

        # In any case, free the blocks allocated for the fake warmup requests
        finally:
            for fs in future_states:
                self.cache.free_blocks(fs.state.request_id)
        return duration
