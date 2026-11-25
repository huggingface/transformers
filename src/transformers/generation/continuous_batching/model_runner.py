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
from math import ceil

import torch
from torch import nn

from ...distributed.parallel_utils import with_fast_tp_all_reduce
from ...generation.configuration_utils import ContinuousBatchingConfig
from ...utils.generic import is_flash_attention_requested
from .cache import PagedAttentionCache
from .cb_logits_processors import ContinuousBatchingLogitsProcessorList, ContinuousBatchingTemperatureLogitsWarper
from ..logits_process import TemperatureLogitsWarper
from .fused_sampler import FusedSampler
from .input_outputs import (
    _CB_MTP_ASYNC,
    _CB_MTP_BT,
    _CB_MTP_K,
    _CB_MTP_MULTI,
    _CB_MTP_QUERY,
    ContinuousBatchingAsyncIOs,
    ContinuousBatchingIOs,
)
from .requests import RequestState, RequestStatus, create_warmup_future_states
from .utils import (
    MLA_PREFILL_MIN_TOKENS,
    accelerator_graph_capture_context,
    create_accelerator_graph,
    get_memory_pools,
    logger,
    memory_pool_context,
    pad_to_interval,
    pad_to_pow2,
    pad_to_token_bucket,
    stream_context,
)


import os as _glue_os

_GLUE_SAMPLE = _glue_os.environ.get("GLUE_SAMPLE", "0") == "1"
# PROTOTYPE of the engine work package (bench only, off by default). CB_PROTO_CAPTURE_THEN_REPLAY=1: a decode batch whose
# padded request count already ran once eagerly (every kernel it uses is compiled and tuned) captures its graph first
# and replays the capture to produce the step's result, instead of an eager forward followed by a capture pass.
_PROTO_CAPTURE_THEN_REPLAY = os.environ.get("CB_PROTO_CAPTURE_THEN_REPLAY", "0") == "1"
# PROTOTYPE of the gap4 zsbench work package (off by default). CB_PROTO_VARLEN_BUCKETS=1: varlen (prefill or mixed)
# batches take a small fixed set of CUDA graph keys, all captured at warm-up (`proto_warmup_varlen_buckets`, on both IO
# pairs) and never in band: the query tokens are padded to a power of 2 (at least CB_PROTO_VARLEN_Q_MIN, 256), the
# cache tokens read to the whole cache (one bucket), max_length_k to a power of 2 (at least 1024) and, in MTP mode, the
# number of sequences to a power of 2 (at least 64). A varlen batch whose key was not captured at warm-up (a longer
# max_length_k than CB_PROTO_VARLEN_MLK_MAX, more sequences than the warm-up's) runs eager, and is logged
_PROTO_MOE_SKIP_PADDING = os.environ.get("PROTO_MOE_SKIP_PADDING", "1") == "1"
_PROTO_VARLEN_BUCKETS = os.environ.get("CB_PROTO_VARLEN_BUCKETS", "0") == "1"
_PROTO_VARLEN_Q_MIN = int(os.environ.get("CB_PROTO_VARLEN_Q_MIN", "256"))
_PROTO_VARLEN_MLK_MAX = int(os.environ.get("CB_PROTO_VARLEN_MLK_MAX", "2048"))
_PROTO_BLAS_READY = __import__("threading").local()  # threads whose cuBLAS handles exist, see compute_batch
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
# [scratch, gap4 zssample] CB_FUSED_SAMPLE=1: sampled requests (with or without logprobs, with or without MTP
# verification) go through `FusedSampler`, two kernels over the bf16 logits instead of an fp32 copy, softmax / log
# softmax, multinomial or noise over the whole vocabulary. Temperature is applied in the kernel; other logits
# processors keep the old path. CB_FUSED_SAMPLE_CHECK=1 also computes the old path's log softmax for the emitted tokens
# and prints the largest difference so far every 50 steps.
_CB_FUSED_SAMPLE = _glue_os.environ.get("CB_FUSED_SAMPLE", "1") == "1"
_CB_FUSED_SAMPLE_CHECK = _glue_os.environ.get("CB_FUSED_SAMPLE_CHECK", "0") == "1"
# CB_FUSED_SAMPLE_SHARDED=1 (with CB_FUSED_SAMPLE=1; no MTP, or MTP spec verification with one draft): when the
# lm_head is vocabulary-sharded (TP plan `colwise_gather_output`), each rank samples its own shard of the logits and
# the ranks all-gather 4 numbers per row instead of the [rows, vocab] logits; the next draft's argmax likewise
_CB_FUSED_SAMPLE_SHARDED = _glue_os.environ.get("CB_FUSED_SAMPLE_SHARDED", "0") == "1"


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
        # [scratch] FA4 decode steps through TRT-LLM's decode kernel: its workspace exists before any graph capture
        if cache.use_varlen_block_table and is_flash_attention_requested(cache.config, version=4):
            from ...modeling_flash_attention_utils import proto_trtllm_decode_workspace

            proto_trtllm_decode_workspace(cache.device)
        self._model_supports_logits_to_keep: bool | None = None  # resolved on first forward
        self._proto_varlen_warm_done = False  # gap4 zsbench prototype, see CB_PROTO_VARLEN_BUCKETS
        self._proto_varlen_eager: dict[tuple[int, ...], int] = {}
        self._proto_eager_seen: set[tuple[bool, int]] = set()  # (decode path, padded query tokens) that ran eagerly
        self._fused_sampler = None
        processors = list(logit_processor.logits_processor)
        temperature_types = (TemperatureLogitsWarper, ContinuousBatchingTemperatureLogitsWarper)
        only_temperature = len(processors) <= 1 and all(isinstance(p, temperature_types) for p in processors)
        if _CB_FUSED_SAMPLE and do_sample and only_temperature:
            # the temperature is applied in the kernel: one for every request, or per request (CB processor)
            self._fused_temperature = processors[0].temperature if processors else None
            self._fused_temperature_per_row = isinstance(processors[0] if processors else None, temperature_types[1])
            # the CB seed is the same on every TP rank (`set_tp_seed`), so every rank samples the same tokens
            self._fused_sampler = FusedSampler(cache.device, torch.initial_seed())
            if _CB_FUSED_SAMPLE_CHECK:
                self._fused_check = torch.zeros(2, dtype=torch.float64, device=cache.device)  # max |dlogprob|, tokens
                self._fused_check_calls = 0

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
        # gap4 zsbench prototype (CB_PROTO_VARLEN_BUCKETS=1): a few fixed keys for the varlen graphs, see the flag
        if not use_decode_fast_path and _PROTO_VARLEN_BUCKETS and self.use_cuda_graph_varlen:
            num_q_tokens = pad_to_pow2(num_q_tokens, max_batch_tokens, _PROTO_VARLEN_Q_MIN)
            max_kv_read = self.cache.max_tokens_read
        # For varlen batches, we pad using interval sizes
        elif not use_decode_fast_path:
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
        # With varlen block tables, any batch can go through the block table: we pad the number of query tokens to its
        # token bucket and use no KV
        elif self.cache.use_varlen_block_table:
            num_q_tokens = pad_to_token_bucket(num_q_tokens, max_batch_tokens)
            max_kv_read = 0
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
        if _CB_FUSED_SAMPLE_CHECK and self._fused_sampler is not None:
            self._fused_check_calls += 1
            if self._fused_check_calls % 50 == 0:  # synchronizes: check mode only
                err, n = self._fused_check.tolist()
                calls = self._fused_check_calls
                print(f"FUSED_SAMPLE_CHECK calls={calls} max_abs_dlogprob={err:.3e} rows={int(n)}", flush=True)
        # This is the stream on which the compute happens
        compute_stream = self.inputs_and_outputs.compute_stream

        # If supported, the model slices hidden states at logits_indices before lm_head, instead of us slicing logits
        if self.supports_logits_to_keep(model) and not _CB_MTP:
            batch_data["logits_to_keep"] = batch_data["logits_indices"]

        # [scratch] the proto_decode experts route the slots of padding tokens nowhere: they read the number of real
        # tokens of the step from the end of cu_seq_lens_q (a fixed address per graph key)
        if _PROTO_MOE_SKIP_PADDING:
            from ...integrations import _proto_bf16_experts

            cu_seq_lens_q = batch_data["cu_seq_lens_q"]
            _proto_bf16_experts.VALID_TOKENS = cu_seq_lens_q[-1:] if self.pad_inputs else None

        # Get the appropriate forward function (compiled or not, based on current path)
        forward_fn, use_cuda_graph = self._get_forward_fn(use_block_table=self.inputs_and_outputs.use_block_table)
        if _CB_MTP:
            forward_fn = self._forward_verify_k if _CB_MTP_MULTI else self._forward_verify_sample_and_draft

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
            # gap4 zsbench prototype: after the bucket warm-up, a varlen key that was not captured runs eager
            elif (
                _PROTO_VARLEN_BUCKETS
                and self._proto_varlen_warm_done
                and not self.inputs_and_outputs.use_block_table
            ):
                io = self.inputs_and_outputs
                device_io = io.io_pairs[io.current_pair].device_io if isinstance(io, ContinuousBatchingAsyncIOs) else io
                key = device_io._get_graph_key()
                count = self._proto_varlen_eager[key] = self._proto_varlen_eager.get(key, 0) + 1
                if count & (count - 1) == 0:  # 1, 2, 4, 8, ... occurrences
                    logger.warning(f"VARLEN_BUCKETS eager varlen batch, key {key} not captured at warm-up ({count} so far)")
                with stream_context(compute_stream):
                    forward_fn(model, batch_data, carry_over_ids, prev_output_ids, output_ids)
            # Otherwise, the graph does not exist, so we create it
            else:
                if _PROTO_VARLEN_BUCKETS and not getattr(_PROTO_BLAS_READY, "done", False):
                    # gap4 zsbench prototype: cuBLAS / cuBLASLt create their handles lazily, per thread, and creating
                    # one inside a capture fails (CUBLAS_STATUS_NOT_INITIALIZED). With the varlen graphs all captured
                    # at warm-up on the caller's thread, the engine thread may capture before it ran any eager GEMM
                    device = torch.device("cuda", torch.cuda.current_device())
                    with stream_context(compute_stream):
                        x = torch.ones(16, 64, device=device, dtype=torch.bfloat16)
                        nn.functional.linear(x, x, x[:, 0])  # addmm with a bias: cuBLASLt
                        torch.mm(x, x.t())  # cuBLAS
                    torch.cuda.current_blas_handle()
                    _PROTO_BLAS_READY.done = True
                args = (model, batch_data, carry_over_ids, prev_output_ids, output_ids)
                self._capture_graph(forward_fn, compute_stream, *args)

    def _get_forward_fn(self, use_block_table: bool) -> tuple[Callable, bool]:
        """Helper function to get the appropriate forward function based on the block table and compile behavior."""
        if use_block_table:
            forward_fn = self._forward_process_and_sample if self._compiled_decode is None else self._compiled_decode
            use_cuda_graph = self.use_cuda_graph_decode
            # [scratch] With varlen block tables, a step that brings several tokens for some sequence is a varlen step:
            # it takes a graph only if the varlen path does (not on models whose layers read lengths on the host)
            io = self.inputs_and_outputs
            host_io = io.io_pairs[io.current_pair].host_io if isinstance(io, ContinuousBatchingAsyncIOs) else io
            if use_cuda_graph and self.cache.use_varlen_block_table and not self.use_cuda_graph_varlen:
                use_cuda_graph = host_io.max_length_q == 1
                # Nor is it compiled: those layers break the graph on every call, and each new batch made dynamo
                # recompile (Qwen3-Next rolling: ~170 ms of host time per mixed step)
                if not use_cuda_graph:
                    forward_fn = self._forward_process_and_sample
            # [scratch] Latent attention: steps with prompts attend through flash attention on decompressed keys, which
            # reads the step's lengths on the host (see `utils.MLA_PREFILL_MIN_TOKENS`): they run eagerly
            if use_cuda_graph and self.cache.latent_attention and host_io.num_q_tokens >= MLA_PREFILL_MIN_TOKENS:
                use_cuda_graph = False
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

        vocab_shard = self._vocab_shard(model)
        if vocab_shard is not None:
            # [gap4 zssample] vocabulary-parallel sampling: this rank's shard of the sampled rows' logits, no all-gather
            kwargs = {k: v for k, v in batch_data.items() if k != "logits_to_keep"}
            hidden_states = model.model(**kwargs).last_hidden_state[:, batch_data["logits_indices"]]
            logits = nn.functional.linear(hidden_states, vocab_shard[0])[0]  # [num_logits, vocab / tp]
            temperature = self._fused_rows_temperature(batch_data, logits.shape[0])
            tokens, logprobs, _, _, _ = self._fused_sampler(logits, temperature, None, vocab_shard[1], vocab_shard[2])
            if _CB_FUSED_SAMPLE_CHECK:
                self._maybe_check_fused(model.lm_head(hidden_states)[0], temperature, tokens, logprobs)
            output_ids[0, : tokens.size(0)].copy_(tokens)
            if self.return_logprobs:
                output_ids[1, : tokens.size(0)].copy_(logprobs.view(dtype=torch.int32))
            return

        # Run model forward pass
        logits = model(**batch_data).logits  # shape [1, seq_len OR num_logits, vocab_size]

        # If it has not been done by the model, extract only the logits that are used to predict new tokens
        logits_indices = batch_data["logits_indices"]  # shape [num_logits]
        if "logits_to_keep" not in batch_data:
            logits = logits[:, logits_indices, :]  # shape [1, num_logits, vocab_size]
        if self._fused_sampler is not None:
            temperature = self._fused_rows_temperature(batch_data, logits.shape[1])
            tokens, logprobs, _, _, _ = self._fused_sampler(logits[0], temperature)
            self._maybe_check_fused(logits[0], temperature, tokens, logprobs)
            output_ids[0, : tokens.size(0)].copy_(tokens)
            if self.return_logprobs:
                output_ids[1, : tokens.size(0)].copy_(logprobs.view(dtype=torch.int32))
            return
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

    @staticmethod
    def _mtp_take_previous_step(batch_data: dict, carry_over_ids: torch.Tensor, prev_output_ids: torch.Tensor) -> None:
        """gap3-mtpa prototype (CB_MTP_ASYNC=1, block table decode batches): a request that was in the previous step
        (still in flight when this step was prepared) takes from that step's outputs its last token (the bonus token
        if the draft was accepted, else the first token) and its new draft, as its two query tokens, and moves its
        positions and key length by one when the draft was accepted (the host counted it as rejected).
        `carry_over_ids[t]` is the previous step's logits row of the request whose first query token is token `t`, -1
        elsewhere. Every index is clamped, so a wrong map can only give wrong values."""
        input_ids = batch_data["input_ids"][0]  # [Q]
        num_tokens = input_ids.shape[0]
        carry = carry_over_ids[:num_tokens].long()
        has_prev = carry >= 0
        prev_row = carry.clamp(0, prev_output_ids.shape[1] - 1)
        accepted = prev_output_ids[4, prev_row]  # int32, 0 for rows without a draft
        last_token = torch.where(accepted != 0, prev_output_ids[2, prev_row], prev_output_ids[0, prev_row])
        new_draft = prev_output_ids[5, prev_row]
        zero = has_prev[:1] & False
        draft_slot = torch.cat([zero, has_prev[:-1]])  # the token right after a carried first token is its draft
        new_ids = torch.where(has_prev, last_token, input_ids)
        new_ids = torch.where(draft_slot, torch.cat([new_draft[:1] * 0, new_draft[:-1]]), new_ids)
        input_ids.copy_(new_ids)
        # the shift of the request's two tokens, then of its key length (cumulated over the requests before it)
        shift = torch.where(has_prev, accepted, 0)
        shift = shift + torch.cat([shift[:1] * 0, shift[:-1]])
        batch_data["position_ids"][0].add_(shift)
        cu_q = batch_data["cu_seq_lens_q"].long()
        first = cu_q[:-1].clamp(max=num_tokens - 1)
        request_shift = torch.where(cu_q[:-1] < num_tokens, torch.where(has_prev, accepted, 0)[first], 0)
        for cu_k in batch_data["cu_seq_lens_k"].values():
            cu_k[1:].add_(torch.cumsum(request_shift, 0).to(cu_k.dtype))

    @staticmethod
    def _mtp_take_previous_step_k(batch_data: dict, carry_over_ids: torch.Tensor, prev_output_ids: torch.Tensor) -> None:
        """gap3-mtpk prototype: `_mtp_take_previous_step` for k drafts. A request carried from the step in flight
        takes its last emitted token (the candidate of verified row `n_acc`), its k new drafts, and moves its positions
        and key length by `n_acc`. Output rows: see `ContinuousBatchingIOs.num_output_rows`."""
        k = _CB_MTP_K
        input_ids = batch_data["input_ids"][0]  # [Q]
        num_tokens = input_ids.shape[0]
        carry = carry_over_ids[:num_tokens].long()
        has_prev = carry >= 0
        prev_row = carry.clamp(0, prev_output_ids.shape[1] - 1)
        n_acc = prev_output_ids[2 * k + 2, prev_row].long().clamp(0, k)
        new_ids = torch.where(has_prev, prev_output_ids[n_acc, prev_row], input_ids)
        first_shift = torch.where(has_prev, n_acc, 0)
        shift = first_shift
        for j in range(1, k + 1):  # the j-th token after a carried first token is the request's j-th draft
            slot = torch.cat([has_prev[:j] & False, has_prev[:-j]])
            draft = prev_output_ids[2 * k + 2 + j, prev_row]
            new_ids = torch.where(slot, torch.cat([draft[:j] * 0, draft[:-j]]), new_ids)
            shift = shift + torch.cat([first_shift[:j] * 0, first_shift[:-j]])
        input_ids.copy_(new_ids)
        position_ids = batch_data["position_ids"][0]
        position_ids.add_(shift.to(position_ids.dtype))
        cu_q = batch_data["cu_seq_lens_q"].long()
        first = cu_q[:-1].clamp(max=num_tokens - 1)
        request_shift = torch.where(cu_q[:-1] < num_tokens, first_shift[first], 0)
        for cu_k in batch_data["cu_seq_lens_k"].values():
            cu_k[1:].add_(torch.cumsum(request_shift, 0).to(cu_k.dtype))

    def _forward_verify_k(
        self,
        model: nn.Module,
        batch_data: dict,
        carry_over_ids: torch.Tensor,
        prev_output_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> None:
        """gap3-mtpk prototype (CB_MTP_K=k): a decoding request brings its last token and k drafts. The target model
        verifies them in one forward (rows 0..k of the request); drafts are accepted left to right, each with
        probability p(draft) under the target distribution (point-mass drafts), the first rejected one replaced by a
        sample of the target distribution without it, and if all are accepted, a bonus token comes from row k. Then
        the MTP layer drafts the next k tokens: once over every query token (as with one draft), then k - 1 times on
        one token per request, each pass taking the previous pass's output and draft, at the next position, through the
        block table. Static shapes; rows without a draft (the step that ends a prefill) verify nothing."""
        if self.logit_processor.do_processing and self._fused_sampler is None:
            raise NotImplementedError("CB_MTP prototype: logits processors are not supported")
        k = _CB_MTP_K
        mtp = batch_data["mtp"]
        if _CB_MTP_ASYNC:
            self._mtp_take_previous_step_k(batch_data, carry_over_ids, prev_output_ids)
        model_kwargs = {key: v for key, v in batch_data.items() if key != "mtp"}
        hidden_states = model.model(**model_kwargs).last_hidden_state  # [1, num_q_tokens, hidden], after the final norm

        input_ids = batch_data["input_ids"][0]
        num_q = input_ids.shape[0]
        row0 = batch_data["logits_indices"].long()  # [R] the request's first query token
        num_rows = row0.shape[0]
        has_draft = mtp["has_draft"].bool()
        steps = torch.arange(k + 1, device=row0.device)
        rows = row0[None] + steps[:, None] * mtp["has_draft"].long()[None]  # [k + 1, R]
        logits = model.lm_head(hidden_states[:, rows.reshape(-1)])[0]  # [(k + 1) * R, vocab], row-major in j
        if self._fused_sampler is None:
            logits = logits.float()
        drafts = input_ids[(row0[None] + steps[1:, None]).clamp(max=num_q - 1)].long()  # [k, R]
        logprobs = None
        if self._fused_sampler is not None:
            # rows 0..k-1 verify their draft (ratio test, residual sample if rejected), row k samples the bonus token
            verify = torch.cat([torch.where(has_draft[None], drafts, -1), torch.full_like(drafts[:1], -1)])
            temperature = self._fused_rows_temperature(batch_data, num_rows, k + 1)
            tokens, token_logprobs, _, _, accepted = self._fused_sampler(logits, temperature, verify.reshape(-1))
            self._maybe_check_fused(logits, temperature, tokens, token_logprobs)
            candidates = tokens.view(k + 1, num_rows)
            accepted = accepted.view(k + 1, num_rows)[:k]
            if self.return_logprobs:
                logprobs = token_logprobs.view(k + 1, num_rows)
        elif self.do_sample:
            logp = nn.functional.log_softmax(logits, dim=-1)
            draft_logp = logp.view(k + 1, num_rows, -1)[:k].gather(2, drafts[..., None]).squeeze(2)  # [k, R]
            accepted = has_draft[None] & (torch.empty_like(draft_logp).uniform_() < draft_logp.exp())
            # a rejected draft is resampled from the target distribution without it (Gumbel-max); row k samples the
            # bonus token from the full distribution, as do rows without a draft
            view = logits.view(k + 1, num_rows, -1)
            kept = torch.where(has_draft[None], float("-inf"), view[:k].gather(2, drafts[..., None]).squeeze(2))
            view[:k].scatter_(2, drafts[..., None], kept[..., None])
            noise = torch.empty_like(logits).exponential_().log_()
            sampled = (logits - noise).argmax(-1).view(k + 1, num_rows)
            candidates = torch.cat([torch.where(accepted, drafts, sampled[:k]), sampled[k:]])  # [k + 1, R]
            if self.return_logprobs:
                logprobs = logp.gather(1, candidates.reshape(-1, 1)).view(k + 1, num_rows)
        elif self.return_logprobs:
            # same operations as `_sample`, so tokens and logprobs compare bit for bit with the non-speculative path
            probs = nn.functional.softmax(logits, dim=-1)
            tokens = torch.argmax(probs, dim=-1, keepdim=True)
            logprobs = probs.gather(dim=1, index=tokens).squeeze(-1).log().view(k + 1, num_rows)
            candidates = tokens.view(k + 1, num_rows)
            accepted = has_draft[None] & (candidates[:k] == drafts)
        else:
            candidates = torch.argmax(logits, dim=-1).view(k + 1, num_rows)
            accepted = has_draft[None] & (candidates[:k] == drafts)
        num_accepted = accepted.int().cumprod(0).sum(0)  # [R] leading accepted drafts

        # Draft pass 1: the MTP layer over every query token, query token j paired with the candidate of row j
        src = mtp["src"].long()
        mtp_input_ids = torch.where(src < 0, mtp["shift_ids"].long(), candidates.reshape(-1)[src.clamp(min=0)])
        mtp_kwargs = {key: v for key, v in model_kwargs.items() if key != "input_ids"}
        mtp_normed, mtp_hidden = model.mtp_forward(hidden_states, mtp_input_ids[None], return_pre_norm=True, **mtp_kwargs)
        last_rows = row0 + num_accepted  # the row of the last emitted token's predecessor: its draft comes next
        draft = model.lm_head(mtp_normed[:, last_rows])[0].argmax(-1)  # [R]
        next_drafts = [draft]
        if k > 1 and batch_data.get("block_table") is None:
            # varlen batch (a prefill in it): the later passes need the block table, so the request's next step gets
            # its first draft only, repeated (valid tokens, verified like any other draft)
            next_drafts = [draft] * k
        elif k > 1:
            # Passes 2..k: one token per request at the next position (the draft of the previous pass as input, its
            # output as previous hidden state), its own key row written through the block table
            prev_hidden = mtp_hidden[:, last_rows]
            ((layer_type, cu_k_main),) = batch_data["cu_seq_lens_k"].items()
            real = (cu_k_main[1:] - cu_k_main[:-1]) > 0
            base = batch_data["position_ids"][0, row0].long() + num_accepted  # position before the last emitted token
            step_kwargs = dict(mtp_kwargs)
            step_kwargs["cu_seq_lens_q"] = torch.arange(num_rows + 1, device=row0.device, dtype=cu_k_main.dtype)
            cache = batch_data["cache"]
            for i in range(2, k + 1):
                position = base + (i - 1)
                seqlen = torch.where(real, position + 1, 0)
                step_kwargs["cu_seq_lens_k"] = {
                    layer_type: torch.cat([seqlen[:1] * 0, torch.cumsum(seqlen, 0)]).to(cu_k_main.dtype)
                }
                step_kwargs["position_ids"] = position[None].to(batch_data["position_ids"].dtype)
                cache._proto_bt_state = None  # the block table rows of this pass, not the main forward's
                normed, prev_hidden = model.mtp_forward(prev_hidden, draft[None], return_pre_norm=True, **step_kwargs)
                draft = model.lm_head(normed)[0].argmax(-1)
                next_drafts.append(draft)

        output_ids[: k + 1, :num_rows].copy_(candidates)
        if logprobs is not None:
            output_ids[k + 1 : 2 * k + 2, :num_rows].copy_(logprobs.float().contiguous().view(dtype=torch.int32))
        output_ids[2 * k + 2, :num_rows].copy_(num_accepted)
        output_ids[2 * k + 3 : 3 * k + 3, :num_rows].copy_(torch.stack(next_drafts))

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
        if self.logit_processor.do_processing and self._fused_sampler is None:
            raise NotImplementedError("CB_MTP prototype: logits processors are not supported")
        mtp = batch_data["mtp"]
        if _CB_MTP_ASYNC:
            self._mtp_take_previous_step(batch_data, carry_over_ids, prev_output_ids)
        model_kwargs = {k: v for k, v in batch_data.items() if k != "mtp"}
        hidden_states = model.model(**model_kwargs).last_hidden_state  # [1, num_q_tokens, hidden], after the final norm

        row_a = batch_data["logits_indices"].long()  # [num_rows]
        has_draft = mtp["has_draft"].bool()
        row_b = row_a + mtp["has_draft"]
        num_rows = row_a.shape[0]
        vocab_shard = self._vocab_shard(model)
        if vocab_shard is not None:
            # this rank's vocabulary shard of the logits, no all-gather
            logits = nn.functional.linear(hidden_states[:, torch.cat([row_a, row_b])], vocab_shard[0])[0]
        else:
            logits = model.lm_head(hidden_states[:, torch.cat([row_a, row_b])])[0]  # [2 * num_rows, vocab]
        if self._fused_sampler is None:
            logits = logits.float()
        logits_a, logits_b = logits[:num_rows], logits[num_rows:]
        draft = batch_data["input_ids"][0, row_b].long()  # only a draft where has_draft

        logprobs_first = logprobs_bonus = None
        if self._fused_sampler is not None:
            # row a verifies the draft (ratio test, residual sample if rejected), row b samples the bonus token
            verify = torch.cat([torch.where(has_draft, draft, -1), torch.full_like(draft, -1)])
            temperature = self._fused_rows_temperature(batch_data, num_rows, 2)
            if vocab_shard is not None:
                _, group, offset = vocab_shard
                tokens, logprobs, logz, _, accepted = self._fused_sampler(logits, temperature, verify, group, offset)
                if _CB_FUSED_SAMPLE_CHECK:
                    full = model.lm_head(hidden_states[:, torch.cat([row_a, row_b])])[0]
                    self._maybe_check_fused(full, temperature, tokens, logprobs)
            else:
                tokens, logprobs, logz, _, accepted = self._fused_sampler(logits, temperature, verify)
                self._maybe_check_fused(logits, temperature, tokens, logprobs)
                # measurement: the target's logprob of the draft the previous step made for this position (read
                # by the "shadow" mode only, not written by the vocabulary-parallel path)
                prev_draft_logit = logits_a.gather(1, mtp["prev_draft"].long()[:, None]).squeeze(1).float()
                if temperature is not None:
                    per_row = self._fused_temperature_per_row
                    prev_draft_logit = prev_draft_logit / (temperature[:num_rows] if per_row else temperature)
                output_ids[6, :num_rows].copy_((prev_draft_logit - logz[:num_rows]).view(dtype=torch.int32))
            first_tokens, bonus_tokens, accepted = tokens[:num_rows], tokens[num_rows:], accepted[:num_rows]
            if self.return_logprobs:
                logprobs_first, logprobs_bonus = logprobs[:num_rows], logprobs[num_rows:]
        elif self.do_sample:
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
        if vocab_shard is not None:
            # argmax of each rank's shard, then of the gathered (max, id) pairs: ties go to the lowest id, as argmax
            weight, group, offset = vocab_shard
            best, index = nn.functional.linear(mtp_hidden_states[:, draft_rows], weight)[0].max(-1)
            pair = torch.stack([best.float(), (index + offset).float()], -1)  # ids below 2^24 are exact in fp32
            pairs = pair.new_empty(torch.distributed.get_world_size(group), *pair.shape)
            torch.distributed.all_gather_into_tensor(pairs, pair, group=group)
            next_draft = pairs[..., 1].gather(0, pairs[..., 0].argmax(0, keepdim=True))[0].long()
        else:
            draft_logits = model.lm_head(mtp_hidden_states[:, draft_rows])[0]
            if _CB_MTP_DRAFT_SAMPLE and self.do_sample:
                draft_logprobs = nn.functional.log_softmax(draft_logits.float(), dim=-1)
                noise = torch.empty_like(draft_logprobs).exponential_().log_()
                next_draft = torch.argmax(draft_logprobs - noise, dim=-1)
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

    def _vocab_shard(self, model: nn.Module):
        """(local lm_head weight, TP group, first vocabulary id of this rank) when CB_FUSED_SAMPLE_SHARDED applies."""
        if not (_CB_FUSED_SAMPLE_SHARDED and self._fused_sampler is not None):
            return None
        from torch.distributed.tensor import DTensor

        weight = model.lm_head.weight
        if not isinstance(weight, DTensor):
            return None
        local = weight.to_local()
        group = weight.device_mesh.get_group()
        assert weight.shape[0] % torch.distributed.get_world_size(group) == 0, "uneven vocabulary shards"
        return local, group, torch.distributed.get_rank(group) * local.shape[0]

    def _fused_rows_temperature(self, batch_data: dict, num_rows: int, copies: int = 1):
        """The temperature of the fused sampler's rows: None, one float, or per row (`copies` blocks of the batch's
        requests, as the MTP paths lay out their logits rows)."""
        if not self._fused_temperature_per_row:
            return self._fused_temperature
        return batch_data["logits_processor_args"][0, :num_rows].view(dtype=torch.float32).repeat(copies)

    def _maybe_check_fused(self, logits, temperature, tokens, logprobs) -> None:
        """CB_FUSED_SAMPLE_CHECK: largest |fused logprob - log_softmax(fp32 logits / T)[token]| so far, on device."""
        if not _CB_FUSED_SAMPLE_CHECK:
            return
        scaled = logits.float()
        if isinstance(temperature, torch.Tensor):
            scaled = scaled / temperature[:, None]
        elif temperature is not None:
            scaled = scaled / temperature
        ref = nn.functional.log_softmax(scaled, dim=-1).gather(1, tokens[:, None]).squeeze(1)
        diff = (ref - logprobs).abs().max().double()
        torch.maximum(self._fused_check[:1], diff[None], out=self._fused_check[:1])
        self._fused_check[1:].add_(tokens.numel())

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

        # [scratch] Dynamo compiles some frames once per layer (guards on the layer index or the module): with the default
        # limit of 8 recompiles, the layers after the 8th ran eager inside the CUDA graphs. Not for FP8 checkpoints: their
        # compiled layers were 2-4x slower than the eager ones (Qwen3-30B-A3B-FP8 H100). Under tensor parallelism too since
        # the all-reduce call counter no longer recompiles every frame (Qwen3-235B-A22B TP2 B300 64 x 1,024: 1,466 / 1,451
        # -> 1,510 / 1,519 tok/s)
        quantization = getattr(model.config, "quantization_config", None)
        quant_method = quantization.get("quant_method") if isinstance(quantization, dict) else None
        quant_method = quant_method or getattr(quantization, "quant_method", None)
        if str(quant_method) not in ("fp8", "QuantizationMethod.FP8"):
            torch._dynamo.config.recompile_limit = max(
                torch._dynamo.config.recompile_limit, int(os.environ.get("PROTO_RECOMPILE_LIMIT", 256))
            )
            torch._dynamo.config.accumulated_recompile_limit = max(
                torch._dynamo.config.accumulated_recompile_limit, 4096
            )

        # [scratch] Blackwell: bf16 MoE experts shuffled once for TRT-LLM's kernel, before any graph capture
        from ...integrations._proto_bf16_experts import trtllm_bf16_prepare_model

        num_prepared = trtllm_bf16_prepare_model(model)
        if num_prepared:
            logger.warning(f"PROTO_TRTLLM_BF16_MOE: {num_prepared} expert modules on TRT-LLM's bf16 MoE")

        # In async mode, each IO pair has its own graph buffer and static tensors, so we warm up both
        total_duration = 0
        iterations = 2 if isinstance(self.inputs_and_outputs, ContinuousBatchingAsyncIOs) else 1
        for _ in range(iterations):
            # With varlen block tables, steps with prompt tokens also go through the block table. Warm up its largest
            # token bucket to get the biggest pool and avoid fragmentation: the other buckets, few and fixed, are
            # captured on first use
            if self.cache.use_varlen_block_table:
                num_q_tokens = self.cache.max_batch_tokens
                total_duration += self.run_one_warmup(model=model, num_q_tokens=num_q_tokens, max_kv_read=0)
            # Otherwise, warm up the varlen path, w/ the largest possible sizes to get the biggest pool and avoid
            # fragmentation
            else:
                num_q_tokens = self.cache.max_batch_tokens
                max_kv_read = max(0, infer_max_single_request_tokens(self.cache) - num_q_tokens)
                total_duration += self.run_one_warmup(model=model, num_q_tokens=num_q_tokens, max_kv_read=max_kv_read)

            # [scratch] Latent attention: its decode-only varlen steps (max_length_q = 1) compile apart, warm them up
            if self.cache.latent_attention and self.cache.max_decode_fast_path_length == 0:
                num_requests = self.cb_config.max_requests_per_batch
                total_duration += self.run_one_warmup(model, num_requests, None, decode_varlen=True)

            # Warm up the decode fast path if it is available
            if self.cache.max_decode_fast_path_length > 0:
                num_requests = 1
                self._warmup_failed = False
                while True:
                    total_duration += self.run_one_warmup(model=model, num_q_tokens=num_requests, max_kv_read=None)
                    # Some models' layers do not compile or capture at decode sizes: their decode steps stay varlen
                    if self._warmup_failed:
                        logger.warning("The decode fast path failed to warm up, decode steps go through the varlen path.")
                        self.cache.max_decode_fast_path_length = 0
                        self.cache.use_varlen_block_table = False
                        break
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
    @torch.no_grad()
    def proto_warmup_varlen_buckets(self, model: nn.Module) -> int:
        """gap4 zsbench prototype (CB_PROTO_VARLEN_BUCKETS=1): captures the varlen graph of every bucket key on the IO
        pair in use (call it once per IO pair under async batching, then set `_proto_varlen_warm_done`). One fake
        request of q tokens with a one-page past for every q bucket (powers of 2 from CB_PROTO_VARLEN_Q_MIN to
        max_batch_tokens) and every max_length_k bucket (powers of 2 from 1024 to CB_PROTO_VARLEN_MLK_MAX). Returns the
        number of graphs captured."""
        if not (_PROTO_VARLEN_BUCKETS and self.use_cuda_graph_varlen):
            return 0
        io = self.inputs_and_outputs
        q_buckets, q = [], _PROTO_VARLEN_Q_MIN
        while q < self.cache.max_batch_tokens:
            q_buckets.append(q)
            q *= 2
        q_buckets.append(self.cache.max_batch_tokens)
        mlk_buckets, m = [], 1024
        while m <= _PROTO_VARLEN_MLK_MAX:
            mlk_buckets.append(m)
            m *= 2
        captured = 0
        for num_q_tokens in q_buckets:
            for max_length_k in mlk_buckets:
                past_length = self.cb_config.page_size
                future_states = create_warmup_future_states(
                    1, RequestStatus.PREFILLING, num_q_tokens, past_length, self.cache
                )
                if not future_states:
                    logger.warning(f"VARLEN_BUCKETS warm-up: no blocks for q={num_q_tokens}")
                    continue
                try:
                    padded_q, padded_kv = self.maybe_pad_inputs(num_q_tokens, past_length, use_decode_fast_path=False)
                    io.prepare_batch_tensors(
                        future_states, self.logit_processor, False, padded_q, padded_kv, use_padding=True
                    )
                    batch_data = io.get_model_kwargs(use_padding=True)
                    device_io = io.io_pairs[io.current_pair].device_io if isinstance(io, ContinuousBatchingAsyncIOs) else io
                    for allocator in self.cache.cache_allocators.values():
                        batch_data["max_length_k"][allocator.layer_type] = max_length_k
                        device_io.max_length_k[allocator.layer_type] = max_length_k
                    if device_io.get_graph() is None:
                        self.compute_batch(model, batch_data)
                        captured += 1
                finally:
                    for fs in future_states:
                        self.cache.free_blocks(fs.state.request_id)
        return captured

    def run_one_warmup(
        self, model: nn.Module, num_q_tokens: int, max_kv_read: int | None, decode_varlen: bool = False
    ) -> float:
        """Warms up the decode fast path (if max_kv_read is None) or varlen path (if max_kv_read is an int) for a
        specific number of query and cache-resident tokens. `max_kv_read` is the number of tokens already in cache,
        matching the terminology used by `prepare_batch_tensors` and the scheduler. With `decode_varlen`, a decode-only
        step of `num_q_tokens` requests goes through the varlen path."""
        # Make up fake request states according to the chosen path
        use_decode_fast_path = not decode_varlen and (max_kv_read is None or self.cache.use_varlen_block_table)
        if decode_varlen:
            num_requests, num_q_tokens = num_q_tokens, 1
            status = RequestStatus.DECODING
            max_kv_read = self.cb_config.page_size
        elif max_kv_read is None:
            num_requests = num_q_tokens
            status = RequestStatus.DECODING
            num_q_tokens = _CB_MTP_QUERY if _CB_MTP_BT else 1  # gap3-mtp prototype: the last token and the draft
            max_kv_read = self.cb_config.page_size
            logger.debug(f"Warming up decode fast path for {num_requests = }.")
        elif use_decode_fast_path:
            # Prefills of num_q_tokens tokens in all through the block table, as few requests as it allows
            num_requests = ceil(
                num_q_tokens / min(self.cache.max_decode_fast_path_length, self.cache.max_request_step_tokens)
            )
            num_q_tokens = num_q_tokens // num_requests
            status = RequestStatus.PREFILLING
            logger.debug(f"Warming up varlen block table path for {num_requests = }, {num_q_tokens = }.")
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
            # The next warmup rewrites the static inputs this forward reads, so wait for it to finish
            if self.inputs_and_outputs.compute_stream is not None:
                self.inputs_and_outputs.compute_stream.synchronize()
            duration = time.perf_counter() - start
            logger.debug(f"Warmup completed in {duration:.2f}s")

        # Exception handling
        except Exception as e:
            duration = 0.0
            self._warmup_failed = True
            logger.warning(f"Failed to warm up: {e}.\nGraph pool may fragment and OOM under load.")

        # In any case, free the blocks allocated for the fake warmup requests
        finally:
            for fs in future_states:
                self.cache.free_blocks(fs.state.request_id)
        return duration
