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
from functools import partial
from itertools import repeat
from typing import TypedDict

import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import ContinuousBatchingConfig

from ...utils import get_available_devices
from .cache import PagedAttentionCache
from .cache_allocators import FULL_ATTENTION, INDEXED_ATTENTION, SLIDING_ATTENTION, FullAttentionCacheAllocator
from .cache_allocators.compressed_latents import (
    _PROTO_MLA_BLOCK_TABLE,
    _PROTO_MLA_BLOCK_TABLE_CHECK,
    MLACacheAllocator,
)
from .cb_logits_processors import ContinuousBatchingLogitsProcessorList
from .requests import TMP_TOKEN_ID, FutureRequestState
from .utils import (
    CudaGraphBuffer,
    aligned_divide,
    attn_mask_is_needed,
    proto_no_mask,
    build_attention_mask,
    create_device_stream,
    logger,
    pad_to_interval,
    pad_to_pow2,
    stream_context,
)


# --- PROTOTYPES of the engine work package (bench only, all off by default) ---------------------------------------
# CB_PROTO_DEVICE_INDEX=1: the read index of full-attention-like allocators is expanded on the device from the block
# ids of the requests (O(requests x blocks) host work) instead of built token by token on the host. "check" also
# builds the host version and asserts both are equal.
_PROTO_DEVICE_INDEX = os.environ.get("CB_PROTO_DEVICE_INDEX", "0")
# CB_PROTO_SKIP_MASK_MAX_Q=<n>: the attention mask is neither reset nor built for batches of at most n query tokens.
# Only valid with a model that does not read the mask on such batches (the GLM prototype with the per-query indexer).
_PROTO_SKIP_MASK_MAX_Q = int(os.environ.get("CB_PROTO_SKIP_MASK_MAX_Q", "0"))
# CB_PROTO_DEVICE_MASK=1: the attention mask is built by one broadcasted comparison on the device instead of one
# fill_ + triu_ per request. "check" also builds the per-request version and asserts both are equal.
_PROTO_DEVICE_MASK = os.environ.get("CB_PROTO_DEVICE_MASK", "0")
# CB_PROTO_ONE_UPLOAD=1 (with CB_PROTO_MLA_BLOCK_TABLE): a block-table decode batch writes its small inputs with one
# host buffer and one copy instead of five, and uploads the block table only when the batch's requests or their block
# lists changed since the previous decode batch.
_PROTO_ONE_UPLOAD = os.environ.get("CB_PROTO_ONE_UPLOAD", "0") == "1"


def _proto_build_attention_mask_on_device(
    attention_mask: torch.Tensor,  # [1, 1, M, N + M], only [:q_len, :k_len] is written
    cumulative_seqlens_q: list[int],
    cumulative_seqlens_k: list[int],
    q_len: int,
    k_len: int,
    arange: torch.Tensor,
) -> None:
    """Same mask as `_reset_static_tensors` + `build_attention_mask` (no sliding window), from the cumulative
    sequence lengths alone: a query attends a key when both belong to the same request and the key is not after it."""
    device = attention_mask.device
    num_requests = len(cumulative_seqlens_q) - 1
    cu_q = torch.tensor(cumulative_seqlens_q, dtype=torch.int64, device=device)
    cu_k = torch.tensor(cumulative_seqlens_k, dtype=torch.int64, device=device)
    q_pos, k_pos = arange[:q_len], arange[:k_len]
    q_req = torch.searchsorted(cu_q[1:], q_pos, right=True)  # num_requests for the padding tokens
    k_req = torch.searchsorted(cu_k[1:], k_pos, right=True)
    q_req_real = q_req.clamp(max=num_requests - 1)
    past = (cu_k[1:] - cu_k[:-1]) - (cu_q[1:] - cu_q[:-1])  # cached tokens of each request
    last_visible = past[q_req_real] + (q_pos - cu_q[q_req_real])  # per query, as an offset in its request's keys
    k_offset = k_pos - cu_k[k_req.clamp(max=num_requests - 1)]
    q_req = q_req.masked_fill(q_req == num_requests, -1)  # padding queries see nothing
    allowed = q_req[:, None] == k_req[None, :]
    allowed &= k_offset[None, :] <= last_visible[:, None]
    block = attention_mask[0, 0, :q_len, :k_len]
    block.fill_(torch.finfo(attention_mask.dtype).min)
    block.masked_fill_(allowed, 0)


# WP prefill (bench only): smallest key-length bucket of prefill batches under CUDA graphs (0: the default 1024)
_PROTO_PREFILL_MIN_LEN_K = int(os.environ.get("CB_PROTO_PREFILL_MIN_LEN_K", "0"))
_PROTO_VARLEN_BUCKETS_IO = os.environ.get("CB_PROTO_VARLEN_BUCKETS", "0") == "1"  # gap4 zsbench, see model_runner.py


class PagedAttentionArgs(TypedDict):
    """The keyword arguments for a forward pass using paged attention, passed directly as model forward kwargs.

    Attributes:
        input_ids: Input token IDs tensor of shape `(1, total_query_tokens)`.
        attention_mask: Attention mask tensor or dictionary mapping layer types to masks. Can be `None` if the
            attention implementation doesn't require explicit masks.
        position_ids: Position IDs tensor of shape `(1, total_query_tokens)`.
        cu_seq_lens_q: Cumulative sequence lengths for queries, used for variable-length batching.
        cu_seq_lens_k: Cumulative sequence lengths for keys/values. It's a dictionary mapping layer types
            (e.g., "full_attention", "sliding_attention") to tensors for hybrid models.
        max_length_q: Maximum query sequence length in the batch.
        max_length_k: Maximum key/value sequence length. It's a dictionary for hybrid models.
        write_index: List of tensors indicating where to write new KV states in the cache, one per attention group.
        read_index: List of tensors indicating which cache positions to read from, one per attention group.
        logits_indices: Tensor indicating which positions in the output should be used for next-token prediction.
        cache: The [`PagedAttentionCache`] instance managing the KV cache.
        block_table: Block table for paged KV cache. If provided, uses `flash_attn_with_kvcache` for fused attention +
            cache update. More information in src/transformers/integrations/flash_attention.py
        logits_processor_args: List of tensors containing the arguments for the logits processors, one per request.
        use_cache: Whether to use caching (always `False` in continuous batching as the cache is managed externally).
        is_causal: Determined internally. SDPA / eager are never causal (custom mask) while flash always is (no mask)
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | dict[str, torch.Tensor] | None
    position_ids: torch.Tensor
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: dict[str, torch.Tensor]
    max_length_q: int
    max_length_k: dict[str, int]
    write_index: list[torch.Tensor]
    read_index: list[torch.Tensor]
    logits_indices: torch.Tensor
    cache: PagedAttentionCache
    block_table: torch.Tensor | None
    logits_processor_args: torch.Tensor
    use_cache: bool
    is_causal: bool


# gap2-mtp prototype (bench only): speculative decoding with the model's MTP module. "shadow" drafts one token per
# step without using it (acceptance statistics only), "spec" feeds the draft back as a second query token and
# verifies it. Off by default.
import os as _mtp_os

_CB_MTP = _mtp_os.environ.get("CB_MTP", "")
if _CB_MTP not in ("", "shadow", "spec"):
    raise ValueError(f"CB_MTP must be 'shadow' or 'spec', got {_CB_MTP!r}")
# gap3-mtp prototype: with the block table decode path (CB_PROTO_MLA_BLOCK_TABLE=1), "spec" decode batches take that
# path too, with `_CB_MTP_QUERY` query tokens per request (the last token, then the draft); padded like the one-token
# path (requests to a power of 2), so the graph key (query tokens, key bucket) also fixes the number of logits rows
_CB_MTP_BT = _CB_MTP == "spec" and _PROTO_MLA_BLOCK_TABLE
# gap3-mtpk prototype: CB_MTP_K=k (block table path) drafts k tokens per step with the same MTP layer applied k times,
# as vLLM and SGLang do with one MTP layer; unset keeps the 1-draft path above
_CB_MTP_K = int(_mtp_os.environ.get("CB_MTP_K", "0"))
_CB_MTP_MULTI = _CB_MTP_BT and _CB_MTP_K >= 1
_CB_MTP_QUERY = 1 + _CB_MTP_K if _CB_MTP_MULTI else 2
# gap3-mtpa prototype: MTP speculative decoding under asynchronous batching (CB_MTP_ASYNC=1, with CB_MTP=spec, the block
# table decode path and "use_async_batching": true). Step N+1 is prepared before step N's outcome is known: the host
# schedules every drafted request as if its draft were rejected (pessimistic length, one slack row allocated), and the
# step itself takes the request's last token (bonus or first token), its new draft and the +1 shift of an accepted
# draft from step N's outputs on the device. The host learns the accepted counts when it updates step N, one step later.
_CB_MTP_ASYNC = _CB_MTP_BT and _mtp_os.environ.get("CB_MTP_ASYNC", "0") == "1"


class ContinuousBatchingIOs:
    """A class to hold inputs and outputs for a continuous batching forward pass, using static tensors as storage. The
    class is meant to be self-contained, so once a set of inputs have been created, the class can be used to update the
    batch alone.
    """

    static_inputs: int = 7  # Number of static inputs always present in the bulk tensor

    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        device: torch.device,
        model_dtype: torch.dtype,
        logit_processor: ContinuousBatchingLogitsProcessorList,
    ) -> None:
        """Initialize the continuous batching I/O manager. Args:
        - cache: The [`PagedAttentionCache`] instance managing the KV cache. Meant to be unique.
        - config: The model's pretrained configuration.
        - continuous_batching_config: The continuous batching configuration.
        - device: The device to allocate tensors on. If the device is CPU, then the memory is pinned.
        - model_dtype: The data type for model computations.
        - logit_processor: The [`ContinuousBatchingLogitsProcessorList`] object used to process the logits.
        """
        # Memoize attributes
        self.cache = cache
        self.device = device
        self.config = config
        self.model_dtype = model_dtype
        self.max_requests_per_batch = continuous_batching_config.max_requests_per_batch
        self.use_cuda_graph_varlen = continuous_batching_config.cuda_graph_booleans[0]
        self.return_logprobs = continuous_batching_config.return_logprobs
        # Setup input-related accumulators
        self.num_q_tokens = 0  # number of query tokens in the batch. Can be padded.
        self.max_kv_read = 0  # number of KV tokens read from cache (maxed across all groups). Can be padded.
        self.num_request_in_batch = 0
        self.true_read_sizes = [0 for _ in cache.cache_allocators]
        self.true_write_sizes = [0 for _ in cache.cache_allocators]
        self.use_block_table = False  # True if all requests in batch have query_length == 1
        # Setup other accumulators
        self.requests_in_batch: list[FutureRequestState] = []
        self.req_id_to_new_token_position: dict[str, int] = {}  # only used for async API
        self.graphs: CudaGraphBuffer = CudaGraphBuffer()
        self._proto_block_table_signature: dict[int, tuple] = {}  # last uploaded block table, per allocator
        # Setup static tensors
        self._setup_static_tensors(logit_processor=logit_processor)
        self._reset_static_tensors(full_reset=True)
        # If the device is an accelerator that supports streams, also create a compute stream
        self.compute_stream = create_device_stream(device) if device.type != "cpu" else None

    def _setup_static_tensors(self, logit_processor: ContinuousBatchingLogitsProcessorList) -> None:
        """Allocates static tensors for generation inputs and outputs. This is called only once at init time, to avoid
        repeated allocations and enable CUDA graphs. All tensors are allocated with maximum possible sizes.
        The allocated tensors are:

        - `_bulk_input_tensor`: Storage for all the small inputs: `input_ids`, `position_ids`, `cumulative_seqlens_q`,
          `logits_indices`, `cumulative_seqlens_k`, `carry_over_ids`.
        - `attention_mask`: Optional attention masks (only for eager/SDPA implementations)
        - `write_index` and `read_index` storage: Cache indexing tensors for each attention group
        - `output_ids`: Storage for generated token IDs and maybe log probabilities if return_logprobs is True
        """
        max_batch_tokens = self.cache.max_batch_tokens
        max_requests_per_batch = self.max_requests_per_batch  # guaranteed to be <= max_batch_tokens
        # Pin memory on CPU only when an accelerator is available, to speed up H2D transfers
        pin_memory = self.device.type == "cpu" and len(get_available_devices()) > 1

        # Small inputs are allocated as slices in a larger tensor aligned to 128 bytes (32 * 4b). This reduces the
        # reduces fragmentation, so it lowers the number of D2H transfers and speeds up transfers.
        bulk_lines = self.static_inputs + logit_processor.tensors_required
        bulk_columns = aligned_divide(max_batch_tokens + 1, 1, 32)
        self._bulk_input_tensor = torch.empty(
            (bulk_lines, bulk_columns), dtype=torch.int32, device=self.device, pin_memory=pin_memory
        )
        # Prepare a tensor to hold the default values for the logits processors
        self.logits_processors_defaults = torch.empty(
            (logit_processor.tensors_required, 1), dtype=torch.int32, device=self.device
        )
        logit_processor.fill_defaults(self.logits_processors_defaults)

        # TODO: update this to use a single and more precise bulk tensor
        self.input_ids = self._bulk_input_tensor[0, :max_batch_tokens]
        self.position_ids = self._bulk_input_tensor[1, :max_batch_tokens]
        self.cumulative_seqlens_q = self._bulk_input_tensor[2, : max_requests_per_batch + 1]
        self.logits_indices = self._bulk_input_tensor[3, :max_requests_per_batch]
        full_attention_cumulative_seqlens_k = self._bulk_input_tensor[4, : max_requests_per_batch + 1]
        sliding_attention_cumulative_seqlens_k = self._bulk_input_tensor[5, : max_requests_per_batch + 1]
        self.carry_over_ids = self._bulk_input_tensor[6, :max_batch_tokens]  # only used for async API

        # For sequence length of KV, the entries in the dict depend on the model
        self.cumulative_seqlens_k: dict[str, torch.Tensor] = {}
        if FULL_ATTENTION in self.cache.cache_allocators:
            self.cumulative_seqlens_k["full_attention"] = full_attention_cumulative_seqlens_k
        if SLIDING_ATTENTION in self.cache.cache_allocators:
            self.cumulative_seqlens_k["sliding_attention"] = sliding_attention_cumulative_seqlens_k
        # Indexed attention layers read their keys like full attention ones, the indexer only narrows the attention
        if INDEXED_ATTENTION in self.cache.cache_allocators:
            self.cumulative_seqlens_k["indexed_attention"] = full_attention_cumulative_seqlens_k

        # Output tensor and scalars
        num_output_rows = 2 if self.return_logprobs else 1
        self.mtp_mode = _CB_MTP
        if self.mtp_mode:
            # rows: first token, its logprob, bonus token, its logprob, draft accepted, next draft, then two
            # measurement rows: target logprob of the previous step's draft, drafter logprob of the next draft
            num_output_rows = 8
            if _CB_MTP_MULTI:
                # gap3-mtpk: rows 0..k the candidate token of each verified row, k+1..2k+1 their logprobs, 2k+2 the
                # number of accepted drafts, 2k+3..3k+2 the next drafts
                num_output_rows = max(8, 3 * _CB_MTP_K + 3)
            # per query token: the token that follows it when it is already known (prompt tokens), and where to take
            # it from among the tokens sampled in the step otherwise (-1: use `mtp_shift_ids`)
            mtp_kwargs = {"dtype": torch.int32, "device": self.device, "pin_memory": pin_memory}
            self.mtp_shift_ids = torch.zeros(max_batch_tokens, **mtp_kwargs)
            self.mtp_src = torch.full((max_batch_tokens,), -1, **mtp_kwargs)
            # per logits row: 1 when the request's last query token is a draft to verify
            self.mtp_has_draft = torch.zeros(max_requests_per_batch, **mtp_kwargs)
            # per logits row: the draft made for this position by the previous step ("shadow" measurement only)
            self.mtp_prev_draft = torch.zeros(max_requests_per_batch, **mtp_kwargs)
        # output_ids are sized to the input_ids to perform carry over, + 1 to have a static 0 at the end for carry over
        self.output_ids = torch.empty(
            (num_output_rows, max_batch_tokens + 1), dtype=torch.int32, device=self.device, pin_memory=pin_memory
        )
        self.output_ids.zero_()
        self.total_seqlen_q = 0
        self.total_seqlen_k: dict[str, int] = dict.fromkeys(self.cumulative_seqlens_k.keys(), 0)
        self.max_length_q = 0
        self.max_length_k: dict[str, int] = dict.fromkeys(self.cumulative_seqlens_k.keys(), 0)

        # If the attention mask is needed, it is allocated separately
        if attn_mask_is_needed(self.config):
            self.attention_mask = {}
            for layer_type in self.cumulative_seqlens_k.keys():
                self.attention_mask[layer_type] = torch.empty(
                    size=(1, 1, max_batch_tokens, self.cache.max_tokens_read + max_batch_tokens),
                    dtype=self.model_dtype,
                    device=self.device,
                    pin_memory=pin_memory,
                )
        else:
            self.attention_mask = None
        # PROTOTYPE (CB_PROTO_NO_MASK=1): the model gets one `None` mask per layer type, as a dict, so that it does not
        # build a causal mask of its own
        self._proto_no_mask_kwarg = dict.fromkeys(self.cumulative_seqlens_k, None) if proto_no_mask(self.config) else None

        # No block table == No elements in the block table tensor
        num_attn_types = len(self.cache.cache_allocators)
        n = num_attn_types if self.cache.max_blocks_per_request > 0 else 0
        self.block_table = torch.empty(
            (n, max_requests_per_batch, self.cache.max_blocks_per_request),
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )

        # For other kwargs, we need a list of tensors with as many tensors as there are groups
        self.write_index_storage = torch.empty(
            (num_attn_types, max_batch_tokens), dtype=torch.int64, device=self.device, pin_memory=pin_memory
        )
        self.read_index_storage = torch.empty(
            (num_attn_types, self.cache.max_tokens_read + max_batch_tokens),
            dtype=torch.int64,
            device=self.device,
            pin_memory=pin_memory,
        )
        # For read index, the +T is because there are sentinel indices for seqlen_q when model uses a sliding window
        if _PROTO_DEVICE_INDEX != "0" or _PROTO_DEVICE_MASK != "0":
            self._proto_arange = torch.arange(self.read_index_storage.size(-1), dtype=torch.int64, device=self.device)

    def _transfer_inputs(
        self, other: "ContinuousBatchingIOs", stream: torch.cuda.Stream, non_blocking: bool = False
    ) -> None:
        # Transfer accumulators
        other.num_q_tokens = self.num_q_tokens
        other.max_kv_read = self.max_kv_read
        other.num_request_in_batch = self.num_request_in_batch
        other.true_read_sizes = self.true_read_sizes[:]
        other.true_write_sizes = self.true_write_sizes[:]
        other.use_block_table = self.use_block_table
        # Transfer scalar attributes
        other.total_seqlen_q = self.total_seqlen_q
        other.total_seqlen_k = dict(self.total_seqlen_k)
        other.max_length_q = self.max_length_q
        other.max_length_k = dict(self.max_length_k)
        if self.use_block_table and _PROTO_MLA_BLOCK_TABLE:
            other._proto_len_k_bucket = self._proto_len_k_bucket
        # Transfer static tensors
        with stream_context(stream):
            if self.mtp_mode:  # gap3-mtpa prototype: the MTP inputs of the step
                other.mtp_shift_ids.copy_(self.mtp_shift_ids, non_blocking=non_blocking)
                other.mtp_src.copy_(self.mtp_src, non_blocking=non_blocking)
                other.mtp_has_draft.copy_(self.mtp_has_draft, non_blocking=non_blocking)
                other.mtp_prev_draft.copy_(self.mtp_prev_draft, non_blocking=non_blocking)
            other._bulk_input_tensor.copy_(self._bulk_input_tensor, non_blocking=non_blocking)  # fast bulk transfer
            # Only transfer block_table for decode-only batches (when it's actually used)
            if self.use_block_table:
                other.block_table.copy_(self.block_table, non_blocking=non_blocking)
            # Otherwise, we transfer the write indices (and read indices if the batch uses any cache reads)
            else:
                other.write_index_storage.copy_(self.write_index_storage, non_blocking=non_blocking)
                if self.max_kv_read > 0:
                    other.read_index_storage.copy_(self.read_index_storage, non_blocking=non_blocking)
            # Transfer the attention masks if needed
            if self.attention_mask is not None and other.attention_mask is not None:
                for layer_type in self.attention_mask.keys():
                    other.attention_mask[layer_type].copy_(self.attention_mask[layer_type], non_blocking=non_blocking)

    @torch.no_grad()
    def _reset_static_tensors(self, full_reset: bool = False) -> None:
        """Reset static tensors for the next batch. For efficiency, this only resets the portions of tensors that were
        actually used in the previous batch, using the attributes num_q_tokens and max_kv_read. If a (full_reset)
        is requested, the entire tensor storage is reset.
        """
        # Compute the slice to reset
        q_len = self.write_index_storage.size(-1) if full_reset else self.num_q_tokens
        kv_len = self.read_index_storage.size(-1) if full_reset else self.max_kv_read
        b_size = self.max_requests_per_batch + 1 if full_reset else min(self.num_q_tokens, self.max_requests_per_batch)

        # PROTOTYPE: on the one-upload decode path, the upload overwrites the whole padded region of the small inputs
        # and of the block table, so only the outputs are reset
        proto_one_upload = self._proto_one_upload() and not full_reset
        if full_reset:
            self._proto_block_table_signature = {}

        # Reset the attributes part of the bulk input tensor in one kernel
        if not proto_one_upload:
            self._bulk_input_tensor[: self.static_inputs, : q_len + 1].zero_()
        if full_reset:
            self._bulk_input_tensor[self.static_inputs :] = self.logits_processors_defaults
        self.max_length_q = 0

        # Reset the logits indices and output ids
        if not proto_one_upload:
            self.logits_indices[:b_size].zero_()
        self.output_ids[:, :b_size].zero_()
        if self.mtp_mode:
            self.mtp_shift_ids[:q_len].zero_()
            self.mtp_src[:q_len].fill_(-1)
            self.mtp_has_draft[:b_size].zero_()
            self.mtp_prev_draft[:b_size].zero_()

        # Reset the attributes that are either tensors or dict of tensors
        for layer_type in self.cumulative_seqlens_k:
            self.max_length_k[layer_type] = 0
            self.total_seqlen_k[layer_type] = 0
            if self.attention_mask is not None and (full_reset or not self._proto_skip_mask_work()):
                self.attention_mask[layer_type][:, :, :q_len, : q_len + kv_len].fill_(
                    torch.finfo(self.model_dtype).min
                )

        # If this is a full reset, we reset every tensors
        if full_reset:
            self.block_table[:, :b_size].fill_(-1)
            self._reset_read_and_write_indices(q_len, kv_len)
        # If this is not a full reset, and we are going to use the block table, we only reset it
        elif self.use_block_table:
            if not proto_one_upload:
                self.block_table[:, :b_size].fill_(-1)
        # Otherwise, the read and write indices are the ones used, so we reset them
        else:
            self._reset_read_and_write_indices(q_len, kv_len)

    def _proto_one_upload(self) -> bool:
        """PROTOTYPE: True when the current batch takes the one-upload decode path."""
        return (
            _PROTO_ONE_UPLOAD
            and _PROTO_MLA_BLOCK_TABLE
            and self.use_block_table
            and len(self.cumulative_seqlens_k) == 1
            and self._bulk_input_tensor.shape[0] == self.static_inputs  # no logits processor rows
        )

    def _proto_skip_mask_work(self) -> bool:
        """PROTOTYPE: True when the per-step mask reset and build are skipped (mask not read by the model, or built
        in one go on the device, which also resets it)."""
        if 0 < self.num_q_tokens <= _PROTO_SKIP_MASK_MAX_Q:
            return True
        if _PROTO_MLA_BLOCK_TABLE and self.use_block_table:
            return True
        return _PROTO_DEVICE_MASK == "1" and getattr(self.config, "sliding_window", None) is None

    def _proto_expand_read_index(
        self, group: int, allocator, blocks: list[int], block_starts: list[int], cumulative_seqlens_k: list[int]
    ) -> int:
        """PROTOTYPE: writes the read index of an allocator group on the device, from the block ids of the requests
        (flattened, with the offset of each request's first block) and their cumulative key lengths. Returns its size.
        Token `o` of a request lives at row `block[o // page] * stride + o % page`."""
        num_tokens = cumulative_seqlens_k[-1]
        if num_tokens > self.read_index_storage.size(-1) or len(block_starts) != len(cumulative_seqlens_k) - 1:
            raise RuntimeError("Inconsistent inputs for the device-side read index expansion")
        cu_k = torch.tensor(cumulative_seqlens_k, dtype=torch.int64, device=self.device)
        blocks_t = torch.tensor(blocks, dtype=torch.int64, device=self.device)
        starts_t = torch.tensor(block_starts, dtype=torch.int64, device=self.device)
        position = self._proto_arange[:num_tokens]
        request = torch.searchsorted(cu_k[1:], position, right=True)
        offset = position - cu_k[request]
        page = allocator.tokens_per_page
        rows = blocks_t[starts_t[request] + offset // page] * allocator.block_physical_stride + offset % page
        self.read_index_storage[group, :num_tokens] = rows
        return num_tokens

    def _reset_read_and_write_indices(self, q_len: int, kv_len: int) -> None:
        """Resets each allocator's row of the index storages to its own trash indices."""
        for i, ca in enumerate(self.cache.cache_allocators.values()):
            self.write_index_storage[i, :q_len].fill_(ca.write_trash_index)
            self.read_index_storage[i, : q_len + kv_len].fill_(ca.read_trash_index)

    def reset(self) -> None:
        """Reset all relevant states for a new generation loop."""
        self._reset_static_tensors(full_reset=True)
        self.requests_in_batch = []
        self.req_id_to_new_token_position = {}
        if self.compute_stream is not None:
            self.compute_stream.synchronize()

    # These getter function help create a common interface for the sync and async IOs
    def get_cumulative_seqlens(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get the cumulative sequence lengths for the current batch."""
        return self.cumulative_seqlens_q, self.cumulative_seqlens_k

    def carry_over_tokens(
        self, input_ids: torch.Tensor, carry_over_ids: torch.Tensor, prev_output_ids: torch.Tensor
    ) -> None:
        pass

    def retrieve_device_outputs(self) -> None:
        if self.compute_stream is not None:
            self.compute_stream.synchronize()

    def prepare_batch_update(self) -> tuple[list[FutureRequestState], list[int], list[float] | None]:
        new_tokens = self.output_ids[0, : self.num_request_in_batch].tolist()
        # If logprobs are generated, we retrieve them from the output tensor and cast them to the right dtype
        if self.return_logprobs:
            logprobs = self.output_ids[1, : self.num_request_in_batch].view(dtype=torch.float32).tolist()
        # Otherwise, we can return an empty list because they won't be used
        else:
            logprobs = None
        return self.requests_in_batch, new_tokens, logprobs

    def prepare_batch_tensors(
        self,
        requests_in_batch: list[FutureRequestState],
        logits_processors: ContinuousBatchingLogitsProcessorList,
        use_decode_fast_path: bool,
        num_q_tokens: int,
        max_kv_read: int,
        use_padding: bool,
    ) -> None:
        """Prepare tensors and metadata for the next model forward pass, using the given requests as data. This method:

        1. Resets the static tensors from the previous batch
        2. Iterates through requests to accumulate input_ids, position_ids, and sequence lengths
        3. Extends read/write indices for cache management
        4. Builds attention masks if needed (for eager/SDPA implementations)
        5. Converts accumulated lists to tensors and copies them to static storage

        This method also modifies the `position_offset` attribute of each request to track progress and adds a
        temporary token at the end of the requests for which there will a new token.
        """
        # Keep track of this requests in the batch, which will be useful to update the batch later
        if not requests_in_batch:
            raise ValueError("No requests in batch")

        # Determine if the block table is used before we start to prepare the batch, to avoid useless preparation
        self.use_block_table = use_decode_fast_path and self.block_table.numel() > 0
        # Memoize the length of Q and KV
        self.num_q_tokens = num_q_tokens
        self.max_kv_read = 0 if self.use_block_table else max_kv_read  # No need to track KV read for decode-fast-path
        self.num_request_in_batch = len(requests_in_batch)
        # Reset the static storage that is going to be used for the next batch
        self._reset_static_tensors()

        # Reset accumulators
        num_attn_types = len(self.cache.cache_allocators)
        self.true_read_sizes = [0 for _ in range(num_attn_types)]
        self.true_write_sizes = [0 for _ in range(num_attn_types)]
        self.requests_in_batch = []
        self.req_id_to_new_token_position = {}

        # Prepare accumulators. For batches with no past cache to read, we leave read_index empty: the cache.update
        # will detect the 0-size indices and skip the read.
        input_ids = []
        position_ids = []
        cumulative_seqlens_q = [0]
        logits_indices = []
        cumulative_seqlens_k = {layer_type: [0] for layer_type in self.cumulative_seqlens_k.keys()}
        write_index = [[] for _ in range(num_attn_types)]
        read_index = None if self.max_kv_read == 0 else [[] for _ in range(num_attn_types)]
        # PROTOTYPE: groups whose read index is expanded on the device only accumulate the block ids of the requests
        allocators = list(self.cache.cache_allocators.values())
        proto_device_index = (
            _PROTO_DEVICE_INDEX != "0"
            and read_index is not None
            and all(type(a).get_read_indices is FullAttentionCacheAllocator.get_read_indices for a in allocators)
        )
        proto_blocks = [[] for _ in range(num_attn_types)]
        proto_block_starts = [[] for _ in range(num_attn_types)]
        host_read_index = read_index if (not proto_device_index or _PROTO_DEVICE_INDEX == "check") else None
        mtp_shift_ids, mtp_src, mtp_has_draft, mtp_prev_draft = [], [], [], []
        mtp_num_rows = self._get_num_sequences(use_padding=use_padding)  # logits rows of the step, padding included

        # Go through all the requests in the batch
        for i, future_state in enumerate(requests_in_batch):
            # First we retrieve the lengths related to the request
            state = future_state.state
            past_length = state.position_offset
            query_length = future_state.query_length

            # gap3-mtp prototype: a block-table decode batch is all requests with their draft (host-side check: the rows
            # and the padding of the step follow from it)
            if self.use_block_table and _CB_MTP_BT and query_length != _CB_MTP_QUERY:
                raise RuntimeError(f"CB_MTP block table path: request {state.request_id} has {query_length} tokens")

            # Update the internal state of the request
            state.position_offset += query_length

            # Then we accumulate for the object used in the kwargs
            input_ids.extend(state.tokens_to_process)
            position_ids.extend(range(past_length, past_length + query_length))
            cumulative_seqlens_q.append(cumulative_seqlens_q[-1] + query_length)
            self.max_length_q = max(self.max_length_q, query_length)

            # Accumulate the key sequence lengths for the current request
            for layer_type, cache_allocator in self.cache.cache_allocators.items():
                seqlen_k = cache_allocator.get_seqlen_k(past_length, query_length)
                cumulative_seqlens_k[layer_type].append(cumulative_seqlens_k[layer_type][-1] + seqlen_k)
                self.max_length_k[layer_type] = max(self.max_length_k[layer_type], seqlen_k)

            # We extend the read and write indices for the cache, or fill the block table if the kernel can read and
            # write the cache itself
            if self.use_block_table and _PROTO_MLA_BLOCK_TABLE:
                pass  # uploaded once for the whole batch after the loop
            elif self.use_block_table:
                self.cache.fill_block_table(state.request_id, past_length, query_length, self.block_table[:, i])
            else:
                self.cache.extend_read_and_write_indices(
                    state.request_id, past_length, query_length, host_read_index, write_index
                )
                if proto_device_index:
                    for group, allocator in enumerate(allocators):
                        num_blocks = -(-(past_length + query_length) // allocator.tokens_per_page)
                        request_blocks = allocator.block_table[state.request_id]
                        # Host-side bound check: the device expansion indexes these lists without one
                        if len(request_blocks) < num_blocks:
                            raise RuntimeError(f"Request {state.request_id} holds fewer blocks than its length needs")
                        proto_block_starts[group].append(len(proto_blocks[group]))
                        proto_blocks[group].extend(request_blocks[:num_blocks])

            if self.mtp_mode:
                # The MTP module pairs the hidden state of position p with the token at p + 1: the next query token
                # of the request, the next prompt token of a split prefill, or a token sampled in this step
                next_prompt_token = state.remaining_prefill_tokens[0] if state.remaining_prefill_tokens else 0
                mtp_shift_ids.extend(state.tokens_to_process[1:] + [next_prompt_token])
                src = [-1] * query_length
                if future_state.has_new_token:
                    row = len(logits_indices)
                    has_draft = state.mtp_has_draft and query_length == _CB_MTP_QUERY
                    if has_draft and _CB_MTP_MULTI:
                        # gap3-mtpk: query token j pairs with the candidate token of verified row j
                        for j in range(_CB_MTP_QUERY):
                            src[j - _CB_MTP_QUERY] = j * mtp_num_rows + row
                    elif has_draft:
                        src[-2], src[-1] = row, mtp_num_rows + row  # first sampled token, then the bonus token
                    else:
                        src[-1] = row
                    mtp_has_draft.append(int(has_draft))
                    mtp_prev_draft.append(state.mtp_draft if state.mtp_draft is not None else 0)
                    future_state.mtp_has_draft = has_draft
                mtp_src.extend(src)

            # If the request has no remaining prefill tokens, it means the next token prediction is relevant
            if future_state.has_new_token:
                logits_indices.append(cumulative_seqlens_q[-1] - 1)
                if self.mtp_mode and future_state.mtp_has_draft:
                    logits_indices[-1] -= _CB_MTP_QUERY - 1  # the row that verifies the first draft; the drafts' follow
                state.tokens_to_process = [TMP_TOKEN_ID]
                if _CB_MTP_MULTI:
                    # gap3-mtpk: as below, for k drafts, in both batching modes: the drafts' rows are counted once the
                    # host knows how many were accepted (asynchronous: the device takes the next query tokens)
                    state.tokens_to_process = [0] * _CB_MTP_QUERY
                    state.mtp_has_draft = True
                    if future_state.mtp_has_draft:
                        state.position_offset -= _CB_MTP_QUERY - 1
                        state.mtp_pending_acc += 1
                elif _CB_MTP_ASYNC:
                    # gap3-mtpa: the request's next query tokens (last token, new draft) are taken on the device from this
                    # step's outputs when it is scheduled in the next step; on the host they are placeholders (valid ids)
                    state.tokens_to_process = [0, 0]
                    state.mtp_has_draft = True
                    if future_state.mtp_has_draft:
                        # pessimistic: the draft's row is not counted until the host knows it was accepted
                        state.position_offset -= 1
                        state.mtp_pending_acc += 1
                self.req_id_to_new_token_position[state.request_id] = logits_indices[-1]

            self.requests_in_batch.append(future_state)

        # Also prepare the tensor arguments for the logits processors
        logits_processors.prepare_tensor_args(
            requests_in_batch=requests_in_batch,
            arg_storage=self._bulk_input_tensor[self.static_inputs :],
        )

        # Scalar attribute update
        self.total_seqlen_q = cumulative_seqlens_q[-1]

        # PROTOTYPE: one block table upload for the batch, and the bucket of the longest request (the prototype
        # attention sizes its per-request key window with it, so it is part of the graph key)
        if self.use_block_table and _PROTO_MLA_BLOCK_TABLE:
            width = self.block_table.shape[-1]
            proto_one_upload = self._proto_one_upload()
            for allocator in allocators:
                if not isinstance(allocator, MLACacheAllocator):
                    raise RuntimeError("CB_PROTO_MLA_BLOCK_TABLE only handles MLA allocators")
                pages_stride = allocator.block_physical_stride // allocator.tokens_per_page
                request_blocks = [allocator.block_table[fs.state.request_id] for fs in requests_in_batch]
                if proto_one_upload:
                    # Same requests holding as many blocks as in the previous decode batch: the table is up to date
                    padded_rows = self._get_num_sequences(use_padding=use_padding)
                    signature = (padded_rows, [fs.state.request_id for fs in requests_in_batch], list(map(len, request_blocks)))
                    if self._proto_block_table_signature.get(allocator.index) == signature:
                        continue
                    self._proto_block_table_signature[allocator.index] = signature
                rows = []
                for block_ids in request_blocks:
                    if len(block_ids) > width:
                        raise RuntimeError("A request holds more blocks than the block table is wide")
                    rows.append([b * pages_stride for b in block_ids] + [-1] * (width - len(block_ids)))
                if proto_one_upload:  # the padding requests too, so nothing has to be reset beforehand
                    rows.extend([[-1] * width] * (padded_rows - len(rows)))
                table = torch.tensor(rows, dtype=self.block_table.dtype)
                if proto_one_upload and os.environ.get("CB_PROTO_ONE_UPLOAD_DEBUG") == "bt_blocking":
                    self.block_table[allocator.index, : len(rows)].copy_(table)
                elif proto_one_upload:
                    # On the compute stream, so the step launched on that stream is ordered after the copy
                    with stream_context(self.compute_stream):
                        self.block_table[allocator.index, : len(rows)].copy_(table, non_blocking=True)
                else:
                    self.block_table[allocator.index, : len(rows)].copy_(table, non_blocking=True)
            longest = max(self.max_length_k.values()) + (1 if _CB_MTP_ASYNC else 0)  # gap3-mtpa: a shift of +1
            if _CB_MTP_MULTI:  # gap3-mtpk: a shift of up to k, and the k - 1 later draft passes
                longest = max(self.max_length_k.values()) + 2 * _CB_MTP_K
            self._proto_len_k_bucket = pad_to_pow2(longest, self.cache.max_decode_fast_path_length, 1024)

        # If needed, build the attention mask with the un-padded sequence lengths
        proto_device_mask = _PROTO_DEVICE_MASK != "0" and getattr(self.config, "sliding_window", None) is None
        if self.attention_mask is not None and proto_device_mask and not 0 < self.num_q_tokens <= _PROTO_SKIP_MASK_MAX_Q:
            k_len = self.num_q_tokens + self.max_kv_read
            for layer_type, layer_type_seqlens_k in cumulative_seqlens_k.items():
                if _PROTO_DEVICE_MASK == "check":
                    build_attention_mask(self.attention_mask[layer_type], cumulative_seqlens_q, layer_type_seqlens_k, 1)
                    expected = self.attention_mask[layer_type][0, 0, : self.num_q_tokens, :k_len].clone()
                _proto_build_attention_mask_on_device(
                    self.attention_mask[layer_type], cumulative_seqlens_q, layer_type_seqlens_k,
                    self.num_q_tokens, k_len, self._proto_arange,
                )
                if _PROTO_DEVICE_MASK == "check":
                    built = self.attention_mask[layer_type][0, 0, : self.num_q_tokens, :k_len]
                    assert torch.equal(built, expected), "device-built attention mask differs from the per-request one"
        elif self.attention_mask is not None and not self._proto_skip_mask_work():
            for layer_type, layer_type_seqlens_k in cumulative_seqlens_k.items():
                sliding_window = (
                    getattr(self.config, "sliding_window", None) if layer_type == "sliding_attention" else 1
                )
                build_attention_mask(
                    attention_mask=self.attention_mask[layer_type],
                    cumulative_seqlens_q=cumulative_seqlens_q,
                    cumulative_seqlens_k=layer_type_seqlens_k,
                    sliding_window=sliding_window,
                )

        # If there is padding, we need to make sure the cumulative_seqlens and total_seqlen are coherent
        if use_padding:
            num_sequences_in_next_batch = self._get_num_sequences(use_padding=use_padding)
            fake_sequences = num_sequences_in_next_batch - self.num_request_in_batch
            cumulative_seqlens_q.extend(repeat(self.total_seqlen_q, fake_sequences))
            # k will be padded in its own loop to avoid multiple loops
        else:
            fake_sequences = 0

        # When looping over request is done, we can build the actual tensors. This is faster than modifying the static
        # tensors inside the loop.
        to_tensor = partial(torch.tensor, dtype=torch.int32, device=self.device)

        if self.mtp_mode:
            # a placeholder token (-1) must never reach an embedding lookup: fail on the host, not on the device
            if min(input_ids) < 0 or min(mtp_shift_ids) < 0:
                raise RuntimeError("CB_MTP prototype: a request was scheduled with a placeholder token")
            self.mtp_shift_ids[: len(mtp_shift_ids)] = to_tensor(mtp_shift_ids)
            self.mtp_src[: len(mtp_src)] = to_tensor(mtp_src)
            self.mtp_has_draft[: len(mtp_has_draft)] = to_tensor(mtp_has_draft)
            self.mtp_prev_draft[: len(mtp_prev_draft)] = to_tensor(mtp_prev_draft)

        if self.use_block_table and self._proto_one_upload():
            # PROTOTYPE: the five small inputs of a decode batch as one host buffer and one copy. Rows follow the layout
            # of the bulk tensor: input ids, position ids, cumulative query lengths, logits indices, cumulative key
            # lengths. Everything is padded to the padded request count + 1, so no reset is needed beforehand.
            ((layer_type, layer_type_seqlens_k),) = cumulative_seqlens_k.items()
            total_seqlen_k = layer_type_seqlens_k[-1]
            self.total_seqlen_k[layer_type] = total_seqlen_k
            layer_type_seqlens_k.extend(repeat(total_seqlen_k, fake_sequences))
            width = self.num_q_tokens + 1
            if max(len(input_ids), len(cumulative_seqlens_q), len(layer_type_seqlens_k)) > width:
                raise RuntimeError("Inconsistent sizes for the one-upload decode path")
            if self.cumulative_seqlens_k[layer_type].data_ptr() != self._bulk_input_tensor[4].data_ptr():
                raise RuntimeError("The one-upload decode path expects the key lengths in row 4 of the bulk tensor")
            rows = [input_ids, position_ids, cumulative_seqlens_q, logits_indices, layer_type_seqlens_k]
            bulk = torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.int32)
            # On the compute stream: the step is launched on that stream, so it is ordered after the copy. Issued on
            # the current stream instead, the copy kernel races with the step (seen: stale inputs on a shared GPU)
            if os.environ.get("CB_PROTO_ONE_UPLOAD_DEBUG") in ("sync", "verify", "bt_blocking"):
                self._bulk_input_tensor[:5, :width].copy_(bulk)
            else:
                with stream_context(self.compute_stream):
                    self._bulk_input_tensor[:5, :width].copy_(bulk, non_blocking=True)
            if os.environ.get("CB_PROTO_ONE_UPLOAD_DEBUG") == "sync" and self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            if os.environ.get("CB_PROTO_ONE_UPLOAD_DEBUG") == "verify" and self.device.type == "cuda":
                torch.cuda.synchronize()
                got = self._bulk_input_tensor[:5, :width].cpu()
                assert torch.equal(got, bulk), "one-upload: device rows differ from the host buffer"
                assert self.input_ids[: len(input_ids)].tolist() == input_ids
                assert self.position_ids[: len(position_ids)].tolist() == position_ids
                assert self.cumulative_seqlens_q[: len(cumulative_seqlens_q)].tolist() == cumulative_seqlens_q
                assert self.logits_indices[: len(logits_indices)].tolist() == logits_indices
                assert self.cumulative_seqlens_k[layer_type][: len(layer_type_seqlens_k)].tolist() == layer_type_seqlens_k
        else:
            # Those kwargs always have the same type regardless of the model
            self.input_ids[: len(input_ids)] = to_tensor(input_ids)
            self.position_ids[: len(position_ids)] = to_tensor(position_ids)
            self.cumulative_seqlens_q[: len(cumulative_seqlens_q)] = to_tensor(cumulative_seqlens_q)
            self.logits_indices[: len(logits_indices)] = to_tensor(logits_indices)

            # Those kwargs are either dict of tensors or tensors, so we need to handle both cases
            for layer_type, layer_type_seqlens_k in cumulative_seqlens_k.items():
                total_seqlen_k = layer_type_seqlens_k[-1]
                self.total_seqlen_k[layer_type] = total_seqlen_k
                layer_type_seqlens_k.extend(repeat(total_seqlen_k, fake_sequences))
                self.cumulative_seqlens_k[layer_type][: len(layer_type_seqlens_k)] = to_tensor(layer_type_seqlens_k)

        # If we are not using the block table, we populate the write indices (and maybe the read indices)
        if not self.use_block_table:
            to_index_tensor = partial(torch.tensor, dtype=torch.int64, device=self.device)
            for i, group_write_indices in enumerate(write_index):
                self.write_index_storage[i, : len(group_write_indices)] = to_index_tensor(group_write_indices)
                self.true_write_sizes[i] = len(group_write_indices)
            if read_index is not None and proto_device_index:
                for i, allocator in enumerate(allocators):
                    seqlens_k = cumulative_seqlens_k[allocator.layer_type][: self.num_request_in_batch + 1]
                    self.true_read_sizes[i] = self._proto_expand_read_index(
                        i, allocator, proto_blocks[i], proto_block_starts[i], seqlens_k
                    )
                    if _PROTO_DEVICE_INDEX == "check":
                        expected = to_index_tensor(read_index[i])
                        built = self.read_index_storage[i, : self.true_read_sizes[i]]
                        assert torch.equal(built, expected), "device-built read index differs from the host one"
            elif read_index is not None:
                for i, group_read_indices in enumerate(read_index):
                    self.read_index_storage[i, : len(group_read_indices)] = to_index_tensor(group_read_indices)
                    self.true_read_sizes[i] = len(group_read_indices)

        if self.use_block_table and _PROTO_MLA_BLOCK_TABLE_CHECK and not (_CB_MTP_ASYNC or _CB_MTP_MULTI):  # rows shift on the device
            self._proto_check_block_table_rows(requests_in_batch, use_padding)

    def _proto_check_block_table_rows(self, requests_in_batch: list[FutureRequestState], use_padding: bool) -> None:
        """PROTOTYPE, check mode: runs the attention's own row computation on the tensors the step will read (block
        table, cumulative key lengths, length bucket) and asserts that every request writes to and reads from the rows
        the allocator's index path would have used, and that every row is inside the smallest layer view."""
        from ...models.glm_moe_dsa.modeling_glm_moe_dsa import _proto_block_table_row_state

        num_sequences = self._get_num_sequences(use_padding=use_padding)
        if _CB_MTP_BT:
            return self._proto_check_block_table_token_rows(requests_in_batch, num_sequences)
        for allocator in self.cache.cache_allocators.values():
            page = allocator.tokens_per_page
            last_safe_row = allocator.cache_tensor.shape[0] - allocator.tokens_per_block + page - 1
            write_rows, rows_all, visible = _proto_block_table_row_state(
                self.block_table[allocator.index, :num_sequences].to(self.cache.device),
                self.cumulative_seqlens_k[allocator.layer_type][: num_sequences + 1].to(self.cache.device),
                self._proto_len_k_bucket,
                page,
                last_safe_row,
                allocator.write_trash_index,
            )
            write_rows, rows_all, visible = write_rows.cpu(), rows_all.cpu(), visible.cpu()
            assert write_rows.shape[0] == num_sequences and rows_all.shape == (num_sequences, self._proto_len_k_bucket)
            assert int(rows_all.min()) >= 0 and int(rows_all.max()) <= last_safe_row, "row outside the cache"
            assert int(write_rows.min()) >= 0 and int(write_rows.max()) <= last_safe_row, "write row outside the cache"
            for i, future_state in enumerate(requests_in_batch):
                state = future_state.state
                past_length = state.position_offset - 1  # the offset was already advanced by the query token
                expected_write = allocator.get_write_indices(state.request_id, past_length, 1)
                expected_read = allocator.get_read_indices(state.request_id, past_length, 1)
                assert [int(write_rows[i])] == expected_write, f"write row of request {i} differs from the index path"
                assert int(visible[i].sum()) == len(expected_read) and bool(visible[i, : len(expected_read)].all())
                assert rows_all[i, : len(expected_read)].tolist() == expected_read, (
                    f"read rows of request {i} differ from the index path"
                )
            # Padding requests see nothing and write to the trash row
            assert not bool(visible[len(requests_in_batch) :].any())
            assert bool((write_rows[len(requests_in_batch) :] == allocator.write_trash_index).all())
        self._proto_checked_steps = getattr(self, "_proto_checked_steps", 0) + 1
        if self._proto_checked_steps in (1, 10, 100, 1000) or self._proto_checked_steps % 5000 == 0:
            print(f"BTCHECK ok steps={self._proto_checked_steps} requests={len(requests_in_batch)} "
                  f"bucket={self._proto_len_k_bucket}", flush=True)
    def _proto_check_block_table_token_rows(self, requests_in_batch: list[FutureRequestState], num_sequences: int):
        """gap3-mtp prototype, check mode: `_proto_check_block_table_rows` when every request brings 1 + k query
        tokens. Every query token must write to the row the allocator's index path gives it and see exactly the rows
        of its request up to itself; padding tokens write to the trash row and see nothing."""
        from ...models.glm_moe_dsa.modeling_glm_moe_dsa import _proto_block_table_token_state

        num_tokens = self.num_q_tokens
        cu_q = self.cumulative_seqlens_q[: num_sequences + 1]
        for allocator in self.cache.cache_allocators.values():
            page = allocator.tokens_per_page
            last_safe_row = allocator.cache_tensor.shape[0] - allocator.tokens_per_block + page - 1
            write_rows, rows_all, visible = _proto_block_table_token_state(
                self.block_table[allocator.index, :num_sequences].to(self.cache.device),
                cu_q.to(self.cache.device),
                self.cumulative_seqlens_k[allocator.layer_type][: num_sequences + 1].to(self.cache.device),
                self._proto_len_k_bucket,
                page,
                last_safe_row,
                allocator.write_trash_index,
                num_tokens,
            )
            write_rows, rows_all, visible = write_rows.cpu(), rows_all.cpu(), visible.cpu()
            assert write_rows.shape[0] == num_tokens and rows_all.shape == (num_tokens, self._proto_len_k_bucket)
            assert int(rows_all.min()) >= 0 and int(rows_all.max()) <= last_safe_row, "row outside the cache"
            assert int(write_rows.min()) >= 0 and int(write_rows.max()) <= last_safe_row, "write row outside the cache"
            t = 0
            for future_state in requests_in_batch:
                state, q = future_state.state, future_state.query_length
                past_length = state.position_offset - q  # the offset was already advanced by the query tokens
                expected_write = allocator.get_write_indices(state.request_id, past_length, q)
                expected_read = allocator.get_read_indices(state.request_id, past_length, q)
                assert write_rows[t : t + q].tolist() == expected_write, f"write rows of {state.request_id} differ"
                for j in range(q):
                    n = past_length + j + 1
                    assert int(visible[t + j].sum()) == n and bool(visible[t + j, :n].all()), "causal visibility"
                    assert rows_all[t + j, :n].tolist() == expected_read[:n], f"read rows of {state.request_id} differ"
                t += q
            assert not bool(visible[t:].any()) and bool((write_rows[t:] == allocator.write_trash_index).all())
        self._proto_checked_steps = getattr(self, "_proto_checked_steps", 0) + 1
        if self._proto_checked_steps in (1, 10, 100, 1000) or self._proto_checked_steps % 5000 == 0:
            print(f"BTCHECK mtp ok steps={self._proto_checked_steps} requests={len(requests_in_batch)} "
                  f"tokens={num_tokens} bucket={self._proto_len_k_bucket}", flush=True)

    def mtp_k_outputs(self) -> torch.Tensor:
        """gap3-mtpk prototype: the output rows of the step (see `num_output_rows`), on the host."""
        return self.output_ids[:, : self.num_request_in_batch].cpu()

    def prepare_mtp_batch_update(self) -> tuple[list, ...]:
        """gap2-mtp prototype: per logits row, the first token and its logprob, the bonus token and its logprob,
        whether the draft was accepted, the draft for the next step, the target logprob of the previous step's draft
        and the drafter's logprob of the next draft."""
        out = self.output_ids[:, : self.num_request_in_batch].cpu()
        return (
            out[0].tolist(),
            out[1].contiguous().view(torch.float32).tolist(),
            out[2].tolist(),
            out[3].contiguous().view(torch.float32).tolist(),
            out[4].tolist(),
            out[5].tolist(),
            out[6].contiguous().view(torch.float32).tolist(),
            out[7].contiguous().view(torch.float32).tolist(),
        )

    def _get_num_sequences(self, use_padding: bool) -> int:
        """Get the number of sequences for the current batch, accounting for padding if there is any."""
        if use_padding:
            if self.mtp_mode and self.use_block_table:
                # gap3-mtp prototype, see `_CB_MTP_BT`; "shadow" decode batches have one token per request
                if _CB_MTP_BT:
                    return self.num_q_tokens // _CB_MTP_QUERY
                return min(self.num_q_tokens, self.max_requests_per_batch)
            if self.mtp_mode:
                # gap2-mtp prototype: a decoding request takes two query tokens, so the logits rows follow the number
                # of requests (padded), not the number of query tokens. It joins the graph key
                if _PROTO_VARLEN_BUCKETS_IO:  # gap4 zsbench prototype: powers of 2, see CB_PROTO_VARLEN_BUCKETS
                    return pad_to_pow2(self.num_request_in_batch, self.max_requests_per_batch, 64)
                return pad_to_interval(self.num_request_in_batch, 64, self.max_requests_per_batch)
            return min(self.num_q_tokens, self.max_requests_per_batch)
        return self.num_request_in_batch

    def get_model_kwargs(self, use_padding: bool = False) -> PagedAttentionArgs:
        """Get model keyword arguments for the current batch, eventually padding the query dimension and KV dimensions
        if use_padding is True. The padding is only useful if we want static shapes, like when using cuda graphs."""
        q_size = self.num_q_tokens
        kv_size = self.max_kv_read + self.num_q_tokens
        num_sequences = self._get_num_sequences(use_padding=use_padding)

        # Prepare the kwargs, the attributes that are either tensors or dict of tensors are initialized to empty dicts.
        kwargs = PagedAttentionArgs(
            input_ids=self.input_ids[:q_size].unsqueeze(0),
            position_ids=self.position_ids[:q_size].unsqueeze(0),
            cu_seq_lens_q=self.cumulative_seqlens_q[: num_sequences + 1],
            max_length_q=self.max_length_q,
            logits_indices=self.logits_indices[:num_sequences],
            logits_processor_args=self._bulk_input_tensor[self.static_inputs :, :num_sequences],
            cu_seq_lens_k={},
            max_length_k={},
            attention_mask=self._proto_no_mask_kwarg if self.attention_mask is None else {},
            read_index=[],
            write_index=[],
            cache=self.cache,
            block_table=self.block_table[:, :num_sequences] if self.use_block_table else None,
            use_cache=False,
            is_causal=self.attention_mask is None,  # False for SDPA and eager, True for flash
        )

        is_prefill_batch = any(fs.query_length > 1 for fs in self.requests_in_batch)
        if self.mtp_mode:
            kwargs["mtp"] = {
                "shift_ids": self.mtp_shift_ids[:q_size],
                "src": self.mtp_src[:q_size],
                "has_draft": self.mtp_has_draft[:num_sequences],
                "prev_draft": self.mtp_prev_draft[:num_sequences],
            }

        # If there is padding, make sure the padding sequences have length 0 (ie. cumulative lengths plateau)
        if use_padding:  # TODO: add per-path padding
            self.max_length_q = q_size  # keep max_length_q > 1 so FA skips the seqlen_q==1 GQA reshape on padded q
            # Additionally, if there are CUDA graphs, we need to pad max_length_k so graph capture will work regardless
            # of the future Q / KV lengths of the next batches
            if not self.use_block_table and self.use_cuda_graph_varlen:
                # WP prefill (bench only), off by default: batches that hold a prefill (some request brings more than
                # one token) get a finer key-length bucket, so short prompts do not attend through 1024-wide lists
                min_length_k = _PROTO_PREFILL_MIN_LEN_K if is_prefill_batch and _PROTO_PREFILL_MIN_LEN_K else 1024
                self.max_length_k = {
                    layer_type: pad_to_pow2(self.max_length_k[layer_type], self.cache.max_tokens_read, min_length_k)
                    for layer_type in self.max_length_k.keys()
                }

        # When using block table, max_length_q and max_length_k are not used by flash_attn_with_kvcache, so we set them
        # to constant `1` to avoid dynamo guards on these changing integer values. This applies throughout this method.
        kwargs["max_length_q"] = 1 if self.use_block_table else self.max_length_q

        # For the attributes that are lists of tensors, we construct list of tensor references
        for i in range(len(self.cache.cache_allocators)):
            write_index_size = q_size if use_padding else self.true_write_sizes[i]
            kwargs["write_index"].append(self.write_index_storage[i, :write_index_size])
            # If there is no cache to read, pass a list of empty tensors so `cache.update` uses the write-only fast path
            if self.max_kv_read == 0:
                read_index_size = 0
            else:
                read_index_size = kv_size if use_padding else self.true_read_sizes[i]
            kwargs["read_index"].append(self.read_index_storage[i, :read_index_size])

        # For the attributes that are dict of tensors, we first fill the dict with the actual values
        for layer_type, seqlens_k in self.cumulative_seqlens_k.items():
            kwargs["cu_seq_lens_k"][layer_type] = seqlens_k[: num_sequences + 1]
            kwargs["max_length_k"][layer_type] = 1 if self.use_block_table else self.max_length_k[layer_type]
            if self.use_block_table and _PROTO_MLA_BLOCK_TABLE:
                kwargs["max_length_k"][layer_type] = self._proto_len_k_bucket
            if self.attention_mask is not None:
                k_len = kv_size if use_padding else self.total_seqlen_k[layer_type]
                kwargs["attention_mask"][layer_type] = self.attention_mask[layer_type][..., :q_size, :k_len]

        # The masking utils expect a single tensor if the model has only one layer type
        if len(self.cumulative_seqlens_k.keys()) == 1 and self.attention_mask is not None:
            kwargs["attention_mask"] = kwargs["attention_mask"].popitem()[1]  # type: ignore

        return kwargs

    def get_cb_kwargs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the tensors used inside the generation step that are not inputs to the model forward pass. In
        synchronous batching, there is no carry over, so the only tensor that will be used is output_ids, but we still
        return 3 tensors to have the same interface as when using async batching."""
        return self.carry_over_ids, self.output_ids, self.output_ids

    def _get_graph_key(self) -> tuple[int, ...]:
        # Keys for decode fast path
        if self.use_block_table and _PROTO_MLA_BLOCK_TABLE:
            return (self.num_q_tokens, self._proto_len_k_bucket)
        if self.use_block_table:
            return (self.num_q_tokens,)
        # Keys for varlen path
        if self.mtp_mode:
            return (self.num_q_tokens, self.max_kv_read, *self.max_length_k.values(), self._get_num_sequences(True))
        return (self.num_q_tokens, self.max_kv_read, *self.max_length_k.values())

    def get_graph(self, prefix: str = "") -> torch.cuda.CUDAGraph | None:
        key = self._get_graph_key()
        graph = self.graphs.get_graph(key)
        # If this point is reached, it means the next step will be a new graph capture
        if graph is None:
            logger.info(f"{prefix}Creating graph for {key = }")
        return graph

    def set_graph(self, graph: torch.cuda.CUDAGraph) -> None:
        key = self._get_graph_key()
        self.graphs.set_graph(key, graph)
        logger.info(f"Setting graph for {key = }")

    def clear_graphs(self) -> None:
        """Drops the captured CUDA graphs, so the next batches capture them again."""
        self.graphs = CudaGraphBuffer()


class HostDeviceIOPair:
    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        device: torch.device,
        model_dtype: torch.dtype,
        logit_processor: ContinuousBatchingLogitsProcessorList,
    ) -> None:
        # The host IO has automatic pinned memory because it is created on the CPU
        self.host_io = ContinuousBatchingIOs(
            cache=cache,
            config=config,
            continuous_batching_config=continuous_batching_config,
            device=torch.device("cpu"),
            model_dtype=model_dtype,
            logit_processor=logit_processor,
        )
        self.device_io = ContinuousBatchingIOs(
            cache=cache,
            config=config,
            continuous_batching_config=continuous_batching_config,
            device=device,
            model_dtype=model_dtype,
            logit_processor=logit_processor,
        )
        self.h2d_over = torch.Event(device, enable_timing=False)
        self.compute_over = torch.Event(device, enable_timing=False)
        self.d2h_over = torch.Event(device, enable_timing=False)

    def reset(self) -> None:
        self.host_io.reset()
        self.device_io.reset()
        for event in [self.h2d_over, self.compute_over, self.d2h_over]:
            event.synchronize()

    def transfer_inputs_h2d(self, stream: torch.cuda.Stream) -> None:
        self.host_io._transfer_inputs(self.device_io, stream=stream, non_blocking=True)

    def transfer_outputs_d2h(self, stream: torch.cuda.Stream | None) -> None:
        with stream_context(stream):
            self.host_io.output_ids.copy_(self.device_io.output_ids, non_blocking=True)


class ContinuousBatchingAsyncIOs:
    """A class to handle the inputs and outputs for the asynchronous API. It uses two IO pairs to avoid race conditions
    between the two batches, which means twice as more VRAM is used for static input tensors and CUDA graph. If your GPU
    is large enough or you want to generate long sequences, this is a good trade-off to make.

    Asynchronous batching works by creating two pairs of host - device inputs and outputs:

                                    inputs
                      ┌──────────┐ ────────► ┌────────────┐
    IO pair object:   │ Host IOs │           │ Device IOs │       (for a CUDA system, Host = CPU and Device = GPU)
                      └──────────┘ ◄──────── └────────────┘
                                    outputs

    Each pair is separate from the other. This means that each pairs has its own CUDA graphs set, because CUDA graphs
    need to have static addresses for input tensors. To have a unique set of CUDA graph, we would need to copy the input
    tensors to a third device-side buffer. This could limit the memory cost of CUDA graphs but would slow down the
    forward pass.
    But the CUDA streams orchestrating the transfer from host to device (H2D) and device to host (D2H) are the same for
    both pairs. Same for the compute stream.
    The order of steps in async batching looks like this (for 3 batches of compute):

         │ ┌────┬────┐                  ┌────┬────┐     ┌────┬────┐       ┌────┐          ┌────┐
    CPU  │ │PR 0│PR 1│                  │UP 0│PR 2│     │UP 1│PR 3│       │UP 2│          │UP 3│
         │ └────┼───┬┴──┐               └────┴────┼───┐ └────┴────┼───┐   └────┘          └────┘
    H2D  │      │0->│1->│               ¦         │2->│ ¦         │3->│   ¦               ¦
         │      └───┼───┴───────────┬─────────────┴─┬─┼───────────┴───┼───────────────┐   ¦
    GPU  │          │   COMPUTE 0   │   COMPUTE 1   │█│   COMPUTE 2   │   COMPUTE 3   │   ¦
         │          └───────────────┼───┬───────────┼─┴─┬─────────────┼───┬───────────┼───┤
    D2H  │                          │0<-│           │1<-│             │2<-│           │3<-│
         │                          └───┘           └───┘             └───┘           └───┘

    with: - CPU: actions happening on the CPU (host-side)
          - GPU: actions happening on the GPU (device-side)
          - H2D: host to device transfer
          - D2H: device to host transfer
    and:
          - PR N: preparation of batch N
          - ->N: host to device transfer of batch N
          - COMPUTE N: compute step for batch N
          - <-N: device to host transfer of batch N
          - UP N: update of batch N

    You can see that the GPU is almost always busy, except where the █ is.
    Proper ordering of steps is ensured through the use of CUDA events and streams.
    """

    _supported_device_types = ("cuda", "xpu")

    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        device: torch.device,
        model_dtype: torch.dtype,
        logit_processor: ContinuousBatchingLogitsProcessorList,
    ) -> None:
        # Check device module is compatible with async batching
        if device.type not in self._supported_device_types:
            raise RuntimeError(
                f"Async batching requires a device type in {self._supported_device_types} but got {device.type = }"
            )
        if not torch.get_device_module(device).is_available():
            raise RuntimeError("Async batching requires an available device.")
        # IO pairs used to avoid race conditions
        self.current_pair = 0
        self.io_pairs = [
            HostDeviceIOPair(
                cache=cache,
                config=config,
                continuous_batching_config=continuous_batching_config,
                device=device,
                model_dtype=model_dtype,
                logit_processor=logit_processor,
            )
            for _ in range(2)
        ]
        # CUDA streams
        self.h2d_stream: torch.cuda.Stream = create_device_stream(device)
        self.d2h_stream: torch.cuda.Stream = create_device_stream(device)
        self.compute_stream: torch.cuda.Stream = create_device_stream(device)
        if self.h2d_stream is None or self.d2h_stream is None or self.compute_stream is None:
            raise RuntimeError(f"Async batching requires stream to function. Stream creation failed with {device = }.")
        # Set all unused compute streams to None
        self.io_pairs[0].host_io.compute_stream = None
        self.io_pairs[0].device_io.compute_stream = None
        self.io_pairs[1].host_io.compute_stream = None
        self.io_pairs[1].device_io.compute_stream = None
        # Used in carry over ids computation
        self.max_batch_tokens = cache.max_batch_tokens

    # These methods are simple wrapper dispatching to the current IO pair
    def get_cumulative_seqlens(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.io_pairs[self.current_pair].host_io.get_cumulative_seqlens()

    # The prepare_batch_tensor method also has to prepare the carry over ids
    def prepare_batch_tensors(
        self,
        requests_in_batch: list[FutureRequestState],
        logits_processors: ContinuousBatchingLogitsProcessorList,
        use_decode_fast_path: bool,
        num_q_tokens: int,
        max_kv_read: int,
        use_padding: bool,
    ) -> None:
        io_pair = self.io_pairs[self.current_pair]
        io_pair.host_io.prepare_batch_tensors(
            requests_in_batch, logits_processors, use_decode_fast_path, num_q_tokens, max_kv_read, use_padding
        )
        carry_over_ids = self.infer_carry_over_ids()
        if _CB_MTP_ASYNC and not io_pair.host_io.use_block_table:
            # gap3-mtpa: only the block table path takes the shift of an accepted draft on the device; a varlen batch must
            # not hold a request whose last step is still in flight (the scheduler keeps prefills apart from decodes)
            prev_positions = self.io_pairs[1 - self.current_pair].host_io.req_id_to_new_token_position
            carried = [r for r in io_pair.host_io.req_id_to_new_token_position if r in prev_positions and not r.startswith("__warmup_")]
            if carried:
                raise RuntimeError(f"CB_MTP_ASYNC: varlen batch with {len(carried)} requests whose last step is in flight")
        io_pair.host_io.carry_over_ids.copy_(carry_over_ids)

    def infer_carry_over_ids(self) -> torch.Tensor:
        """Infers the ids of the tokens to carry over from batch N to batch N+1. In asynchronous batching mode, we can
        schedule a request for batch N+1 without knowing the token predicted for that request in batch N. For that
        reason, we might need to carry over tokens just predicted in batch N before launching the forward pass of batch
        N+1. This method computes the ids of the tokens to carry over."""
        next_req_id_to_new_token_position = self.io_pairs[self.current_pair].host_io.req_id_to_new_token_position
        prev_req_id_to_new_token_position = self.io_pairs[1 - self.current_pair].host_io.req_id_to_new_token_position
        carry_over_ids = [-1 for _ in range(self.max_batch_tokens)]
        # Carry over happens after the raw predictions have been indexed with logits_indices. So output_ids contains the
        # a sequence of contiguous new tokens in the order the request were added to the batch. Eg:
        # output_ids = [new_tok_req3, new_tok_req1, new_tok_req2]
        # Since it's also the order of req_id_to_new_token_position, we just iterate over the old positions and look for
        # a request_id match: if there is one, we carry the predicted token over to its new position.
        for i, req_id in enumerate(prev_req_id_to_new_token_position.keys()):
            new_token_position = next_req_id_to_new_token_position.get(req_id)
            if new_token_position is not None:
                carry_over_ids[new_token_position] = i
        return torch.tensor(carry_over_ids, dtype=torch.int32)

    # The get_model_kwargs method is where the H2D transfer happens
    def get_model_kwargs(self, use_padding: bool = False) -> PagedAttentionArgs:
        io_pair = self.io_pairs[self.current_pair]
        io_pair.transfer_inputs_h2d(self.h2d_stream)
        self.h2d_stream.record_event(io_pair.h2d_over)
        self.compute_stream.wait_event(io_pair.h2d_over)
        return io_pair.device_io.get_model_kwargs(use_padding=use_padding)

    def get_cb_kwargs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the tensors used inside the generation step that are not inputs to the model forward pass. Those
        tensors could be retrieved using this object, but it would trigger a recompile if using torch.compile. They are:
        - output_ids: the output ids of the current batch
        - prev_output_ids: the output ids of the previous batch, required to carry over outputs tokens of the previous
            batch to the input tokens of the next batch.
        - carry_over_ids: a mask representing how to carry over tokens.
        """
        current_pair = self.io_pairs[self.current_pair]
        previous_pair = self.io_pairs[1 - self.current_pair]
        return (
            current_pair.device_io.carry_over_ids,
            previous_pair.device_io.output_ids,
            current_pair.device_io.output_ids,
        )

    # TODO: this should be moved to the model runner, and have it called conditioned on optional carry over ids
    def carry_over_tokens(
        self,
        input_ids: torch.Tensor,  # shape [1, seq_len]
        carry_over_ids: torch.Tensor,  # shape [seq_len]
        prev_output_ids: torch.Tensor,  # shape [1, max_batch_tokens + 1]
    ) -> None:
        """As explained in the infer_carry_over_ids method, we might need to carry over tokens just predicted in batch N
        before launching the forward pass of batch N+1. This method performs the carry over, and is recorded in CUDA
        graphs if they are enabled."""
        # Compute tokens to carry over and the corresponding mask
        carried_over_ids = prev_output_ids[0, carry_over_ids]  # shape [seq_len]
        carried_over_mask = (carry_over_ids != -1).int()  # shape [seq_len]
        # Truncate everything to the right size
        carried_over_ids = carried_over_ids[: input_ids.size(1)]  # shape [seq_len]
        carried_over_mask = carried_over_mask[: input_ids.size(1)]  # shape [seq_len]
        # Perform the carry over
        input_ids[0] = carried_over_ids * carried_over_mask + input_ids[0] * (1 - carried_over_mask)

    # This is called during compute, so we always pick the device IO in the IO pair
    @property
    def output_ids(self) -> torch.Tensor:
        # The output ids are used to copy_ the inferred tokens: they need to be on the device
        return self.io_pairs[self.current_pair].device_io.output_ids

    def get_graph(self) -> torch.cuda.CUDAGraph | None:
        prefix = f"(IO {self.current_pair})"
        return self.io_pairs[self.current_pair].device_io.get_graph(prefix=prefix)

    def set_graph(self, graph: torch.cuda.CUDAGraph) -> None:
        self.io_pairs[self.current_pair].device_io.set_graph(graph)

    def clear_graphs(self) -> None:
        """Drops the CUDA graphs captured by both IO pairs, so the next batches capture them again."""
        for io_pair in self.io_pairs:
            io_pair.device_io.clear_graphs()

    @property
    def use_block_table(self) -> bool:
        return self.io_pairs[self.current_pair].host_io.use_block_table

    # The retrieve_device_outputs method is where the D2H transfer happens AND where we switch IO pair
    def retrieve_device_outputs(self) -> None:
        io_pair = self.io_pairs[self.current_pair]
        # Wait for compute to finish before starting D2H transfer
        self.compute_stream.record_event(io_pair.compute_over)
        self.d2h_stream.wait_event(io_pair.compute_over)
        # Transfer the outputs to the host
        io_pair.transfer_outputs_d2h(self.d2h_stream)
        self.d2h_stream.record_event(io_pair.d2h_over)
        # Swap IO pair
        self.swap_io_pairs()

    def swap_io_pairs(self) -> None:
        """Switch to the other IO pair so the next batch reads from / writes into a fresh set of static buffers."""
        self.current_pair = 1 - self.current_pair

    # This method is called after the switch and not during the first batch
    def prepare_batch_update(self) -> tuple[list[FutureRequestState], list[int], list[float] | None]:
        io_pair = self.io_pairs[self.current_pair]
        io_pair.d2h_over.synchronize()  # ty:ignore[unresolved-attribute]  <- this is always a CUDA event
        return io_pair.host_io.prepare_batch_update()

    def prepare_mtp_batch_update(self) -> tuple[list, ...]:
        """gap3-mtpa prototype: the MTP output rows of the step whose outputs were just retrieved."""
        io_pair = self.io_pairs[self.current_pair]
        io_pair.d2h_over.synchronize()  # ty:ignore[unresolved-attribute]
        return io_pair.host_io.prepare_mtp_batch_update()

    def mtp_k_outputs(self) -> torch.Tensor:
        """gap3-mtpk prototype: the output rows of the step whose outputs were just retrieved, on the host."""
        io_pair = self.io_pairs[self.current_pair]
        io_pair.d2h_over.synchronize()  # ty:ignore[unresolved-attribute]
        return io_pair.host_io.mtp_k_outputs()

    @property
    def requests_in_batch(self) -> list[FutureRequestState]:
        """gap3-mtpa prototype: the requests of the step whose outputs were just retrieved."""
        return self.io_pairs[self.current_pair].host_io.requests_in_batch

    def reset(self) -> None:
        """Reset all state for a new generation session. Used in persistent mode between sessions."""
        self.current_pair = 0
        for io_pair in self.io_pairs:
            io_pair.reset()
        self.h2d_stream.synchronize()
        self.d2h_stream.synchronize()
        self.compute_stream.synchronize()
