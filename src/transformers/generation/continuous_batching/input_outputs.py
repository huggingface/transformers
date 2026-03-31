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
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from itertools import count
from typing import Any

import torch

from transformers.configuration_utils import PretrainedConfig

from ...utils import get_available_devices
from ...utils.metrics import traced
from .cache import PagedAttentionCache
from .requests import TMP_TOKEN_ID, FutureRequestState, logger
from .utils import CudaGraphBuffer, aligned_divide, attn_mask_is_needed, build_attention_mask


@dataclass
class PagedAttentionArgs:
    """Dataclass containing the keyword arguments for a forward pass using paged attention.

    Attributes:
        input_ids: Input token IDs tensor of shape `(1, total_query_tokens)`.
        attention_mask: Attention mask tensor or dictionary mapping layer types to masks. Can be `None` if the
            attention implementation doesn't require explicit masks.
        position_ids: Position IDs tensor of shape `(1, total_query_tokens)`.
        cu_seq_lens_q: Cumulative sequence lengths for queries, used for variable-length batching.
        cu_seq_lens_k: Cumulative sequence lengths for keys/values. Can be a tensor or dictionary mapping layer
            types (e.g., "full_attention", "sliding_attention") to tensors for hybrid models.
        max_seqlen_q: Maximum query sequence length in the batch.
        max_seqlen_k: Maximum key/value sequence length. Can be an int or dictionary for hybrid models.
        write_index: List of tensors indicating where to write new KV states in the cache, one per attention group.
        read_index: List of tensors indicating which cache positions to read from, one per attention group.
        logits_indices: Tensor indicating which positions in the output should be used for next-token prediction.
        cache: The [`PagedAttentionCache`] instance managing the KV cache.
        block_table: Block table for paged KV cache. If provided, uses `flash_attn_with_kvcache` for fused attention +
            cache update. More information in src/transformers/integrations/flash_paged.py
        use_cache: Whether to use caching (always `False` in continuous batching as the cache is managed externally).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor | dict[str, torch.Tensor] | None
    position_ids: torch.Tensor
    cu_seq_lens_q: torch.Tensor
    cu_seq_lens_k: torch.Tensor | dict[str, torch.Tensor]
    max_seqlen_q: int
    max_seqlen_k: int | dict[str, int]
    write_index: list[torch.Tensor]
    read_index: list[torch.Tensor]
    logits_indices: torch.Tensor
    cache: PagedAttentionCache
    block_table: torch.Tensor | None
    use_cache: bool = False

    def asdict(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
            "cu_seq_lens_q": self.cu_seq_lens_q,
            "cu_seq_lens_k": self.cu_seq_lens_k,
            "max_seqlen_q": self.max_seqlen_q,
            "max_seqlen_k": self.max_seqlen_k,
            "write_index": self.write_index,
            "read_index": self.read_index,
            "logits_indices": self.logits_indices,
            "cache": self.cache,
            "block_table": self.block_table,
            "use_cache": self.use_cache,
        }


class ContinuousBatchingIOs:
    """A class to hold inputs and outputs for a continuous batching forward pass, using static tensors as storage. The
    class is meant to be self-contained, so once a set of inputs have been created, the class can be used to update the
    batch alone.
    """

    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        device: torch.device,
        model_dtype: torch.dtype,
        max_graphs: int,
        return_logprobs: bool,
    ) -> None:
        """Initialize the continuous batching I/O manager. Args:
        - cache: The [`PagedAttentionCache`] instance managing the KV cache. Meant to be unique.
        - config: The model's pretrained configuration.
        - device: The device to allocate tensors on. If the device is CPU, then the memory is pinned.
        - model_dtype: The data type for model computations.
        - max_graphs: Maximum number of CUDA graphs to cache. Uses LRU eviction when full.
        - return_logprobs: Whether to return log probabilities along with the token IDs.
        """
        # Memoize attributes
        self.cache = cache
        self.device = device
        self.config = config
        self.model_dtype = model_dtype
        self.sliding_window = 1 if getattr(config, "sliding_window", None) is None else config.sliding_window
        self.return_logprobs = return_logprobs
        # Setup input-related accumulators
        self.num_q_tokens = 0  # number of query tokens in the batch. Can be padded.
        self.max_kv_read = 0  # number of KV tokens read from cache (maxed across all groups). Can be padded.
        self.true_batch_size = 0
        self.true_read_sizes = [0 for _ in range(cache.num_groups)]
        self.true_write_sizes = [0 for _ in range(cache.num_groups)]
        self.use_block_table = False  # True if all requests in batch have query_length == 1
        # Setup other accumulators
        self.requests_in_batch: list[FutureRequestState] = []
        self.req_id_to_new_token_position: dict[str, int] = {}  # only used for async API
        self.graphs: CudaGraphBuffer = CudaGraphBuffer(max_graphs)
        # Setup static tensors and compute stream
        self._setup_static_tensors()
        self._reset_static_tensors(full_reset=True)
        self.compute_stream = torch.cuda.Stream(device=self.device) if device.type == "cuda" else None

    @traced(standalone=True)
    def _setup_static_tensors(self) -> None:
        """Allocates static tensors for generation inputs and outputs. This is called only once at init time, to avoid
        repeated allocations and enable CUDA graphs. All tensors are allocated with maximum possible sizes.
        The allocated tensors are:

        - `_bulk_input_tensor`: Storage for all the small inputs: `input_ids`, `position_ids`, `cumulative_seqlens_q`,
          `logits_indices`, `cumulative_seqlens_k`, `carry_over_ids`.
        - `attention_mask`: Optional attention masks (only for eager/SDPA implementations)
        - `write_index` and `read_index` storage: Cache indexing tensors for each attention group
        - `output_ids`: Storage for generated token IDs and maybe log probabilities if return_logprobs is True
        """
        num_groups = self.cache.num_groups
        max_batch_tokens = self.cache.max_batch_tokens
        num_pages = self.cache.num_blocks * self.cache.block_size
        # Pin memory on CPU only when an accelerator is available, to speed up H2D transfers
        pin_memory = self.device.type == "cpu" and len(get_available_devices()) > 1

        # Small inputs are allocated as slices in a larget tensor aligned to 128 bytes (32 * 4b). This reduces the
        # reduces fragmentation, so it lowers the number of D2H transfers and speeds up transfers.
        bulk_size = aligned_divide(max_batch_tokens + 1, 1, 32)
        self._bulk_input_tensor = torch.empty(
            (7, bulk_size), dtype=torch.int32, device=self.device, pin_memory=pin_memory
        )

        self.input_ids = self._bulk_input_tensor[0, :max_batch_tokens]
        self.position_ids = self._bulk_input_tensor[1, :max_batch_tokens]
        self.cumulative_seqlens_q = self._bulk_input_tensor[2, : max_batch_tokens + 1]
        self.logits_indices = self._bulk_input_tensor[3, :max_batch_tokens]
        full_attention_cumulative_seqlens_k = self._bulk_input_tensor[4, : max_batch_tokens + 1]
        sliding_attention_cumulative_seqlens_k = self._bulk_input_tensor[5, : max_batch_tokens + 1]
        self.carry_over_ids = self._bulk_input_tensor[6, :max_batch_tokens]  # only used for async API

        # For sequence length of KV, the entries in the dict depend on the model
        self.cumulative_seqlens_k: dict[str, torch.Tensor] = {}
        if self.cache.num_full_attention_groups:
            self.cumulative_seqlens_k["full_attention"] = full_attention_cumulative_seqlens_k
        if self.cache.num_sliding_attention_groups:
            self.cumulative_seqlens_k["sliding_attention"] = sliding_attention_cumulative_seqlens_k

        # Output tensor and scalars
        num_output_rows = 2 if self.return_logprobs else 1
        self.output_ids = torch.empty(
            (num_output_rows, max_batch_tokens + 1), dtype=torch.int32, device=self.device, pin_memory=pin_memory
        )
        # Last output token is never changed and set to 0 for async carry on purpose
        self.output_ids.zero_()
        self.total_seqlen_q = 0
        self.max_seqlen_q = 0
        self.max_seqlen_k = dict.fromkeys(self.cumulative_seqlens_k.keys(), 0)

        # If the attention mask is needed, it is allocated separately
        if attn_mask_is_needed(self.config):
            self.attention_mask = {}
            for layer_type in self.cumulative_seqlens_k.keys():
                self.attention_mask[layer_type] = torch.empty(
                    size=(1, 1, max_batch_tokens, num_pages + max_batch_tokens),
                    dtype=self.model_dtype,
                    device=self.device,
                    pin_memory=pin_memory,
                )
        else:
            self.attention_mask = None

        # No block table == No elements in the block table tensor
        n = num_groups if self.cache.max_blocks_per_request > 0 else 0
        self.block_table = torch.empty(
            (n, max_batch_tokens, self.cache.max_blocks_per_request),
            dtype=torch.int32,
            device=self.device,
            pin_memory=pin_memory,
        )

        # For other kwargs, we need a list of tensors with as many tensors as there are groups
        self.write_index_storage = torch.empty(
            (num_groups, max_batch_tokens), dtype=torch.int32, device=self.device, pin_memory=pin_memory
        )
        self.read_index_storage = torch.empty(
            (num_groups, num_pages + max_batch_tokens), dtype=torch.int32, device=self.device, pin_memory=pin_memory
        )
        # For read index, the +T is because there are -1 for seqlen_q when model uses a sliding window

    def _transfer_inputs(
        self, other: "ContinuousBatchingIOs", stream: torch.cuda.Stream, non_blocking: bool = False
    ) -> None:
        # Transfer accumulators
        other.num_q_tokens = self.num_q_tokens
        other.max_kv_read = self.max_kv_read
        other.true_batch_size = self.true_batch_size
        other.true_read_sizes = self.true_read_sizes[:]
        other.true_write_sizes = self.true_write_sizes[:]
        other.use_block_table = self.use_block_table
        # Transfer scalar attributes
        other.total_seqlen_q = self.total_seqlen_q
        other.max_seqlen_q = self.max_seqlen_q
        other.max_seqlen_k = dict(self.max_seqlen_k.items())
        # Transfer static tensors
        maybe_stream = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with maybe_stream:
            other._bulk_input_tensor.copy_(self._bulk_input_tensor, non_blocking=non_blocking)  # fast bulk transfer
            # Only transfer block_table for decode-only batches (when it's actually used)
            if self.use_block_table:
                other.block_table.copy_(self.block_table, non_blocking=non_blocking)
            # Otherwise, we transfer the read and write indices
            else:
                other.write_index_storage.copy_(self.write_index_storage, non_blocking=non_blocking)
                other.read_index_storage.copy_(self.read_index_storage, non_blocking=non_blocking)
            # Transfer the attention masks if needed
            if self.attention_mask is not None and other.attention_mask is not None:
                for layer_type in self.attention_mask.keys():
                    other.attention_mask[layer_type].copy_(self.attention_mask[layer_type], non_blocking=non_blocking)

    @traced
    @torch.no_grad()
    def _reset_static_tensors(self, full_reset: bool = False) -> None:
        """Reset static tensors for the next batch. For efficiency, this only resets the portions of tensors that were
        actually used in the previous batch, using the attributes num_q_tokens and max_kv_read. If a (full_reset)
        is requested, the entire tensor storage is reset.
        """
        # Compute the slice to reset
        q_len = self.write_index_storage.size(-1) if full_reset else self.num_q_tokens
        kv_len = self.read_index_storage.size(-1) if full_reset else self.max_kv_read

        # Reset the attributes part of the bulk input tensor in one kernel
        self._bulk_input_tensor[:, : q_len + 1].zero_()
        self.max_seqlen_q = 0

        # Reset the logits indices and output ids
        self.logits_indices[:q_len].zero_()
        self.output_ids[:, :q_len].zero_()

        # Reset the attributes that are either tensors or dict of tensors
        for layer_type in self.cumulative_seqlens_k:
            self.max_seqlen_k[layer_type] = 0
            if self.attention_mask is not None:
                self.attention_mask[layer_type][:, :, :q_len, : q_len + kv_len].fill_(
                    torch.finfo(self.model_dtype).min
                )

        # If this is a full reset, we reset every tensors
        if full_reset:
            self.block_table[:, :q_len].fill_(-1)
            self.write_index_storage[:, :q_len].fill_(-2)  # -1 is used to let the cache where new states go
            self.read_index_storage[:, : q_len + kv_len].fill_(-2)  # same
        # If this is not a full reset, and we are going to use the block table, we only reset it
        elif self.use_block_table:
            self.block_table[:, :q_len].fill_(-1)
        # Otherwise, the read and write indices are the ones used, so we reset them
        else:
            self.write_index_storage[:, :q_len].fill_(-2)  # -1 is used to let the cache where new states go
            self.read_index_storage[:, : q_len + kv_len].fill_(-2)  # same

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
        requests_in_batch = self.requests_in_batch
        new_tokens = self.output_ids[0, : len(self.requests_in_batch)].tolist()
        # If logprobs are generated, we retrieve them from the output tensor and cast them to the right dtype
        if self.return_logprobs:
            logprobs = self.output_ids[1, : len(self.requests_in_batch)].view(dtype=torch.float32).tolist()
        # Otherwise, we can return an empty list because they wont be used
        else:
            logprobs = None
        return requests_in_batch, new_tokens, logprobs

    @traced
    def prepare_batch_tensors(
        self,
        requests_in_batch: list[FutureRequestState],
        use_decode_fast_path: bool,
        num_q_tokens: int,
        max_kv_read: int,
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
        self.true_batch_size = len(requests_in_batch)
        # Reset the static storage that is going to be used for the next batch
        self._reset_static_tensors()

        # Reset accumulators
        self.true_read_sizes = [0 for _ in range(self.cache.num_groups)]
        self.true_write_sizes = [0 for _ in range(self.cache.num_groups)]
        self.requests_in_batch = []
        self.req_id_to_new_token_position = {}

        # Prepare accumulators
        input_ids = []
        position_ids = []
        cumulative_seqlens_q = [0]
        logits_indices = []
        cumulative_seqlens_k = {layer_type: [0] for layer_type in self.cumulative_seqlens_k.keys()}
        read_index = [[] for _ in range(self.cache.num_groups)]
        write_index = [[] for _ in range(self.cache.num_groups)]

        # Go through all the requests in the batch
        for i, future_state in enumerate(requests_in_batch):
            # First we retrieve the lengths related to the request
            state = future_state.state
            past_length = state.position_offset
            query_length = len(state.tokens_to_process)
            seqlens_k = self.cache.get_seqlens_k(past_length, query_length)

            # Update the internal state of the request
            state.position_offset += query_length

            # Then we accumulate for the object used in the kwargs
            input_ids.extend(state.tokens_to_process)
            position_ids.extend(range(past_length, past_length + query_length))
            cumulative_seqlens_q.append(cumulative_seqlens_q[-1] + query_length)
            self.max_seqlen_q = max(self.max_seqlen_q, query_length)

            # Accumulate the key sequence lengths for the current request
            for layer_type, layer_type_seqlen_k in seqlens_k.items():
                cumulative_seqlens_k[layer_type].append(cumulative_seqlens_k[layer_type][-1] + layer_type_seqlen_k)
                self.max_seqlen_k[layer_type] = max(self.max_seqlen_k[layer_type], layer_type_seqlen_k)

            # We extend the read and write indices for the cache, or fill the block table for decode-only batches
            if self.use_block_table:
                self.cache.fill_block_table(state.request_id, past_length, query_length, self.block_table[:, i])
            else:
                self.cache.extend_read_and_write_indices(
                    state.request_id, past_length, query_length, read_index, write_index
                )

            # If the request has no remaining prefill tokens, it means the next token prediction is relevant
            if future_state.has_new_token:
                logits_indices.append(cumulative_seqlens_q[-1] - 1)
                state.tokens_to_process = [TMP_TOKEN_ID]
                self.req_id_to_new_token_position[state.request_id] = logits_indices[-1]

            self.requests_in_batch.append(future_state)

        # When looping over request is done, we can build the actual tensors. This is faster than modifying the static
        # tensors inside the loop.
        to_tensor = partial(torch.tensor, dtype=torch.int32, device=self.device)

        # Those kwargs always have the same type regardless of the model
        self.input_ids[: len(input_ids)] = to_tensor(input_ids)
        self.position_ids[: len(position_ids)] = to_tensor(position_ids)
        self.cumulative_seqlens_q[: len(cumulative_seqlens_q)] = to_tensor(cumulative_seqlens_q)
        self.logits_indices[: len(logits_indices)] = to_tensor(logits_indices)
        self.total_seqlen_q = cumulative_seqlens_q[-1]

        # Those kwargs are either dict of tensors or tensors, so we need to handle both cases
        for layer_type, layer_type_seqlens_k in cumulative_seqlens_k.items():
            self.cumulative_seqlens_k[layer_type][: len(layer_type_seqlens_k)] = to_tensor(layer_type_seqlens_k)
            if self.attention_mask is not None:
                build_attention_mask(
                    attention_mask=self.attention_mask[layer_type],
                    cumulative_seqlens_q=cumulative_seqlens_q,
                    cumulative_seqlens_k=layer_type_seqlens_k,
                    sliding_window=self.sliding_window if layer_type == "sliding_attention" else 1,
                )

        # If we are not using the block table, we populate the read and write indices
        if not self.use_block_table:
            for i, group_read_indices, group_write_indices in zip(count(), read_index, write_index):
                self.read_index_storage[i, : len(group_read_indices)] = to_tensor(group_read_indices)
                self.write_index_storage[i, : len(group_write_indices)] = to_tensor(group_write_indices)
                self.true_read_sizes[i] = len(group_read_indices)
                self.true_write_sizes[i] = len(group_write_indices)

    def get_model_kwargs(self, use_padding: bool = False) -> dict[str, Any]:
        """Get model keyword arguments for the current batch, eventually padding the query dimension to (padded_q_size)
        and the keys/values dimension to (padded_kv_cache_size). The padding is only useful if we want static shapes,
        like when using cuda graphs AND only activated if both Q and KV are padded."""
        q_size = self.num_q_tokens
        kv_size = self.max_kv_read + self.num_q_tokens
        batch_size = self.num_q_tokens if use_padding else self.true_batch_size

        # Prepare the kwargs, the attributes that are either tensors or dict of tensors are initialized to empty dicts.
        kwargs = PagedAttentionArgs(
            input_ids=self.input_ids[:q_size].unsqueeze(0),
            position_ids=self.position_ids[:q_size].unsqueeze(0),
            cu_seq_lens_q=self.cumulative_seqlens_q[: batch_size + 1],
            max_seqlen_q=self.max_seqlen_q,
            logits_indices=self.logits_indices[:q_size],
            cu_seq_lens_k={},
            max_seqlen_k={},
            attention_mask={},
            read_index=[],
            write_index=[],
            cache=self.cache,
            block_table=self.block_table[:, :batch_size] if self.use_block_table else None,
            use_cache=False,
        )

        # If we use constant-sized slicing, there are some "padding" queries tokens which FA has some issues with. In
        # some models like Qwen3-4B-Instruct-2507, if we don't include these tokens in cumulative_seqlens_q, there are
        # some NaNs in the output logits even for non-padded tokens.
        if use_padding:
            self.max_seqlen_q = max(self.max_seqlen_q, q_size - self.total_seqlen_q)
            kwargs.max_seqlen_q = self.max_seqlen_q
            self.cumulative_seqlens_q[self.true_batch_size + 1 :] = q_size
            # FIXME: is there another way to avoid this? It has a very slight impact on performance (~5 tok/s)

        # When using block table, max_seqlen_q and max_seqlen_k are not used by flash_attn_with_kvcache, so we set them
        # to constant `1` to avoid dynamo guards on these changing integer values. This applies throughout this method.
        if self.use_block_table:
            kwargs.max_seqlen_q = 1

        # For the attributes that are lists of tensors, we construct list of tensor references
        for i in range(self.cache.num_groups):
            read_index_size = kv_size if use_padding else self.true_read_sizes[i]
            write_index_size = q_size if use_padding else self.true_write_sizes[i]
            kwargs.read_index.append(self.read_index_storage[i, :read_index_size])
            kwargs.write_index.append(self.write_index_storage[i, :write_index_size])

        # For the attributes that are dict of tensors, we replace the dict with a tensor if there is only one entry
        # When using block table, max_seqlen_k is not used, so we set it to a constant to avoid dynamo guards
        layer_types = list(self.cumulative_seqlens_k.keys())
        if len(layer_types) > 1:
            kwargs.max_seqlen_k: dict[str, int] = {}
            kwargs.cu_seq_lens_k: dict[str, torch.Tensor] = {}
            kwargs.attention_mask: dict[str, torch.Tensor] = {}
            for layer_type, seqlens_k in self.cumulative_seqlens_k.items():
                kwargs.cu_seq_lens_k[layer_type] = seqlens_k[: batch_size + 1]
                kwargs.max_seqlen_k[layer_type] = 1 if self.use_block_table else self.max_seqlen_k[layer_type]
                if self.attention_mask is not None:
                    k_len = kv_size if use_padding else seqlens_k[batch_size]
                    kwargs.attention_mask[layer_type] = self.attention_mask[layer_type][..., :q_size, :k_len]
        else:
            layer_type = layer_types[0]
            kwargs.cu_seq_lens_k = self.cumulative_seqlens_k[layer_type][: batch_size + 1]
            kwargs.max_seqlen_k = 1 if self.use_block_table else self.max_seqlen_k[layer_type]
            if self.attention_mask is not None:
                k_len = kv_size if use_padding else self.cumulative_seqlens_k[layer_type][batch_size]
                kwargs.attention_mask = self.attention_mask[layer_type][..., :q_size, :k_len]

        if self.attention_mask is None:
            kwargs.attention_mask = None
        return kwargs.asdict()  # TODO: this is imperfect, check if there is no better way to juggle dict / dataclass

    def get_cb_kwargs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the tensors used inside the generation step that are not inputs to the model forward pass. In
        synchronous batching, there is no carry over, so the only tensor that will be used is output_ids, but we still
        return 3 tensors to have the same interface as when using async batching."""
        return self.carry_over_ids, self.output_ids, self.output_ids

    def get_graph(self) -> torch.cuda.CUDAGraph | None:
        graph = self.graphs.get_graph(self.num_q_tokens, self.max_kv_read)
        # If this point is reached, it means the next step will be a new graph capture
        if graph is None:
            self.graphs.plan_for_new_graph()
            logger.info(f"Creating graph for {(self.num_q_tokens, self.max_kv_read) = }")
        return graph

    def set_graph(self, graph: torch.cuda.CUDAGraph) -> None:
        self.graphs.set_graph(self.num_q_tokens, self.max_kv_read, graph)


class HostDeviceIOPair:
    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        device: torch.device,
        model_dtype: torch.dtype,
        max_graphs: int,
        return_logprobs: bool,
    ) -> None:
        # The host IO has automatic pinned memory because it is created on the CPU
        self.host_io = ContinuousBatchingIOs(
            cache, config, torch.device("cpu"), model_dtype, max_graphs, return_logprobs
        )
        self.device_io = ContinuousBatchingIOs(cache, config, device, model_dtype, max_graphs, return_logprobs)
        # Create events only on CUDA devices
        self.h2d_over = torch.cuda.Event() if torch.cuda.is_available() else None
        self.compute_over = torch.cuda.Event() if torch.cuda.is_available() else None
        self.d2h_over = torch.cuda.Event() if torch.cuda.is_available() else None

    def reset(self) -> None:
        self.host_io.reset()
        self.device_io.reset()
        for event in [self.h2d_over, self.compute_over, self.d2h_over]:
            if event is not None:
                event.synchronize()

    def transfer_inputs_h2d(self, stream: torch.cuda.Stream) -> None:
        self.host_io._transfer_inputs(self.device_io, stream=stream, non_blocking=True)

    def transfer_outputs_d2h(self, stream: torch.cuda.Stream | None) -> None:
        maybe_stream = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with maybe_stream:
            self.host_io.output_ids.copy_(self.device_io.output_ids, non_blocking=True)


class ContinuousBatchingAsyncIOs:
    """A class to handle the inputs and outputs for the asynchronous API. It uses two IO pairs to avoid race conditions
    between the two batches, which means twice as more VRAM is used for static input tensors and CUDA graph. If your GPU
    is large enough or you want to generate long sequences, this is a good trade-off to make.

    Asynchronous batching works by creating two pairs of host - device inputs and ouputs:

                                    inputs
                      ┌──────────┐ ────────► ┌────────────┐
    IO pair object:   │ Host IOs │           │ Device IOs │       (for a CUDA sytem, Host = CPU and Device = GPU)
                      └──────────┘ ◄──────── └────────────┘
                                    outputs

    Each pair is separate from the other. This means that each pairs has its own CUDA graphs set, because CUDA graphs
    need to have static adresses for input tensors. To have a unique set of CUDA graph, we would need to copy the input
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

    You can see that the GPU is almost always busy, execpt where the █ is.
    Proper ordering of steps is ensured through the use of CUDA events and streams.
    """

    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        device: torch.device,
        model_dtype: torch.dtype,
        max_graphs: int,
        return_logprobs: bool,
    ) -> None:
        # Async batching needs streams to function, so check is CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError(f"Async batching requires CUDA, but {torch.cuda.is_available() = }")
        # IO pairs used to avoid race conditions
        self.current_pair = 0
        self.io_pairs = [
            HostDeviceIOPair(cache, config, device, model_dtype, max_graphs, return_logprobs) for _ in range(2)
        ]
        # CUDA streams
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.d2h_stream = torch.cuda.Stream(device=device)
        self.compute_stream = torch.cuda.Stream(device=device)
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
        use_decode_fast_path: bool,
        num_q_tokens: int,
        max_kv_read: int,
    ) -> None:
        io_pair = self.io_pairs[self.current_pair]
        io_pair.host_io.prepare_batch_tensors(requests_in_batch, use_decode_fast_path, num_q_tokens, max_kv_read)
        io_pair.host_io.carry_over_ids.copy_(self.infer_carry_over_ids())

    def infer_carry_over_ids(self) -> torch.Tensor:
        """Infers the ids of the tokens to carry over from batch N to batch N+1. In asynchronous batching mode, we can
        schedule a request for batch N+1 without knowing the token predicted for that request in batch N. For that
        reason, we might need to carry over tokens just predicted in batch N before launching the forwar pass of batch
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
    def get_model_kwargs(self, use_padding: bool = False) -> dict[str, Any]:
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

    def carry_over_tokens(
        self, input_ids: torch.Tensor, carry_over_ids: torch.Tensor, prev_output_ids: torch.Tensor
    ) -> None:
        """As explained in the infer_carry_over_ids method, we might need to carry over tokens just predicted in batch N
        before launching the forwar pass of batch N+1. This method performs the carry over, and is recorded in CUDA
        graphs if they are enabled."""
        # Compute tokens to carry over and the corresponding mask
        carried_over_ids = prev_output_ids[0, carry_over_ids]
        carried_over_mask = (carry_over_ids != -1).int()
        # Truncate everything to the right size
        carried_over_ids = carried_over_ids[: input_ids.size(1)]
        carried_over_mask = carried_over_mask[: input_ids.size(1)]
        # Perform the carry over
        input_ids[0] = carried_over_ids * carried_over_mask + input_ids[0] * (1 - carried_over_mask)

    # This is called during compute, so we always pick the device IO in the IO pair
    @property
    def output_ids(self) -> torch.Tensor:
        # The output ids are used to copy_ the infered tokens: they need to be on the device
        return self.io_pairs[self.current_pair].device_io.output_ids

    def get_graph(self) -> torch.cuda.CUDAGraph | None:
        return self.io_pairs[self.current_pair].device_io.get_graph()

    def set_graph(self, graph: torch.cuda.CUDAGraph) -> None:
        self.io_pairs[self.current_pair].device_io.set_graph(graph)

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
        # Switch IO pair
        self.current_pair = 1 - self.current_pair

    # This method is called after the switch and not during the first batch
    def prepare_batch_update(self) -> tuple[list[FutureRequestState], list[int], list[float] | None]:
        io_pair = self.io_pairs[self.current_pair]
        io_pair.d2h_over.synchronize()  # ty:ignore[unresolved-attribute]  <- this is always a CUDA event
        return io_pair.host_io.prepare_batch_update()

    def reset(self) -> None:
        """Reset all state for a new generation session. Used in persistent mode between sessions."""
        self.current_pair = 0
        for io_pair in self.io_pairs:
            io_pair.reset()
        self.h2d_stream.synchronize()
        self.d2h_stream.synchronize()
        self.compute_stream.synchronize()
