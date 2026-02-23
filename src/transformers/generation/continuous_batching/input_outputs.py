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
from dataclasses import dataclass
from functools import partial
from itertools import count
from typing import Any

import torch

from transformers.configuration_utils import PretrainedConfig

from ...utils.metrics import traced
from .cache import PagedAttentionCache
from .requests import TMP_TOKEN_ID, FutureRequestState
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
        max_graphs: int = 32,
    ) -> None:
        """Initialize the continuous batching I/O manager. Args:
        - cache: The [`PagedAttentionCache`] instance managing the KV cache. Meant to be unique.
        - config: The model's pretrained configuration.
        - device: The device to allocate tensors on. If the device is CPU, then the memory is pinned.
        - model_dtype: The data type for model computations.
        - max_graphs: Maximum number of CUDA graphs to cache. Uses LRU eviction when full.
        """
        # Memoize attributes
        self.cache = cache
        self.device = device
        self.config = config
        self.model_dtype = model_dtype
        self.sliding_window = 1 if getattr(config, "sliding_window", None) is None else config.sliding_window
        # Setup input-related accumulators
        self.actual_query_length = 0
        self.actual_key_length = 0
        self.actual_batch_size = 0
        self.actual_read_sizes = [0 for _ in range(cache.num_groups)]
        self.actual_write_sizes = [0 for _ in range(cache.num_groups)]
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
        - `output_ids`: Storage for generated token IDs
        """
        num_groups = self.cache.num_groups
        max_batch_tokens = self.cache.max_batch_tokens
        num_pages = self.cache.num_blocks * self.cache.block_size
        pin_memory = self.device.type == "cpu"

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
        self.output_ids = torch.empty(
            (max_batch_tokens + 1,), dtype=torch.int32, device=self.device, pin_memory=pin_memory
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
        other.actual_query_length = self.actual_query_length
        other.actual_key_length = self.actual_key_length
        other.actual_batch_size = self.actual_batch_size
        other.actual_read_sizes = self.actual_read_sizes[:]
        other.actual_write_sizes = self.actual_write_sizes[:]
        # Transfer scalar attributes
        other.total_seqlen_q = self.total_seqlen_q
        other.max_seqlen_q = self.max_seqlen_q
        other.max_seqlen_k = dict(self.max_seqlen_k.items())
        # Transfer static tensors
        with torch.cuda.stream(stream):
            other._bulk_input_tensor.copy_(self._bulk_input_tensor, non_blocking=non_blocking)  # fast bulk transfer
            other.write_index_storage.copy_(self.write_index_storage, non_blocking=non_blocking)
            other.read_index_storage.copy_(self.read_index_storage, non_blocking=non_blocking)
            if self.attention_mask is not None and other.attention_mask is not None:
                for layer_type in self.attention_mask.keys():
                    other.attention_mask[layer_type].copy_(self.attention_mask[layer_type], non_blocking=non_blocking)

    @traced
    @torch.no_grad()
    def _reset_static_tensors(self, full_reset: bool = False) -> None:
        """Reset static tensors for the next batch. For efficiency, this only resets the portions of tensors that were
        actually used in the previous batch, using the attributes actual_query_length, actual_key_length, and
        actual_batch_size. If a (full_reset) is requested, the entire tensor storage is reset.
        """
        # Compute the slice to reset
        q_len = self.write_index_storage.size(-1) if full_reset else self.actual_query_length
        k_len = self.read_index_storage.size(-1) if full_reset else self.actual_key_length

        # Reset the attributes part of the bulk input tensor in one kernel
        self._bulk_input_tensor[:, : q_len + 1].zero_()
        self.max_seqlen_q = 0

        # Reset the logits indices and output ids
        self.logits_indices[:q_len].zero_()
        self.output_ids[:q_len].zero_()

        # Reset the attributes that are either tensors or dict of tensors
        for layer_type in self.cumulative_seqlens_k:
            self.max_seqlen_k[layer_type] = 0
            if self.attention_mask is not None:
                self.attention_mask[layer_type][:, :, :q_len, :k_len].fill_(torch.finfo(self.model_dtype).min)

        # Reset the attributes that are lists of tensors
        self.write_index_storage[:, :q_len].fill_(-2)  # -1 is used to let the cache where new states go
        self.read_index_storage[:, : q_len + k_len].fill_(-2)  # same

    # These getter function help create a common interface for the sync and async IOs
    def get_cumulative_seqlens(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Get the cumulative sequence lengths for the current batch."""
        return self.cumulative_seqlens_q, self.cumulative_seqlens_k

    def get_actual_lengths(self) -> tuple[int, int, int, list[int], list[int]]:
        return (
            self.actual_query_length,
            self.actual_key_length,
            self.actual_batch_size,
            self.actual_read_sizes,
            self.actual_write_sizes,
        )

    def carry_over_tokens(self, input_ids: torch.Tensor) -> None:
        pass

    def retrieve_device_outputs(self) -> None:
        if self.compute_stream is not None:
            self.compute_stream.synchronize()

    def prepare_batch_update(self) -> tuple[list[FutureRequestState], list[int]]:
        requests_in_batch = self.requests_in_batch
        new_tokens = self.output_ids[: len(self.requests_in_batch)].tolist()
        return requests_in_batch, new_tokens

    @traced
    def prepare_batch_tensors(self, requests_in_batch: list[FutureRequestState]) -> None:
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

        # Reset the static tensors used for storage
        self._reset_static_tensors()  # FIXME: why does this make the generation faster?
        # Reset accumulators
        self.actual_query_length = 0
        self.actual_key_length = 0
        self.actual_batch_size = 0
        self.actual_read_sizes = [0 for _ in range(self.cache.num_groups)]
        self.actual_write_sizes = [0 for _ in range(self.cache.num_groups)]
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
        for future_state in requests_in_batch:
            # First we retrieve the lengths related to the request
            state = future_state.state
            past_length = state.position_offset
            query_length = len(state.tokens_to_process)
            seqlens_k = self.cache.get_seqlens_k(past_length, query_length)

            # Then we update the total lengths that are used for slicing
            self.actual_query_length += query_length
            # total_key_length is used to slice the keys so we need to take the max of all the key lengths
            self.actual_key_length += max(seqlens_k.values())
            self.actual_batch_size += 1
            # And the attribute tracking the position in the request object
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

            # We extend the read and write indices for the cache
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

        # The index only contain references to the storage tensors, so we update the storage and their references
        self.read_index = []
        self.write_index = []
        for i, group_read_indices, group_write_indices in zip(count(), read_index, write_index):
            self.read_index_storage[i, : len(group_read_indices)] = to_tensor(group_read_indices)
            self.write_index_storage[i, : len(group_write_indices)] = to_tensor(group_write_indices)
            self.actual_read_sizes[i] = len(group_read_indices)
            self.actual_write_sizes[i] = len(group_write_indices)

    def get_model_kwargs(self, padded_q_size: int = 0, padded_kv_cache_size: int = 0) -> dict[str, Any]:
        """Get model keyword arguments for the current batch, eventually padding the query dimension to (padded_q_size)
        and the keys/values dimension to (padded_kv_cache_size). The padding is only useful if we want static shapes,
        like when using cuda graphs AND only activated if both Q and KV are padded."""
        # Compute the slice to return, with the given padding if we are using cuda graphs
        use_padding = padded_q_size > 0 and padded_kv_cache_size > 0
        q_len = padded_q_size if use_padding else self.actual_query_length
        b_size = padded_q_size if use_padding else self.actual_batch_size
        # If there is padding, the size of the KV is the nb of padded Q tokens + the size padded of the padded KV cache
        padded_kv_size = padded_q_size + padded_kv_cache_size

        # Prepare the kwargs, the attributes that are either tensors or dict of tensors are initialized to empty dicts
        kwargs = PagedAttentionArgs(
            input_ids=self.input_ids[:q_len].unsqueeze(0),
            position_ids=self.position_ids[:q_len].unsqueeze(0),
            cu_seq_lens_q=self.cumulative_seqlens_q[: b_size + 1],
            max_seqlen_q=self.max_seqlen_q,
            logits_indices=self.logits_indices[:q_len],
            cu_seq_lens_k={},
            max_seqlen_k={},
            attention_mask={},
            read_index=[],
            write_index=[],
            cache=self.cache,
            use_cache=False,
        )

        # If we use constant-sized slicing, there are some "padding" queries tokens which FA has some issues with. In
        # some models like Qwen3-4B-Instruct-2507, if we don't include these tokens in cumulative_seqlens_q, there are
        # some NaNs in the output logits even for non-padded tokens.
        if use_padding:
            self.max_seqlen_q = max(self.max_seqlen_q, q_len - self.total_seqlen_q)
            kwargs.max_seqlen_q = self.max_seqlen_q
            self.cumulative_seqlens_q[self.actual_batch_size + 1 :] = q_len
            # FIXME: is there another way to avoid this? It has a very slight impact on performance (~5 tok/s)

        # For the attributes that are lists of tensors, we construct list of tensor references
        for i in range(self.cache.num_groups):
            read_index_size = padded_kv_size if use_padding else self.actual_read_sizes[i]
            write_index_size = padded_q_size if use_padding else self.actual_write_sizes[i]
            kwargs.read_index.append(self.read_index_storage[i, :read_index_size])
            kwargs.write_index.append(self.write_index_storage[i, :write_index_size])

        # For the attributes that are dict of tensors, we replace the dict with a tensor if there is only one entry
        layer_types = list(self.cumulative_seqlens_k.keys())
        if len(layer_types) > 1:
            kwargs.max_seqlen_k: dict[str, int] = {}
            kwargs.cu_seq_lens_k: dict[str, torch.Tensor] = {}
            kwargs.attention_mask: dict[str, torch.Tensor] = {}
            for layer_type, seqlens_k in self.cumulative_seqlens_k.items():
                kwargs.cu_seq_lens_k[layer_type] = seqlens_k[: b_size + 1]
                kwargs.max_seqlen_k[layer_type] = self.max_seqlen_k[layer_type]
                if self.attention_mask is not None:
                    k_len = padded_kv_size if use_padding else seqlens_k[b_size]
                    kwargs.attention_mask[layer_type] = self.attention_mask[layer_type][..., :q_len, :k_len]
        else:
            layer_type = layer_types[0]
            kwargs.cu_seq_lens_k = self.cumulative_seqlens_k[layer_type][: b_size + 1]
            kwargs.max_seqlen_k = self.max_seqlen_k[layer_type]
            if self.attention_mask is not None:
                k_len = padded_kv_size if use_padding else self.cumulative_seqlens_k[layer_type][b_size]
                kwargs.attention_mask = self.attention_mask[layer_type][..., :q_len, :k_len]

        if self.attention_mask is None:
            kwargs.attention_mask = None
        return kwargs.asdict()  # TODO: this is imperfect, check if there is no better way to juggle dict / dataclass


class HostDeviceIOPair:
    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        device: torch.device,
        model_dtype: torch.dtype,
        max_graphs: int = 32,
    ) -> None:
        # The host IO has automatic pinned memory because it is created on the CPU
        self.host_io = ContinuousBatchingIOs(cache, config, torch.device("cpu"), model_dtype, max_graphs)
        self.device_io = ContinuousBatchingIOs(cache, config, device, model_dtype, max_graphs)
        self.h2d_over = torch.cuda.Event()
        self.compute_over = torch.cuda.Event()
        self.d2h_over = torch.cuda.Event()

    def transfer_inputs_h2d(self, stream: torch.cuda.Stream) -> None:
        self.host_io._transfer_inputs(self.device_io, stream=stream, non_blocking=True)

    def transfer_outputs_d2h(self, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
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
        max_graphs: int = 32,
    ) -> None:
        # IO pairs used to avoid race conditions
        self.current_pair = 0
        self.io_pairs = [HostDeviceIOPair(cache, config, device, model_dtype, max_graphs) for _ in range(2)]
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

    def get_actual_lengths(self) -> tuple[int, int, int, list[int], list[int]]:
        return self.io_pairs[self.current_pair].host_io.get_actual_lengths()

    # The prepare_batch_tensor method also has to prepare the carry over ids
    def prepare_batch_tensors(self, requests_in_batch: list[FutureRequestState]) -> None:
        io_pair = self.io_pairs[self.current_pair]
        io_pair.host_io.prepare_batch_tensors(requests_in_batch)
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
    def get_model_kwargs(self, padded_q_size: int = 0, padded_kv_cache_size: int = 0) -> dict[str, Any]:
        io_pair = self.io_pairs[self.current_pair]
        io_pair.transfer_inputs_h2d(self.h2d_stream)
        self.h2d_stream.record_event(io_pair.h2d_over)
        self.compute_stream.wait_event(io_pair.h2d_over)
        return io_pair.device_io.get_model_kwargs(padded_q_size, padded_kv_cache_size)

    def carry_over_tokens(self, input_ids: torch.Tensor) -> None:
        """As explained in the infer_carry_over_ids method, we might need to carry over tokens just predicted in batch N
        before launching the forwar pass of batch N+1. This method performs the carry over, and is recorded in CUDA
        graphs if they are enabled."""
        # Retrieve previous batch output ids
        prev_output_ids = self.io_pairs[1 - self.current_pair].device_io.output_ids
        # Retrieve the carry over ids and mask
        carry_over_ids = self.io_pairs[self.current_pair].device_io.carry_over_ids
        # Compute tokens to carry over and the corresponding mask
        carried_over_ids = prev_output_ids[carry_over_ids]
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

    @property
    def graphs(self) -> CudaGraphBuffer:
        return self.io_pairs[self.current_pair].device_io.graphs

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
    def prepare_batch_update(self) -> tuple[list[FutureRequestState], list[int]]:
        io_pair = self.io_pairs[self.current_pair]
        io_pair.d2h_over.synchronize()
        return io_pair.host_io.prepare_batch_update()
