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
from .requests import TMP_TOKEN_ID, RequestState


def attn_mask_is_needed(config: PretrainedConfig) -> bool:
    """Checks if attention mask is needed for the given (config)."""
    return config._attn_implementation in ["paged|eager", "paged|sdpa"]


def build_attention_mask(
    attention_mask: torch.Tensor,
    cumulative_seqlens_q: list[int],
    cumulative_seqlens_k: list[int],
    sliding_window: int = 1,
) -> None:
    """Builds an attention mask inplace using the cumulative seqlens of the query and key. If given a sliding window, it
    will also apply a sliding window mask on top. The attention mask is not boolean, it uses zeroes and -inf (or its
    equivalent) so it's more of an attention score bias tensor.
    The attention mask is a block-diagonal matrix, with each block an attention mask for a single query-key pair.
    Each of those block is built from a causal mask and, if there is a sliding window, a sliding window mask.

    An example is represented below, with seqlen_k = 8, seqlen_q = 4 and sliding_window = 6:

    CAUSAL MASK:

           █ █ █ █ █ ░ ░ ░
           █ █ █ █ █ █ ░ ░
           █ █ █ █ █ █ █ ░
           █ █ █ █ █ █ █ █

    SLIDING WINDOW MASK:
         ┌──────────────────────── seqlen_k - seqlen_q - sliding_window = 8 - 4 - 6 = -2 offset to the left
       <─┴─>
     ░ █ | █ █ █ █ █ █ █ █
     ░ ░ | █ █ █ █ █ █ █ █
     ░ ░ | ░ █ █ █ █ █ █ █
     ░ ░ | ░ ░ █ █ █ █ █ █

    ATTENTION MASK (sum of causal and sliding window masks):

           █ █ █ █ █ ░ ░ ░
           █ █ █ █ █ █ ░ ░
           ░ █ █ █ █ █ █ ░
           ░ ░ █ █ █ █ █ █

    Another example with seqlen_k = 5, seqlen_q = 3 and sliding_window = 2:

    CAUSAL MASK:

           █ █ █ ░ ░
           █ █ █ █ ░
           █ █ █ █ █

    SLIDING WINDOW MASK:
         ┌──────────────────────── seqlen_k - seqlen_q - sliding_window = 5 - 3 - 2 = 0 offset to the left
        <┴>
         | ░ █ █ █ █
         | ░ ░ █ █ █
         | ░ ░ ░ █ █

    ATTENTION MASK (sum of causal and sliding window masks):

           ░ █ █ ░ ░
           ░ ░ █ █ ░
           ░ ░ ░ █ █

    """
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
            sliding_diagonal = seqlen_k - seqlen_q - sliding_window
            masked += torch.tril(minus_inf, diagonal=sliding_diagonal)
        # Replace in attention mask
        attention_mask[..., query_range, key_range] = masked


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
    """Manages input/output tensors for continuous batching generation. This class handles the allocation and management
    of static tensors used during generation steps in continuous batching mode. Allocation is done once at init time.

    The class is responsible for:
    - Setting up static tensor storage for all generation inputs/outputs
    - Preparing batch tensors from a list of request states before each forward pass
    - Building model keyword arguments with optional padding for CUDA graphs/torch.compile
    - Resetting tensors between batches while minimizing memory operations

    It keeps track of the requests in the current batch as well as the actual number of tokens (Q and KV), sequences in
    the batch and sizes of indices. This is useful when using padded inputs, for CUDA graphs and/or torch.compile.
    """

    def __init__(
        self, cache: PagedAttentionCache, config: PretrainedConfig, device: torch.device, model_dtype: torch.dtype
    ) -> None:
        """Initialize the continuous batching I/O manager.

        Args:
            cache: The [`PagedAttentionCache`] instance managing the KV cache.
            config: The model's pretrained configuration.
            device: The device to allocate tensors on.
            model_dtype: The data type for model computations.
        """
        # Memoize attributes
        self.cache = cache
        self.device = device
        self.config = config
        self.model_dtype = model_dtype
        self.sliding_window = 1 if getattr(config, "sliding_window", None) is None else config.sliding_window
        # Setup accumulators
        self.requests_in_batch: list[RequestState] = []
        self.actual_query_length = 0
        self.actual_key_length = 0
        self.actual_batch_size = 0
        self.actual_index_sizes = [(0, 0) for _ in range(cache.num_groups)]
        # Setup static tensors
        self.setup_static_tensors()
        self.reset_static_tensors(full_reset=True)

    @traced(standalone=True)
    def setup_static_tensors(self) -> None:
        """Allocates static tensors for generation inputs and outputs. This is called only once at init time, to avoid
        repeated allocations and enable CUDA graphs. All tensors are allocated with maximum possible sizes.
        The allocated tensors are:

        - `input_ids` and `position_ids`: Query token information
        - `cumulative_seqlens_q` and `cumulative_seqlens_k`: Sequence length tracking for FlashAttention-style batching
        - `attention_mask`: Optional attention masks (only for eager/SDPA implementations)
        - `write_index` and `read_index` storage: Cache indexing tensors for each attention group
        - `output_ids`: Storage for generated token IDs
        """
        num_pages = self.cache.num_blocks * self.cache.block_size

        # Some tensors always have the same shape regardless of the model
        self.input_ids = torch.empty((1, self.cache.max_batch_tokens), dtype=torch.int32, device=self.device)
        self.position_ids = torch.empty((1, self.cache.max_batch_tokens), dtype=torch.int32, device=self.device)
        self.cumulative_seqlens_q = torch.empty(
            (self.cache.max_batch_tokens + 1,), dtype=torch.int32, device=self.device
        )
        self.max_seqlen_q = 0
        self.logits_indices = torch.empty((self.cache.max_batch_tokens,), dtype=torch.int32, device=self.device)
        self.output_ids = torch.empty((self.cache.max_batch_tokens,), dtype=torch.int32, device=self.device)

        # For some kwargs, we have a dict of tensors with as many items as there are attention types
        self.cumulative_seqlens_k: dict[str, torch.Tensor] = {}
        if self.cache.num_full_attention_groups:
            self.cumulative_seqlens_k["full_attention"] = torch.empty(
                (self.cache.max_batch_tokens + 1,), dtype=torch.int32, device=self.device
            )
        if self.cache.num_sliding_attention_groups:
            self.cumulative_seqlens_k["sliding_attention"] = torch.empty(
                (self.cache.max_batch_tokens + 1,), dtype=torch.int32, device=self.device
            )
        self.max_seqlen_k = dict.fromkeys(self.cumulative_seqlens_k.keys(), 0)

        if attn_mask_is_needed(self.config):
            self.attention_mask = {}
            for layer_type in self.cumulative_seqlens_k.keys():
                self.attention_mask[layer_type] = torch.empty(
                    size=(1, 1, self.cache.max_batch_tokens, num_pages + self.cache.max_batch_tokens),
                    dtype=self.model_dtype,
                    device=self.device,
                )
        else:
            self.attention_mask = None

        # For other kwargs, we need a list of tensors with as many tensors as there are groups
        self.write_index_storage = [
            torch.empty((self.cache.max_batch_tokens,), dtype=torch.int32, device=self.device)
            for _ in range(self.cache.num_groups)
        ]
        self.read_index_storage = [
            torch.empty((num_pages + self.cache.max_batch_tokens), dtype=torch.int32, device=self.device)
            for _ in range(self.cache.num_groups)
        ]
        # For read index, the +T is because there are -1 for seqlen_q when model uses a sliding window

    @traced
    @torch.no_grad()
    def reset_static_tensors(self, full_reset: bool = False) -> None:
        """Reset static tensors for the next batch. For efficiency, this only resets the portions of tensors that were
        actually used in the previous batch, using the attributes actual_query_length, actual_key_length, and
        actual_batch_size. If a (full_reset) is requested, the entire tensor storage is reset.
        """
        # Compute the slice to reset
        q_len = self.write_index_storage[0].size(-1) if full_reset else self.actual_query_length
        k_len = self.read_index_storage[0].size(-1) if full_reset else self.actual_key_length
        b_size = self.write_index_storage[0].size(0) if full_reset else self.actual_batch_size

        # Reset the attributes that always have the same shape
        self.input_ids[:, :q_len].zero_()
        self.position_ids[:, :q_len].zero_()
        self.cumulative_seqlens_q[: b_size + 1].zero_()
        self.max_seqlen_q = 0
        self.logits_indices[:q_len].fill_(-1)
        self.output_ids[:q_len].fill_(-1)

        # Reset the attributes that are either tensors or dict of tensors
        for layer_type in self.cumulative_seqlens_k:
            self.cumulative_seqlens_k[layer_type][: b_size + 1].zero_()
            self.max_seqlen_k[layer_type] = 0
            if self.attention_mask is not None:
                self.attention_mask[layer_type][:, :, :q_len, :k_len].fill_(torch.finfo(self.model_dtype).min)

        # Reset the attributes that are lists of tensors
        for i in range(self.cache.num_groups):
            self.write_index_storage[i][:q_len].fill_(-2)  # -1 is used to let the cache where new states go
            self.read_index_storage[i][: q_len + k_len].fill_(-2)  # same

    @traced
    def prepare_batch_tensors(self, requests_in_batch: list[RequestState]) -> None:
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
        self.requests_in_batch = requests_in_batch
        if not self.requests_in_batch:
            raise ValueError("No requests in batch")

        # Reset the static tensors used for storage
        self.reset_static_tensors()  # FIXME: why does this make the generation faster?
        # Reset accumulators
        self.actual_query_length = 0
        self.actual_key_length = 0
        self.actual_batch_size = 0

        # Prepare accumulators
        input_ids = []
        position_ids = []
        cumulative_seqlens_q = [0]
        logits_indices = []
        cumulative_seqlens_k = {layer_type: [0] for layer_type in self.cumulative_seqlens_k.keys()}
        read_index = [[] for _ in range(self.cache.num_groups)]
        write_index = [[] for _ in range(self.cache.num_groups)]

        # Go through all the requests in the batch
        for state in self.requests_in_batch:
            # First we retrieve the lengths related to the request
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
            if not state.remaining_prefill_tokens:
                logits_indices.append(cumulative_seqlens_q[-1] - 1)
                state.generated_tokens.append(TMP_TOKEN_ID)

        # When looping over request is done, we can build the actual tensors. This is faster than modifying the static
        # tensors inside the loop.
        to_tensor = partial(torch.tensor, dtype=torch.int32, device=self.device)

        # Those kwargs always have the same type regardless of the model
        self.input_ids[:, : len(input_ids)] = to_tensor(input_ids)
        self.position_ids[:, : len(position_ids)] = to_tensor(position_ids)
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
            self.read_index_storage[i][: len(group_read_indices)] = to_tensor(group_read_indices)
            self.write_index_storage[i][: len(group_write_indices)] = to_tensor(group_write_indices)
            self.actual_index_sizes[i] = (len(group_read_indices), len(group_write_indices))

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
            input_ids=self.input_ids[:, :q_len],
            position_ids=self.position_ids[:, :q_len],
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
        for i, (read_index_size, write_index_size) in enumerate(self.actual_index_sizes):
            read_index_size = padded_kv_size if use_padding else read_index_size
            write_index_size = padded_q_size if use_padding else write_index_size
            kwargs.read_index.append(self.read_index_storage[i][:read_index_size])
            kwargs.write_index.append(self.write_index_storage[i][:write_index_size])

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
