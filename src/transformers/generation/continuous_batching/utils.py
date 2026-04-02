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
from collections import OrderedDict
from math import ceil
from typing import Any

import torch

from transformers.configuration_utils import PretrainedConfig

from .requests import FutureRequestState, RequestState, RequestStatus, logger


class CudaGraphBuffer:
    """A fixed-size dict for CUDA graphs with LRU eviction when full."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, but got {max_size}")
        self.max_size = max_size
        self._storage: OrderedDict[tuple[int, int], torch.cuda.CUDAGraph] = OrderedDict()

    def __del__(self) -> None:
        original_max_size = self.max_size
        self.max_size = 1  # 0 would cause an infinite loop, 1 is enough to clear all graphs
        self.plan_for_new_graph(silent=True)
        self.max_size = original_max_size

    def get_graph(self, q_len: int, kv_len: int) -> torch.cuda.CUDAGraph | None:
        graph = self._storage.get((q_len, kv_len))
        if graph is not None:
            self._storage.move_to_end((q_len, kv_len))
        return graph

    def plan_for_new_graph(self, silent: bool = False) -> None:
        while len(self._storage) >= self.max_size:
            evicted_key, evicted_graph = self._storage.popitem(last=False)
            if not silent:
                logger.info(f"Evicting graph for {evicted_key = }")
            evicted_graph.reset()

    def set_graph(self, q_len: int, kv_len: int, graph: torch.cuda.CUDAGraph) -> None:
        # In our use case, this should not have any effect because we plan for a new graph before it is captured
        self.plan_for_new_graph()
        logger.info(f"Setting graph for {q_len = }, {kv_len = }")
        self._storage[(q_len, kv_len)] = graph


def attn_mask_is_needed(config: PretrainedConfig) -> bool:
    """Checks if attention mask is needed for the given (config)."""
    return config._attn_implementation in ["paged|eager", "paged|sdpa"]


def pad_to_interval(size: int, interval_size: int, max_value: int) -> int:
    """Return the smallest multiple of (interval_size) >= (size), capped at (max_value)."""
    if interval_size <= 0:
        return max_value
    padded = ceil(size / interval_size) * interval_size if size > 0 else interval_size
    return min(padded, max_value)


def aligned_divide(x: int, divide_by: int, align_to: int) -> int:
    x = int(ceil(x / divide_by))
    if x % align_to:
        x += align_to - (x % align_to)
    return x


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


def create_warmup_future_states(
    num: int,
    status: RequestStatus,
    num_query_tokens: int,
    num_cache_tokens: int,
    cache: Any,  # not annotated to avoid circular import
) -> list[FutureRequestState]:
    """An utility function to create a list of FutureRequestStates for the warmup of CB."""
    # Setup
    request_ids = [f"__warmup_{status.name}_{i}__" for i in range(num)]
    total_tokens = num_query_tokens + num_cache_tokens
    blocks_needed = ceil(total_tokens / cache.block_size)
    # Main loop
    future_states = []
    for req_id in request_ids:
        state = RequestState(request_id=req_id, initial_tokens=[0] * total_tokens, max_new_tokens=1)
        state._status = status  # bypass the property setter to avoid the lifecycle side effects
        state.tokens_to_process = [0] * num_query_tokens
        state.position_offset = num_cache_tokens
        # Stop if allocation fails for any request
        allocated = cache.allocate_blocks(blocks_needed, state.request_id, 0)
        if allocated is None:
            return future_states
        future_states.append(FutureRequestState(state, has_new_token=True, complete_blocks=0))
    return future_states
