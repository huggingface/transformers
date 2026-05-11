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
import queue
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, log2
from typing import Any, TypeVar

import torch
import torch.distributed as dist
from torch.distributed.tensor.device_mesh import DeviceMesh

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import is_torch_greater_or_equal

from .requests import FutureRequestState, RequestState, RequestStatus, logger


T = TypeVar("T")


class CudaGraphBuffer:
    """A fixed-size dict for CUDA graphs with LRU eviction when full."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, but got {max_size}")
        self.max_size = max_size
        self._storage: OrderedDict[tuple[int, ...], torch.cuda.CUDAGraph] = OrderedDict()

    def __del__(self) -> None:
        original_max_size = self.max_size
        self.max_size = 1  # 0 would cause an infinite loop, 1 is enough to clear all graphs
        self.plan_for_new_graph(silent=True)
        self.max_size = original_max_size

    def get_graph(self, key: tuple[int, ...]) -> torch.cuda.CUDAGraph | None:
        graph = self._storage.get(key)
        if graph is not None:
            self._storage.move_to_end(key)
        return graph

    def plan_for_new_graph(self, silent: bool = False) -> None:
        while len(self._storage) >= self.max_size:
            evicted_key, evicted_graph = self._storage.popitem(last=False)
            if not silent:
                logger.info(f"Evicting graph for {evicted_key = }")
            evicted_graph.reset()

    def set_graph(self, key: tuple[int, ...], graph: torch.cuda.CUDAGraph) -> None:
        # In our use case, this should not have any effect because we plan for a new graph before it is captured
        self.plan_for_new_graph()
        self._storage[key] = graph


@dataclass
class WorkloadHints:
    """A tiny dataclass containing hints to help choose good continuous batching defaults"""

    max_prompt_length: int = 0
    max_generated_length: int = 0


def attn_mask_is_needed(config: PretrainedConfig) -> bool:
    """Checks if attention mask is needed for the given (config)."""
    return config._attn_implementation in ["paged|eager", "paged|sdpa"]


def pad_to_interval(size: int, interval_size: int, max_value: int) -> int:
    """Return the smallest multiple of (interval_size) >= (size), capped at (max_value)."""
    if interval_size <= 0:
        return max_value
    padded = ceil(size / interval_size) * interval_size if size > 0 else interval_size
    return min(padded, max_value)


def pad_to_pow2(value: int, max_value: int, min_value: int = 0) -> int:
    """Return the smallest power of 2 >= (value), capped at (max_value). If a minimum value is provided, the value is at
    least padded to that value."""
    value = max(value, max(1, min_value))
    padded = 2 ** int(ceil(log2(value)))
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
    num_q_tokens: int,
    max_kv_read: int,
    cache: Any,  # not annotated to avoid circular import
) -> list[FutureRequestState]:
    """A utility function to create a list of FutureRequestStates for the warmup of CB."""
    # Setup
    request_ids = [f"__warmup_{status.name}_{i}__" for i in range(num)]
    total_tokens = num_q_tokens + max_kv_read
    blocks_needed = ceil(total_tokens / cache.block_size)
    # Main loop
    future_states = []
    for req_id in request_ids:
        state = RequestState(request_id=req_id, initial_tokens=[0] * total_tokens, max_new_tokens=1)
        state._status = status  # bypass the property setter to avoid the lifecycle side effects
        state.tokens_to_process = [0] * num_q_tokens
        state.position_offset = max_kv_read
        # Stop if allocation fails for any request
        allocated = cache.allocate_blocks(blocks_needed, state.request_id, 0)
        if allocated is None:
            return future_states
        future_states.append(
            FutureRequestState(state, has_new_token=True, complete_blocks=0, query_length=num_q_tokens)
        )
    return future_states


def drain_queue(request_queue: queue.Queue) -> list[RequestState]:
    """Drains a queue and returns a list of RequestStates."""
    new_states: list[RequestState] = []
    while not request_queue.empty():
        try:
            state = request_queue.get_nowait()
            if state is not None:
                new_states.append(state)
        except queue.Empty:
            break
    return new_states


class DistributedHelper:
    """A helper class to handle distributed-related operations. Notably, it does not crash when distributed is off."""

    def __init__(self, device_mesh: DeviceMesh | None) -> None:
        self.device_mesh = device_mesh
        self.dist_on = dist.is_available() and dist.is_initialized()

        # These attributes depend on the global dist state
        self.global_rank = dist.get_rank() if self.dist_on else 0
        self.world_size = dist.get_world_size() if self.dist_on else 1

        # These attributes depend on the TP state
        if self.dist_on and self.device_mesh is not None:
            self.tp_size = self.device_mesh.size()
            self.tp_group = self.device_mesh.get_group()
            self.tp_root_global_rank = dist.get_global_rank(self.tp_group, 0)
            self.tp_local_rank = self.device_mesh.get_local_rank()
            self.is_tp_driver = self.tp_local_rank == 0
            # If TP is on, we create a dedicate CPU group
            tp_ranks = dist.get_process_group_ranks(self.tp_group)
            self.ingress_group = dist.new_group(ranks=tp_ranks, backend="gloo")
        else:
            self.tp_size = 1
            self.tp_group = None
            self.tp_root_global_rank = 0
            self.tp_local_rank = 0
            self.is_tp_driver = False
            self.ingress_group = None

        # These attributes depend on the DP state
        self.dp_rank = self.global_rank // self.tp_size
        self.dp_size = self.world_size // self.tp_size

    def destroy_ingress_group(self) -> None:
        """Destroys the ingress group."""
        if self.ingress_group is not None:
            dist.destroy_process_group(self.ingress_group)
            self.ingress_group = None

    def tp_broadcast_from_rank_0(self, value: torch.Tensor) -> torch.Tensor:
        """Inside each TP group, broadcasts the given value from rank 0 to all other ranks."""
        if self.tp_size > 1:
            dist.broadcast(value, src=self.tp_root_global_rank, async_op=False, group=self.tp_group)
        return value

    def tp_broadcast_cpu_from_rank_0(self, value: torch.Tensor) -> torch.Tensor:
        """Inside each TP group, broadcasts a CPU tensor from rank 0 over the gloo ingress group."""
        if self.tp_size > 1:
            dist.broadcast(value, src=self.tp_root_global_rank, async_op=False, group=self.ingress_group)
        return value

    def tp_all_reduce_min(self, value: torch.Tensor) -> torch.Tensor:
        """Inside each TP group, all-reduces a tensor with the MIN op. No-op when TP is off."""
        if self.tp_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MIN, group=self.tp_group)
        return value

    def tp_broadcast_object(self, obj):
        """Inside each TP group, broadcasts an arbitrary picklable Python object from TP-rank 0 to all other ranks.
        Used to keep request ingress and cancellations consistent across TP workers without requiring all ranks to
        receive the same external request stream. Uses a dedicated CPU (gloo) `ingress_group` for broadcast."""
        if self.tp_size <= 1:
            return obj
        holder = [obj] if self.is_tp_driver else [None]
        dist.broadcast_object_list(
            holder, src=self.tp_root_global_rank, group=self.ingress_group, device=torch.device("cpu")
        )
        return holder[0]

    def maybe_warn_nccl_graph_mixing(self) -> None:
        """Throws a warning if TP is on and NCCL's graph mixing support was supposed to be disabled but isn't. That can
        happen if the distributed group is created before graph mixing is disabled. Typically, if the model is
        initialized before the ContinuousBatchingConfig is created."""
        tp_on = self.tp_size > 1
        graph_mixing_not_disabled = os.environ.get("NCCL_GRAPH_MIXING_SUPPORT") != "0"
        if tp_on and graph_mixing_not_disabled:
            logger.warning(
                "NCCL_GRAPH_MIXING_SUPPORT was not set to '0' before init_process_group: performance will be harmed. "
                "Construct your `ContinuousBatchingConfig(...)` BEFORE calling `from_pretrained(tp_plan='auto')`, or "
                "set NCCL_GRAPH_MIXING_SUPPORT=0 in the launch environment."
            )

    def set_tp_seed(self, seed: int | None, model_device: torch.device) -> None:
        # Get an integer seed for the TP group
        if seed is None:
            tp_seed_tensor = torch.randint(0, 2**32 - 1, (1,), dtype=torch.int64, device=model_device)
        else:
            tp_seed_tensor = torch.tensor(seed, dtype=torch.int64, device=model_device)
        # Broadcast the seed to all ranks from rank 0 and memoize it
        tp_seed_tensor = self.tp_broadcast_from_rank_0(tp_seed_tensor)
        tp_seed = tp_seed_tensor.item()
        if self.global_rank == 0 and seed is None:
            logger.info(f"Found no user-specified seed in the config. Setting the config seed to: {tp_seed}.")
        # Set the seed while accounting for DP replicas
        torch.manual_seed(tp_seed + self.dp_rank)
