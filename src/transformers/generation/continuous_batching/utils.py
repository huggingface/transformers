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
import queue
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, log2
from typing import Any

import torch

from transformers.configuration_utils import PretrainedConfig

from .requests import FutureRequestState, RequestState, RequestStatus


SUPPORTED_GRAPH_ACCELERATOR_TYPES = ("cuda", "xpu")


def _get_graph_class_name(device_type: str) -> str:
    if device_type == "cuda":
        return "CUDAGraph"
    if device_type == "xpu":
        return "XPUGraph"
    raise RuntimeError(f"Expected one of {SUPPORTED_GRAPH_ACCELERATOR_TYPES}, but got {device_type = }.")


def get_torch_device_module(device: torch.device) -> Any:
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.cuda
    if device_type == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu
    raise RuntimeError(f"Expected one of {SUPPORTED_GRAPH_ACCELERATOR_TYPES}, but got {device_type = }.")


def is_accelerator_graph_available(device: torch.device | None = None) -> bool:
    device_types = SUPPORTED_GRAPH_ACCELERATOR_TYPES if device is None else (torch.device(device).type,)
    for device_type in device_types:
        try:
            device_module = get_torch_device_module(torch.device(device_type))
        except RuntimeError:
            continue
        is_available = getattr(device_module, "is_available", None)
        required_attrs = (_get_graph_class_name(device_type), "graph", "MemPool", "use_mem_pool")
        if callable(is_available) and is_available() and all(hasattr(device_module, attr) for attr in required_attrs):
            return True
    return False


def get_accelerator_graph(device: torch.device) -> Any:
    device_type = torch.device(device).type
    device_module = get_torch_device_module(device)
    graph_class_name = _get_graph_class_name(device_type)
    graph_class = getattr(device_module, graph_class_name, None)
    if graph_class is None:
        raise RuntimeError(f"Graph capture on {device_type} requires torch.{device_type}.{graph_class_name}.")
    return graph_class()


class AcceleratorGraphBuffer:
    """A dict for accelerator graphs with a special __del__ method to make sure the graphs are properly reset."""

    def __init__(self) -> None:
        self._storage: dict[tuple[int, ...], Any] = {}

    def __del__(self) -> None:
        self.clear()

    def clear(self) -> None:
        while self._storage:
            _, graph = self._storage.popitem()
            graph.reset()

    def get_graph(self, key: tuple[int, ...]) -> Any | None:
        return self._storage.get(key)

    def set_graph(self, key: tuple[int, ...], graph: Any) -> None:
        self._storage[key] = graph


@dataclass
class WorkloadHints:
    """A tiny dataclass containing hints to help choose good continuous batching defaults"""

    max_prompt_length: int = 0
    max_generated_length: int = 0
    num_requests: int = 0


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


def get_accelerator_pools(device: torch.device) -> tuple:
    """Returns a tuple of (mem_pool, graph_pool_id) for accelerator graphs."""
    device_module = get_torch_device_module(device)
    mem_pool = device_module.MemPool()
    graph_pool_id = mem_pool.id
    return mem_pool, graph_pool_id


@contextmanager
def mem_pool_ctx(device: torch.device, mem_pool):
    """A context manager to use an accelerator mem pool."""
    device_module = get_torch_device_module(device)
    with device_module.use_mem_pool(mem_pool):
        yield


@contextmanager
def graph_capture_ctx(device: torch.device, graph, stream, graph_pool_id):
    device_type = torch.device(device).type
    device_module = get_torch_device_module(device)
    kwargs = {"stream": stream, "pool": graph_pool_id}
    if device_type == "cuda":
        kwargs["capture_error_mode"] = "thread_local"
    with device_module.graph(graph, **kwargs):
        yield
