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
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import ceil, log2
from typing import Any

import torch

from ...configuration_utils import PreTrainedConfig
from .requests import FutureRequestState, RequestState, RequestStatus


SUPPORTED_CUDA_GRAPH_DEVICE_TYPES = ("cuda", "xpu")


def _get_graph_class_name(device_type: str) -> str:
    if device_type == "cuda":
        return "CUDAGraph"
    if device_type == "xpu":
        return "XPUGraph"
    raise RuntimeError(f"Expected one of {SUPPORTED_CUDA_GRAPH_DEVICE_TYPES}, but got {device_type = }.")


def device_stream_ctx(stream: torch.cuda.Stream | None):
    if stream is None:
        return nullcontext()
    return torch.get_device_module(stream.device).stream(stream)


def is_cuda_graph_available(device: torch.device | None = None) -> bool:
    device_types = SUPPORTED_CUDA_GRAPH_DEVICE_TYPES if device is None else (torch.device(device).type,)
    for device_type in device_types:
        try:
            device_module = torch.get_device_module(device_type)
        except RuntimeError:
            continue
        is_available = getattr(device_module, "is_available", None)
        required_attrs = (_get_graph_class_name(device_type), "graph", "MemPool", "use_mem_pool")
        if callable(is_available) and is_available() and all(hasattr(device_module, attr) for attr in required_attrs):
            return True
    return False


def get_cuda_graph(device: torch.device) -> torch.cuda.CUDAGraph:
    device_type = torch.device(device).type
    device_module = torch.get_device_module(device)
    graph_class_name = _get_graph_class_name(device_type)
    graph_class = getattr(device_module, graph_class_name, None)
    if graph_class is None:
        raise RuntimeError(f"Graph capture on {device_type} requires torch.{device_type}.{graph_class_name}.")
    return graph_class()


class CudaGraphBuffer:
    """A dict for CUDA graphs with a special __del__ method to make sure the graphs are properly reset."""

    def __init__(self) -> None:
        self._storage: dict[tuple[int, ...], torch.cuda.CUDAGraph] = {}

    def __del__(self) -> None:
        self.clear()

    def clear(self) -> None:
        while self._storage:
            _, graph = self._storage.popitem()
            graph.reset()

    def get_graph(self, key: tuple[int, ...]) -> torch.cuda.CUDAGraph | None:
        return self._storage.get(key)

    def set_graph(self, key: tuple[int, ...], graph: torch.cuda.CUDAGraph) -> None:
        self._storage[key] = graph


@dataclass
class WorkloadHints:
    """A tiny dataclass containing hints to help choose good continuous batching defaults"""

    max_prompt_length: int = 0
    max_generated_length: int = 0
    num_requests: int = 0


class ThreadLocalCounter(threading.local):
    def __init__(self) -> None:
        self.value = 0


def attn_mask_is_needed(config: PreTrainedConfig) -> bool:
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
        # Apply the causal mask fully in place: a request block can be as large as [max_batch_tokens, whole cache],
        # so materializing [seqlen_q, seqlen_k] temporaries here can cost tens of GB during warmup
        block = attention_mask[..., query_range, key_range]
        block.fill_(min_value)
        block.triu_(causal_diagonal)  # zeroes everything strictly below the causal diagonal
        # Apply sliding window mask if needed. This branch keeps a temporary, but its size is bounded by the window.
        if sliding_window > 1:
            sliding_diagonal = seqlen_k - seqlen_q - sliding_window
            block.add_(torch.tril(torch.full_like(block, min_value), diagonal=sliding_diagonal))


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
    # Main loop
    future_states = []
    for req_id in request_ids:
        state = RequestState(request_id=req_id, initial_tokens=[0] * total_tokens, max_new_tokens=1)
        state._status = status  # bypass the property setter to avoid the lifecycle side effects
        state.tokens_to_process = [0] * num_q_tokens
        state.position_offset = max_kv_read
        # Stop if allocation fails for any request. Since position_offset acts as the past length, this allocates
        # enough cache for the whole fake request (past + query).
        if not cache.can_store_request_tokens(state, num_q_tokens):
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


def get_cuda_graph_pools(device: torch.device) -> tuple:
    """Returns a tuple of (mem_pool, graph_pool_id) for CUDA graphs."""
    device_module = torch.get_device_module(device)
    mem_pool = device_module.MemPool()
    graph_pool_id = mem_pool.id
    return mem_pool, graph_pool_id


@contextmanager
def mem_pool_ctx(device: torch.device, mem_pool):
    """A context manager to use a CUDA graph mem pool."""
    device_module = torch.get_device_module(device)
    with device_module.use_mem_pool(mem_pool):
        yield


@contextmanager
def graph_capture_ctx(device: torch.device, graph, stream, graph_pool_id):
    device_type = torch.device(device).type
    device_module = torch.get_device_module(device)
    kwargs = {"stream": stream, "pool": graph_pool_id}
    if device_type == "cuda":
        kwargs["capture_error_mode"] = "thread_local"
    with device_module.graph(graph, **kwargs):
        yield


def find_num_key_value_heads(config: PreTrainedConfig) -> int:
    """Finds the number of key-value heads for the given config."""
    # If the model supports GQA, we leverage it by using the num_key_value_heads attribute
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is not None:
        return kv_heads
    # Otherwise, the number of KV heads is the same as the number of attention heads
    kv_heads = getattr(config, "num_attention_heads", None)
    if kv_heads is not None:
        return kv_heads
    raise ValueError(f"num_key_value_heads or num_attention_heads could not be found in the config:\n{config}")


def find_head_dim(config: PreTrainedConfig) -> int:
    """Finds the head dimension for the given config."""
    # If the model has the head_dim attribute, there is nothing to do but return it
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    # If it is missing, we may reconstruct it from the hidden size and the number of attention heads
    hidden_size = getattr(config, "hidden_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    if hidden_size is not None and num_attention_heads is not None:
        return hidden_size // num_attention_heads
    raise ValueError(f"head_dim or (hidden_size and num_attention_heads) could not be found in the config:\n{config}")


def exact_div(a: int, b: int) -> int:
    """Divide an integer a by a integer b and error out if there is a remainder."""
    quotient, remainder = divmod(a, b)
    if remainder:
        raise ValueError(f"Division of {a} by {b} is not exact: {remainder = } != 0")
    return quotient
