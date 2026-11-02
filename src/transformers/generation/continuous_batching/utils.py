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
import re
import threading
import warnings
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from math import ceil, log2
from typing import Any, TypeVar

import torch

from ...configuration_utils import PreTrainedConfig
from ...utils import is_psutil_available, is_torch_xpu_available
from ...utils.logging import logging


if is_psutil_available():
    import psutil


# NOTE: AMD GPUs are supported under the "cuda" module
DEVICE_TYPE_TO_GRAPH_NAME = {"cuda": "CUDAGraph", "xpu": "XPUGraph"}

T = TypeVar("T")

# NUMA policy as shown in /proc/<pid>/numa_maps: "<mode>[=<flags>]:<nodes>", e.g. "bind:0-1,3" or "bind=static:0"
_NUMA_BIND_POLICY_PATTERN = re.compile(r"bind(=[a-z|]+)?:(?P<nodes>\d+(-\d+)?(,\d+(-\d+)?)*)")
# First line of /sys/devices/system/node/node<N>/meminfo
_NUMA_NODE_MEMTOTAL_PATTERN = re.compile(r"Node \d+ MemTotal:\s+(?P<size_kb>\d+) kB")


# We centralize the logger here to coordinate between logging and progress bar
logger = logging.getLogger("ContinuousBatchingLogger")
# Add a handler to the logger to print the logs to the console. Only happens once thanks to setting propagate to False.
if logger.propagate:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


class CudaGraphBuffer:
    """A dict for CUDA graphs with a special __del__ method to make sure the graphs are properly reset."""

    def __init__(self) -> None:
        self._storage: dict[tuple[int, ...], torch.cuda.CUDAGraph] = {}

    def __del__(self) -> None:
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
    return config._attn_implementation in ["paged|eager", "sdpa"]


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


def drain_queue(q: queue.Queue[T]) -> list[T]:
    """Drains a queue and returns its content as a list."""
    elements: list[T] = []
    while not q.empty():
        try:
            elem = q.get_nowait()
            if elem is not None:
                elements.append(elem)
        except queue.Empty:
            break
    return elements


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


# ---------------------------------------------- DEVICE AGNOSTIC UTILS ----------------------------------------------- #


def get_available_accelerator_module() -> Any | None:
    """Returns the accelerator module if there is one and it is available, None otherwise."""
    device_module = torch.get_device_module()  # the device is never specified, so that we always get the same module
    if device_module.__name__.endswith("cpu") or not torch.accelerator.is_available():
        return None
    return device_module


def create_device_stream(device: torch.device) -> torch.cuda.Stream | None:
    """If the given device is available and supports streams, returns a new stream. Otherwise, returns None."""
    device_module = torch.get_device_module()
    if torch.accelerator.is_available() and hasattr(device_module, "Stream"):
        return device_module.Stream(device=device)
    return None


def stream_context(stream: torch.cuda.Stream | None) -> torch.cuda.StreamContext:
    """If the stream is not None, returns a context manager to use the stream. Otherwise, returns a null context.
    This function assumes that the current device supports streams."""
    if stream is None:
        return nullcontext()
    return torch.get_device_module().stream(stream)


def get_accelerator_graph_name() -> str | None:
    """If an accelerator is available and has cuda graph-like objects, returns their name. Otherwise, returns None."""
    # Stop if no accelerator is available
    accelerator_module = get_available_accelerator_module()
    if accelerator_module is None:
        return None
    # Stop if the accelerator does not support graphs no matter the version (eg. MPS)
    accelerator_type = accelerator_module.__name__.split(".")[-1]
    graph_name = DEVICE_TYPE_TO_GRAPH_NAME.get(accelerator_type)
    if graph_name is None:
        return None
    # Stop if the accelerator does not support the graph for this version (eg. XPU before 2.11)
    return graph_name if hasattr(accelerator_module, graph_name) else None


def create_accelerator_graph() -> torch.cuda.CUDAGraph:
    """If an accelerator is available and supports graphs, returns a new graph. Otherwise, raises an error."""
    graph_name = get_accelerator_graph_name()
    if graph_name is None:
        raise RuntimeError(
            f"The current device is unavailable or does not support graphs: {torch.get_device_module().__name__ = }."
        )
    graph_cls = getattr(torch.get_device_module(), graph_name)
    return graph_cls()


def accelerator_graph_capture_context(
    graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream | None, pool: int
) -> torch.cuda.graph:
    """Returns a context manager used to capture the graph. This function assumes that the current device supports
    graphs."""
    device_module = torch.get_device_module()
    capture_kwargs = {"stream": stream, "pool": pool}
    if device_module.__name__.endswith("cuda"):
        capture_kwargs["capture_error_mode"] = "thread_local"
    return device_module.graph(graph, **capture_kwargs)


def get_memory_pools() -> tuple[Any, int]:
    """If an accelerator is available and supports graphs, returns a tuple of (mem_pool, graph_pool_id) for the
    accelerator graphs. Otherwise, raises an error."""
    if get_accelerator_graph_name() is None:
        raise RuntimeError(
            "No graph-capture backend is available for the current accelerator: memory pools cannot be used."
        )
    mem_pool = torch.get_device_module().MemPool()
    graph_pool_id = mem_pool.id
    return mem_pool, graph_pool_id


def memory_pool_context(mem_pool: Any) -> AbstractContextManager:
    """A context manager to use a specific memory pool for allocations. This function assumes that the current device
    supports memory pools."""
    return torch.get_device_module().use_mem_pool(mem_pool)


# --------------------------------------------------- MEMORY UTILS --------------------------------------------------- #


def _parse_numa_node_list(node_list: str) -> set[int]:
    """Parses a node list in the kernel format, e.g. "0-1,3" -> {0, 1, 3}."""
    nodes = set()
    for node_range in node_list.split(","):
        first_node, _, last_node = node_range.partition("-")
        nodes.update(range(int(first_node), int(last_node or first_node) + 1))
    return nodes


def _get_allowed_numa_nodes() -> set[int]:
    """Returns the NUMA nodes the process can allocate memory from, i.e. the nodes allowed both by its cpuset (cgroups,
    `docker --cpuset-mems`, ...) and by its memory policy (`numactl --membind`, ...). Returns an empty set if they
    cannot be read, e.g. if we are not on Linux or there is no NUMA support."""
    allowed_sets: list[set[int]] = []

    # Retrieve the allowed nodes from the cpuset
    mems_allowed = None
    try:
        # The cpuset restricts memory to the nodes in "Mems_allowed_list: <nodes>"
        with open("/proc/self/status", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Mems_allowed_list:"):
                    mems_allowed = line.split(":")[1].strip()
                    break
    except OSError:
        pass

    if mems_allowed is not None:
        allowed_sets.append(_parse_numa_node_list(mems_allowed))

    # Retrieve the allowed nodes from the numa_maps
    first_mapping = None
    try:
        # Each line of numa_maps is "<address> <policy> ...". Usually, the <policy> is the same for all address ranges,
        # because it's the policy of the process: thus we only need to read the first line to learn the process' policy
        with open("/proc/self/numa_maps", encoding="utf-8", errors="replace") as f:
            first_mapping = f.readline()
    except OSError:
        pass

    if first_mapping is not None:
        # Only the "bind" mode is a hard limit: other modes (preferred, interleave, ...) fall back to any allowed node
        fields = first_mapping.split()
        policy_match = _NUMA_BIND_POLICY_PATTERN.fullmatch(fields[1]) if len(fields) > 1 else None
        if policy_match is not None:
            allowed_sets.append(_parse_numa_node_list(policy_match["nodes"]))

    # The process can only use nodes allowed by both the cpuset and the numa policy
    return set.intersection(*allowed_sets) if allowed_sets else set()


def _get_numa_nodes_memory(nodes: set[int]) -> int | None:
    """Returns the total memory of the given NUMA nodes in bytes, or None if the size of one of them cannot be read."""
    total_memory = 0
    for node in sorted(nodes):
        try:
            with open(f"/sys/devices/system/node/node{node}/meminfo", encoding="utf-8") as f:
                memtotal_match = _NUMA_NODE_MEMTOTAL_PATTERN.match(f.readline())
        except OSError:
            memtotal_match = None
        if memtotal_match is None:
            logger.warning(f"Cannot read the size of NUMA node {node}, ignoring the NUMA restrictions.")
            return None
        total_memory += int(memtotal_match["size_kb"]) * 1024  # the kernel's kB are KiB
    return total_memory


def get_device_and_memory_breakdown() -> tuple[torch.device, int, int, int]:
    """Returns the device and memory breakdown for the current device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Use mem_get_info to get actual free memory: device_properties().total_memory returns the physical device
        # total which ignores CUDA context and driver overhead (~0.5 GiB), leading to overcommit.
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = total_memory - free_memory
    elif is_torch_xpu_available():
        device = torch.device("xpu")
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
        total_memory = torch.xpu.get_device_properties(device).total_memory
        reserved_memory = torch.xpu.memory_reserved(device)
        allocated_memory = torch.xpu.memory_allocated(device)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        total_memory = torch.mps.recommended_max_memory()
        allocated_memory = torch.mps.current_allocated_memory()
        reserved_memory = torch.mps.driver_allocated_memory()
    else:
        device = torch.device("cpu")
        if is_psutil_available():
            total_memory = psutil.virtual_memory().total
            allocated_memory = reserved_memory = psutil.Process().memory_info().rss
        else:
            total_memory = allocated_memory = reserved_memory = 0
            logger.error("Cannot get memory breakdown on CPU without psutil: please install psutil.")
        # If the process is restricted to some NUMA nodes (numactl, cgroups), it can only use their memory
        numa_memory = _get_numa_nodes_memory(_get_allowed_numa_nodes())
        if numa_memory is not None and numa_memory > 0:
            total_memory = min(total_memory, numa_memory) if total_memory else numa_memory

    return device, total_memory, reserved_memory, allocated_memory


# ------------------------------------------------- DEPRECATED UTILS ------------------------------------------------- #


def get_cuda_pools() -> tuple:
    warnings.warn(
        "Deprecated. Use get_memory_pools() instead. Will be removed in transformers 5.23",
        FutureWarning,
        stacklevel=2,
    )
    return get_memory_pools()


def mem_pool_ctx(mem_pool: Any):
    warnings.warn(
        "Deprecated. Use memory_pool_context() instead. Will be removed in transformers 5.23",
        FutureWarning,
        stacklevel=2,
    )
    return memory_pool_context(mem_pool)


def create_warmup_future_states(num: int, status, num_q_tokens: int, max_kv_read: int, cache) -> list:
    """A deprecated utility function to create a list of FutureRequestStates for the warmup of CB."""
    from .requests import create_warmup_future_states as requests_version

    warnings.warn(
        "`create_warmup_future_states` has been moved to continuous_batching/requests.py and will be deleted from "
        "utils.py in v5.24. Please import it from there instead.",
        FutureWarning,
        stacklevel=2,
    )

    return requests_version(num, status, num_q_tokens, max_kv_read, cache)
