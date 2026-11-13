# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Stream MoE experts from disk through a shared device cache. Inference only, on one device."""

import heapq
import json
import math
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open

from ..utils import logging
from .moe import ExpertsInterface


try:
    import fcntl
except ImportError:  # Windows
    fcntl = None

logger = logging.get_logger(__name__)

_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
_PACKED_DTYPES = {**_DTYPES, "U32": torch.uint32, "U8": torch.uint8}
_QUANTIZED_FORMAT = "mlx-affine-expert-blob"
_MLX_QUANTIZATION_KERNEL = ("kernels-community/mlx-quantization-metal-kernels", 2)
_COUNTER_LOCK = threading.Lock()


def _read_exact(path, offset, size, bypass_page_cache, what):
    """Read a byte range, optionally bypassing or dropping the OS page cache."""
    # TODO(portability): implement Windows cache bypass; report unsupported OS/filesystem combinations.
    data = bytearray(size)
    with path.open("rb", buffering=0) as handle:
        if bypass_page_cache and hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(handle.fileno(), fcntl.F_NOCACHE, 1)
        handle.seek(offset)
        view = memoryview(data)
        done = 0
        while done < size:
            count = handle.readinto(view[done:])
            if not count:
                raise OSError(f"Short checkpoint read: {path}, {what}")
            done += count
        # TODO(portability): distinguish best-effort cache eviction from uncached reads.
        if bypass_page_cache and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(handle.fileno(), offset, size, os.POSIX_FADV_DONTNEED)
    return data


class SafetensorsReader:
    """Validate safetensors headers and read tensors or contiguous expert slices without persistent mmap."""

    def __init__(self, directory, bypass_page_cache=False, packed=False):
        directory = Path(directory)
        self.bypass_page_cache = bypass_page_cache
        self.dtypes = _PACKED_DTYPES if packed else _DTYPES
        index = directory / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            filenames = sorted(set(weight_map.values()))
        else:
            weight_map = None
            filenames = ["model.safetensors"]
        self.tensors = {}
        self.read_bytes = 0
        self.read_calls = 0
        for filename in filenames:
            path = (directory / filename).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {path}")
            # Validate offsets, sizes and overlaps before reading the header.
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
            with path.open("rb") as handle:
                header_size = int.from_bytes(handle.read(8), "little")
                header = json.loads(handle.read(header_size))
            for name in keys:
                info = header[name]
                if name in self.tensors:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                self.tensors[name] = (path, 8 + header_size + info["data_offsets"][0], info)
                if weight_map is not None and weight_map.get(name) != filename:
                    raise ValueError(f"Checkpoint index disagrees with shard for {name}")
        if weight_map is not None and set(weight_map) != set(self.tensors):
            raise ValueError("Checkpoint index contains missing tensors")

    def shape(self, name):
        return tuple(self.tensors[name][2]["shape"])

    def read(self, name, expert=None):
        path, offset, info = self.tensors[name]
        shape = tuple(info["shape"])
        if info["dtype"] not in self.dtypes:
            raise ValueError(f"Only F32/F16/BF16 (and packed) tensors can be read: {name} is {info['dtype']}")
        dtype = self.dtypes[info["dtype"]]
        item_size = torch.empty((), dtype=dtype).element_size()
        if expert is not None:
            if len(shape) < 2 or not 0 <= expert < shape[0]:
                raise ValueError(f"Invalid expert slice {expert} for {name}: {shape}")
            shape = shape[1:]
            offset += expert * math.prod(shape) * item_size
        size = math.prod(shape) * item_size
        data = _read_exact(path, offset, size, self.bypass_page_cache, f"tensor {name}")
        with _COUNTER_LOCK:
            self.read_bytes += size
            self.read_calls += 1
        return torch.frombuffer(data, dtype=dtype).reshape(shape)

    def read_expert(self, prefix, expert):
        if f"{prefix}.gate_up_proj" in self.tensors:
            return self.read(f"{prefix}.gate_up_proj", expert), self.read(f"{prefix}.down_proj", expert)
        gate_up = torch.cat([self.read(f"{prefix}.{expert}.{p}_proj.weight") for p in ("gate", "up")])
        return gate_up, self.read(f"{prefix}.{expert}.down_proj.weight")


class ExpertStore:
    """Shared device pool with LRU or decaying LFU admission, for one inference request at a time.

    Staging and uncached experts can exceed the cache budget."""

    def __init__(
        self,
        reader,
        cache_bytes,
        device,
        dtype,
        policy="lru",
        decay_every=4096,
        io_threads=0,
        quantized=None,
    ):
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        if policy not in ("lru", "lfu"):
            raise ValueError(f"Unknown cache policy {policy!r}; use 'lru' or 'lfu'")
        self.policy = policy
        self.decay_every = decay_every
        self.counts = {}
        self.heap = []  # (count, sequence, key); discard stale entries lazily
        self.sequence = 0
        self.lookups = 0
        self.not_admitted = 0
        self.reader = reader
        self.cache_bytes = cache_bytes
        self.device = torch.device(device)
        self.dtype = dtype
        self.cache = OrderedDict()
        self.resident_bytes = 0
        self.peak_cache_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self.gate_up_pool = self.down_pool = None
        self.pools = None
        self.quantized = None
        if quantized is not None:
            info = json.loads((Path(quantized) / "quantization.json").read_text(encoding="utf-8"))
            if info.get("format") != _QUANTIZED_FORMAT:
                raise ValueError(f"Unsupported quantized experts format {info.get('format')!r}")
            self.quantized = (
                SafetensorsReader(quantized, reader.bypass_page_cache, packed=True),
                info["bits"],
                info["group_size"],
            )
            self.gather_qmm = _load_gather_qmm(self.device)
        self.num_slots = 0
        self.free_slots = []
        self.slot_tables = {}
        self.total_experts = 0
        self.compute = "grouped_mm"
        self.executor = ThreadPoolExecutor(io_threads, thread_name_prefix="experts-offload-io") if io_threads else None
        self.pending = {}
        self.window = 2 * io_threads
        self.prefetched = self.prefetch_used = self.installed_ahead = 0
        # Retain host buffers until the next routing copy synchronizes the device.
        self.inflight = []

    def register(self, prefix, module):
        shapes = {
            "gate_up_proj": (module.num_experts, 2 * module.intermediate_dim, module.hidden_dim),
            "down_proj": (module.num_experts, module.hidden_dim, module.intermediate_dim),
        }
        if self.quantized is not None and f"{prefix}.experts" in self.quantized[0].tensors:
            candidates = [f"{prefix}.gate_up_proj", f"{prefix}.down_proj"] + [
                f"{prefix}.{expert}.{projection}.weight"
                for expert in range(module.num_experts)
                for projection in ("gate_proj", "up_proj", "down_proj")
            ]
            names = {name: None for name in candidates if name in self.reader.tensors}
        elif f"{prefix}.gate_up_proj" in self.reader.tensors:
            names = {f"{prefix}.{suffix}": shape for suffix, shape in shapes.items()}
        else:
            names = {}
            for expert in range(module.num_experts):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    shape = (module.intermediate_dim, module.hidden_dim)
                    if projection == "down_proj":
                        shape = shape[::-1]
                    names[f"{prefix}.{expert}.{projection}.weight"] = shape
        for name, shape in names.items():
            if shape is None:
                continue
            if name not in self.reader.tensors or self.reader.shape(name) != shape:
                raise ValueError(f"Missing or incompatible expert tensor {name}; expected {shape}")
        if self.quantized is not None:
            self._check_quantized(prefix, module)
        self._register_slots(prefix, module)
        return set(names)

    def _quantized_shapes(self, module):
        """Per-expert shapes and dtypes of the quantized tensors, in pool order."""
        _, bits, group_size = self.quantized
        shapes = []
        for rows, columns in (
            (2 * module.intermediate_dim, module.hidden_dim),
            (module.hidden_dim, module.intermediate_dim),
        ):
            if columns % group_size or (columns * bits) % 32:
                raise ValueError(f"Input size {columns} does not fit {bits}-bit groups of {group_size}")
            shapes += [
                ((rows, columns * bits // 32), torch.uint32),
                ((rows, columns // group_size), self.dtype),
                ((rows, columns // group_size), self.dtype),
            ]
        return shapes

    def _check_quantized(self, prefix, module):
        reader, name = self.quantized[0], f"{prefix}.experts"
        size = sum(math.prod(shape) * dtype.itemsize for shape, dtype in self._quantized_shapes(module))
        if name not in reader.tensors or reader.shape(name) != (module.num_experts, size):
            raise ValueError(
                f"Missing or incompatible quantized experts {name}; expected {module.num_experts} x {size} bytes "
                "(quantized from a checkpoint in the execution dtype)"
            )

    def _register_slots(self, prefix, module):
        if self.quantized is not None:
            shapes = self._quantized_shapes(module)
        else:
            shapes = [
                ((2 * module.intermediate_dim, module.hidden_dim), self.dtype),
                ((module.hidden_dim, module.intermediate_dim), self.dtype),
            ]
        if self.pools is None:
            self._pool_shapes = shapes
            self.slot_bytes = sum(math.prod(shape) * dtype.itemsize for shape, dtype in shapes)
            self.num_slots = self.cache_bytes // self.slot_bytes
            if self.quantized is not None:
                self.slot_buffer = torch.empty(
                    (self.num_slots, self.slot_bytes), device=self.device, dtype=torch.uint8
                )
                self.pools = list(_split_expert(self.slot_buffer, shapes))
            else:
                self.pools = [
                    torch.empty((self.num_slots, *shape), device=self.device, dtype=dtype) for shape, dtype in shapes
                ]
            if self.quantized is None:
                self.gate_up_pool, self.down_pool = self.pools
            self.free_slots = list(range(self.num_slots - 1, -1, -1))
        elif [tuple(pool.shape[1:]) for pool in self.pools] != [shape for shape, _ in shapes]:
            raise ValueError("All MoE layers must have the same expert shapes to share the expert pool")
        self.slot_tables[prefix] = torch.full((module.num_experts,), self.num_slots, device=self.device)
        self.total_experts += module.num_experts

    def all_resident(self):
        """Whether every expert is cached, allowing routing to stay on the device."""
        return len(self.cache) == self.total_experts

    def _count(self, key):
        self.lookups += 1
        if self.policy != "lfu":
            return
        if self.decay_every and self.lookups % self.decay_every == 0:
            self.counts = {k: v / 2 for k, v in self.counts.items() if v >= 0.25}
            self.heap = []
            for resident in self.cache:
                self._push(resident)
        self.counts[key] = self.counts.get(key, 0) + 1
        if key in self.cache:
            self._push(key)

    def _push(self, key):
        self.sequence += 1
        heapq.heappush(self.heap, (self.counts.get(key, 0), self.sequence, key))

    def _least_used(self):
        while self.heap:
            count, _, key = self.heap[0]
            if key in self.cache and count == self.counts.get(key, 0):
                return key, count
            heapq.heappop(self.heap)
        return None, None

    def _evict(self, victim):
        entry = self.cache.pop(victim)
        self.resident_bytes -= entry[-1]
        self.evictions += 1
        self.free_slots.append(entry[0])
        self.slot_tables[victim[0]][victim[1]] = self.num_slots

    def _make_room(self, key, size, pinned=()):
        """Evict unpinned experts until size fits; return False if admission is declined."""
        if size > self.cache_bytes:
            return False
        if self.policy == "lfu":
            while self.cache and self.resident_bytes + size > self.cache_bytes:
                victim, count = self._least_used()
                if victim is None or victim in pinned or self.counts.get(key, 0) <= count:
                    self.not_admitted += 1
                    return False
                heapq.heappop(self.heap)
                self._evict(victim)
            return True
        while self.cache and self.resident_bytes + size > self.cache_bytes:
            victim = next((k for k in self.cache if k not in pinned), None)
            if victim is None:
                self.not_admitted += 1
                return False
            self._evict(victim)
        return True

    def _read_host(self, prefix, expert):
        """Read expert bytes or floating weights on the host; safe to run on I/O threads."""
        if self.quantized is not None:
            return (self.quantized[0].read(f"{prefix}.experts", expert),)
        return tuple(t.to(self.dtype) for t in self.reader.read_expert(prefix, expert))

    def prefetch(self, prefix, experts):
        """Start reading the experts of ``prefix`` that are neither resident nor already being read."""
        if self.executor is None:
            return
        for expert in experts:
            key = (prefix, expert)
            if key not in self.cache and key not in self.pending:
                self.pending[key] = self.executor.submit(self._read_host, prefix, expert)
                self.prefetched += 1

    def discard(self, prefix):
        """Drop the reads started for ``prefix`` that its forward call did not use (wrong predictions)."""
        for key in [key for key in self.pending if key[0] == prefix]:
            self.pending.pop(key).cancel()

    def _take_host(self, prefix, expert):
        """One expert in host memory: from a read already started by :meth:`prefetch`, else read now."""
        future = self.pending.pop((prefix, expert), None)
        if future is not None and not future.cancelled():
            self.prefetch_used += 1
            return future.result()
        return self._read_host(prefix, expert)

    # TODO(portability): share pinned host staging for asynchronous accelerator copies with _install.
    def _read(self, prefix, expert):
        """One expert on the device, for a single use."""
        tensors = self._take_host(prefix, expert)
        self.inflight.append(tensors)
        if self.quantized is not None:
            return _split_expert(tensors[0].to(self.device, non_blocking=True), self._pool_shapes)
        return tuple(t.to(self.device, non_blocking=True) for t in tensors)

    def release_host_buffers(self):
        """Called right after a host synchronization: every queued copy has read its host buffer."""
        # TODO(portability): validate copy completion and slot reuse on each backend before releasing buffers.
        self.inflight.clear()

    def install_ahead(self, prefix, experts):
        """Install prefetched experts while queued device work runs, without counting lookups."""
        for expert in experts:
            key = (prefix, expert)
            if key in self.cache or key not in self.pending:
                continue
            if self._install(key, self.slot_bytes) is not None:
                self.installed_ahead += 1

    def _install(self, key, size, pinned=()):
        if not self._make_room(key, size, pinned):
            return None
        slot = self.free_slots.pop()
        # Keep host buffers alive until the next host synchronization.
        tensors = self._take_host(*key)
        self.inflight.append(tensors)
        if self.quantized is not None:
            self.slot_buffer[slot].copy_(tensors[0], non_blocking=True)
        else:
            for pool, tensor in zip(self.pools, tensors):
                pool[slot].copy_(tensor, non_blocking=True)
        self.slot_tables[key[0]][key[1]] = slot
        self.cache[key] = (slot, size)
        self.resident_bytes += size
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.resident_bytes)
        if self.policy == "lfu":
            self._push(key)
        return slot

    def acquire(self, prefix, expert, pinned=()):
        """Make an expert resident in the pool; returns its slot, or None when it runs once without being cached."""
        key = (prefix, expert)
        self._count(key)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key][0]
        self.misses += 1
        return self._install(key, self.slot_bytes, pinned)

    def get(self, prefix, expert):
        """Return the device tensors of one expert (pool views when cached, else read for this use only)."""
        slot = self.acquire(prefix, expert)
        if slot is None:
            return self._read(prefix, expert)
        return tuple(pool[slot] for pool in self.pools)

    def _readers(self):
        return [self.reader] + ([self.quantized[0]] if self.quantized else [])

    def preload(self):
        """Load every expert now (reads in parallel), when the cache holds them all: no storage reads afterwards."""
        if self.num_slots < self.total_experts:
            raise ValueError(
                f"preload needs room for all {self.total_experts} experts; the cache has {self.num_slots} slots"
            )
        for prefix, table in self.slot_tables.items():
            experts = list(range(table.numel()))
            self.prefetch(prefix, experts)
            for expert in experts:
                self.acquire(prefix, expert)
            # TODO(portability): synchronize other accelerator backends before releasing host buffers.
            if self.device.type == "mps":
                torch.mps.synchronize()
            elif self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.release_host_buffers()
        self.hits = self.misses = 0

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "not_admitted": self.not_admitted,
            "policy": self.policy,
            "slots": self.num_slots,
            "prefetched": self.prefetched,
            "prefetch_used": self.prefetch_used,
            "installed_ahead": self.installed_ahead,
            "resident_bytes": self.resident_bytes,
            "peak_cache_bytes": self.peak_cache_bytes,
            "read_bytes": sum(reader.read_bytes for reader in self._readers()),
            "read_calls": sum(reader.read_calls for reader in self._readers()),
        }


class _PoolExperts:
    """Expose pool banks to MoE kernels; slot num_slots uses their expert-parallel sentinel handling."""

    def __init__(self, module, store):
        self.module = module
        self.num_experts = store.num_slots
        self.gate_up_proj = store.gate_up_pool
        self.down_proj = store.down_pool
        self.act_fn = module.act_fn
        self.has_gate = getattr(module, "has_gate", True)
        self.has_bias = getattr(module, "has_bias", False)
        self.is_transposed = getattr(module, "is_transposed", False)
        self.has_post_expert_norm = getattr(module, "has_post_expert_norm", False)
        self.post_expert_norm = getattr(module, "post_expert_norm", None)
        self._is_expert_parallel = True

    def _apply_gate(self, gate_up_out):
        return self.module._apply_gate(gate_up_out)


def _run_expert(module, result, hidden_states, top_k_weights, token, slot, gate_up, down):
    # Bound temporary activations during prefill.
    for start in range(0, token.numel(), module.offload_token_chunk_size):
        indices = token[start : start + module.offload_token_chunk_size]
        slots = slot[start : start + module.offload_token_chunk_size]
        activated = module._apply_gate(torch.nn.functional.linear(hidden_states[indices], gate_up))
        output = torch.nn.functional.linear(activated, down)
        output = output * top_k_weights[indices, slots, None]
        result.index_add_(0, indices, output.to(result.dtype))


def _split_expert(blob, shapes):
    """Views of one expert's bytes as its quantized tensors: gate_up (weight, scales, biases), then down."""
    # TODO(portability): validate uint32 views and strided packed banks on each accelerator backend.
    tensors, offset = [], 0
    for shape, dtype in shapes:
        size = math.prod(shape) * dtype.itemsize
        if offset % dtype.itemsize:
            raise ValueError("Misaligned quantized expert layout")
        tensors.append(blob[..., offset : offset + size].view(dtype).view(*blob.shape[:-1], *shape))
        offset += size
    return tuple(tensors)


def _load_gather_qmm(device):
    """MLX's quantized MoE matmul from the Hub (Metal) on MPS; elsewhere the slow PyTorch reference below."""
    if device.type == "mps":
        try:
            from .hub_kernels import get_kernel

            return get_kernel(_MLX_QUANTIZATION_KERNEL[0], version=_MLX_QUANTIZATION_KERNEL[1]).gather_qmm
        except Exception as error:  # the kernels package is optional
            logger.warning(f"Using the PyTorch reference for quantized experts; the Metal kernel failed: {error}")
    # TODO(portability): add optimized non-Metal kernels; retain the reference as a correctness fallback.
    return _gather_qmm_reference


def _gather_qmm_reference(x, w, scales, biases, rhs_indices, group_size, bits, **_):
    """`gather_qmm(x [S, 1, K], w [E, N, K * bits / 32], ..., rhs_indices [S], transpose=True)` -> [S, 1, N]."""
    from .metal_quantization import _affine_dequantize_tensor

    output = x.new_empty(*x.shape[:-1], w.shape[1])
    for expert in torch.unique(rhs_indices).tolist():
        rows = rhs_indices == expert
        weight = _affine_dequantize_tensor(w[expert], scales[expert], biases[expert], group_size, bits)
        output[rows] = x[rows] @ weight.to(x.dtype).T
    return output


def _quantized_mlp(module, store, x, banks, rhs, *, sorted_indices=False):
    """Apply the quantized expert selected by rhs for each input row."""
    _, bits, group_size = store.quantized

    def matmul(inputs, weight, scales, biases):
        return store.gather_qmm(
            inputs[:, None, :],
            weight,
            scales,
            biases,
            rhs_indices=rhs,
            transpose=True,
            group_size=group_size,
            bits=bits,
            sorted_indices=sorted_indices,
        ).squeeze(1)

    return matmul(module._apply_gate(matmul(x, *banks[:3])), *banks[3:])


def _sorted_kernel_applies(rows, experts):
    """Whether MLX selects its batched sorted kernel (at least 16 rows and 4 per expert)."""
    # TODO(portability): select this threshold and the float32 workaround by kernel/backend.
    return rows >= 16 and rows >= 4 * experts


def _quantized_sorted_forward(module, store, hidden_states, top_k_index, top_k_weights, experts, banks):
    """Sort routes for prefill. Use float32 to avoid incorrect fp16/bf16 results in the v2 M5 kernel."""
    num_tokens, top_k = top_k_index.shape
    lookup = torch.zeros(module.num_experts, dtype=torch.long, device=hidden_states.device)
    lookup[torch.tensor(experts, device=hidden_states.device)] = torch.arange(
        len(experts), device=hidden_states.device
    )
    rhs, order = torch.sort(lookup[top_k_index.reshape(-1)], stable=True)
    banks = [bank.float() if bank.is_floating_point() else bank for bank in banks]

    x = hidden_states.float()[order // top_k]
    output = _quantized_mlp(module, store, x, banks, rhs, sorted_indices=True)
    output = output * top_k_weights.reshape(-1)[order, None].float()
    routed = torch.zeros_like(output).index_copy_(0, order, output)
    return routed.view(num_tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


def _quantized_pool_forward(module, store, hidden_states, slots, top_k_weights):
    """Resident quantized experts through `gather_qmm`; non-resident routes (slot ``num_slots``) contribute zero."""
    num_tokens, top_k = slots.shape
    rhs = slots.reshape(-1)
    valid = rhs < store.num_slots
    weights = (top_k_weights.reshape(-1) * valid)[:, None]
    output = _quantized_mlp(module, store, hidden_states.repeat_interleave(top_k, dim=0), store.pools, rhs * valid)
    return (output * weights.to(output.dtype)).view(num_tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


def _pool_forward(module, store, hidden_states, top_k_index, top_k_weights):
    slots = store.slot_tables[module.offload_prefix][top_k_index]
    if store.quantized is not None:
        return _quantized_pool_forward(module, store, hidden_states, slots, top_k_weights)
    kernel = ExpertsInterface._global_mapping[store.compute]
    return kernel(_PoolExperts(module, store), hidden_states, slots, top_k_weights)


def experts_offload_forward(module, hidden_states, top_k_index, top_k_weights):
    """The ``experts_offload`` experts implementation: resident experts from the cache, missing ones from disk."""
    if module.training or torch.is_grad_enabled():
        raise RuntimeError("experts_offload is inference-only: use eval() and torch.no_grad() / inference_mode()")
    store = module.offload_store
    if hidden_states.device != store.device or hidden_states.dtype != store.dtype:
        raise ValueError(
            "Moving/casting a streamed model after loading is unsupported; reload on the target device/dtype"
        )
    prefix = module.offload_prefix
    if store.all_resident():
        store.hits += top_k_index.numel()
        return _pool_forward(module, store, hidden_states, top_k_index, top_k_weights)
    # Copy routing and next-layer predictions to the host together.
    next_layer = getattr(module, "offload_next", None)
    predict = next_layer is not None and store.executor is not None
    predict = predict and top_k_index.numel() <= module.offload_predict_max_routes
    after_next = getattr(module, "offload_after_next", None) if predict else None
    if predict:
        next_prefix, router = next_layer
        guesses = [torch.nn.functional.linear(hidden_states, router.weight).topk(router.top_k, dim=-1).indices]
        if after_next is not None:
            guesses.append(
                torch.nn.functional.linear(hidden_states, after_next[1].weight).topk(router.top_k, dim=-1).indices
            )
        routes = torch.cat([top_k_index.reshape(-1)] + [guess.reshape(-1) for guess in guesses]).cpu()
        routing = routes[: top_k_index.numel()].view(top_k_index.shape)
    else:
        routing = top_k_index.cpu()
    # The routing copy also waits for earlier host-to-device copies.
    store.release_host_buffers()
    experts = torch.unique(routing).tolist()
    if experts and not (experts[0] >= 0 and experts[-1] < module.num_experts):
        raise ValueError("Expert parallelism and sentinel expert IDs are unsupported")
    store.prefetch(prefix, experts[: store.window])
    width = top_k_index.shape[0] * (next_layer[1].top_k if predict else 0)
    guessed = torch.unique(routes[top_k_index.numel() : top_k_index.numel() + width]).tolist() if predict else []
    if predict:
        store.prefetch(next_prefix, guessed)
    if after_next is not None:
        store.prefetch(after_next[0], torch.unique(routes[top_k_index.numel() + width :]).tolist())
    try:
        result = _offloaded_experts(module, store, prefix, hidden_states, top_k_index, top_k_weights, routing, experts)
    finally:
        store.discard(prefix)
    if guessed:
        store.install_ahead(next_prefix, guessed)
    return result


def _offloaded_experts(module, store, prefix, hidden_states, top_k_index, top_k_weights, routing, experts):
    pinned, uncached = set(), []
    for position, expert in enumerate(experts):
        store.prefetch(prefix, experts[position + 1 : position + 1 + store.window])
        if store.acquire(prefix, expert, pinned) is None:
            uncached.append(expert)
        else:
            pinned.add((prefix, expert))
    if store.quantized is not None and _sorted_kernel_applies(top_k_index.numel(), len(experts)):
        per_expert = []
        for expert in experts:
            if (prefix, expert) in pinned:
                slot = store.cache[(prefix, expert)][0]
                per_expert.append([pool[slot] for pool in store.pools])
            else:
                per_expert.append(store._read(prefix, expert))
        banks = [torch.stack(parts) for parts in zip(*per_expert)]
        return _quantized_sorted_forward(module, store, hidden_states, top_k_index, top_k_weights, experts, banks)
    if pinned:
        result = _pool_forward(module, store, hidden_states, top_k_index, top_k_weights)
    else:
        result = torch.zeros_like(hidden_states)
    for position, expert in enumerate(uncached):
        store.prefetch(prefix, uncached[position + 1 : position + 1 + store.window])
        token, slot = (t.to(hidden_states.device) for t in torch.where(routing == expert))
        tensors = store._read(prefix, expert)
        if store.quantized is not None:
            banks = [tensor[None] for tensor in tensors]
            rhs = torch.zeros_like(token)
            output = _quantized_mlp(module, store, hidden_states[token], banks, rhs) * top_k_weights[token, slot, None]
            result.index_add_(0, token, output.to(result.dtype))
        else:
            _run_expert(module, result, hidden_states, top_k_weights, token, slot, *tensors)
    return result


def _checkpoint_prefix(reader, name, quantized_reader=None):
    """Find a unique packed, per-expert, or quantized checkpoint prefix, allowing a different model root."""
    suffix = name.split(".", 1)[1] if name.startswith("model.") else name
    found = set()
    for key in reader.tensors:
        if key.endswith(f"{suffix}.gate_up_proj"):
            found.add(key.removesuffix(".gate_up_proj"))
        elif key.endswith(f"{suffix}.0.gate_proj.weight"):
            found.add(key.removesuffix(".0.gate_proj.weight"))
    if not found and quantized_reader is not None:
        found = {key.removesuffix(".experts") for key in quantized_reader.tensors if key.endswith(f"{suffix}.experts")}
    if len(found) != 1:
        raise ValueError(f"Cannot find the checkpoint experts of {name} (candidates: {sorted(found)})")
    return found.pop()


_OPTIONS = {
    "cache_bytes",
    "quantized_experts",
    "cache_policy",
    "io_threads",
    "prefetch_depth",
    "preload",
    "bypass_page_cache",
    "experts_implementation",
    "token_chunk_size",
    "predict_max_routes",
}


def prepare_experts_offload(model, checkpoint_files, *, dtype, device, options):
    """Replace meta expert banks with a shared disk-backed cache and return checkpoint keys to skip.

    Options:
        cache_bytes (`int`, required): device memory for cached experts.
        quantized_experts (`str` or `os.PathLike`, *optional*): a directory written by :func:`quantize_experts`; the
            experts are then read and cached quantized and run by `gather_qmm`. The checkpoint may omit its own experts.
        cache_policy (`str`, defaults to `"lru"`): `"lru"` or `"lfu"`.
        io_threads (`int`, defaults to 4): threads reading experts; 0 reads synchronously and disables prefetch.
        prefetch_depth (`int`, defaults to 2): how many layers ahead to guess and read experts (0 disables).
        preload (`bool`, defaults to `False`): read every expert at load time; the cache must hold them all.
        bypass_page_cache (`bool`, defaults to `False`): keep streamed bytes out of the OS page cache.
        experts_implementation (`str`, defaults to `"grouped_mm"`): kernel for cached BF16 experts, or `"batched_mm"`.
        token_chunk_size (`int`, defaults to 256): row chunk for experts that run uncached.
        predict_max_routes (`int`, defaults to 64): prefetch only for calls with at most this many routes (decode).
    """
    options = dict(options)
    unknown = set(options) - _OPTIONS
    if unknown or "cache_bytes" not in options:
        raise ValueError(f"experts_offload needs `cache_bytes`; unknown options: {sorted(unknown)}")
    if not checkpoint_files or not all(str(f).endswith(".safetensors") for f in checkpoint_files):
        raise ValueError("experts_offload reads experts from safetensors checkpoints only")
    if dtype not in _DTYPES.values():
        raise ValueError("experts_offload needs a floating dtype (float32/float16/bfloat16)")
    if options.get("experts_implementation", "grouped_mm") not in ("grouped_mm", "batched_mm"):
        raise ValueError("experts_offload supports experts_implementation 'grouped_mm' or 'batched_mm'")
    ExpertsInterface.register("experts_offload", experts_offload_forward)
    device = torch.empty(0, device=device).device
    reader = SafetensorsReader(Path(checkpoint_files[0]).parent, options.get("bypass_page_cache", False))
    store = ExpertStore(
        reader,
        options["cache_bytes"],
        device,
        dtype,
        policy=options.get("cache_policy", "lru"),
        io_threads=options.get("io_threads", 4),
        quantized=options.get("quantized_experts"),
    )
    store.compute = options.get("experts_implementation", "grouped_mm")
    experts = [
        (name, module)
        for name, module in model.named_modules()
        if hasattr(module, "_apply_gate") and isinstance(getattr(module, "gate_up_proj", None), torch.nn.Parameter)
    ]
    if not experts:
        raise ValueError(f"{type(model).__name__} has no experts module that can be offloaded")
    consumed = set()
    for name, module in experts:
        if module.has_bias or module.is_transposed or not module.is_concatenated or module.has_post_expert_norm:
            raise ValueError(
                f"experts_offload supports gated, concatenated, untransposed experts without bias: {name}"
            )
        prefix = _checkpoint_prefix(reader, name, store.quantized[0] if store.quantized else None)
        consumed.update(store.register(prefix, module))
        module.register_parameter("gate_up_proj", None)
        module.register_parameter("down_proj", None)
        module._is_hf_initialized = True  # Banks remain in the checkpoint.
        module.config._experts_implementation = "experts_offload"
        module.offload_store = store
        module.offload_prefix = prefix
        module.offload_token_chunk_size = options.get("token_chunk_size", 256)
        module.offload_predict_max_routes = options.get("predict_max_routes", 64)
        module.offload_next = module.offload_after_next = None
    depth = options.get("prefetch_depth", 2)
    routers = []
    for name, module in experts:
        router = getattr(model.get_submodule(name.rpartition(".")[0]), "gate", None)
        usable = router is not None and hasattr(router, "weight") and hasattr(router, "top_k")
        routers.append((module.offload_prefix, router) if usable else None)
    for index, (_, module) in enumerate(experts):
        if depth >= 1 and index + 1 < len(routers):
            module.offload_next = routers[index + 1]
        if depth >= 2 and index + 2 < len(routers):
            module.offload_after_next = routers[index + 2]
    if options.get("preload", False):
        store.preload()
    model.experts_offload_store = store
    return consumed


def quantize_experts(checkpoint, output, *, bits=4, group_size=64, device="cpu"):
    """Write MLX affine experts as uint8 rows: gate_up then down, each with weights, scales and biases.

    Uses the Metal kernel on MPS, otherwise PyTorch. Peak memory is one experts module."""
    from safetensors.torch import save_file

    from .metal_quantization import _affine_quantize_tensor

    device = torch.device(device)
    quantize = lambda weight: _affine_quantize_tensor(weight, group_size, bits)  # noqa: E731
    quantizer = "pytorch-reference"
    if device.type == "mps":
        from .hub_kernels import get_kernel

        kernel = get_kernel(_MLX_QUANTIZATION_KERNEL[0], version=_MLX_QUANTIZATION_KERNEL[1])
        quantize = lambda weight: kernel.quantize(weight, group_size=group_size, bits=bits)  # noqa: E731
        quantizer = "mlx-quantization-metal-kernels"
    reader = SafetensorsReader(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    packed = sorted(name.removesuffix(".gate_up_proj") for name in reader.tensors if name.endswith(".gate_up_proj"))
    per_expert = sorted(
        name.removesuffix(".0.gate_proj.weight") for name in reader.tensors if name.endswith(".0.gate_proj.weight")
    )
    weight_map = {}
    for index, prefix in enumerate(packed + per_expert):
        if prefix in packed:
            experts = reader.shape(f"{prefix}.gate_up_proj")[0]
        else:
            experts = sum(
                1 for name in reader.tensors if re.fullmatch(rf"{re.escape(prefix)}\.\d+\.gate_proj\.weight", name)
            )

        blobs = []
        for expert in range(experts):
            parts = []
            for weight in reader.read_expert(prefix, expert):
                weight_q, scales, biases = quantize(weight.to(device))
                parts += [weight_q, scales.to(weight.dtype), biases.to(weight.dtype)]
            blobs.append(torch.cat([part.cpu().contiguous().view(torch.uint8).reshape(-1) for part in parts]))
        filename = f"experts-{index:05d}.safetensors"
        save_file({f"{prefix}.experts": torch.stack(blobs)}, output / filename, {"format": "pt"})
        weight_map[f"{prefix}.experts"] = filename
        logger.info(f"quantized {prefix}: {experts} experts")
    (output / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    info = {"format": _QUANTIZED_FORMAT, "bits": bits, "group_size": group_size, "quantizer": quantizer}
    (output / "quantization.json").write_text(json.dumps(info, indent=2))
    return output
