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


def _read_exact(path, offset, size, bypass_page_cache, what, out=None, fd=None):
    """Read a byte range into ``out`` (or a new buffer), optionally bypassing or dropping the OS page cache.

    ``fd`` is an open descriptor of ``path``; positional reads are thread-safe."""
    # TODO(portability): implement Windows cache bypass; report unsupported OS/filesystem combinations.
    data = bytearray(size) if out is None else out
    own = fd is None
    if own:
        fd = os.open(path, os.O_RDONLY)
        if bypass_page_cache and hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
    try:
        view = memoryview(data).cast("B")
        done = 0
        while done < size:
            count = os.preadv(fd, [view[done:size]], offset + done)
            if not count:
                raise OSError(f"Short checkpoint read: {path}, {what}")
            done += count
        # TODO(portability): distinguish best-effort cache eviction from uncached reads.
        if bypass_page_cache and hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
    finally:
        if own:
            os.close(fd)
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
        self._fds = {}
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

    def read(self, name, expert=None, out=None):
        """Read a tensor, or one expert slice of it, into ``out`` (a CPU uint8 tensor of its size) when given."""
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
        if out is not None:
            if out.dtype != torch.uint8 or out.numel() != size or out.device.type != "cpu":
                raise ValueError(f"`out` must be a CPU uint8 tensor of {size} bytes")
            _read_exact(path, offset, size, self.bypass_page_cache, f"tensor {name}", out.numpy(), self._fd(path))
            data = out
        else:
            data = torch.frombuffer(
                _read_exact(path, offset, size, self.bypass_page_cache, f"tensor {name}", fd=self._fd(path)),
                dtype=torch.uint8,
            )
        with _COUNTER_LOCK:
            self.read_bytes += size
            self.read_calls += 1
        return data.view(dtype).reshape(shape)

    def _fd(self, path):
        """One read-only descriptor per shard, opened on first use and shared by the I/O threads."""
        fd = self._fds.get(path)
        if fd is None:
            with _COUNTER_LOCK:
                fd = self._fds.get(path)
                if fd is None:
                    fd = os.open(path, os.O_RDONLY)
                    if self.bypass_page_cache and hasattr(fcntl, "F_NOCACHE"):
                        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
                    self._fds[path] = fd
        return fd

    def __del__(self):
        for fd in getattr(self, "_fds", {}).values():
            try:
                os.close(fd)
            except OSError:
                pass

    def read_expert(self, prefix, expert):
        if f"{prefix}.gate_up_proj" in self.tensors:
            return self.read(f"{prefix}.gate_up_proj", expert), self.read(f"{prefix}.down_proj", expert)
        gate_up = torch.cat([self.read(f"{prefix}.{expert}.{p}_proj.weight") for p in ("gate", "up")])
        return gate_up, self.read(f"{prefix}.{expert}.down_proj.weight")


def _layout_bytes(layout):
    return sum(math.prod(shape) * dtype.itemsize for shape, dtype in layout)


class _Pool:
    """Device slots shared by the MoE layers whose experts have one layout; LRU (or LFU) within the pool."""

    def __init__(self, layout, num_slots, packed):
        self.layout = layout  # [(shape, dtype)] of one expert's tensors, in storage order
        self.slot_bytes = _layout_bytes(layout)
        self.num_slots = num_slots
        self.packed = packed
        self.slot_buffer = self.banks = None  # see `materialize`
        self.free_slots = list(range(num_slots - 1, -1, -1))
        self.cache = OrderedDict()  # key -> slot, least recently used first
        self.heap = []  # LFU: (count, sequence, key); stale entries are discarded lazily
        self.num_experts = 0  # over the pool's layers

    def materialize(self, device):
        if self.banks is not None:
            return
        if self.packed:
            # One row of bytes per slot, viewed as the expert's tensors: one copy per install.
            self.slot_buffer = torch.empty((self.num_slots, self.slot_bytes), device=device, dtype=torch.uint8)
            self.banks = list(_split_expert(self.slot_buffer, self.layout))
        else:
            self.banks = [
                torch.empty((self.num_slots, *shape), device=device, dtype=dtype) for shape, dtype in self.layout
            ]


class ExpertStore:
    """Device pools of expert slots with LRU or decaying LFU admission, for one inference request at a time.

    Layers whose experts share a layout share a pool; with several layouts (mixed quantization, an external expert
    source), the budget gives every layer the same number of slots. Staging and uncached experts can exceed the budget.

    Experts come from the checkpoint (floating point), from `quantized_experts` (MLX affine bytes), or from an
    ``expert_source``: an object with ``layout(prefix)`` (``[(shape, dtype)]`` of one expert's tensors, stored
    contiguously in that order), ``read(prefix, expert, out)`` (fill ``out``, a CPU uint8 tensor of the expert's size)
    and ``read_bytes`` / ``read_calls`` counters; optionally ``prefix(module_name)`` (its key for an experts module)
    and ``pool_key(prefix)`` (layers with equal keys share a pool; defaults to the layout)."""

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
        source=None,
    ):
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        if policy not in ("lru", "lfu"):
            raise ValueError(f"Unknown cache policy {policy!r}; use 'lru' or 'lfu'")
        if source is not None and quantized is not None:
            raise ValueError("Use either quantized_experts or expert_source, not both")
        self.policy = policy
        self.decay_every = decay_every
        self.counts = {}
        self.sequence = 0
        self.lookups = 0
        self.not_admitted = 0
        self.reader = reader
        self.source = source
        self.cache_bytes = cache_bytes
        self.device = torch.device(device)
        self.dtype = dtype
        self.resident_bytes = 0
        self.peak_cache_bytes = 0
        self.hits = self.misses = self.evictions = 0
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
        # Layers register first (`register`), then `allocate` sizes the pools from the whole set of layouts.
        self.layouts = {}  # prefix -> (pool key, layout, num_experts)
        self.pools = []
        self.pool_of = {}
        # expert -> slot per layer (the pool's num_slots when absent), kept on the host (installs and evictions are host
        # bookkeeping) and copied to the device only when a device-side lookup needs it.
        self.slot_tables = {}
        self.host_slot_tables = {}
        self.stale_slot_tables = set()
        self.total_experts = 0
        self.compute = "grouped_mm"
        self.expert_compute = None  # see `experts_offload_forward`
        self.executor = ThreadPoolExecutor(io_threads, thread_name_prefix="experts-offload-io") if io_threads else None
        self.pending = {}
        self.window = 2 * io_threads
        self.prefetched = self.prefetch_used = self.installed_ahead = 0
        # Long prefill calls (`cold_prefill_routes`): hits do not refresh, and the experts installed for a layer move to
        # the cold end of its pool once that layer's call is done, so one long prompt does not flush the decode set.
        self.cold = False
        self.cold_installed = {}
        # Retain host buffers until the next routing copy synchronizes the device.
        self.inflight = []
        # Recycled host buffers for expert bytes, all of the largest expert size, see `_host_buffer`.
        self.free_host_buffers = []
        self.host_buffer_bytes = 0
        self.commit = False  # see `_commit_mps`
        # Optional host tier (expert bytes, one layout): a pool of expert rows in host memory between storage and the
        # device cache, see `allocate_host_cache`. `host_busy` holds the rows handed out and not yet released.
        self.host_pool = None
        self.host_slots = 0
        self.host_cache = OrderedDict()
        self.host_free = []
        self.host_busy = set()
        self.host_lock = threading.Lock()
        self.host_hits = self.host_misses = 0

    def register(self, prefix, module):
        """Validate one experts module's tensors and record its layout; returns the checkpoint keys it replaces."""
        if self.pools:
            raise ValueError("Register every experts module before `allocate`")
        shapes = {
            "gate_up_proj": (module.num_experts, 2 * module.intermediate_dim, module.hidden_dim),
            "down_proj": (module.num_experts, module.hidden_dim, module.intermediate_dim),
        }
        per_expert = [
            f"{prefix}.{expert}.{projection}.weight"
            for expert in range(module.num_experts)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ]
        if self.source is not None or (
            self.quantized is not None and f"{prefix}.experts" in self.quantized[0].tensors
        ):
            # Experts come from elsewhere: skip whatever the checkpoint holds for them.
            candidates = [f"{prefix}.{suffix}" for suffix in shapes] + per_expert
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
        if self.source is not None:
            layout = [(tuple(shape), dtype) for shape, dtype in self.source.layout(prefix)]
            key = self.source.pool_key(prefix) if hasattr(self.source, "pool_key") else None
        elif self.quantized is not None:
            self._check_quantized(prefix, module)
            layout, key = self._quantized_shapes(module), None
        else:
            layout = [
                ((2 * module.intermediate_dim, module.hidden_dim), self.dtype),
                ((module.hidden_dim, module.intermediate_dim), self.dtype),
            ]
            key = None
        self.layouts[prefix] = (key if key is not None else tuple(layout), layout, module.num_experts)
        self.total_experts += module.num_experts
        return set(names)

    def allocate(self, materialize=True):
        """Size the device pools: one per layout, each layer getting the same number of slots. Their memory is taken
        now, or with ``materialize=False`` at the first lookup (after loading, so the loading peak excludes them)."""
        groups = {}
        for prefix, (key, layout, _) in self.layouts.items():
            groups.setdefault(key, (layout, []))[1].append(prefix)
        layer_bytes = [_layout_bytes(layout) for _, layout, _ in self.layouts.values()]
        if not groups:
            raise ValueError("No experts module registered")
        if len(groups) == 1:
            sizes = [self.cache_bytes // layer_bytes[0]]
        else:
            per_layer = self.cache_bytes // sum(layer_bytes)
            sizes = [per_layer * len(prefixes) for _, prefixes in groups.values()]
        packed = self.quantized is not None
        for (layout, prefixes), num_slots in zip(groups.values(), sizes):
            pool = _Pool(layout, num_slots, packed)
            self.pools.append(pool)
            for prefix in prefixes:
                self.pool_of[prefix] = pool
                experts = self.layouts[prefix][2]
                pool.num_experts += experts
                self.slot_tables[prefix] = torch.full((experts,), num_slots, device=self.device)
                self.host_slot_tables[prefix] = torch.full((experts,), num_slots)
        self.host_buffer_bytes = max(pool.slot_bytes for pool in self.pools)
        self.materialized = False
        if materialize:
            self.materialize()

    def materialize(self):
        """Take the pools' device memory (once)."""
        if not self.materialized:
            for pool in self.pools:
                pool.materialize(self.device)
            self.materialized = True

    # Single-pool views, kept for callers that predate size classes.
    @property
    def num_slots(self):
        return sum(pool.num_slots for pool in self.pools)

    @property
    def slot_bytes(self):
        if len(self.pools) != 1:
            raise ValueError("slot_bytes is defined for a single pool; see `pools`")
        return self.pools[0].slot_bytes

    @property
    def cache(self):
        """Every resident key -> slot (read-only view over the pools)."""
        return {key: slot for pool in self.pools for key, slot in pool.cache.items()}

    @property
    def gate_up_pool(self):
        return self.pools[0].banks[0] if len(self.pools) == 1 and self.quantized is None else None

    @property
    def down_pool(self):
        return self.pools[0].banks[1] if len(self.pools) == 1 and self.quantized is None else None

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

    def all_resident(self):
        """Whether every expert is cached, allowing routing to stay on the device."""
        return sum(len(pool.cache) for pool in self.pools) == self.total_experts

    def _count(self, key):
        self.lookups += 1
        if self.policy != "lfu":
            return
        if self.decay_every and self.lookups % self.decay_every == 0:
            self.counts = {k: v / 2 for k, v in self.counts.items() if v >= 0.25}
            for pool in self.pools:
                pool.heap = []
                for resident in pool.cache:
                    self._push(resident)
        self.counts[key] = self.counts.get(key, 0) + 1
        if key in self.pool_of[key[0]].cache:
            self._push(key)

    def _push(self, key):
        self.sequence += 1
        heapq.heappush(self.pool_of[key[0]].heap, (self.counts.get(key, 0), self.sequence, key))

    def _least_used(self, pool):
        while pool.heap:
            count, _, key = pool.heap[0]
            if key in pool.cache and count == self.counts.get(key, 0):
                return key, count
            heapq.heappop(pool.heap)
        return None, None

    def _evict(self, pool, victim):
        slot = pool.cache.pop(victim)
        self.resident_bytes -= pool.slot_bytes
        self.evictions += 1
        pool.free_slots.append(slot)
        self.host_slot_tables[victim[0]][victim[1]] = pool.num_slots
        self.stale_slot_tables.add(victim[0])

    def _make_room(self, key, pinned=()):
        """Evict unpinned experts of the key's pool until a slot is free; return False if admission is declined."""
        pool = self.pool_of[key[0]]
        if not pool.num_slots:
            return False
        if self.policy == "lfu":
            while not pool.free_slots:
                victim, count = self._least_used(pool)
                if victim is None or victim in pinned or self.counts.get(key, 0) <= count:
                    self.not_admitted += 1
                    return False
                heapq.heappop(pool.heap)
                self._evict(pool, victim)
            return True
        while not pool.free_slots:
            victim = next((k for k in pool.cache if k not in pinned), None)
            if victim is None:
                self.not_admitted += 1
                return False
            self._evict(pool, victim)
        return True

    def demote(self, prefix):
        """Move the experts installed for ``prefix`` during a cold call to the cold end of their pool."""
        cache = self.pool_of[prefix].cache
        for key in self.cold_installed.pop(prefix, ()):
            if key in cache:
                cache.move_to_end(key, last=False)

    def allocate_host_cache(self, host_cache_bytes):
        """Keep up to ``host_cache_bytes`` of expert bytes in (pinned) host memory, between storage and the device
        cache: a device-cache miss then costs one host-to-device copy instead of a storage read. Worth it when host
        memory is larger than device memory (discrete GPUs); on unified memory the device cache already is host RAM."""
        if self.quantized is None and self.source is None:
            raise ValueError("host_cache_bytes needs quantized_experts or an expert_source")
        if len(self.pools) != 1:
            raise ValueError("host_cache_bytes needs experts of a single layout")
        self.host_slots = host_cache_bytes // self.slot_bytes
        if not self.host_slots:
            return
        self.host_pool = torch.empty(
            (self.host_slots, self.slot_bytes), dtype=torch.uint8, pin_memory=self.device.type != "cpu"
        )
        self.host_free = list(range(self.host_slots))

    def _host_row(self, tensor):
        """The host-pool row ``tensor`` is, or None."""
        if self.host_pool is None:
            return None
        offset = tensor.data_ptr() - self.host_pool.data_ptr()
        if 0 <= offset < self.host_pool.numel() and offset % self.slot_bytes == 0:
            return offset // self.slot_bytes
        return None

    def _release_host_row(self, tensor):
        row = self._host_row(tensor)
        if row is not None:
            with self.host_lock:
                self.host_busy.discard(row)
        return row is not None

    def _read_bytes(self, prefix, expert, out):
        """Read one expert's bytes into ``out`` (a CPU uint8 tensor of its pool's slot size)."""
        if self.source is not None:
            self.source.read(prefix, expert, out)
        else:
            self.quantized[0].read(f"{prefix}.experts", expert, out=out)
        return out

    def _read_host_cached(self, prefix, expert):
        """An expert through the host tier: a hit hands out its row, a miss reads into a free (or the least recently
        used, not busy) row. Returns None when every row is busy. Safe on I/O threads."""
        key = (prefix, expert)
        with self.host_lock:
            row = self.host_cache.get(key)
            if row is not None:
                self.host_cache.move_to_end(key)
                self.host_busy.add(row)
                self.host_hits += 1
                return (self.host_pool[row],)
            if self.host_free:
                row = self.host_free.pop()
            else:
                victim = next((k for k, r in self.host_cache.items() if r not in self.host_busy), None)
                if victim is None:
                    return None
                row = self.host_cache.pop(victim)
            self.host_busy.add(row)
            self.host_misses += 1
        try:
            self._read_bytes(prefix, expert, self.host_pool[row])
        except BaseException:
            with self.host_lock:
                self.host_busy.discard(row)
                self.host_free.append(row)
            raise
        with self.host_lock:
            self.host_cache[key] = row
        return (self.host_pool[row],)

    def preload_host(self):
        """Read every expert into the host tier now (in parallel when there are I/O threads)."""
        keys = [(prefix, expert) for prefix, table in self.host_slot_tables.items() for expert in range(table.numel())]
        if self.host_slots < len(keys):
            raise ValueError(
                f"host preload needs room for all {len(keys)} experts; the host tier has {self.host_slots}"
            )
        if self.executor is not None:
            results = [f.result() for f in [self.executor.submit(self._read_host_cached, *key) for key in keys]]
        else:
            results = [self._read_host_cached(*key) for key in keys]
        for tensors in results:
            self._release_host_row(tensors[0])
        self.host_hits = self.host_misses = 0

    def _read_host(self, prefix, expert):
        """Read expert bytes or floating weights on the host; safe to run on I/O threads."""
        if self.host_pool is not None:
            tensors = self._read_host_cached(prefix, expert)
            if tensors is not None:
                return tensors
        if self.quantized is not None or self.source is not None:
            # Straight into a recycled pinned buffer: copying pinned memory to the device costs the host ~5x less
            # than pageable memory (0.03 vs 0.14 ms for a 2 MB expert on an M5).
            return (self._read_bytes(prefix, expert, self._host_buffer()[: self.pool_of[prefix].slot_bytes]),)
        return tuple(t.to(self.dtype) for t in self.reader.read_expert(prefix, expert))

    def _host_buffer(self):
        """A host buffer for one expert (of the largest size, so one size is recycled), pinned when the device can
        use it; called from I/O threads."""
        try:
            return self.free_host_buffers.pop()
        except IndexError:
            return torch.empty(self.host_buffer_bytes, dtype=torch.uint8, pin_memory=self.device.type != "cpu")

    def prefetch(self, prefix, experts):
        """Start reading the experts of ``prefix`` that are neither resident nor already being read."""
        if self.executor is None:
            return
        cache = self.pool_of[prefix].cache
        for expert in experts:
            key = (prefix, expert)
            if key not in cache and key not in self.pending:
                self.pending[key] = self.executor.submit(self._read_host, prefix, expert)
                self.prefetched += 1

    def discard(self, prefix):
        """Drop the reads started for ``prefix`` that its forward call did not use (wrong predictions)."""
        for key in [key for key in self.pending if key[0] == prefix]:
            future = self.pending.pop(key)
            if not future.cancel() and self.host_pool is not None:
                # Already read (or reading): its host-tier row was handed out and will not be copied.
                future.add_done_callback(self._release_unused)

    def _release_unused(self, future):
        if not future.cancelled() and future.exception() is None:
            self._release_host_row(future.result()[0])

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
        if self.quantized is not None or self.source is not None:
            return _split_expert(tensors[0].to(self.device, non_blocking=True), self.pool_of[prefix].layout)
        return tuple(t.to(self.device, non_blocking=True) for t in tensors)

    def release_host_buffers(self):
        """Called right after a host synchronization: every queued copy has read its host buffer."""
        # TODO(portability): validate copy completion and slot reuse on each backend before releasing buffers.
        if self.quantized is not None or self.source is not None:
            # list.append / pop are atomic, so I/O threads can take buffers back while this runs.
            for tensors in self.inflight:
                if self._release_host_row(tensors[0]):
                    continue  # a host-tier row: it stays cached, now free to be evicted
                buffer = tensors[0] if tensors[0]._base is None else tensors[0]._base
                if buffer.dtype == torch.uint8 and buffer.numel() == self.host_buffer_bytes:
                    self.free_host_buffers.append(buffer)
        self.inflight.clear()

    def synchronize(self):
        """Wait for queued device work, then release the host buffers of its copies."""
        # TODO(portability): synchronize other accelerator backends.
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.release_host_buffers()

    def device_slot_table(self, prefix):
        """The layer's expert -> slot table on the device, refreshed from the host copy when it changed."""
        if prefix in self.stale_slot_tables:
            snapshot = self.host_slot_tables[prefix].clone()
            self.inflight.append((snapshot,))  # read by a queued non-blocking copy
            self.slot_tables[prefix].copy_(snapshot, non_blocking=True)
            self.stale_slot_tables.discard(prefix)
        return self.slot_tables[prefix]

    def device_slots(self, prefix, routing):
        """Slots of host-side ``routing`` as a device tensor: one small copy instead of a device lookup."""
        slots = self.host_slot_tables[prefix][routing]
        self.inflight.append((slots,))
        return slots.to(self.device, non_blocking=True)

    def install_ahead(self, prefix, experts):
        """Install prefetched experts while queued device work runs, without counting lookups."""
        if not self.materialized:
            self.materialize()
        cache = self.pool_of[prefix].cache
        for expert in experts:
            key = (prefix, expert)
            # Only finished reads: waiting here would delay the next layer's launches for guesses that may go unused.
            if key in cache or key not in self.pending or not self.pending[key].done():
                continue
            if self._install(key) is None:
                break  # the pool is full of pinned experts
            self.installed_ahead += 1

    def _install(self, key, pinned=()):
        if not self._make_room(key, pinned):
            return None
        pool = self.pool_of[key[0]]
        # Keep host buffers alive until the next host synchronization.
        tensors = self._take_host(*key)
        slot = pool.free_slots.pop()
        self.inflight.append(tensors)
        if pool.slot_buffer is not None:
            pool.slot_buffer[slot].copy_(tensors[0], non_blocking=True)
        elif self.source is not None:
            offset = 0
            for bank in pool.banks:
                size = bank[slot].numel() * bank.element_size()
                bank[slot].view(-1).view(torch.uint8).copy_(tensors[0][offset : offset + size], non_blocking=True)
                offset += size
        else:
            for bank, tensor in zip(pool.banks, tensors):
                bank[slot].copy_(tensor, non_blocking=True)
        self.host_slot_tables[key[0]][key[1]] = slot
        self.stale_slot_tables.add(key[0])
        pool.cache[key] = slot
        if self.cold:
            self.cold_installed.setdefault(key[0], []).append(key)
        self.resident_bytes += pool.slot_bytes
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.resident_bytes)
        if self.policy == "lfu":
            self._push(key)
        return slot

    def acquire(self, prefix, expert, pinned=()):
        """Make an expert resident in its pool; returns its slot, or None when it runs once without being cached."""
        if not self.materialized:
            self.materialize()
        key = (prefix, expert)
        self._count(key)
        cache = self.pool_of[prefix].cache
        if key in cache:
            self.hits += 1
            if not self.cold:
                cache.move_to_end(key)
            return cache[key]
        self.misses += 1
        return self._install(key, pinned)

    def expert_tensors(self, prefix, expert):
        """The device tensors of one expert without counting a lookup: pool views when resident, else read for this use."""
        pool = self.pool_of[prefix]
        slot = pool.cache.get((prefix, expert))
        if slot is None:
            return self._read(prefix, expert)
        return tuple(bank[slot] for bank in pool.banks)

    def get(self, prefix, expert):
        """Return the device tensors of one expert (pool views when cached, else read for this use only)."""
        slot = self.acquire(prefix, expert)
        if slot is None:
            return self._read(prefix, expert)
        return tuple(bank[slot] for bank in self.pool_of[prefix].banks)

    def _readers(self):
        return [self.reader] + ([self.quantized[0]] if self.quantized else []) + ([self.source] if self.source else [])

    def preload(self):
        """Load every expert now (reads in parallel), when the cache holds them all: no storage reads afterwards."""
        for pool in self.pools:
            if pool.num_slots < pool.num_experts:
                raise ValueError(
                    f"preload needs room for all {pool.num_experts} experts of a pool; it has {pool.num_slots} slots"
                )
        for prefix, table in self.host_slot_tables.items():
            experts = list(range(table.numel()))
            self.prefetch(prefix, experts)
            for expert in experts:
                self.acquire(prefix, expert)
            self.synchronize()
        self.hits = self.misses = 0

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "not_admitted": self.not_admitted,
            "policy": self.policy,
            "slots": self.num_slots,
            "pools": len(self.pools),
            "prefetched": self.prefetched,
            "prefetch_used": self.prefetch_used,
            "installed_ahead": self.installed_ahead,
            "resident_bytes": self.resident_bytes,
            "peak_cache_bytes": self.peak_cache_bytes,
            "read_bytes": sum(reader.read_bytes for reader in self._readers()),
            "read_calls": sum(reader.read_calls for reader in self._readers()),
            "host_slots": self.host_slots,
            "host_hits": self.host_hits,
            "host_misses": self.host_misses,
        }


class _PoolExperts:
    """Expose a pool's banks to MoE kernels; slot num_slots uses their expert-parallel sentinel handling."""

    def __init__(self, module, pool):
        self.module = module
        self.num_experts = pool.num_slots
        self.gate_up_proj, self.down_proj = pool.banks
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


# Experts per sorted `gather_qmm` call in prefill: bounds the transient bank when a long prompt routes to nearly every
# expert of a layer (64 x 2.15 MB for Qwen3.8's 3-bit experts instead of 512).
PREFILL_EXPERT_GROUP = 64


def _quantized_sorted_forward(module, store, hidden_states, top_k_index, top_k_weights, experts, banks, result):
    """Add the routes to ``experts`` (sorted by expert, over ``banks`` holding only them) into ``result`` (float32).

    Uses float32 to avoid incorrect fp16/bf16 results in the v2 M5 kernel."""
    top_k = top_k_index.shape[1]
    lookup = torch.full((module.num_experts,), -1, dtype=torch.long, device=hidden_states.device)
    lookup[torch.tensor(experts, device=hidden_states.device)] = torch.arange(
        len(experts), device=hidden_states.device
    )
    flat = lookup[top_k_index.reshape(-1)]
    selected = torch.nonzero(flat >= 0).squeeze(1)
    rhs, permutation = torch.sort(flat[selected], stable=True)
    order = selected[permutation]
    banks = [bank.float() if bank.is_floating_point() else bank for bank in banks]

    x = hidden_states.float()[order // top_k]
    output = _quantized_mlp(module, store, x, banks, rhs, sorted_indices=True)
    output = output * top_k_weights.reshape(-1)[order, None].float()
    result.index_add_(0, order // top_k, output)


def _quantized_pool_forward(module, store, pool, hidden_states, slots, top_k_weights):
    """Resident quantized experts through `gather_qmm`; non-resident routes (slot ``num_slots``) contribute zero."""
    num_tokens, top_k = slots.shape
    rhs = slots.reshape(-1)
    valid = rhs < pool.num_slots
    weights = (top_k_weights.reshape(-1) * valid)[:, None]
    output = _quantized_mlp(module, store, hidden_states.repeat_interleave(top_k, dim=0), pool.banks, rhs * valid)
    return (output * weights.to(output.dtype)).view(num_tokens, top_k, -1).sum(dim=1).to(hidden_states.dtype)


def _pool_forward(module, store, hidden_states, top_k_index, top_k_weights, slots=None):
    pool = store.pool_of[module.offload_prefix]
    if slots is None:
        slots = store.device_slot_table(module.offload_prefix)[top_k_index]
    if store.quantized is not None:
        return _quantized_pool_forward(module, store, pool, hidden_states, slots, top_k_weights)
    kernel = ExpertsInterface._global_mapping[store.compute]
    return kernel(_PoolExperts(module, pool), hidden_states, slots, top_k_weights)


def _commit_mps(store):
    """Hand the queued MPS work to the GPU without waiting for it.

    PyTorch's MPS stream keeps small kernels in a command buffer that it commits at the next synchronization, so after
    each layer's routing copy the GPU would idle while the host encodes the next layer. Recording an event commits the
    buffer.
    """
    # TODO(portability): MPS-only workaround for PyTorch's lazy command-buffer commits; check whether other backends
    # (CUDA streams submit eagerly) need an equivalent, and drop it once PyTorch's MPS stream commits on its own.
    if store.device.type == "mps":
        torch.mps.Event().record()


def experts_offload_forward(module, hidden_states, top_k_index, top_k_weights):
    """The ``experts_offload`` experts implementation: resident experts from the cache, missing ones from disk.

    With an ``expert_compute`` callable, this forward does the routing copy, reads, read-ahead, next-layer guesses and
    installs, and the callable computes the output: ``expert_compute(module, store, hidden_states, top_k_index,
    top_k_weights, routing, experts, pinned)``, where ``routing`` is ``top_k_index`` on the host, ``experts`` the sorted
    routed experts and ``pinned`` the keys made resident for this call (the others are read with
    ``store.expert_tensors``)."""
    if module.training or torch.is_grad_enabled():
        raise RuntimeError("experts_offload is inference-only: use eval() and torch.no_grad() / inference_mode()")
    store = module.offload_store
    if hidden_states.device != store.device or hidden_states.dtype != store.dtype:
        raise ValueError(
            "Moving/casting a streamed model after loading is unsupported; reload on the target device/dtype"
        )
    prefix = module.offload_prefix
    if store.expert_compute is None and store.all_resident():
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
    readahead = module.offload_readahead_routes
    readahead_next = readahead and module.offload_next_prefix is not None and top_k_index.numel() >= readahead
    if readahead_next:
        # A long prompt routes to nearly every expert of every layer: read the next layer's while this one computes.
        store.prefetch(module.offload_next_prefix, range(module.num_experts))
    width = top_k_index.shape[0] * (next_layer[1].top_k if predict else 0)
    guessed = torch.unique(routes[top_k_index.numel() : top_k_index.numel() + width]).tolist() if predict else []
    if predict:
        store.prefetch(next_prefix, guessed)
    if after_next is not None:
        store.prefetch(after_next[0], torch.unique(routes[top_k_index.numel() + width :]).tolist())
    cold_routes = module.offload_cold_prefill_routes
    store.cold = bool(cold_routes) and store.policy == "lru" and top_k_index.numel() >= cold_routes
    try:
        result = _offloaded_experts(module, store, prefix, hidden_states, top_k_index, top_k_weights, routing, experts)
    finally:
        store.discard(prefix)
        # Also demotes what a previous cold call installed ahead for this layer; `store.cold` stays set until the next
        # call, so the next layer's installs below are demoted after that layer runs.
        store.demote(prefix)
    if store.commit:
        _commit_mps(store)
    if guessed:
        store.install_ahead(next_prefix, guessed)
    if readahead_next and module.offload_prefill_install_ahead:
        # The next layer's reads started above: queue the finished ones' copies behind this layer's compute, so that
        # layer finds them installed instead of copying while the device idles after its routing copy.
        store.install_ahead(module.offload_next_prefix, range(module.num_experts))
    return result


def _offloaded_experts(module, store, prefix, hidden_states, top_k_index, top_k_weights, routing, experts):
    pool = store.pool_of[prefix]
    pinned, uncached = set(), []
    for position, expert in enumerate(experts):
        store.prefetch(prefix, experts[position + 1 : position + 1 + store.window])
        if store.acquire(prefix, expert, pinned) is None:
            uncached.append(expert)
        else:
            pinned.add((prefix, expert))
    if store.expert_compute is not None:
        return store.expert_compute(module, store, hidden_states, top_k_index, top_k_weights, routing, experts, pinned)
    if store.quantized is not None and _sorted_kernel_applies(top_k_index.numel(), len(experts)):
        # A group of experts at a time: a prefill that routes to every expert never holds a whole layer of them.
        result = torch.zeros(hidden_states.shape, dtype=torch.float32, device=hidden_states.device)
        for start in range(0, len(experts), PREFILL_EXPERT_GROUP):
            group = experts[start : start + PREFILL_EXPERT_GROUP]
            per_expert = []
            for position, expert in enumerate(group):
                if (prefix, expert) in pinned:
                    slot = pool.cache[(prefix, expert)]
                    per_expert.append([bank[slot] for bank in pool.banks])
                else:
                    store.prefetch(prefix, experts[start + position + 1 : start + position + 1 + store.window])
                    per_expert.append(store._read(prefix, expert))
            banks = [torch.stack(parts) for parts in zip(*per_expert)]
            del per_expert
            _quantized_sorted_forward(module, store, hidden_states, top_k_index, top_k_weights, group, banks, result)
            if store.inflight:
                store.synchronize()
        return result.to(hidden_states.dtype)
    if pinned:
        slots = store.device_slots(prefix, routing)
        result = _pool_forward(module, store, hidden_states, top_k_index, top_k_weights, slots)
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


def split_experts_device_map(device_map, experts_offload):
    """Turn ``device_map`` entries that send the experts to ``"disk"`` into expert streaming.

    An entry whose key names experts modules (``"experts.*"``, ``"model.layers.*.mlp.experts"``, ...) and whose value is
    ``"disk"`` streams every experts module from disk through the device cache, instead of Accelerate's disk offload
    (which would read every expert of a layer at each forward). The rest of the map must place the model on one device.
    Returns the device map without those entries and the ``experts_offload`` options (``cache_bytes`` required)."""
    if not isinstance(device_map, dict):
        return device_map, experts_offload
    keys = [key for key, value in device_map.items() if value == "disk" and "experts" in str(key)]
    if not keys:
        return device_map, experts_offload
    rest = {key: value for key, value in device_map.items() if key not in keys}
    if not rest:
        raise ValueError('Map the rest of the model to a device too, e.g. {"": "mps", "experts.*": "disk"}')
    if "disk" in rest.values():
        raise ValueError("Streamed experts cannot be combined with disk offload of other modules")
    options = dict(experts_offload or {})
    if "cache_bytes" not in options:
        raise ValueError(
            'Experts on "disk" stream through a device cache: pass its size, e.g. experts_offload={"cache_bytes": 8 << 30}'
        )
    return rest, options


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
    "mps_commit",
    "host_cache_bytes",
    "host_preload",
    "prefill_readahead_routes",
    "prefill_install_ahead",
    "cold_prefill_routes",
    "expert_source",
    "expert_compute",
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
        prefill_readahead_routes (`int`, defaults to 4096): calls with at least this many routes (long prompts) also
            start reading every expert of the next MoE layer, which such a prompt nearly all uses, so storage keeps
            reading while the device computes. 0 disables.
        host_cache_bytes (`int`, defaults to 0): host memory (pinned on accelerators) kept as a tier of quantized
            experts between storage and the device cache; for discrete GPUs with more host than device memory.
        host_preload (`bool`, defaults to `False`): read every expert into the host tier at load time.
        prefill_install_ahead (`bool`, defaults to `False`): with `prefill_readahead_routes`, also install the next
            layer's finished reads behind this layer's compute.
        cold_prefill_routes (`int`, defaults to 0): calls with at least this many routes (long prompts) do not refresh
            the LRU order on hits, and the experts they install move to the cold end once the layer is done, so a long
            prompt does not evict the experts decoding keeps using. 0 disables. LRU only.
        mps_commit (`bool`, defaults to `True`): on MPS, commit the queued work after each decoder-layer sub-block and
            after the experts, so the GPU runs it while the host encodes what follows.
        expert_source (*optional*): read experts from an external store instead of the checkpoint (see
            :class:`ExpertStore`); needs `expert_compute`. Layers may differ in layout: each layout gets its own pool and
            every layer the same number of slots.
        expert_compute (`Callable`, *optional*): computes the experts output over the pools (see
            :func:`experts_offload_forward`); required with `expert_source`, where the experts' format is the source's.
    """
    options = dict(options)
    unknown = set(options) - _OPTIONS
    if unknown or "cache_bytes" not in options:
        raise ValueError(f"experts_offload needs `cache_bytes`; unknown options: {sorted(unknown)}")
    if not checkpoint_files or not all(str(f).endswith(".safetensors") for f in checkpoint_files):
        raise ValueError("experts_offload reads experts from safetensors checkpoints only")
    if dtype not in _DTYPES.values():
        raise ValueError("experts_offload needs a floating dtype (float32/float16/bfloat16)")
    source = options.get("expert_source")
    if (source is None) != (options.get("expert_compute") is None):
        raise ValueError("expert_source and expert_compute go together")
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
        source=source,
    )
    store.compute = options.get("experts_implementation", "grouped_mm")
    store.expert_compute = options.get("expert_compute")
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
        if source is not None:
            prefix = source.prefix(name) if hasattr(source, "prefix") else name
        else:
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
        module.offload_readahead_routes = options.get("prefill_readahead_routes", 4096)
        module.offload_prefill_install_ahead = options.get("prefill_install_ahead", False)
        module.offload_cold_prefill_routes = options.get("cold_prefill_routes", 0)
    store.allocate(materialize=False)
    depth = options.get("prefetch_depth", 2)
    routers = []
    for name, module in experts:
        router = getattr(model.get_submodule(name.rpartition(".")[0]), "gate", None)
        usable = router is not None and hasattr(router, "weight") and hasattr(router, "top_k")
        routers.append((module.offload_prefix, router) if usable else None)
    for index, (_, module) in enumerate(experts):
        module.offload_next_prefix = experts[index + 1][1].offload_prefix if index + 1 < len(experts) else None
        if depth >= 1 and index + 1 < len(routers):
            module.offload_next = routers[index + 1]
        if depth >= 2 and index + 2 < len(routers):
            module.offload_after_next = routers[index + 2]
    if options.get("mps_commit", True):
        store.commit = True

        def commit_hook(module, args, output):
            _commit_mps(store)  # returns nothing: a forward hook's return value would replace the output

        for name, _ in experts:
            layer = model.get_submodule(name.rpartition(".")[0].rpartition(".")[0])
            for child in layer.children():
                child.register_forward_hook(commit_hook)
    if options.get("host_cache_bytes", 0):
        store.allocate_host_cache(options["host_cache_bytes"])
    if options.get("host_preload", False):
        store.preload_host()
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
    # TODO(portability): Metal quantizer on MPS only; other accelerators use the PyTorch reference.
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
