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
"""Stream MoE experts from the checkpoint through a device cache of expert slots (inference only).

```python
model = AutoModelForCausalLM.from_pretrained(
    repo, device_map={"": "mps", "experts": "disk"}, max_memory={"mps": "20GiB"}
)
```

The experts named `"disk"` in the `device_map` are never materialized. The loader hands their checkpoint slices to
an `ExpertsOffloadStore` (see `core_model_loading.py`), and every experts module gets an `ExpertsOffloadLayer`:

```text
layer L forward
  routing (+ router guesses for L+1..L+lookahead) ──► one host copy
  hits   → slots in the pool (LRU)          ─► pass 1 on the pool, misses as EP sentinels   (GPU busy)
  misses → reads already started by L-1     ─► install into slots ─► pass 2, hits as sentinels
  reads for L+1..L+lookahead start now, overlapping L's compute
```

The experts module's own forward (any `experts_implementation`) runs on the pool: its banks are the pool banks,
`num_experts` is the number of slots and ids `>= num_slots` are expert-parallel sentinels.
"""

from __future__ import annotations

import bisect
import collections
import ctypes
import fnmatch
import itertools
import json
import os
import queue
import struct
import threading
import time
import weakref
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

from ..integrations.moe import batched_mm_experts_forward
from ..utils import logging
from .tensor_parallel import TensorParallelLayer


if TYPE_CHECKING:
    from torch import nn


logger = logging.get_logger(__name__)


def expert_params(module: nn.Module) -> list[str]:
    """The module's per-expert params (leading dim `num_experts`): banks, and e.g. their quantization scales."""
    return [
        name
        for name, param in module.named_parameters(recurse=False)
        if param.dim() >= 2 and param.shape[0] == module.num_experts
    ]


def split_streamed_modules(device_map):
    """Pop the `{"<experts module pattern>": "disk"}` entries of `device_map`; returns `(device_map, patterns)`."""
    if not isinstance(device_map, dict):
        return device_map, []
    patterns = [key for key, value in device_map.items() if value == "disk"]
    rest = {key: value for key, value in device_map.items() if key not in patterns}
    if patterns and not rest:
        raise ValueError(
            'Streamed modules need the rest of the model on one device, e.g. {"": "mps", "experts": "disk"}'
        )
    return rest, patterns


class _Pool:
    """Device banks of `num_slots` experts of one layout, with one LRU over every layer that uses it."""

    def __init__(self, layout: dict[str, tuple[tuple[int, ...], torch.dtype]], num_slots: int, policy: str = "lru"):
        self.layout = layout
        self.num_slots = num_slots
        self.policy = policy
        self.tables: dict[str, torch.Tensor] = {}  # layer -> host expert -> slot table (`num_slots`: not resident)
        self.dirty: set[str] = set()  # layers whose device table is stale
        self.counts: dict[tuple[str, int], float] = {}  # lookups per expert, halved every `num_slots` lookups (lfu)
        self.lookups = 0
        self.banks = None
        self.cache: OrderedDict[tuple[str, int], int] = OrderedDict()  # (layer, expert) -> slot, coldest first
        self.free = list(range(num_slots))
        self.owner = None  # the layer of a pinned pool (exclusive), see `complete`

    def materialize(self, device):
        if self.banks is None:
            self.banks = {
                name: torch.zeros(
                    self.num_slots, *shape, dtype=dtype, device=device
                )  # unfilled slots: zeros, never NaN
                for name, (shape, dtype) in self.layout.items()
            }
        # bf16 / fp32 banks, run by the experts interface; packed (quantized) banks have their own kernels
        self.floating = all(dtype.is_floating_point for _, dtype in self.layout.values())

    def count(self, keys):
        """Record lookups (lfu): every lookup counts, hit or miss, and old counts decay."""
        if self.policy != "lfu":
            return
        for key in keys:
            self.counts[key] = self.counts.get(key, 0) + 1
        self.lookups += len(keys)
        if self.lookups >= self.num_slots:
            self.lookups = 0
            self.counts = {k: v / 2 for k, v in self.counts.items() if v >= 0.5}

    def slot_for(self, key, pinned) -> int | None:
        """A free slot for `key`, evicting an unpinned expert (least recently used, or with `lfu` the least counted of
        the 64 least recently used); None if all are pinned."""
        if not self.free:
            # ponytail: approximate LFU over a window of the LRU's cold end; a heap if the window misses hot experts
            candidates = (k for k in self.cache if k not in pinned)
            if self.policy == "lfu":
                window = [k for _, k in zip(range(64), candidates)]
                victim = min(window, key=lambda k: self.counts.get(k, 0), default=None)
            else:
                victim = next(candidates, None)
            if victim is None:
                return None
            self.free.append(self.cache.pop(victim))
            self.tables[victim[0]][victim[1]] = self.num_slots
            self.dirty.add(victim[0])
        slot = self.free.pop()
        self.cache[key] = slot
        self.tables[key[0]][key[1]] = slot
        self.dirty.add(key[0])
        return slot

    def complete(self, layer) -> bool:
        """Every expert of `layer` is resident and can never be evicted (its own pool, see `pin_layers`)."""
        return self.owner == layer and not self.free


class ExpertsOffloadStore:
    """Expert slot pools, checkpoint readers and the read schedule of every offloaded experts module."""

    def __init__(
        self,
        device,
        cache_bytes: int,
        io_threads: int = 4,
        lookahead: int = 1,
        policy: str = "lru",
        pin_layers: int = 0,
    ):
        self.pin_layers = pin_layers  # the first layers keep all their experts: shallow routing predicts worst
        if policy not in ("lru", "lfu"):
            raise ValueError(f"Unknown expert cache policy {policy!r}: use 'lru' or 'lfu'")
        self.policy = policy
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        # CUDA: safetensors' prefetch loader (chunked parallel reads into a pinned pool, async copies to the device).
        # MPS (unified memory): the loader's `pread` slices land on the device directly.
        if self.device.type == "cuda" and not hasattr(safe_open, "prefetch"):
            raise ImportError(
                "Streaming experts on CUDA needs safetensors' prefetch loader: `pip install 'safetensors>=0.9'`"
            )
        self.cache_bytes = cache_bytes
        self.lookahead = lookahead
        self.executor = ThreadPoolExecutor(io_threads, thread_name_prefix="experts-offload")
        self.layers: dict[str, nn.Module] = {}  # module name -> experts module, in forward order
        self.routers: dict[str, nn.Module | None] = {}  # module name -> its router, to guess its experts ahead
        self.params: dict[str, list[str]] = {}  # module name -> its streamed params (per-expert, or `weight`)
        self.embeddings: dict[str, nn.Module] = {}  # streamed embeddings: only the looked-up rows are read
        self.row_readers: dict[str, _RowReader] = {}
        self.key_files: dict[str, str] | None = None  # checkpoint key -> file, set by the loader
        self.handles = {}  # file -> safe_open handle of the prefetch loaders
        self.order: list[str] = []
        self.readers: dict[str, list] = {}  # module name -> [read(expert) -> {param: tensor}]
        self.pools: dict[str, _Pool] = {}
        self.pool_of: dict[str, _Pool] = {}  # module name -> its pool
        self.device_tables: dict[str, torch.Tensor] = {}  # module name -> expert -> slot, on the device
        self.pending = {}  # (layer, expert) -> Future of its device tensors
        self.pending_experts = collections.defaultdict(set)  # layer -> its experts in `pending`
        self.host_routes = {}  # layer -> (routing on the host, its copy's MPS event), from `start_routing`
        self.mps_router: _MPSHostRouter | None = None  # MPS: misses resolved while the device waits, see the class
        self.pass2_tables: dict[str, torch.Tensor] = {}  # MPS: layer -> its last call's misses (expert -> slot)
        self.hits = self.misses = self.reads = self.covered = 0
        self.trace: list | None = None  # set to [] to record (time, thread, phase, layer, info) events, see `_mark`
        # Host time blocked on the routing copy (device sync) and on expert reads, see `stats`
        self.sync_seconds = self.wait_seconds = 0.0

    def _mark(self, phase, layer=None, **info):
        if self.trace is not None:
            self.trace.append((time.perf_counter(), threading.get_ident(), phase, layer, info))

    def stats(self) -> dict:
        return {
            "slots": sum(pool.num_slots for pool in self.pools.values()),
            "hits": self.hits,
            "misses": self.misses,
            "reads": self.reads,
            "covered": self.covered,  # misses whose read the lookahead had already started
            "sync_seconds": self.sync_seconds,
            "wait_seconds": self.wait_seconds,
        }

    def reset_stats(self):
        self.hits = self.misses = self.reads = self.covered = 0
        self.sync_seconds = self.wait_seconds = 0.0

    def owns(self, param_name: str) -> bool:
        module_name, _, param = param_name.rpartition(".")
        return param in self.params.get(module_name, ())

    def add_layer(self, name: str, module: nn.Module, router: nn.Module | None):
        self.layers[name] = module
        self.params[name] = expert_params(module)
        self.routers[name] = router
        self.order.append(name)

    def allocate(self):
        """Split `cache_bytes - resident_bytes` between the expert layouts, in proportion to their layer count.

        Checkpoints are in the transformers format, so a layer's meta params are exactly what its reads return."""
        layouts = {}
        for name, module in self.layers.items():
            layout = {p: (tuple(getattr(module, p).shape[1:]), getattr(module, p).dtype) for p in self.params[name]}
            layouts.setdefault(repr(layout), (layout, []))[1].append(name)
        budget = self.cache_bytes - self.resident_bytes
        # Pinned layers: an exclusive pool each, of all their experts
        for name in self.order[: self.pin_layers]:
            layout = next(layout for layout, names in layouts.values() if name in names)
            names = next(names for _, names in layouts.values() if name in names)
            names.remove(name)
            pool = _Pool(layout, self.layers[name].num_experts, self.policy)
            pool.owner = name
            budget -= pool.num_slots * sum(
                torch.Size(shape).numel() * dtype.itemsize for shape, dtype in layout.values()
            )
            self.pools[f"pinned {name}"] = pool
            self.pool_of[name] = pool
        layouts = {key: value for key, value in layouts.items() if value[1]}
        shared = sum(len(names) for _, names in layouts.values())
        for key, (layout, names) in layouts.items():
            expert_bytes = sum(torch.Size(shape).numel() * dtype.itemsize for shape, dtype in layout.values())
            total = sum(self.layers[name].num_experts for name in names)
            num_slots = min(total, int(budget * len(names) / shared) // expert_bytes)
            if num_slots < 1:
                raise ValueError(f"No room for one expert in {budget} bytes of `max_memory` for streamed experts")
            self.pools[key] = pool = _Pool(layout, num_slots, self.policy)
            for name in names:
                self.pool_of[name] = pool
        for name, pool in self.pool_of.items():
            num_experts = self.layers[name].num_experts
            pool.tables[name] = torch.full((num_experts,), pool.num_slots, dtype=torch.long)
            self.device_tables[name] = pool.tables[name].to(self.device)
        logger.info(f"Streaming experts through {[p.num_slots for p in self.pools.values()]} device slots")
        if self.device.type == "mps" and os.environ.get("TRANSFORMERS_EXPERTS_OFFLOAD_MPS_EVENTS", "1") == "1":
            from ..integrations.mps_events import load

            ext = load()
            if ext is not None and self.key_files is not None:
                torch.mps.synchronize()
                self.mps_router = _MPSHostRouter(self, ext)

    def add_reader(self, target_name: str, mapping, reads: dict[str, list], hf_quantizer=None):
        """Register how to read one expert of `target_name` from the checkpoint slices the loader collected.

        `mapping` is the `WeightConverter` / `WeightRenaming` of that target and `reads` its collected
        `{source pattern: [(slice, dtype), ...]}`. A converter whose ops merge per-expert tensors
        (`MergeModulelist`, then e.g. `Concatenate`) is replayed on one expert's tensors; a renaming (the checkpoint
        already holds the stacked bank) is sliced on its first dimension."""

        module_name = target_name.rpartition(".")[0]
        if module_name in self.embeddings:
            self.row_readers[module_name] = _RowReader(reads, "cpu" if self.device.type == "cuda" else self.device)
            return
        self.readers.setdefault(module_name, []).append(_ExpertReader(target_name, mapping, reads, hf_quantizer))

    def _read(self, layer, expert):
        """One expert's rows in host memory (MPS, CPU). Runs on an io thread, so it never touches the device: the
        loader's own slices may be MPS handles, and MPS work issued from several threads at once is not safe. The main
        thread copies the rows into the slot (`_install`); on unified memory that is one memcpy."""
        self.reads += 1
        out = {}
        for reader in self.readers[layer]:
            for param, slice_, rows, dtype, key in reader.sources(expert):
                if self.key_files is not None:
                    slice_ = self._handle(self.key_files[key]).get_slice(key)  # a CPU handle
                out[param] = slice_[rows].to("cpu", dtype)[0]
        return out

    def fetch(self, layer, experts):
        """Start reading the `experts` of `layer` that are neither resident nor already being read; returns the
        Futures of all of `experts` that are not resident. On CUDA one prefetch plan covers the batch."""
        cache = self.pool_of[layer].cache
        new = [e for e in dict.fromkeys(experts) if (layer, e) not in self.pending and (layer, e) not in cache]
        if self.device.type == "cuda" and new:
            futures = {e: Future() for e in new}
            self.pending.update({(layer, e): future for e, future in futures.items()})
            self.pending_experts[layer].update(futures)
            self.executor.submit(self._prefetch_batch, layer, futures)
        else:
            for e in new:
                self.pending[(layer, e)] = self.executor.submit(self._read, layer, e)
                self.pending_experts[layer].add(e)
        return [self.pending[(layer, e)] for e in experts if (layer, e) in self.pending]

    def _handle(self, file):
        if file not in self.handles:
            self.handles[file] = safe_open(file, framework="pt")
        return self.handles[file]

    def _prefetch_batch(self, layer, futures):
        """Read a batch of experts with one prefetch loader per file (a list of row slices per bank) and hand each
        expert out as soon as all its params have landed, in readiness order."""
        try:
            self._mark("batch_start", layer, experts=len(futures))
            plans, where = {}, {}  # file -> {key: [rows]}; (key, position) -> (expert, param, dtype)
            for expert in futures:
                for reader in self.readers[layer]:
                    for param, _, rows, dtype, key in reader.sources(expert):
                        slices = plans.setdefault(self.key_files[key], {}).setdefault(key, [])
                        where[(key, len(slices))] = (expert, param, dtype)
                        slices.append(rows)
            loaders = [
                self._handle(file).prefetch(plan, device=str(self.device), threads=16) for file, plan in plans.items()
            ]
            self._mark("loaders_started", layer, loaders=len(loaders))
            outs = {expert: {} for expert in futures}
            num_params = sum(len(reader.banks) for reader in self.readers[layer])
            for loader in loaders:
                for (key, position), tensor in loader:
                    expert, param, dtype = where[(key, position)]
                    outs[expert][param] = tensor[0].to(dtype)
                    if len(outs[expert]) == num_params and not futures[expert].cancelled():
                        self.reads += 1
                        self._mark(
                            "take_end", layer, expert=expert, bytes=sum(t.nbytes for t in outs[expert].values())
                        )
                        futures[expert].set_result(outs.pop(expert))
                loader.close()
        except BaseException as error:
            for future in futures.values():
                if not future.done():
                    future.set_exception(error)

    def setup_layer(self, name):
        """First call of a layer: size the pools, then swap its meta banks for the pool's (as buffers, so compile sees
        them as inputs) with the expert -> slot table."""
        if not self.pools:
            self.allocate()
        module, pool = self.layers[name], self.pool_of[name]
        if self.params[name][0] not in module._parameters:
            return
        pool.materialize(self.device)
        for param in pool.layout:
            module._parameters.pop(param, None)
            module.register_buffer(param, pool.banks[param], persistent=False)
        module.register_buffer("offload_table", self.device_tables[name], persistent=False)
        if self.mps_router is not None:  # the host thread writes slots and tables from the CPU: no queued writes left
            pass2 = torch.full_like(self.device_tables[name], pool.num_slots)  # this call's misses -> slot, for pass 2
            module.register_buffer("offload_pass2", pass2, persistent=False)
            self.pass2_tables[name] = pass2
            torch.mps.synchronize()
        module.num_experts = pool.num_slots
        module._is_expert_parallel = True
        if name in self.order[: self.pin_layers]:  # pinned: all its experts now, never a miss after
            futures = self.fetch(name, range(module.offload_num_experts))
            for expert, future in enumerate(futures):
                self._install(pool, pool.slot_for((name, expert), ()), future)
                self._pop_pending(name, expert)
            self._refresh_tables(pool)

    def _install(self, pool, slot, future):
        """Copy a read expert into `slot` once its read is done (one device copy per param)."""
        start = time.perf_counter()
        self._mark("wait_start")
        tensors = future.result()
        self.wait_seconds += time.perf_counter() - start
        self._mark("wait_end")
        for param, tensor in tensors.items():
            pool.banks[param][slot].copy_(tensor, non_blocking=True)

    def _refresh_tables(self, pool):
        """Copy the host slot tables changed by installs and evictions to the device (queued, in stream order)."""
        for layer in pool.dirty:
            snapshot = pool.tables[layer].clone()
            if self.device.type == "cuda":
                snapshot = snapshot.pin_memory()
            self.device_tables[layer].copy_(snapshot, non_blocking=True)
        pool.dirty.clear()

    def misses_pass(self, name, module, original_forward, hidden_states, top_k_index, top_k_weights, output):
        pool = self.pool_of[name]
        num_experts = module.offload_num_experts
        if pool.complete(name):
            self.hits += top_k_index.numel()
            return
        # 1. The routing + next layers' guesses, copied to the host by `start_routing` before pass 1 was queued: wait
        # for that copy only, pass 1 keeps the device busy meanwhile
        ahead = self.next_layers(name)
        decode = top_k_index.numel() < num_experts
        guessed = [n for n in ahead if decode and self.routers[n] is not None]
        if self.device.type == "mps":
            torch.mps.Event().record()  # submit pass 1 now: the device runs it while the host handles the misses
        self._mark("sync_start", name)
        start = time.perf_counter()
        routes = self.host_routes.pop(name, None)
        if routes is None:  # `start_routing` did not run (e.g. a direct call): copy now
            routes = self._routes(name, hidden_states, top_k_index).cpu()
        else:
            routes, event = routes
            if event is not None:
                event.synchronize()
        self.sync_seconds += time.perf_counter() - start
        self._mark("sync_end", name)
        routing = routes[: top_k_index.numel()].view(top_k_index.shape)
        experts = torch.unique(routing).tolist()
        misses = [e for e in experts if (name, e) not in pool.cache]
        self.hits += len(experts) - len(misses)
        self.misses += len(misses)
        for e in experts:
            if (name, e) in pool.cache:
                pool.cache.move_to_end((name, e))
        pool.count([(name, e) for e in experts])
        self.covered += sum((name, e) in self.pending for e in misses)  # already read ahead by the lookahead
        self._mark(
            "misses",
            name,
            misses=len(misses),
            experts=len(experts),
            covered=sum((name, e) in self.pending for e in misses),
        )
        futures = self.fetch(name, misses)
        # 2. Reads for the next layers overlap this layer's compute
        for index, next_name in enumerate(guessed):
            guesses = routes[(index + 1) * top_k_index.numel() : (index + 2) * top_k_index.numel()]
            self.fetch(next_name, torch.unique(guesses).tolist())
        if not decode and ahead:  # a prompt routes to nearly every expert: read the whole next layer
            self.fetch(ahead[0], range(self.layers[ahead[0]].offload_num_experts))
        self._mark("guesses_fetched", name)
        # 3. Install the misses as their reads land, as many as fit at a time, and add their routes
        while misses:
            batch = []
            for expert, future in zip(misses, futures):
                slot = pool.slot_for((name, expert), {(name, e) for e in batch})
                if slot is None:
                    break
                self._install(pool, slot, future)
                self._pop_pending(name, expert)
                if not decode:  # a prompt's experts go to the cold end, the decode set stays
                    pool.cache.move_to_end((name, expert), last=False)
                batch.append(expert)
            table = torch.full((num_experts,), pool.num_slots, dtype=top_k_index.dtype)
            table[batch] = torch.tensor([pool.cache[(name, e)] for e in batch], dtype=top_k_index.dtype)
            self._mark("pass2", name, experts=len(batch))
            if decode and routing.shape[0] == 1:
                # One token: only its routes to this batch's experts (the others are resident or later batches)
                positions = [k for k, e in enumerate(routing[0].tolist()) if e in set(batch)]
                index = torch.tensor(positions, device=top_k_weights.device)
                slots = table[routing[:, positions]].to(self.device, non_blocking=True)
                weights = top_k_weights.index_select(-1, index)
            else:
                slots, weights = table[routing].to(self.device, non_blocking=True), top_k_weights
            output.add_(_run_pool(module, original_forward, hidden_states, slots, weights, {}))
            misses, futures = misses[len(batch) :], futures[len(batch) :]
        self._mark("misses_done", name)
        self._refresh_tables(pool)
        self._mark("tables_refreshed", name, dirty=len(pool.dirty))
        # Guesses for this layer that the routing did not pick
        unused = list(self.pending_experts.get(name, ()))
        for expert in unused:
            self._pop_pending(name, expert).cancel()
        self._mark("layer_end", name, unused_guesses=len(unused))
        if self.device.type == "mps":
            torch.mps.Event().record()  # submit pass 2 / table copies: they run while the host issues the next ops

    def _pop_pending(self, layer, expert):
        self.pending_experts[layer].discard(expert)
        return self.pending.pop((layer, expert), None)

    def _routes(self, name, hidden_states, top_k_index):
        """On the device: this layer's routing, then each next layer's guess (its router on this layer's input)."""
        routes = [top_k_index.reshape(-1)]
        if top_k_index.numel() < self.layers[name].offload_num_experts:  # decode: guesses for the lookahead
            for next_name in self.next_layers(name):
                if self.routers[next_name] is not None:
                    routes.append(_route(self.routers[next_name], hidden_states, top_k_index.shape[-1]).flatten())
        return torch.cat(routes)

    def start_routing(self, name, hidden_states, top_k_index):
        """Queue the routing's copy to the host before pass 1: on MPS an async copy + an event the misses wait on
        (not a drain of the queue); elsewhere the misses copy it themselves, after pass 1 is queued."""
        if self.device.type != "mps" or self.pool_of[name].complete(name):
            return
        routes = self._routes(name, hidden_states, top_k_index).to("cpu", non_blocking=True)
        event = torch.mps.Event()
        event.record()
        self.host_routes[name] = (routes, event)

    def next_layers(self, layer):
        index = self.order.index(layer)
        return self.order[index + 1 : index + 1 + self.lookahead]


def _read_into(job) -> bool:
    fd, address, size, offset = job
    return os.preadv(fd, [memoryview((ctypes.c_char * size).from_address(address)).cast("B")], offset) == size


class _MPSHostRouter:
    """MPS: misses resolved by a host thread while the device waits on a Metal shared event, so the host never waits on
    the device mid-step. Per streamed layer call: the device copies the routing into the layer's shared buffer and
    waits; this thread spins (C++, GIL released) until the routing lands, reads the missing experts from their files
    straight into free pool slots (`preadv` into the slots' unified memory: no copy, no device op), writes the slot
    table in place and signals. Pools and tables live in shared storage, written by the CPU while the device waits."""

    MAX_AHEAD = 16
    COMMIT_EVERY = int(os.environ.get("TRANSFORMERS_EXPERTS_OFFLOAD_COMMIT_EVERY", "2"))

    def __init__(self, store, ext):
        self.store, self.ext = store, ext
        self.slot_bases = {}  # (pool, param) -> (CPU address of slot 0, slot bytes)
        self.stamps: dict[int, torch.Tensor] = {}  # pool -> last use per slot (job counter), written by the hit path
        self.owners: dict[int, dict[int, tuple[str, int]]] = {}  # pool -> slot -> (layer, expert)
        self.deferred: list[tuple[str, list[int]]] = []  # resolved calls whose tables publish later
        self.times: dict[
            int, dict[str, float]
        ] = {}  # debug: per job, when it was routed / waited / picked up / landed
        self.spin_seconds = self.work_seconds = 0.0  # host thread: waiting for routings / resolving them
        self.jobs_done = self.ahead_reads = 0
        self.event = ext.event_new()
        self.counter = 0
        self.jobs = queue.SimpleQueue()
        self.buffers: dict[str, torch.Tensor] = {}  # layer -> routing buffer (int32, -1 = not landed)
        self.files: dict[str, int] = {}  # path -> fd
        self.layout: dict[tuple[str, str], tuple[int, int, int]] = {}  # (layer, param) -> (fd, base offset, row bytes)
        self.error = None
        threading.Thread(target=self._run, daemon=True, name="experts-offload-mps").start()

    @staticmethod
    def _host_written(tensor):
        """`tensor` on MPS, safe for CPU writes: the caching allocator may hand out memory that queued device work
        of its previous owner still writes, so wait for the queue first (the host thread keeps signalling meanwhile,
        the wait releases the GIL)."""
        tensor = tensor.to("mps")
        torch.mps.synchronize()
        return tensor

    def _locate(self, layer):
        """File, byte offset and row size of each param bank of `layer` (rows are experts, stored contiguous)."""
        for reader in self.store.readers[layer]:
            for param, _, dtype, key in reader.banks:
                path = self.store.key_files[key]
                if path not in self.files:
                    self.files[path] = os.open(path, os.O_RDONLY)
                    with open(path, "rb") as f:
                        size = struct.unpack("<Q", f.read(8))[0]
                        self.files[(path, "header")] = (8 + size, json.loads(f.read(size)))
                start, header = self.files[(path, "header")]
                info = header[key]
                rows = info["shape"][0]
                row_bytes = (info["data_offsets"][1] - info["data_offsets"][0]) // rows
                self.layout[(layer, param)] = (self.files[path], start + info["data_offsets"][0], row_bytes)

    def guess(self, layer, hidden_states, top_k_index):
        """The next streamed layer's experts as its router would pick them on this layer's input (decode only; an
        empty tensor otherwise): what the host reads ahead."""
        ahead = self.store.next_layers(layer)
        router = self.store.routers.get(ahead[0]) if ahead else None
        if router is None or top_k_index.shape[0] != 1:
            return top_k_index.new_empty(0)
        return _route(router, hidden_states, top_k_index.shape[-1]).reshape(-1).to(top_k_index.dtype)

    def route(self, layer, top_k_index, slots, guess):
        """Issue side, before pass 1: one copy of [routing | the slots pass 1 reads | next-layer guess] into the
        layer's shared buffer (-1 = not landed), then queue the job (no wait)."""
        n, g = top_k_index.numel(), guess.numel()
        buffer = self.buffers.get(layer)
        if buffer is None or buffer.numel() < 2 * n + g:  # first call, or a longer prompt
            buffer = self.buffers[layer] = self._host_written(torch.empty(max(2 * n + g, 1 << 16), dtype=torch.int32))
            ctypes.memset(self.ext.cpu_address(buffer), 0xFF, buffer.numel() * 4)
        if (layer, self.store.params[layer][0]) not in self.layout:
            self._locate(layer)
        buffer[: 2 * n + g].copy_(torch.cat([top_k_index.reshape(-1), slots.reshape(-1), guess]))
        self.counter += 1
        # Each wait commits a command buffer at most every `COMMIT_EVERY`: past ~64 in flight Metal stops the queue,
        # so the host stays at most `MAX_AHEAD` layers ahead of the device
        while self.counter - self.ext.signaled(self.event) > self.MAX_AHEAD:
            if self.error is not None:
                break
            time.sleep(0)
        self.jobs.put((layer, n, g, self.counter, self.ext.cpu_address(buffer)))
        self.ext.flush()  # the copy's writes reach the host at its encoder's end: end it now, before pass 1
        if self.error is not None:
            raise RuntimeError("experts offload host thread failed") from self.error

    def wait(self, layer):
        """Issue side, after pass 1: the device waits until the host thread resolved this call (pass 2 table set)."""
        self.times.setdefault(self.counter, {})["wait"] = time.perf_counter()
        self.ext.gpu_wait(self.event, self.counter, self.counter % self.COMMIT_EVERY == 0)

    def _run(self):
        try:
            while True:
                job = self.jobs.get()
                with (
                    torch.inference_mode()
                ):  # the pool's host tables are inference tensors (made in the first forward)
                    self._resolve(*job)
        except BaseException as error:  # surfaced on the next submit; release the device so it does not hang
            logger.error("experts offload host thread failed", exc_info=error)
            self.error = error
            self.ext.cpu_signal(self.event, 1 << 62)

    def _slot_address(self, pool, param, slot):
        base = self.slot_bases.get((id(pool), param))
        if base is None:
            bank = pool.banks[param]
            base = self.slot_bases[(id(pool), param)] = (self.ext.cpu_address(bank), bank[0].nbytes)
        return base[0] + slot * base[1]

    def _stamps(self, pool):
        stamps = self.stamps.get(id(pool))
        if stamps is None:
            stamps = self.stamps[id(pool)] = torch.zeros(pool.num_slots, dtype=torch.int64)
            self.owners[id(pool)] = {slot: key for key, slot in pool.cache.items()}
        return stamps

    def _evict_into(self, pool, key, keep):
        """A slot for `key`: a free one, else the least recently used (by stamp) whose expert is not in `keep`."""
        owners, stamps = self.owners[id(pool)], self.stamps[id(pool)]
        if pool.free:
            slot = pool.free.pop()
        else:
            ranked = stamps.clone()
            ranked[[pool.cache[k] for k in keep if k in pool.cache]] = torch.iinfo(torch.int64).max
            slot = int(ranked.argmin())
            victim = owners.pop(slot)
            del pool.cache[victim]
            pool.tables[victim[0]][victim[1]] = pool.num_slots
            pool.dirty.add(victim[0])
        owners[slot] = key
        pool.cache[key] = slot
        pool.tables[key[0]][key[1]] = slot
        pool.dirty.add(key[0])
        return slot

    def _reset_pass2(self, exclude):
        """Pass-2 entries of earlier calls back to "not a miss" (their pass 2 ran: a later routing landed)."""
        keep = []
        for name, experts in self.deferred:
            if name == exclude:
                keep.append((name, experts))
                continue
            pool = self.store.pool_of[name]
            pass2 = (ctypes.c_int64 * pool.tables[name].numel()).from_address(
                self.ext.cpu_address(self.store.pass2_tables[name])
            )
            for e in experts:
                pass2[e] = pool.num_slots
        self.deferred = keep

    def _install(self, pool, layer, e, keep):
        """Read expert `e` of `layer` into a slot and publish it (data first, then the table entry)."""
        slot = self._evict_into(pool, (layer, e), keep)
        for param in pool.layout:
            fd, base, row_bytes = self.layout[(layer, param)]
            if not _read_into((fd, self._slot_address(pool, param, slot), row_bytes, base + e * row_bytes)):
                raise OSError(f"short read of expert {e} of {layer}")
        self.store.reads += 1
        self._publish(pool)
        return slot

    def _publish(self, pool):
        """The changed slot tables (installs, and the entries of evicted experts) to the device. Safe at any time: a
        call's pass 1 reports the slots it read, so the host always knows what it missed."""
        for name in pool.dirty:
            entries = pool.tables[name]
            ctypes.memmove(self.ext.cpu_address(self.store.device_tables[name]), entries.data_ptr(), entries.nbytes)
        pool.dirty.clear()

    def _resolve(self, layer, n, g, counter, address):
        store, pool = self.store, self.store.pool_of[layer]
        stamps = self._stamps(pool)
        start = time.perf_counter()
        # Hit path in C++ (no GIL): all landed, every slot pass 1 read is resident -> stamps, reset, signal
        status = self.ext.resolve_hits(
            address, n, g, pool.num_slots, stamps.data_ptr(), counter, self.event, counter, 10_000_000
        )
        if status == -1:
            raise TimeoutError(f"routing of {layer} never landed")
        landed = time.perf_counter()
        self.spin_seconds += landed - start
        view = (ctypes.c_int32 * (2 * n + g)).from_address(address)
        values = view[:]
        experts, slots, guesses = values[:n], values[n : 2 * n], values[2 * n :]
        keep = {(layer, e) for e in experts}
        if status == 1:
            ctypes.memset(address, 0xFF, (2 * n + g) * 4)
            store.hits += n
        else:
            misses = sorted({e for e, slot in zip(experts, slots) if slot == pool.num_slots})
            pass2 = (ctypes.c_int64 * pool.tables[layer].numel()).from_address(
                self.ext.cpu_address(store.pass2_tables[layer])
            )
            for e in misses:
                slot = pool.cache.get((layer, e))  # installed since pass 1 gathered (read ahead): reuse it
                pass2[e] = self._install(pool, layer, e, keep) if slot is None else slot
            stamps[[pool.cache[(layer, e)] for e in set(experts)]] = counter
            ctypes.memset(address, 0xFF, (2 * n + g) * 4)
            self.ext.cpu_signal(self.event, counter)
            self.deferred.append((layer, misses))
            store.hits += n - len(misses)
            store.misses += len(misses)
        # Off the critical path: reset the pass-2 entries of earlier calls, read the next layer's guess ahead
        self._reset_pass2(exclude=layer)
        ahead = store.next_layers(layer)
        if guesses and ahead:
            nxt = ahead[0]
            npool = store.pool_of[nxt]
            self._stamps(npool)
            wanted = {(nxt, e) for e in guesses}
            for e in dict.fromkeys(guesses):
                if (nxt, e) not in npool.cache:
                    self._install(npool, nxt, e, keep | wanted)
                    self.ahead_reads += 1
        self.work_seconds += time.perf_counter() - landed
        self.jobs_done += 1


class ExpertsOffloadLayer(TensorParallelLayer):
    """`TensorParallelLayer` style for an experts module streamed through an `ExpertsOffloadStore` (passed where TP
    passes its mesh): `shard_param` leaves the meta banks for the loader to skip, `install_forward` runs the module on
    the pool."""

    def shard_param(self, module, param, store):
        # The meta bank stays until the first forward: the loader sees `store.owns(name)` and only records the reads
        pass

    def install_forward(self, module, store):
        original_forward = module.forward
        name = next(n for n, m in store.layers.items() if m is module)
        handle = next(_HANDLES)
        module.offload_store, module.offload_name, module.offload_experts_forward = store, name, original_forward
        _LAYERS[handle] = module
        setup = torch.compiler.disable(store.setup_layer)

        def offload_forward(hidden_states, top_k_index, top_k_weights, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError("Streamed experts are inference-only: use torch.no_grad() / inference_mode()")
            if not hasattr(module, "offload_table"):  # first call only: no graph break once set up
                setup(name)
            if store.mps_router is not None:
                # MPS: pass 1 (resident experts) runs while a host thread reads what it used (the gathered slots) and
                # installs the misses; the device then waits (usually already signalled) and pass 2 adds the misses.
                # The host also gets a guess of the next layer's experts and reads them ahead.
                slots = module.offload_table[top_k_index]
                guess = store.mps_router.guess(name, hidden_states, top_k_index)
                torch.ops.transformers.experts_offload_mps_route(handle, top_k_index, slots, guess)
                output = _run_pool(module, original_forward, hidden_states, slots, top_k_weights, kwargs)
                # Only the small pass-2 table is declared mutated: it orders pass 2 after the wait. The banks the host
                # writes are not (compile would copy a mutated input around the op: the whole pool, per layer)
                torch.ops.transformers.experts_offload_mps_wait(handle, [module.offload_pass2])
                slots = module.offload_pass2[top_k_index]
                if top_k_index.shape[0] == 1 and hasattr(module, "masked_decode"):
                    return output + module.masked_decode(hidden_states, slots, top_k_weights)
                return output + _run_pool(module, original_forward, hidden_states, slots, top_k_weights, kwargs)
            torch.ops.transformers.experts_offload_route(handle, hidden_states, top_k_index)
            # Pass 1, all on the device (compilable): resident experts through the slot table, misses are sentinels
            slots = module.offload_table[top_k_index]
            output = _run_pool(module, original_forward, hidden_states, slots, top_k_weights, kwargs)
            # Misses: one opaque op, the only host round trip (none for a layer whose experts are all resident)
            state = [module.offload_table, *(getattr(module, p) for p in store.pool_of[name].layout)]
            torch.ops.transformers.experts_offload_misses(
                handle, hidden_states, top_k_index, top_k_weights, output, state
            )
            return output

        module.forward = offload_forward
        return module


_HANDLES = itertools.count()
_LAYERS = weakref.WeakValueDictionary()  # op handle -> experts module (weak: models free their pools)


def _run_pool(module, original_forward, hidden_states, slots, top_k_weights, kwargs):
    """The experts on the pool for `slots` (ids `>= num_slots` contribute zero)."""
    if top_k_weights.numel() < module.offload_num_experts and module.gate_up_proj.is_floating_point():
        # Gathers only the routed slots: `grouped_mm` iterates every slot on the host where it has no device kernel
        # (sm < 90), and the pool has thousands of them
        return batched_mm_experts_forward(module, hidden_states, slots, top_k_weights)
    if not module.gate_up_proj.is_floating_point():  # quantized kernels index the pool directly
        return original_forward(hidden_states, slots, top_k_weights, **kwargs)
    # A prompt: the kernel over only the routed slots (a compact bank), not every slot of the pool
    used, compact = torch.unique(slots, return_inverse=True)
    used = used[used < module.num_experts]
    if not used.numel():  # every route is a miss: pass 2 computes them
        return torch.zeros_like(hidden_states)
    banks = {name: getattr(module, name) for name in ("gate_up_proj", "down_proj")}
    num_slots = module.num_experts
    try:
        for name, bank in banks.items():
            setattr(module, name, bank[used])
        module.num_experts = used.numel()  # sentinels (ids >= num_slots) sort last in `unique`: still >= used.numel()
        return original_forward(hidden_states, compact, top_k_weights, **kwargs)
    finally:
        for name, bank in banks.items():
            setattr(module, name, bank)
        module.num_experts = num_slots


@torch.library.custom_op("transformers::experts_offload_mps_route", mutates_args=())
def experts_offload_mps_route(
    handle: int, top_k_index: torch.Tensor, slots: torch.Tensor, guess: torch.Tensor
) -> None:
    """MPS: publish the routing, the slots pass 1 reads and the next layer's guess to the host thread."""
    module = _LAYERS[handle]
    module.offload_store.mps_router.route(module.offload_name, top_k_index, slots, guess)


@experts_offload_mps_route.register_fake
def _(handle, top_k_index, slots, guess) -> None:
    return None


@torch.library.custom_op("transformers::experts_offload_mps_wait", mutates_args=("state",))
def experts_offload_mps_wait(handle: int, state: list[torch.Tensor]) -> None:
    """MPS: the device waits until the host thread resolved the call (`state`: the pass-2 table and the banks)."""
    module = _LAYERS[handle]
    module.offload_store.mps_router.wait(module.offload_name)


@experts_offload_mps_wait.register_fake
def _(handle, state) -> None:
    return None


@torch.library.custom_op("transformers::experts_offload_route", mutates_args=(), tags=(torch.Tag.cudagraph_unsafe,))
def experts_offload_route(handle: int, hidden_states: torch.Tensor, top_k_index: torch.Tensor) -> None:
    """Start copying the layer's routing (and next-layer guesses) to the host, ahead of pass 1."""
    module = _LAYERS[handle]
    module.offload_store.start_routing(module.offload_name, hidden_states, top_k_index)


@experts_offload_route.register_fake
def _(handle, hidden_states, top_k_index) -> None:
    return None


@torch.library.custom_op(
    "transformers::experts_offload_misses", mutates_args=("output", "state"), tags=(torch.Tag.cudagraph_unsafe,)
)
def experts_offload_misses(
    handle: int,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    output: torch.Tensor,
    state: list[torch.Tensor],
) -> None:
    """Add the routes to non-resident experts into `output`, after reading and installing them (`state`: the slot
    table and pool banks it writes). Also starts the reads of the next layers."""
    module = _LAYERS[handle]
    module.offload_store.misses_pass(
        module.offload_name, module, module.offload_experts_forward, hidden_states, top_k_index, top_k_weights, output
    )


@experts_offload_misses.register_fake
def _(handle, hidden_states, top_k_index, top_k_weights, output, state) -> None:
    return None


class _ExpertReader:
    """The params read for one target of an experts module, one expert at a time: expert `e` is row `e` of each of the
    checkpoint's stacked banks (the transformers format), copied as is into a pool slot. Nothing is converted per read.
    A target can carry several banks: a quantized param loads with its serialized companions (`down_proj`,
    `down_proj_scales`, ...) through an identity converter."""

    def __init__(self, target_name, mapping, reads, hf_quantizer):
        from ..core_model_loading import WeightConverter, _IdentityOp

        converts = isinstance(mapping, WeightConverter) and not all(
            isinstance(op, _IdentityOp) for op in mapping.operations
        )
        needs_quantization = hf_quantizer is not None and getattr(mapping, "quantization_operation", None) is not None
        if converts or needs_quantization:
            raise ValueError(
                f"Streamed experts read each expert as one row of a stacked bank, but `{target_name}` would have to be "
                f"converted ({type(mapping).__name__}: {getattr(mapping, 'operations', None)}, quantize on load: "
                f"{needs_quantization}) or quantized on every read. Load the model once and save it in the transformers "
                "format: `model.save_pretrained(path, save_original_format=False)`, then stream from `path`."
            )
        # (param, bank slice, dtype, checkpoint key) per bank
        self.banks = [(key.rpartition(".")[2], bank, dtype, key) for ((bank, dtype, key),) in reads.values()]

    def sources(self, expert) -> list[tuple]:
        """`(param, slice, rows, dtype, checkpoint key)` of the expert's row in each bank."""
        return [(param, bank, slice(expert, expert + 1), dtype, key) for param, bank, dtype, key in self.banks]


class _RowReader:
    """Rows of a table the checkpoint stores as slices concatenated on dim 0 (one slice, or e.g. 128 shards), with an
    optional dequantization scale per shard (or one for all: an FP8 table's scalar `weight_scale_inv`)."""

    def __init__(self, reads: dict[str, list], device):
        entries = [entry for slices in reads.values() for entry in slices]
        # The table's own rank (e.g. 2-d rows); lower-rank sources are its dequantization scales (FP8 `weight_scale_inv`)
        rank = max(len(entry[0].get_shape()) for entry in entries)
        self.slices = [entry for entry in entries if len(entry[0].get_shape()) == rank]
        scales = [
            s for slice_, *_ in entries if len(slice_.get_shape()) < rank for s in slice_[...].flatten().tolist()
        ]
        if scales and len(scales) not in (1, len(self.slices)):
            raise ValueError(
                f"Cannot stream rows of {list(reads)}: {len(scales)} scales for {len(self.slices)} shards"
            )
        self.scales = scales  # a few values: read once
        widths = {tuple(slice_.get_shape()[1:]) for slice_, *_ in self.slices}
        if len(widths) != 1:
            raise ValueError(f"Cannot stream rows of {list(reads)}: their sources differ in row shape {widths}")
        self.offsets = [0]
        for slice_, *_ in self.slices:
            self.offsets.append(self.offsets[-1] + slice_.get_shape()[0])
        self.device = device

    def read(self, rows: list[int]) -> torch.Tensor:
        out = []
        for row in rows:
            shard = bisect.bisect_right(self.offsets, row) - 1
            slice_, dtype, _ = self.slices[shard]
            local = row - self.offsets[shard]
            values = slice_[local : local + 1].to(self.device)
            if self.scales:
                values = values.float() * self.scales[shard if len(self.scales) > 1 else 0]
            out.append(values.to(dtype))
        return torch.cat(out)


def _streamed_embedding_forward(store, name, module, model_dtype):
    # Rows come out in the compute dtype: the embedding's own, unless it is a quantized (e.g. FP8) one, then the model's
    weight = module.weight
    dtype = weight.dtype if weight.is_floating_point() and weight.element_size() > 1 else model_dtype

    @torch.compiler.disable
    def forward(input_ids):
        unique, inverse = torch.unique(input_ids.reshape(-1), return_inverse=True)
        rows = store.row_readers[name].read(unique.tolist()).to(store.device, dtype)
        return rows[inverse].view(*input_ids.shape, -1)

    return forward


def _route(router: nn.Module, hidden_states: torch.Tensor, top_k: int) -> torch.Tensor:
    """The experts `router` picks for `hidden_states`: its own routing (the integer output of its forward), else the
    top-k of its logits."""
    out = router(hidden_states)
    for tensor in out if isinstance(out, (tuple, list)) else (out,):
        if isinstance(tensor, torch.Tensor) and not tensor.is_floating_point():
            return tensor
    logits = out[0] if isinstance(out, (tuple, list)) else out
    return logits.topk(top_k, dim=-1).indices


def _find_router(parent: nn.Module, experts: nn.Module):
    """The router next to `experts` (its parent's child with a `(num_experts, hidden)` weight), if any."""
    for child in parent.children():
        weight = getattr(child, "weight", None)
        if child is not experts and isinstance(weight, torch.Tensor) and weight.dim() == 2:
            if weight.shape[0] == experts.num_experts:
                return child
    return None


def apply_experts_offload(model: nn.Module, patterns: list[str], device, cache_bytes: int, **kwargs):
    """Attach an `ExpertsOffloadStore` to `model` and an `ExpertsOffloadLayer` to every experts module matching
    `patterns` (globs over module names, e.g. `"experts"` or `"model.layers.*.mlp.experts"`)."""
    store = ExpertsOffloadStore(device, cache_bytes, **kwargs)
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if (
            getattr(module, "num_experts", None)
            and expert_params(module)
            and any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name, f"*.{p}") for p in patterns)
        ):
            module.offload_num_experts = module.num_experts
            store.add_layer(name, module, _find_router(modules[name.rpartition(".")[0]], module))
        elif isinstance(module, torch.nn.Embedding) and any(
            fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name, f"*.{p}") for p in patterns
        ):
            store.embeddings[name] = module
            store.params[name] = ["weight"]
            module.forward = _streamed_embedding_forward(store, name, module, model.dtype)
    if not store.layers and not store.embeddings:
        raise ValueError(f"No experts module or embedding matches {patterns}")
    # Tied params (e.g. `lm_head.weight`) are not tied yet on the meta model: count them once
    tied = set(getattr(model, "all_tied_weights_keys", None) or {})
    resident = sum(
        p.numel() * p.element_size() for n, p in model.named_parameters() if not store.owns(n) and n not in tied
    ) + sum(b.numel() * b.element_size() for b in model.buffers())
    store.resident_bytes = resident  # pools are sized at the first forward, from what a read returns
    style = ExpertsOffloadLayer()
    for module in store.layers.values():
        style.install_forward(module, store)
    model._experts_offload = store
    return store
