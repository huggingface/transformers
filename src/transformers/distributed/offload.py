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
import copy
import fnmatch
import time
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

    def materialize(self, device):
        if self.banks is None:
            self.banks = {
                name: torch.empty(self.num_slots, *shape, dtype=dtype, device=device)
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

    def complete(self, layer, num_experts) -> bool:
        """Every expert of `layer` is resident and can never be evicted (its own pool, see `pin_layers`)."""
        return self.num_slots == num_experts and len(self.cache) == num_experts


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
        # disk -> RAM -> VRAM on CUDA: reads land in host memory, then pinned, then copy on their own stream so they
        # overlap compute. On MPS (unified memory) the loader's `pread` slices already land on the device.
        self.read_device = torch.device("cpu") if self.device.type == "cuda" else self.device
        self.copy_stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
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
        self.hits = self.misses = self.reads = self.covered = 0
        # Host time blocked on the routing copy (device sync) and on expert reads, see `stats`
        self.sync_seconds = self.wait_seconds = 0.0

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

        A layout is what a read returns (e.g. quantized on the fly), so one expert of each kind of layer is read."""
        layouts, probes = {}, {}
        for name, module in self.layers.items():
            kind = (type(module), tuple(getattr(module, p).shape for p in self.params[name]))
            if kind not in probes:
                tensors, _ = self._read(name, 0)
                probes[kind] = {p: (tuple(t.shape), t.dtype) for p, t in tensors.items()}
            layouts.setdefault(repr(probes[kind]), (probes[kind], []))[1].append(name)
        self.reads = 0
        budget = self.cache_bytes - self.resident_bytes
        # Pinned layers: an exclusive pool each, of all their experts
        for name in self.order[: self.pin_layers]:
            layout = next(layout for layout, names in layouts.values() if name in names)
            names = next(names for _, names in layouts.values() if name in names)
            names.remove(name)
            pool = _Pool(layout, self.layers[name].num_experts, self.policy)
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

    def add_reader(self, target_name: str, mapping, reads: dict[str, list], hf_quantizer=None):
        """Register how to read one expert of `target_name` from the checkpoint slices the loader collected.

        `mapping` is the `WeightConverter` / `WeightRenaming` of that target and `reads` its collected
        `{source pattern: [(slice, dtype), ...]}`. A converter whose ops merge per-expert tensors
        (`MergeModulelist`, then e.g. `Concatenate`) is replayed on one expert's tensors; a renaming (the checkpoint
        already holds the stacked bank) is sliced on its first dimension."""

        module_name = target_name.rpartition(".")[0]
        if module_name in self.embeddings:
            self.row_readers[module_name] = _RowReader(reads, self.read_device)
            return
        self.readers.setdefault(module_name, []).append(_ExpertReader(target_name, mapping, reads, hf_quantizer))

    def _read(self, layer, expert):
        """One expert's device tensors and the event their copy records (None when they were read on the device)."""
        self.reads += 1
        out = {}
        for reader in self.readers[layer]:
            tensors = [
                (slice_[...] if rows is None else slice_[rows]).to(self.read_device, dtype)
                for slice_, rows, dtype, _ in reader.sources(expert)
            ]
            out.update(reader.assemble(tensors))
        if self.copy_stream is None:
            return out, None
        with torch.cuda.stream(self.copy_stream):
            out = {name: tensor.pin_memory().to(self.device, non_blocking=True) for name, tensor in out.items()}
            copied = torch.cuda.Event()
            copied.record(self.copy_stream)
        return out, copied

    def fetch(self, layer, expert):
        """Start reading one expert (once); returns its Future."""
        key = (layer, expert)
        if key not in self.pending:
            self.pending[key] = self.executor.submit(self._read, layer, expert)
        return self.pending[key]

    def prefetch(self, layer, experts):
        cache = self.pool_of[layer].cache
        self.fetch_many(layer, [e for e in experts if (layer, e) not in cache])

    @property
    def use_prefetch(self) -> bool:
        """safetensors' prefetch loader (chunked parallel reads into a pinned pool, async copies): CUDA only."""
        return self.device.type == "cuda" and self.key_files is not None and hasattr(safe_open, "prefetch")

    def fetch_many(self, layer, experts):
        """Start reading `experts` of `layer` (each once); returns their Futures. On CUDA one prefetch plan per file
        covers them all."""
        new = [e for e in dict.fromkeys(experts) if (layer, e) not in self.pending]
        if new and self.use_prefetch:
            futures = {e: Future() for e in new}
            for e, future in futures.items():
                self.pending[(layer, e)] = future
            self.executor.submit(self._prefetch_batch, layer, futures)
        else:
            for e in new:
                self.fetch(layer, e)
        return [self.pending[(layer, e)] for e in experts]

    def _handle(self, file):
        if file not in self.handles:
            self.handles[file] = safe_open(file, framework="pt")
        return self.handles[file]

    def _prefetch_batch(self, layer, futures):
        try:
            # One loader per file (and per round when a key is wanted with several row ranges: stacked banks)
            rounds = []  # [{file: {key: rows}}]
            for expert in futures:
                for reader in self.readers[layer]:
                    for _, rows, _, key in reader.sources(expert):
                        file = self.key_files[key]
                        index = next(i for i, plan in enumerate(rounds + [{}]) if key not in plan.get(file, {}))
                        if index == len(rounds):
                            rounds.append({})
                        rounds[index].setdefault(file, {})[key] = rows
            loaders = {}
            for index, plan in enumerate(rounds):
                for file, keys in plan.items():
                    loaders[(index, file)] = self._handle(file).prefetch(keys, device=str(self.device), threads=16)
            taken = {}
            for expert, future in futures.items():
                self.reads += 1
                out = {}
                for reader in self.readers[layer]:
                    tensors = []
                    for _, rows, dtype, key in reader.sources(expert):
                        file = self.key_files[key]
                        index = taken.get((file, key), 0)
                        taken[(file, key)] = index + 1
                        tensors.append(loaders[(index, file)].take(key).to(dtype))
                    out.update(reader.assemble(tensors))
                future.set_result((out, None))
            for loader in loaders.values():
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
        module.num_experts = pool.num_slots
        module._is_expert_parallel = True
        if name in self.order[: self.pin_layers]:  # pinned: all its experts now, never a miss after
            futures = self.fetch_many(name, range(module.offload_num_experts))
            for expert, future in enumerate(futures):
                self._install(pool, pool.slot_for((name, expert), ()), future)
                self.pending.pop((name, expert), None)
            self._refresh_tables(pool)

    def _install(self, pool, slot, future):
        """Copy a read expert into `slot` once its read (and host-to-device copy) is done."""
        start = time.perf_counter()
        tensors, copied = future.result()
        self.wait_seconds += time.perf_counter() - start
        if copied is not None:  # the device waits for the copy, the host does not
            torch.cuda.current_stream(self.device).wait_event(copied)
        for param, tensor in tensors.items():
            if copied is not None:  # allocated on the copy stream, freed after this stream used it
                tensor.record_stream(torch.cuda.current_stream(self.device))
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
        if pool.complete(name, num_experts):
            self.hits += top_k_index.numel()
            return
        # 1. One host copy (pass 1 is already queued, the device works meanwhile): routing + next layers' guesses
        routes = [top_k_index.reshape(-1)]
        ahead = self.next_layers(name)
        decode = top_k_index.numel() < num_experts
        guessed = [n for n in ahead if decode and self.routers[n] is not None]
        for next_name in guessed:
            logits = torch.nn.functional.linear(hidden_states, self.routers[next_name].weight)
            routes.append(logits.topk(top_k_index.shape[-1], dim=-1).indices.flatten())
        start = time.perf_counter()
        routes = torch.cat(routes).cpu()
        self.sync_seconds += time.perf_counter() - start
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
        futures = self.fetch_many(name, misses)
        # 2. Reads for the next layers overlap this layer's compute
        for index, next_name in enumerate(guessed):
            guesses = routes[(index + 1) * top_k_index.numel() : (index + 2) * top_k_index.numel()]
            self.prefetch(next_name, torch.unique(guesses).tolist())
        if not decode and ahead:  # a prompt routes to nearly every expert: read the whole next layer
            self.prefetch(ahead[0], range(self.layers[ahead[0]].offload_num_experts))
        # 3. Install the misses as their reads land, as many as fit at a time, and add their routes
        while misses:
            batch = []
            for expert, future in zip(misses, futures):
                slot = pool.slot_for((name, expert), {(name, e) for e in batch})
                if slot is None:
                    break
                self._install(pool, slot, future)
                self.pending.pop((name, expert), None)
                if not decode:  # a prompt's experts go to the cold end, the decode set stays
                    pool.cache.move_to_end((name, expert), last=False)
                batch.append(expert)
            table = torch.full((num_experts,), pool.num_slots, dtype=top_k_index.dtype)
            table[batch] = torch.tensor([pool.cache[(name, e)] for e in batch], dtype=top_k_index.dtype)
            slots = table[routing].to(self.device, non_blocking=True)
            output.add_(_run_pool(module, original_forward, hidden_states, slots, top_k_weights, {}))
            misses, futures = misses[len(batch) :], futures[len(batch) :]
        self._refresh_tables(pool)
        # Guesses for this layer that the routing did not pick
        for key in [key for key in self.pending if key[0] == name]:
            self.pending.pop(key).cancel()

    def next_layers(self, layer):
        index = self.order.index(layer)
        return self.order[index + 1 : index + 1 + self.lookahead]


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
        handle = len(_LAYERS)
        _LAYERS.append((store, name, module, original_forward))
        setup = torch.compiler.disable(store.setup_layer)

        def offload_forward(hidden_states, top_k_index, top_k_weights, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError("Streamed experts are inference-only: use torch.no_grad() / inference_mode()")
            if not hasattr(module, "offload_table"):  # first call only: no graph break once set up
                setup(name)
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


_LAYERS: list = []  # op handle -> (store, module name, module, its experts forward)


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
    store, name, module, original_forward = _LAYERS[handle]
    store.misses_pass(name, module, original_forward, hidden_states, top_k_index, top_k_weights, output)


@experts_offload_misses.register_fake
def _(handle, hidden_states, top_k_index, top_k_weights, output, state) -> None:
    return None


class _ExpertReader:
    """How to read one expert of a target param: its checkpoint `sources` and how to `assemble` them.

    A converter whose ops merge per-expert tensors (`MergeModulelist`, then e.g. `Concatenate`) is replayed on one
    expert's tensors (and quantized on the fly if the plan says so); a renaming (the checkpoint already holds the
    stacked bank) is the expert's row of the bank."""

    def __init__(self, target_name, mapping, reads, hf_quantizer):
        from ..core_model_loading import WeightConverter

        self.target_name = target_name
        self.mapping = mapping if isinstance(mapping, WeightConverter) else None
        self.reads = reads  # {source pattern: [(slice, dtype, checkpoint key), ...]}
        self.hf_quantizer = hf_quantizer

    def sources(self, expert) -> list[tuple]:
        """`(slice, rows, dtype, checkpoint key)` per tensor to read, in pattern order."""
        if self.mapping is None:
            ((bank, dtype, key),) = next(iter(self.reads.values()))
            return [(bank, slice(expert, expert + 1), dtype, key)]
        return [(slices[expert][0], None, slices[expert][1], slices[expert][2]) for slices in self.reads.values()]

    def assemble(self, tensors) -> dict[str, torch.Tensor]:
        if self.mapping is None:
            return {self.target_name.rpartition(".")[2]: tensors[0][0]}
        one = copy.copy(self.mapping)
        one.collected_tensors = {pattern: [tensor] for pattern, tensor in zip(self.reads, tensors)}
        out = one.convert(self.target_name, hf_quantizer=self.hf_quantizer)  # quantized on the fly if planned
        return {name.rpartition(".")[2]: tensor[0] for name, tensor in out.items()}


class _RowReader:
    """Rows of a table the checkpoint stores as slices concatenated on dim 0 (one slice, or e.g. 128 shards)."""

    def __init__(self, reads: dict[str, list], device):
        ((_, slices),) = reads.items()
        self.slices = list(slices)
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
            out.append(slice_[local : local + 1].to(self.device, dtype))
        return torch.cat(out)


def _streamed_embedding_forward(store, name, module):
    @torch.compiler.disable
    def forward(input_ids):
        unique, inverse = torch.unique(input_ids.reshape(-1), return_inverse=True)
        rows = store.row_readers[name].read(unique.tolist()).to(store.device)
        return rows[inverse].view(*input_ids.shape, -1)

    return forward


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
            module.forward = _streamed_embedding_forward(store, name, module)
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
