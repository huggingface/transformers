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

import copy
import fnmatch
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

from ..utils import logging
from .tensor_parallel import TensorParallelLayer


if TYPE_CHECKING:
    from torch import nn


logger = logging.get_logger(__name__)

EXPERT_PARAMS = ("gate_up_proj", "down_proj")


def split_offloaded_experts(device_map):
    """Pop the `{"<experts module pattern>": "disk"}` entries of `device_map`; returns `(device_map, patterns)`."""
    if not isinstance(device_map, dict):
        return device_map, []
    patterns = [key for key, value in device_map.items() if value == "disk" and "experts" in key]
    rest = {key: value for key, value in device_map.items() if key not in patterns}
    if patterns and (not rest or "disk" in rest.values()):
        raise ValueError(
            'Streamed experts need the rest of the model on one device, e.g. {"": "mps", "experts": "disk"}'
        )
    return rest, patterns


class _Pool:
    """Device banks of `num_slots` experts of one layout, with one LRU over every layer that uses it."""

    def __init__(self, layout: dict[str, tuple[tuple[int, ...], torch.dtype]], num_slots: int):
        self.layout = layout
        self.num_slots = num_slots
        self.banks = None
        self.cache: OrderedDict[tuple[str, int], int] = OrderedDict()  # (layer, expert) -> slot, coldest first
        self.free = list(range(num_slots))

    def materialize(self, device):
        if self.banks is None:
            self.banks = {
                name: torch.empty(self.num_slots, *shape, dtype=dtype, device=device)
                for name, (shape, dtype) in self.layout.items()
            }

    def slot_for(self, key, pinned) -> int | None:
        """A free slot for `key`, evicting the least recently used unpinned expert; None if all are pinned."""
        if not self.free:
            victim = next((k for k in self.cache if k not in pinned), None)
            if victim is None:
                return None
            self.free.append(self.cache.pop(victim))
        slot = self.free.pop()
        self.cache[key] = slot
        return slot


class ExpertsOffloadStore:
    """Expert slot pools, checkpoint readers and the read schedule of every offloaded experts module."""

    def __init__(self, device, cache_bytes: int, io_threads: int = 4, lookahead: int = 1):
        self.device = torch.device(device)
        self.cache_bytes = cache_bytes
        self.lookahead = lookahead
        self.executor = ThreadPoolExecutor(io_threads, thread_name_prefix="experts-offload")
        self.layers: dict[str, nn.Module] = {}  # module name -> experts module, in forward order
        self.routers: dict[str, nn.Module | None] = {}  # module name -> its router, to guess its experts ahead
        self.order: list[str] = []
        self.readers: dict[str, list] = {}  # module name -> [read(expert) -> {param: tensor}]
        self.pools: dict[str, _Pool] = {}
        self.pool_of: dict[str, _Pool] = {}  # module name -> its pool
        self.pending = {}  # (layer, expert) -> Future of its device tensors
        self.hits = self.misses = self.reads = 0
        # Host time blocked on the routing copy (device sync) and on expert reads, see `stats`
        self.sync_seconds = self.wait_seconds = 0.0

    def stats(self) -> dict:
        return {
            "slots": sum(pool.num_slots for pool in self.pools.values()),
            "hits": self.hits,
            "misses": self.misses,
            "reads": self.reads,
            "sync_seconds": self.sync_seconds,
            "wait_seconds": self.wait_seconds,
        }

    def reset_stats(self):
        self.hits = self.misses = self.reads = 0
        self.sync_seconds = self.wait_seconds = 0.0

    def owns(self, param_name: str) -> bool:
        module_name, _, param = param_name.rpartition(".")
        return param in EXPERT_PARAMS and module_name in self.layers

    def add_layer(self, name: str, module: nn.Module, router: nn.Module | None):
        self.layers[name] = module
        self.routers[name] = router
        self.order.append(name)

    def allocate(self, resident_bytes: int = 0):
        """Split `cache_bytes - resident_bytes` between the expert layouts, in proportion to their layer count."""
        layouts = {}
        for name, module in self.layers.items():
            layout = {p: (tuple(getattr(module, p).shape[1:]), getattr(module, p).dtype) for p in EXPERT_PARAMS}
            layouts.setdefault(repr(layout), (layout, []))[1].append(name)
        budget = self.cache_bytes - resident_bytes
        for key, (layout, names) in layouts.items():
            expert_bytes = sum(torch.Size(shape).numel() * dtype.itemsize for shape, dtype in layout.values())
            total = sum(self.layers[name].num_experts for name in names)
            num_slots = min(total, int(budget * len(names) / len(self.layers)) // expert_bytes)
            if num_slots < 1:
                raise ValueError(f"No room for one expert in {budget} bytes of `max_memory` for streamed experts")
            self.pools[key] = pool = _Pool(layout, num_slots)
            for name in names:
                self.pool_of[name] = pool
        logger.info(f"Streaming experts through {[p.num_slots for p in self.pools.values()]} device slots")

    def add_reader(self, target_name: str, mapping, reads: dict[str, list], dtype=None):
        """Register how to read one expert of `target_name` from the checkpoint slices the loader collected.

        `mapping` is the `WeightConverter` / `WeightRenaming` of that target and `reads` its collected
        `{source pattern: [(slice, dtype), ...]}`. A converter whose ops merge per-expert tensors
        (`MergeModulelist`, then e.g. `Concatenate`) is replayed on one expert's tensors; a renaming (the checkpoint
        already holds the stacked bank) is sliced on its first dimension."""
        from ..core_model_loading import WeightConverter

        module_name = target_name.rpartition(".")[0]
        if isinstance(mapping, WeightConverter):

            def read(expert):
                one = copy.copy(mapping)
                one.collected_tensors = {
                    pattern: [lambda s=slices[expert]: s[0][...].to(self.device, s[1])]
                    for pattern, slices in reads.items()
                }
                out = one.convert(target_name)
                return {name.rpartition(".")[2]: tensor[0] for name, tensor in out.items()}
        else:
            ((bank, bank_dtype),) = next(iter(reads.values()))
            param = target_name.rpartition(".")[2]

            def read(expert):
                return {param: bank[expert : expert + 1].to(self.device, bank_dtype)[0]}

        self.readers.setdefault(module_name, []).append(read)

    def _read(self, layer, expert):
        self.reads += 1
        out = {}
        for read in self.readers[layer]:
            out.update(read(expert))
        return out

    def fetch(self, layer, expert):
        """Start reading one expert (once); returns its Future."""
        key = (layer, expert)
        if key not in self.pending:
            self.pending[key] = self.executor.submit(self._read, layer, expert)
        return self.pending[key]

    def prefetch(self, layer, experts):
        cache = self.pool_of[layer].cache
        for expert in experts:
            if (layer, expert) not in cache:
                self.fetch(layer, expert)

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

        def offload_forward(hidden_states, top_k_index, top_k_weights, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError("Streamed experts are inference-only: use torch.no_grad() / inference_mode()")
            pool = store.pool_of[name]
            if "gate_up_proj" in module._parameters:  # first call: the meta banks become the pool's
                _use_pool(module, pool, store.device)
            num_experts = module.offload_num_experts
            # 1. One host copy: this layer's routing and the guesses for the next layers
            routes = [top_k_index.reshape(-1)]
            ahead = store.next_layers(name)
            decode = top_k_index.numel() < num_experts
            if decode:
                for next_name in ahead:
                    router = store.routers[next_name]
                    if router is not None:
                        k = top_k_index.shape[-1]
                        logits = torch.nn.functional.linear(hidden_states, router.weight)
                        routes.append(logits.topk(k, dim=-1).indices.flatten())
            start = time.perf_counter()
            routes = torch.cat(routes).cpu()
            store.sync_seconds += time.perf_counter() - start
            routing = routes[: top_k_index.numel()].view(top_k_index.shape)
            experts = torch.unique(routing).tolist()
            hits = [e for e in experts if (name, e) in pool.cache]
            misses = [e for e in experts if (name, e) not in pool.cache]
            store.hits += len(hits)
            store.misses += len(misses)
            for e in hits:
                pool.cache.move_to_end((name, e))
            futures = [store.fetch(name, e) for e in misses]
            # 2. Reads for the next layers overlap this layer's compute
            if decode:
                offset = top_k_index.numel()
                for next_name in ahead:
                    if store.routers[next_name] is None:
                        continue
                    guesses = routes[offset : offset + top_k_index.numel()]
                    offset += top_k_index.numel()
                    store.prefetch(next_name, torch.unique(guesses).tolist())
            elif ahead:  # a prompt routes to nearly every expert: read the whole next layer
                store.prefetch(ahead[0], range(store.layers[ahead[0]].offload_num_experts))

            def run(selected):
                table = torch.full((num_experts,), pool.num_slots, dtype=top_k_index.dtype)
                table[selected] = torch.tensor([pool.cache[(name, e)] for e in selected], dtype=top_k_index.dtype)
                slots = table[routing].to(store.device, non_blocking=True)
                return original_forward(hidden_states, slots, top_k_weights, **kwargs)

            # 3. Pass 1 on the resident experts is queued before waiting on any read
            output = run(hits) if hits else torch.zeros_like(hidden_states)
            # 4. Install the misses as their reads land, as many as fit at a time
            while misses:
                batch = []
                for expert, future in zip(misses, futures):
                    slot = pool.slot_for((name, expert), {(name, e) for e in batch})
                    if slot is None:
                        break
                    start = time.perf_counter()
                    tensors = future.result()
                    store.wait_seconds += time.perf_counter() - start
                    for param, tensor in tensors.items():
                        pool.banks[param][slot].copy_(tensor, non_blocking=True)
                    store.pending.pop((name, expert), None)
                    if not decode:  # a prompt's experts go to the cold end, the decode set stays
                        pool.cache.move_to_end((name, expert), last=False)
                    batch.append(expert)
                output = output + run(batch)
                misses, futures = misses[len(batch) :], futures[len(batch) :]
            # Guesses for this layer that the routing did not pick
            for key in [key for key in store.pending if key[0] == name]:
                store.pending.pop(key).cancel()
            return output

        module.forward = offload_forward
        return module


def _find_router(parent: nn.Module, experts: nn.Module):
    """The router next to `experts` (its parent's child with a `(num_experts, hidden)` weight), if any."""
    for child in parent.children():
        weight = getattr(child, "weight", None)
        if child is not experts and isinstance(weight, torch.Tensor) and weight.dim() == 2:
            if weight.shape[0] == experts.num_experts:
                return child
    return None


def _use_pool(module, pool, device):
    """Swap the module's meta banks for the pool banks: its forward then indexes slots, not experts."""
    pool.materialize(device)
    for param in EXPERT_PARAMS:
        module._parameters.pop(param, None)
        setattr(module, param, pool.banks[param])
    module.num_experts = pool.num_slots
    module._is_expert_parallel = True


def apply_experts_offload(model: nn.Module, patterns: list[str], device, cache_bytes: int, **kwargs):
    """Attach an `ExpertsOffloadStore` to `model` and an `ExpertsOffloadLayer` to every experts module matching
    `patterns` (globs over module names, e.g. `"experts"` or `"model.layers.*.mlp.experts"`)."""
    store = ExpertsOffloadStore(device, cache_bytes, **kwargs)
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if all(hasattr(module, p) for p in EXPERT_PARAMS) and any(
            fnmatch.fnmatch(name, p) or fnmatch.fnmatch(name, f"*.{p}") for p in patterns
        ):
            module.offload_num_experts = module.num_experts
            store.add_layer(name, module, _find_router(modules[name.rpartition(".")[0]], module))
    if not store.layers:
        raise ValueError(f"No experts module matches {patterns}")
    # Tied params (e.g. `lm_head.weight`) are not tied yet on the meta model: count them once
    tied = set(getattr(model, "all_tied_weights_keys", None) or {})
    resident = sum(
        p.numel() * p.element_size() for n, p in model.named_parameters() if not store.owns(n) and n not in tied
    ) + sum(b.numel() * b.element_size() for b in model.buffers())
    store.allocate(resident)
    style = ExpertsOffloadLayer()
    for module in store.layers.values():
        style.install_forward(module, store)
    model._experts_offload = store
    return store
