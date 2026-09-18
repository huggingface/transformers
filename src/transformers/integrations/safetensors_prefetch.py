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
"""Load safetensors shards straight into CUDA memory with safetensors' prefetch engine.

Opt-in through `from_pretrained(..., prefetch=True)`. Each shard is opened with the `pread` backend on the
target device; its tensors are read and copied to the device in the background from the moment one of them
is first materialised, and every tensor is handed out once, as a zero-copy device tensor. The loader keeps
working on the same three calls it makes on a `safe_open.get_slice` object: `get_dtype`, `get_shape` and
indexing, so nothing downstream changes.
"""

import sys
import threading
from collections import OrderedDict

import torch
from safetensors import safe_open


def is_prefetch_available() -> bool:
    """safetensors >= 0.9.0rc1 exposes `safe_open.prefetch`; the engine needs Linux and a CUDA runtime."""
    return sys.platform == "linux" and torch.cuda.is_available() and hasattr(safe_open, "prefetch")


def prefetch_target_device(device_map) -> torch.device | None:
    """The single CUDA device every entry of `device_map` resolves to, or `None`.

    One shard handle loads onto one device, so the prefetch path is used only when the whole model lands on
    the same CUDA device (also the case for every rank of a tensor-parallel load).
    """
    if not device_map:
        return None
    devices = set()
    for value in device_map.values():
        if isinstance(value, int):
            value = f"cuda:{value}"
        try:
            device = torch.device(value)
        except (TypeError, RuntimeError):
            return None  # "disk" and friends
        if device.type != "cuda":
            return None
        index = device.index
        if index is None:
            index = torch.cuda.current_device() if torch.cuda.is_available() else 0
        devices.add(torch.device("cuda", index))
    return devices.pop() if len(devices) == 1 else None


class PrefetchedShard:
    """One open shard whose background load starts the first time a tensor of it is materialised.

    `copy_full` makes even whole-tensor reads compact copies. The engine allocates a shard in ranges that
    hold many neighbouring tensors and frees a range only when its last view is gone; on a rank that keeps
    just a slice of most tensors (tensor parallelism), a zero-copy view of a small replicated tensor would
    pin its whole range, so nothing is kept as a view there.
    """

    def __init__(self, file_pointer, copy_full: bool = False):
        self._file_pointer = file_pointer
        self.copy_full = copy_full
        self._lock = threading.Lock()
        self._started = False

    def meta(self, name: str):
        return self._file_pointer.get_tensor_meta(name)

    def tensor(self, name: str) -> torch.Tensor:
        with self._lock:
            if not self._started:
                self._file_pointer.prefetch()
                self._started = True
        return self._file_pointer.get_tensor(name)


class _MaterializedTensors:
    """The last few whole tensors handed out by the engine, so that repeated indexing of one source
    within a loading step (e.g. `_slice_and_cat` reading several intervals) reuses the same tensor.

    Bounded: a source that is only partially kept by this rank (tensor-parallel shards) must not stay
    resident once its step is done, and the loader keeps the proxies alive until the whole load ends.
    """

    def __init__(self, capacity: int = 16):
        self._items: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def get_or_load(self, key: int, load) -> torch.Tensor:
        with self._lock:
            tensor = self._items.get(key)
            if tensor is not None:
                self._items.move_to_end(key)
                return tensor
        tensor = load()
        with self._lock:
            self._items[key] = tensor
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return tensor


_materialized = _MaterializedTensors()


class PrefetchedTensor:
    """Stands in for `safe_open.get_slice(name)`.

    `get_dtype` and `get_shape` come from the header and never touch the data. Indexing materialises the
    whole tensor on the device (once per loading step, see `_MaterializedTensors`): `[...]` returns it as
    is, so a parameter created from it is a zero-copy view of the engine's allocation, unless the shard
    was opened with `copy_full`; a partial index always returns a compact copy.
    """

    def __init__(self, shard: PrefetchedShard, name: str):
        self._shard = shard
        self._name = name
        self._meta = shard.meta(name)

    def get_dtype(self) -> str:
        return self._meta.dtype

    def get_shape(self) -> list[int]:
        return list(self._meta.shape)

    def __getitem__(self, index) -> torch.Tensor:
        tensor = _materialized.get_or_load(id(self), lambda: self._shard.tensor(self._name))
        result = tensor[index]
        if self._shard.copy_full or result.numel() < tensor.numel():
            result = result.clone()
        return result
