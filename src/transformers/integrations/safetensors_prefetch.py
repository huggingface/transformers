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
"""Load checkpoints on device with safetensors' prefetch loader.

The default mmap loader lazily loads weights when `__getitem__` (`tensor[...]`) is called, and the pipeline looks
like so:

    mmapped disk region
      -> page fault I/O
      -> kernel & user shared buffer (the page cache, mapped into the process)
      -> `.to(device)`
           -> the CUDA driver copies it to its own pinned RAM buffer
           -> `cudaMemcpy` from pinned host buffer to device buffer

The new prefetch loader, instead of lazily loading when slicing the tensor, loads in the background. Calls to
`take` yield the bytes or wait for them to be loaded, but the rest of loading keeps going while bytes load. You
can precisely define which tensors, and which rows of them, you want when you call `prefetch` on the `safe_open`
handle. The prefetch pipeline looks like so:

    chunked disk `pread`
      -> page cache
      -> copy to a process-wide shared pinned memory slab
      -> `cudaMemcpyAsync`
      -> device buffer

The copies are the same, but the bytes move differently: large reads (16 MiB chunks) on several threads, with
several files in flight, instead of page faults as each tensor is sliced. The other thing that makes this really
fast is the fact that the disk read part is decoupled from the device copy. Both happen concurrently, which makes
sure that the disk read part of the pipeline is never waiting for bytes to have finished copying before reading.

Calling `take` on a prefetch loader either returns instantly if the bytes are loaded on device or waits until
loading for that given tensor/slice is finished.

How transformers uses it: `from_pretrained` first works out, for every checkpoint tensor, which of the model's
weights it becomes, on which device, and which rows this process keeps (one process per GPU under tensor
parallelism). Each checkpoint file then gets a `FilePrefetch`, which starts one prefetch loader per device with
exactly those tensors and rows, the first time one of them is needed. Tensors that don't go to a CUDA device, and
every tensor of a file whose loaders fail to start, are read the usual way from the same handle.
"""

import sys
import threading
from collections import defaultdict
from functools import partial

import torch
from safetensors import SafetensorError, safe_open

from ..distributed.sharding_utils import read_intervals
from ..utils import is_rocm_platform, logging


logger = logging.get_logger(__name__)


def is_prefetch_available() -> bool:
    """Needs safetensors >= 0.9.0rc1 (`safe_open.prefetch`), Linux and CUDA (not ROCm)."""
    return (
        sys.platform == "linux"
        and torch.cuda.is_available()
        and not is_rocm_platform()
        and hasattr(safe_open, "prefetch")
    )


def cuda_device(device) -> torch.device | None:
    """Normalize a `device_map` value (int, str or `torch.device`) to an indexed CUDA device, `None` if it isn't
    one."""
    try:
        device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    except (TypeError, RuntimeError):
        return None
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.device("cuda", index)


def has_cuda_target(device_map) -> bool:
    return bool(device_map) and any(cuda_device(device) is not None for device in device_map.values())


def should_prefetch(load_config, state_dict, checkpoint_files) -> bool:
    """Whether `from_pretrained` loads with prefetch (see its `prefetch` argument). Logs the decision."""
    if load_config.prefetch is False:
        return False
    applies = (
        state_dict is None
        and bool(checkpoint_files)
        and checkpoint_files[0].endswith(".safetensors")
        and (load_config.hf_quantizer is None or load_config.hf_quantizer.pre_quantized)
        and is_prefetch_available()
        and has_cuda_target(load_config.device_map)
    )
    if applies:
        logger.info("Loading checkpoint files with safetensors prefetch")
    elif load_config.prefetch:
        logger.warning(
            "prefetch=True was requested but does not apply to this load (see the `prefetch` argument of "
            "`from_pretrained`); loading with the default loader."
        )
    return applies


def attach_prefetch(loads, handles: dict, copy_full: bool) -> list["FilePrefetch"]:
    """Read every load going to a CUDA device through a prefetch loader of its file. `handles` maps checkpoint
    keys to the `safe_open` handle of their file."""
    by_file = defaultdict(list)
    for load in loads:
        handle = handles.get(load.source_key)
        device = cuda_device(load.device)
        if load.owned and handle is not None and device is not None:
            by_file[id(handle)].append((load, device))
    return [
        FilePrefetch(handles[file_loads[0][0].source_key], file_loads, copy_full) for file_loads in by_file.values()
    ]


class FilePrefetch:
    """The prefetch loaders of one checkpoint file, one per CUDA device its tensors go to.

    They start the first time one of the file's tensors is taken, so files start in the order the loader needs
    them. If they fail to start, the file's tensors are read as usual.

    With `copy_full` (tensor parallelism), every take is a copy rather than a view of safetensors' memory. That
    memory is freed per range of neighbouring tensors once no view of it is left, and a process under tensor
    parallelism keeps only slices of most tensors, so keeping views would keep whole ranges alive for as long as
    the model exists.
    """

    def __init__(self, handle, loads, copy_full: bool):
        self._handle = handle
        self._copy_full = copy_full
        self._plans: dict[torch.device, dict[str, slice | None]] = defaultdict(dict)
        self._loaders: dict | None = None
        self._lock = threading.Lock()
        for load, device in loads:
            rows = self._planned_rows(load)
            self._plans[device][load.source_key] = rows
            load.reader = partial(self.take, device, rows)

    def _planned_rows(self, load) -> slice | None:
        """The rows this rank keeps when they're one contiguous run, `None` to read the whole tensor."""
        if not load.intervals or len(load.intervals[0]) != 1:
            return None
        start, end = load.intervals[0][0]
        if (start, end) == (0, self._handle.get_tensor_meta(load.source_key).shape[0]):
            return None
        return slice(start, end)

    def take(self, device: torch.device, rows: slice | None, load) -> torch.Tensor | None:
        """`load`'s tensor from its loader, `None` if prefetch isn't running."""
        with self._lock:
            if self._loaders is None:
                self._loaders = self._start()
        loader = self._loaders.get(device)
        if loader is None:
            return None
        tensor = loader.take(load.source_key)
        intervals = load.intervals
        if rows is not None:
            # we only got these rows, shift the intervals to match
            intervals = [[(s - rows.start, e - rows.start) for s, e in intervals[0]], *intervals[1:]]
        result = read_intervals(lambda index: tensor[index], intervals)
        shares_storage = result.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        if shares_storage and (self._copy_full or result.numel() < tensor.numel()):
            result = result.clone()  # a view would keep the whole range alive
        return result.to(dtype=load.dtype)  # already on its device

    def _start(self) -> dict:
        loaders = {}
        try:
            for device, plan in self._plans.items():
                loaders[device] = self._handle.prefetch(plan, device=str(device))
        except SafetensorError as e:
            for loader in loaders.values():
                loader.close()
            logger.warning_once(f"safetensors prefetch failed to start ({e}), using the default loader")
            return {}
        return loaders

    def close(self) -> None:
        for loader in (self._loaders or {}).values():
            loader.close()
