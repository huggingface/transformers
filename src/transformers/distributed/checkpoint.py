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
from __future__ import annotations

import json
import os
import queue
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING

from huggingface_hub import split_torch_state_dict_into_shards

from ..utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    is_torch_available,
    is_torch_greater_or_equal,
    logging,
)
from .utils import (
    _check_distributed_checkpointing_available,
    _distributed_barrier,
    _get_torch_distributed_rank,
    _get_torch_distributed_world_size,
    _is_torch_distributed_initialized,
    is_dtensor,
)


if TYPE_CHECKING:
    from huggingface_hub import StateDictSplit

    from ...modeling_utils import PreTrainedModel
    from ..quantizers import HfQuantizer


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from safetensors.torch import save as safe_save
    from safetensors.torch import save_file as safe_save_file
    from torch.utils._pytree import tree_map

    from ..core_model_loading import get_reverse_conversion_groups, revert_weight_conversion

if _check_distributed_checkpointing_available():
    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader, HuggingFaceStorageWriter
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.placement_types import _StridedShard


def _redistribute_state_dict(state_dict, reference_state_dict: dict | None = None):
    """
    The DTensor hooks used by DCP to save/load a state dict are broken for `_StridedShard` placements.

    Example:
        Context:
            - Global tensor: [10, 11, 12, 13, 14, 15, 16, 17]
            - Placement: _StridedShard(dim=0, split_factor=2)
            - Mesh size:    2
            - Rank 0 local: [10, 11, 14, 15]

        Torch DCP machinery uses DTensor hooks (`__create_write_items__`, `__create_chunk_list__`, and
        `__get_tensor_shard__`) to save/load a state dict.

        The issue is that they expose the local storage as a single contiguous region, which would be saved
        as a single chunk:
        ```
            __create_write_items__:
                [WriteItem(name="weight", global_shape=[8], offset=[0], size=[4])]

            __create_chunk_list__:
                [ChunkStorageMetadata(offsets=[0], sizes=[4])]

            __get_tensor_shard__(MetadataIndex("weight", offset=[0])):
                [10, 11, 14, 15]
        ```
        While the data is correct, the chunk metadata is misleading because it implies that the local storage
        corresponds to a single contiguous region of the global tensor, which is not the case.

        The hooks should expose the local storage as two disjoint regions, which should be saved as two chunks:
        ```
            __create_write_items__:
                [WriteItem(name="weight", global_shape=[8], offset=[0], size=[2]),
                 WriteItem(name="weight", global_shape=[8], offset=[4], size=[2])]
            __create_chunk_list__:
                [ChunkStorageMetadata(offsets=[0], sizes=[2]),
                 ChunkStorageMetadata(offsets=[4], sizes=[2])]
            __get_tensor_shard__(MetadataIndex("weight", offset=[0])):
                [10, 11]
            __get_tensor_shard__(MetadataIndex("weight", offset=[4])):
                [14, 15]
        ```

        A solution would be to fix this machinary in PyTorch or by extending the DCP API to support it.
        Until then, one workaround is to redistribute the DTensors with `_StridedShard` placements to equivalent
        `Shard` placements before saving the state dict.

    If `reference_state_dict` is given, the DTensors are instead redistributed back to the placements of the matching
    DTensors in `reference_state_dict`, which undoes a previous call (e.g. after loading).
    """
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.8` with `torch.distributed` available.")

    def redistribute(value, reference=None):
        if not is_dtensor(value):
            return value
        if reference is not None:
            placements = reference.placements
        else:
            placements = tuple(Shard(p.dim) if isinstance(p, _StridedShard) else p for p in value.placements)
        if placements == value.placements:
            return value
        return value.redistribute(placements=placements)

    # tree_map allows to map a function on arbitrarily nested structures.
    if reference_state_dict is None:
        return tree_map(redistribute, state_dict)
    return tree_map(redistribute, state_dict, reference_state_dict)


def _save_model_checkpoint_distributed(model, checkpoint_dir: str | os.PathLike) -> None:
    """Save rank-local model shards as safetensors with DCP."""
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.8` with `torch.distributed` available.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    # DCP describes each DTensor as one rectangular chunk, which cannot represent strided shards.
    # We redistribute any strided shards to contiguous shards so DCP can write them out.
    # Sub-optimal compared to a future DCP that can write strided shards directly, but works for now.
    state_dict = _redistribute_state_dict(get_model_state_dict(model))

    writer = HuggingFaceStorageWriter(
        path=checkpoint_dir,
        save_distributed=True,
        enable_consolidation=False,
    )
    dcp.save(state_dict, storage_writer=writer)

    # All ranks wait until the checkpoint is ready for loading or consolidation.
    _distributed_barrier()


def save_sharded_state_dict(
    model,
    state_dict: dict[str, torch.Tensor],
    save_directory: str | os.PathLike,
    *,
    filename_pattern: str,
    max_shard_size: int | str,
    metadata: dict[str, str],
    save_original_format: bool,
) -> tuple[StateDictSplit, dict[str, str]]:
    """
    Write a TP- or FSDP-sharded `state_dict` as regular safetensors files, with every rank writing some of the files.
    Every rank must call this function.

        1. Everything is planned on meta tensors with the global shapes, so every rank gets the same plan without
           moving any data: the groups of weights that are reverted together, the reverted weights, their split
           into files of at most `max_shard_size`, and the byte offset of every weight in its file.
        2. File `i` is written by rank `i % world_size`.
        3. All ranks join the `full_tensor()` gathers of every group, in the same order. The owners of the files
           the group's weights go to revert the weight conversions on device and hand their weights to a background
           thread, which streams them through pinned buffers into their files, at their planned offsets.

    Returns the split of the reverted state dict and the weight map of the files that were written.
    """
    if not _is_torch_distributed_initialized():
        raise ValueError(
            "Saving a TP- or FSDP-sharded model requires torch.distributed to be initialized. "
            "Call save_pretrained from every rank after init_process_group."
        )
    rank = _get_torch_distributed_rank()
    world_size = _get_torch_distributed_world_size()

    def to_meta(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Fresh meta tensors: the reverted weights can be views of a shared storage, which would skew the file sizes.
        return {name: torch.empty(tensor.shape, dtype=tensor.dtype, device="meta") for name, tensor in tensors.items()}

    def revert(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return revert_weight_conversion(model, tensors) if save_original_format else tensors

    meta_state_dict = to_meta(state_dict)
    if save_original_format:
        groups = get_reverse_conversion_groups(model, meta_state_dict)
    else:
        groups = [[name] for name in meta_state_dict]
    planned_outputs = [to_meta(revert({name: meta_state_dict[name] for name in group})) for group in groups]
    reverted_state_dict = {name: tensor for outputs in planned_outputs for name, tensor in outputs.items()}

    state_dict_split = split_torch_state_dict_into_shards(
        reverted_state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )
    weight_map = state_dict_split.tensor_to_filename
    headers, locations = _plan_safetensors_files(state_dict_split.filename_to_tensors, reverted_state_dict, metadata)

    # A group whose weights go to files of several ranks is reverted by each of them.
    file_owner = {filename: index % world_size for index, filename in enumerate(headers)}
    group_owners = [{file_owner[weight_map[name]] for name in outputs} for outputs in planned_outputs]
    # The gathers go through the groups of every rank at the same pace in bytes, so that every rank's writer always
    # has work queued.
    order, queued_bytes = [], [0] * world_size
    for group_index, outputs in enumerate(planned_outputs):
        owner = min(group_owners[group_index], default=0)
        order.append((queued_bytes[owner], owner, group_index))
        queued_bytes[owner] += sum(tensor.nbytes for tensor in outputs.values())
    order = [group_index for _, _, group_index in sorted(order)]

    for filename, (header, size) in headers.items():
        if file_owner[filename] == rank:
            _create_safetensors_file(os.path.join(save_directory, filename), header, size)

    writer = _SafetensorsWriter(save_directory, locations)
    try:
        with torch.no_grad():
            for group_index in order:
                owners = group_owners[group_index]
                tensors = {}
                for name in groups[group_index]:
                    tensor = state_dict[name]
                    # Collective: every rank must gather, even if it does not own the group.
                    if is_dtensor(tensor):
                        tensor = tensor.full_tensor()
                    if rank in owners:
                        tensors[name] = tensor.detach()
                if rank not in owners:
                    continue
                outputs = revert(tensors)
                planned = planned_outputs[group_index]
                if {name: (t.shape, t.dtype) for name, t in outputs.items()} != {
                    name: (t.shape, t.dtype) for name, t in planned.items()
                }:
                    raise RuntimeError(
                        f"The weights reverted from {groups[group_index]} do not match the ones planned on meta tensors."
                    )
                for name, tensor in outputs.items():
                    if file_owner[weight_map[name]] == rank:
                        writer.submit(name, tensor)
                del tensors, outputs
    finally:
        writer.close()

    # The files are only complete once every rank is done writing.
    _distributed_barrier()
    return state_dict_split, weight_map


# Bytes of gathered weights a rank can queue for its writer thread before the gathers wait for it.
_MAX_PENDING_WRITE_BYTES = 2**30
# Size of each of the two pinned buffers that stream a device tensor to its file.
_STAGING_BUFFER_BYTES = 64 * 2**20


@lru_cache
def _safetensors_dtype(dtype: torch.dtype) -> str:
    # Serialize an empty tensor to read the dtype name safetensors uses, rather than relying on its private tables.
    data = safe_save({"tensor": torch.empty(0, dtype=dtype)})
    header_size = int.from_bytes(data[:8], "little")
    return json.loads(data[8 : 8 + header_size])["tensor"]["dtype"]


def _plan_safetensors_files(
    filename_to_tensors: dict[str, list[str]], tensors: dict[str, torch.Tensor], metadata: dict[str, str] | None
) -> tuple[dict[str, tuple[bytes, int]], dict[str, tuple[str, int]]]:
    """
    Build the header and size of every safetensors file from meta `tensors`, and the location (file and absolute
    byte offset) of every tensor. Like `safetensors`, larger dtypes come first so that every tensor is aligned on its dtype.
    """
    headers, locations = {}, {}
    for filename, names in filename_to_tensors.items():
        names = sorted(names, key=lambda name: (-tensors[name].element_size(), name))
        header = {"__metadata__": metadata} if metadata else {}
        offsets, offset = {}, 0
        for name in names:
            tensor = tensors[name]
            header[name] = {
                "dtype": _safetensors_dtype(tensor.dtype),
                "shape": list(tensor.shape),
                "data_offsets": [offset, offset + tensor.nbytes],
            }
            offsets[name] = offset
            offset += tensor.nbytes
        header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        # The data starts on an 8-byte boundary, as in files written by `safetensors`.
        header += b" " * (-len(header) % 8)
        header = len(header).to_bytes(8, "little") + header
        headers[filename] = (header, len(header) + offset)
        locations.update({name: (filename, len(header) + tensor_offset) for name, tensor_offset in offsets.items()})
    return headers, locations


def _open_for_writing(path: str, flags: int = 0) -> int:
    # `O_BINARY` only exists, and is needed, on Windows.
    return os.open(path, os.O_WRONLY | getattr(os, "O_BINARY", 0) | flags)


def _write_at(fd: int, data: memoryview, offset: int) -> None:
    # Each file descriptor is only used by one thread, so seeking then writing is safe and also works on Windows.
    os.lseek(fd, offset, os.SEEK_SET)
    while data:
        data = data[os.write(fd, data) :]


def _create_safetensors_file(path: str, header: bytes, size: int) -> None:
    fd = _open_for_writing(path, os.O_CREAT | os.O_TRUNC)
    try:
        _write_at(fd, memoryview(header), 0)
        os.ftruncate(fd, size)
    finally:
        os.close(fd)


class _SafetensorsWriter:
    """
    Writes tensors at their planned location from a background thread, so that the gathers of the next weights do
    not wait for the copies to CPU and the writes. A device tensor is copied through two reused pinned buffers: one is
    written to the file while the next chunk is copied into the other.
    """

    def __init__(self, directory: str | os.PathLike, locations: dict[str, tuple[str, int]]):
        self._directory = directory
        self._locations = locations
        self._fds: dict[str, int] = {}
        self._queue: queue.Queue = queue.Queue()
        self._pending_bytes = 0
        self._condition = threading.Condition()
        self._error: BaseException | None = None
        self._stream = None
        self._buffers = None
        self._thread = threading.Thread(target=self._run, name="safetensors-writer", daemon=True)
        self._thread.start()

    def submit(self, name: str, tensor: torch.Tensor) -> None:
        # Flat bytes, made on the current stream so the writer only has to wait for it.
        data = tensor.contiguous().reshape(-1).view(torch.uint8)
        with self._condition:
            while (
                self._error is None
                and self._pending_bytes > 0
                and self._pending_bytes + data.numel() > _MAX_PENDING_WRITE_BYTES
            ):
                self._condition.wait()
            if self._error is not None:
                raise RuntimeError("Writing the safetensors files failed.") from self._error
            self._pending_bytes += data.numel()
        ready = None
        if data.device.type == "cuda":
            ready = torch.cuda.Event()
            ready.record()
        self._queue.put((name, data, ready))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        for fd in self._fds.values():
            os.close(fd)
        if self._error is not None:
            raise RuntimeError("Writing the safetensors files failed.") from self._error

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            name, data, ready = item
            try:
                if self._error is None:
                    filename, offset = self._locations[name]
                    if filename not in self._fds:
                        self._fds[filename] = _open_for_writing(os.path.join(self._directory, filename))
                    self._write(self._fds[filename], data, offset, ready)
            except BaseException as error:
                self._error = error
            with self._condition:
                self._pending_bytes -= data.numel()
                self._condition.notify_all()
            del item, data, ready

    def _write(self, fd: int, data: torch.Tensor, offset: int, ready) -> None:
        if data.device.type != "cuda":
            _write_at(fd, memoryview(data.cpu().numpy()), offset)
            return
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=data.device)
            self._buffers = [torch.empty(_STAGING_BUFFER_BYTES, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        copied = [torch.cuda.Event(), torch.cuda.Event()]
        previous = None
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(ready)
            for index, start in enumerate(range(0, data.numel(), _STAGING_BUFFER_BYTES)):
                chunk = data[start : start + _STAGING_BUFFER_BYTES]
                # The buffer is free: its previous chunk was written before this loop iteration.
                buffer = self._buffers[index % 2][: chunk.numel()]
                buffer.copy_(chunk, non_blocking=True)
                copied[index % 2].record()
                # Write the previous chunk while this one is copied.
                if previous is not None:
                    previous_buffer, previous_start, previous_event = previous
                    previous_event.synchronize()
                    _write_at(fd, memoryview(previous_buffer.numpy()), offset + previous_start)
                previous = (buffer, start, copied[index % 2])
        if previous is not None:
            previous_buffer, previous_start, previous_event = previous
            previous_event.synchronize()
            _write_at(fd, memoryview(previous_buffer.numpy()), offset + previous_start)


def consolidate_distributed_checkpoint(
    checkpoint_dir: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
    max_shard_size: int | str = "50GB",
    model: PreTrainedModel = None,
    num_workers: int | None = None,
) -> None:
    """
    Merge the rank-local safetensors files of a distributed checkpoint into regular safetensors files of at most
    `max_shard_size` each, like `save_pretrained` writes them. Call it from a single process, or
    from every rank of an initialized process group: file `i` is then written by rank `i % world_size`.

    `model` is the model the checkpoint was saved from: its weight conversions are reverted and its tied weights
    are dropped. Without it, the tensors are written as they are stored, which might not be the expected format.

    Each process writes its files in parallel with `num_workers` threads, one per file by default. Each thread holds
    one file in CPU memory, so the memory used per process is about `num_workers * max_shard_size`.
    """
    if not _check_distributed_checkpointing_available() or not is_torch_greater_or_equal("2.9"):
        raise OSError(
            "Consolidating a distributed checkpoint requires `torch>=2.9` with `torch.distributed` available."
        )

    # Import here because otherwise it emits a warning every time it's imported on some hardware
    import torch.distributed.checkpoint as dcp

    from ..modeling_utils import remove_tied_weights_from_state_dict

    if output_dir is None:
        output_dir = checkpoint_dir

    if model is None:
        logger.warning(
            "No `model` was given to `consolidate_distributed_checkpoint`: the weight conversions are not reverted, so "
            "the consolidated checkpoint might not be in the expected format. Pass the model the checkpoint was saved "
            "from to get the original format."
        )

    reader = HuggingFaceStorageReader(str(checkpoint_dir))
    # Plan the files on meta tensors with the global shapes, like `save_pretrained` does.
    meta_state_dict = {
        name: torch.empty(entry.size, dtype=entry.properties.dtype, device="meta")
        for name, entry in reader.read_metadata().state_dict_metadata.items()
    }
    if model is not None:
        # Drop the tied weights (e.g. `lm_head`), found through the parameters of the model.
        meta_state_dict = remove_tied_weights_from_state_dict(meta_state_dict, model)
    state_dict_split = split_torch_state_dict_into_shards(
        meta_state_dict,
        filename_pattern=SAFE_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors"),
        max_shard_size=max_shard_size,
    )

    def write_file(filename: str, names: list[str]) -> None:
        shard = {name: torch.empty(meta_state_dict[name].shape, dtype=meta_state_dict[name].dtype) for name in names}
        # Only the tensors in `shard` are read from the checkpoint. One reader per thread, as readers keep state.
        dcp.load(shard, storage_reader=HuggingFaceStorageReader(str(checkpoint_dir)), no_dist=True)
        if model is not None:
            shard = revert_weight_conversion(model, shard)
        shard = {name: tensor.contiguous() for name, tensor in shard.items()}
        safe_save_file(shard, os.path.join(output_dir, filename), metadata={"format": "pt"})

    rank, world_size = 0, 1
    if _is_torch_distributed_initialized():
        rank, world_size = _get_torch_distributed_rank(), _get_torch_distributed_world_size()
    files = list(state_dict_split.filename_to_tensors.items())
    owned_files = files[rank::world_size]

    os.makedirs(output_dir, exist_ok=True)
    if owned_files:
        with ThreadPoolExecutor(max_workers=num_workers or len(owned_files)) as pool:
            # `result()` raises the exception of a failed file.
            for future in [pool.submit(write_file, filename, names) for filename, names in owned_files]:
                future.result()

    if rank == 0 and state_dict_split.is_sharded:
        # The final names are known without the data: revert the conversions on the meta tensors of each file.
        weight_map = {}
        for filename, names in files:
            if model is not None:
                names = revert_weight_conversion(model, {name: meta_state_dict[name] for name in names})
            weight_map.update(dict.fromkeys(names, filename))
        index = {"metadata": state_dict_split.metadata, "weight_map": weight_map}
        with open(os.path.join(output_dir, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")

    # The checkpoint is only complete once every rank is done writing.
    if world_size > 1:
        _distributed_barrier()


def is_sharded_checkpoint(checkpoint_dir: str | os.PathLike) -> bool:
    """Return True if the checkpoint directory contains sharded safetensors files."""
    pattern = r"^shard-[0-9]{5}-model-[0-9]{5}-of-[0-9]{5}\.safetensors$"
    if not os.path.isdir(checkpoint_dir):
        return False
    return any(re.match(pattern, name) for name in os.listdir(checkpoint_dir))


def _load_model_checkpoint_distributed(model, checkpoint_dir: str | os.PathLike, strict: bool = True) -> None:
    """Load rank-local DCP files from checkpoint_dir into an initialized model, preserving its mesh and placements."""
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.8` with `torch.distributed` available.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

    reader = HuggingFaceStorageReader(str(checkpoint_dir))

    original_state = get_model_state_dict(model)
    if any(value.is_meta for value in original_state.values() if isinstance(value, torch.Tensor)):
        raise ValueError("Materialize the model's tensors before loading a distributed checkpoint.")
    state = _redistribute_state_dict(original_state)
    # `allow_partial_load=False` (i.e. strict) raises if a key in `state` (the model's own params) has no
    # matching entry in the checkpoint, checkpoint keys absent from `state` are always silently ignored.
    dcp.load(state, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=not strict))
    state = _redistribute_state_dict(state, reference_state_dict=original_state)
    set_model_state_dict(model, state)


def save_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Save optimizer state via DCP. All ranks must call."""
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.8` with `torch.distributed` available.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    # Flatten optimizer state and group options into fully qualified name entries.
    # A mesh change can alter which parameters are DTensors and therefore how optimizer groups are constructed, fully
    # qualified name keys make the checkpoint independent of grouping.
    options = StateDictOptions(flatten_optimizer_state_dict=True)
    optimizer_state_dict = _redistribute_state_dict(get_optimizer_state_dict(model, optimizer, options=options))
    dcp.save({"optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir)


def load_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Load optimizer state from a DCP directory."""
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.8` with `torch.distributed` available.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    # Use the same fully qualified name keys format as for saving so PyTorch can map the checkpoint into the destination
    # optimizer's current groups, even when a mesh change has altered their structure.
    options = StateDictOptions(flatten_optimizer_state_dict=True)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer, options=options)
    checkpoint_state_dict = _redistribute_state_dict(optimizer_state_dict)
    dcp.load({"optimizer": checkpoint_state_dict}, checkpoint_id=checkpoint_dir)
    optimizer_state_dict = _redistribute_state_dict(checkpoint_state_dict, reference_state_dict=optimizer_state_dict)
    set_optimizer_state_dict(model, optimizer, optimizer_state_dict)


def get_distributed_checkpoint_dir(
    pretrained_model_name_or_path: str | os.PathLike | None,
    subfolder: str,
    gguf_file: str | None,
    state_dict: dict | None,
) -> str | os.PathLike | None:
    if pretrained_model_name_or_path is None or gguf_file is not None or state_dict is not None:
        return None

    candidate_dir = os.path.join(pretrained_model_name_or_path, subfolder)
    if is_sharded_checkpoint(candidate_dir):
        return candidate_dir
    return None


def _get_resolved_distributed_checkpoint_files(
    distributed_checkpoint_dir: str | os.PathLike,
    hf_quantizer: HfQuantizer | None = None,
) -> list[str]:
    if not _check_distributed_checkpointing_available():
        raise OSError("Loading a distributed checkpoint requires torch>=2.8.")
    if hf_quantizer is not None:
        raise ValueError("Quantization is not supported when loading a distributed checkpoint.")
    return sorted(
        os.path.join(distributed_checkpoint_dir, name)
        for name in os.listdir(distributed_checkpoint_dir)
        if name.endswith(".safetensors")
    )
