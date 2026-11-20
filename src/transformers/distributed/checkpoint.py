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
import re
from concurrent.futures import ThreadPoolExecutor
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
    from safetensors.torch import save_file as safe_save_file
    from torch.utils._pytree import tree_map

    from ..core_model_loading import revert_weight_conversion

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

        1. The files are planned on meta tensors with the global shapes, so every rank gets the same plan without
           moving any data.
        2. File `i` is written by rank `i % world_size`, in rounds of `world_size` files: all ranks join the
           `full_tensor()` gathers of the round and each owner keeps its file on CPU. Then all owners revert the
           weight conversions and write their file in parallel. CPU memory per rank is bounded by one file.

    Returns the split of the (non-reverted) state dict and the weight map of the files that were written.
    """
    if not _is_torch_distributed_initialized():
        raise ValueError(
            "Saving a TP- or FSDP-sharded model requires torch.distributed to be initialized. "
            "Call save_pretrained from every rank after init_process_group."
        )
    rank = _get_torch_distributed_rank()
    world_size = _get_torch_distributed_world_size()

    meta_state_dict = {
        name: torch.empty(tensor.shape, dtype=tensor.dtype, device="meta") for name, tensor in state_dict.items()
    }
    state_dict_split = split_torch_state_dict_into_shards(
        meta_state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )
    files = list(state_dict_split.filename_to_tensors.items())

    # The final names of the weights in each file, after reverting the weight conversions.
    weight_map = {}
    for filename, names in files:
        if save_original_format:
            names = revert_weight_conversion(model, {name: meta_state_dict[name] for name in names})
        weight_map.update(dict.fromkeys(names, filename))

    with torch.no_grad():
        for round_start in range(0, len(files), world_size):
            round_files = files[round_start : round_start + world_size]
            # The files of the round are gathered in lockstep: at step `k`, all ranks join the gathers of the `k`-th
            # tensor of every file, then each owner copies its tensor to CPU. Gathering one file after the other
            # would make the copies run one rank at a time, since an owner blocked in `.cpu()` delays the gathers
            # of the next file.
            shard = {}
            for k in range(max(len(names) for _, names in round_files)):
                owned_tensor = None
                for owner, (_, names) in enumerate(round_files):
                    if k >= len(names):
                        continue
                    tensor = state_dict[names[k]]
                    # Collective: every rank must gather, even if it does not own the file.
                    if is_dtensor(tensor):
                        tensor = tensor.full_tensor()
                    if owner == rank:
                        owned_tensor = (names[k], tensor)
                if owned_tensor is not None:
                    name, tensor = owned_tensor
                    shard[name] = tensor.detach().cpu()

            if rank >= len(round_files):
                continue
            filename = round_files[rank][0]
            if save_original_format:
                shard = revert_weight_conversion(model, shard)
            expected_names = {name for name, file in weight_map.items() if file == filename}
            if set(shard) != expected_names:
                raise RuntimeError(
                    f"The weights reverted for {filename} do not match the ones planned on meta tensors: "
                    f"{sorted(set(shard) ^ expected_names)}."
                )
            shard = {name: tensor.contiguous() for name, tensor in shard.items()}
            safe_save_file(shard, os.path.join(save_directory, filename), metadata=metadata)
            del shard

    # The files are only complete once every owner is done writing.
    _distributed_barrier()
    return state_dict_split, weight_map


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
