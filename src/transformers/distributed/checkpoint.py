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

import os
import re
from typing import TYPE_CHECKING

from huggingface_hub import split_torch_state_dict_into_shards

from ..utils import (
    is_torch_available,
    is_torch_greater_or_equal,
    logging,
)
from ..utils.hub import convert_file_size_to_int
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

    from ..quantizers import HfQuantizer


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from safetensors.torch import save_file as safe_save_file
    from torch.utils._pytree import tree_map

    from ..core_model_loading import _revert_weight_conversion_per_transform, revert_weight_conversion

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
           moving any data: the groups of weights reverted together by one weight conversion, the reverted weights,
           and their split into files. The files are at most `max_shard_size`, and for large models at most a
           `world_size`-th of the model, so that there are at least as many files as ranks.
        2. File `i` is written by rank `i % world_size`, in rounds of `world_size` files: all ranks join the
           `full_tensor()` gathers of the groups of the round, and each owner reverts the groups of its file and
           keeps the weights on CPU. Then all owners write their file in parallel. CPU memory per rank is bounded by
           one file.

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

    # The split is planned on the reverted weights, so that the weights reverted together by a conversion never
    # have to be in the same file.
    meta_state_dict = to_meta(state_dict)
    if save_original_format:
        reverted_groups = _revert_weight_conversion_per_transform(model, meta_state_dict)
    else:
        reverted_groups = [([name], {name: tensor}) for name, tensor in meta_state_dict.items()]
    groups = [input_names for input_names, _ in reverted_groups]
    planned_outputs = [to_meta(reverted) for _, reverted in reverted_groups]
    reverted_state_dict = {name: tensor for outputs in planned_outputs for name, tensor in outputs.items()}

    # At least one file per rank, so that every rank writes: the files are at most a `world_size`-th of the model.
    # `max_shard_size` is only an upper bound, so smaller files still honor it. Small models keep their files, since
    # writing them takes little time anyway.
    total_bytes = sum(tensor.nbytes for tensor in reverted_state_dict.values())
    file_size = min(
        convert_file_size_to_int(max_shard_size),
        max(-(-total_bytes // world_size), _MIN_CAPPED_FILE_BYTES),
    )
    state_dict_split = split_torch_state_dict_into_shards(
        reverted_state_dict, filename_pattern=filename_pattern, max_shard_size=file_size
    )
    weight_map = state_dict_split.tensor_to_filename
    files = list(state_dict_split.filename_to_tensors)

    # The groups each file needs, in the order of their first weight in the file. A group whose weights go to
    # several files is gathered and reverted for each of them.
    output_group = {name: index for index, outputs in enumerate(planned_outputs) for name in outputs}
    file_groups = {filename: [] for filename in files}
    for filename, names in state_dict_split.filename_to_tensors.items():
        for name in names:
            if output_group[name] not in file_groups[filename]:
                file_groups[filename].append(output_group[name])

    with torch.no_grad():
        for round_start in range(0, len(files), world_size):
            round_files = files[round_start : round_start + world_size]
            # The files of the round are gathered in lockstep: at step `k`, all ranks join the gathers of the `k`-th
            # group of every file, then each owner reverts its group and copies it to CPU. Gathering one file after
            # the other would make the copies run one rank at a time, since an owner blocked in `.cpu()` delays the
            # gathers of the next file.
            shard = {}
            for k in range(max(len(file_groups[filename]) for filename in round_files)):
                owned_group = None
                for owner, filename in enumerate(round_files):
                    if k >= len(file_groups[filename]):
                        continue
                    group_index = file_groups[filename][k]
                    tensors = {}
                    for name in groups[group_index]:
                        tensor = state_dict[name]
                        # Collective: every rank must gather, even if it does not own the file.
                        if is_dtensor(tensor):
                            tensor = tensor.full_tensor()
                        if owner == rank:
                            tensors[name] = tensor.detach()
                    if owner == rank:
                        owned_group = (filename, group_index, tensors)
                if owned_group is None:
                    continue
                filename, group_index, tensors = owned_group
                outputs = revert(tensors)
                planned = planned_outputs[group_index]
                if {name: (t.shape, t.dtype) for name, t in outputs.items()} != {
                    name: (t.shape, t.dtype) for name, t in planned.items()
                }:
                    raise RuntimeError(
                        f"The weights reverted from {groups[group_index]} do not match the ones planned on meta tensors."
                    )
                for name, tensor in outputs.items():
                    if weight_map[name] == filename:
                        shard[name] = tensor.contiguous().cpu()
                del tensors, outputs

            if rank < len(round_files):
                safe_save_file(shard, os.path.join(save_directory, round_files[rank]), metadata=metadata)
            del shard

    # The files are only complete once every owner is done writing.
    _distributed_barrier()
    return state_dict_split, weight_map


# When the files are made smaller to give every rank its own file, they are not made smaller than this.
_MIN_CAPPED_FILE_BYTES = 2**30


def consolidate_distributed_checkpoint(
    checkpoint_dir: str | os.PathLike, output_dir: str | os.PathLike | None = None
) -> None:
    """Consolidate rank-local safetensors files. Call from a single process or rank."""
    if not _check_distributed_checkpointing_available() or not is_torch_greater_or_equal("2.9"):
        raise OSError(
            "Consolidating a distributed checkpoint requires `torch>=2.9` with `torch.distributed` available."
        )

    from torch.distributed.checkpoint._consolidate_hf_safetensors import consolidate_safetensors_files

    if output_dir is None:
        output_dir = checkpoint_dir

    metadata = HuggingFaceStorageReader(str(checkpoint_dir)).read_metadata()
    os.makedirs(output_dir, exist_ok=True)
    consolidate_safetensors_files(
        input_dir=str(checkpoint_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping=dict.fromkeys(metadata.state_dict_metadata, 1),
    )


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
