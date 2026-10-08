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

import safetensors.torch

from ..utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    is_torch_available,
    is_torch_greater_or_equal,
    logging,
)
from .sharding_utils import DtensorShardOperation, _dtensor_from_local_like
from .utils import (
    _check_distributed_checkpointing_available,
    _distributed_barrier,
    is_dtensor,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from torch.utils._pytree import tree_map

if _check_distributed_checkpointing_available():
    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader, HuggingFaceStorageWriter
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.placement_types import _StridedShard


def _prepare_state_dict_for_dcp(state_dict):
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

    """
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.7` with `torch.distributed` available.")

    def prepare(value):
        if is_dtensor(value) and any(isinstance(p, _StridedShard) for p in value.placements):
            placements = tuple(Shard(p.dim) if isinstance(p, _StridedShard) else p for p in value.placements)
            return value.redistribute(placements=placements)
        return value

    # tree_map allows to map a function on arbitrarily nested structures.
    return tree_map(prepare, state_dict)


def save_model_checkpoint_distributed(model, checkpoint_dir: str, *, consolidate: bool = True) -> None:
    """Save rank-local model shards as safetensors with DCP, optionally consolidating them.

    With `consolidate=True`, rank-local files are kept in `sharded/` and complete weights
    are written at the root. Otherwise, load the rank-local files with
    `load_distributed_checkpoint`; they are not `from_pretrained` checkpoints.
    """
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.7` with `torch.distributed` available.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    # DCP describes each DTensor as one rectangular chunk, which cannot represent strided shards.
    # We redistribute any strided shards to contiguous shards so DCP can write them out.
    # Sub-optimal compared to a future DCP that can write strided shards directly, but works for now.
    state_dict = _prepare_state_dict_for_dcp(get_model_state_dict(model))

    writer = HuggingFaceStorageWriter(
        path=checkpoint_dir,
        save_distributed=True,
        enable_consolidation=consolidate,
    )
    dcp.save(state_dict, storage_writer=writer)

    # All ranks wait until consolidated weights are ready for loading.
    _distributed_barrier()


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


def _distribute_tensor_for_load(tensor: torch.Tensor, destination: torch.Tensor):
    """Convert a full tensor into a DTensor matching destination's placement."""
    if not is_dtensor(destination):
        return tensor
    if tensor.shape != destination.shape:
        raise ValueError(f"Cannot load tensor of shape {tensor.shape} into destination of shape {destination.shape}")

    shard = DtensorShardOperation(destination).shard_tensor(tensor, device=destination.device, dtype=destination.dtype)
    return _dtensor_from_local_like(shard, destination)


def distribute_state_dict_for_load(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Convert a state dict of full tensors into a state dict of DTensors matching the destination's placements."""

    model_state_dict = model.state_dict()

    for name, tensor in state_dict.items():
        destination = model_state_dict.get(name)
        if is_dtensor(destination) and not is_dtensor(tensor):
            state_dict[name] = _distribute_tensor_for_load(tensor, destination)

    return state_dict


def _load_consolidated_checkpoint_in_distributed_model(
    model, checkpoint_files: str | os.PathLike | list[str | os.PathLike], strict: bool = True
):
    """
    Load one or more consolidated safetensors files into a distributed model, preserving its current mesh and
    placements.
    """
    if isinstance(checkpoint_files, (str, os.PathLike)):
        checkpoint_files = [checkpoint_files]
    state_dict = {}
    for checkpoint_file in checkpoint_files:
        state_dict.update(safetensors.torch.load_file(checkpoint_file, device="cpu"))
    distribute_state_dict_for_load(model, state_dict)
    model.load_state_dict(state_dict, strict=strict)


def _load_sharded_checkpoint_in_distributed_model(model, checkpoint_dir: str | os.PathLike, strict: bool = True):
    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict

    reader = HuggingFaceStorageReader(str(checkpoint_dir))

    original_state = get_model_state_dict(model)
    if any(value.is_meta for value in original_state.values() if isinstance(value, torch.Tensor)):
        raise ValueError("Materialize the model's tensors before loading a distributed checkpoint.")
    state = _prepare_state_dict_for_dcp(original_state)
    # `allow_partial_load=False` (i.e. strict) raises if a key in `state` (the model's own params) has no
    # matching entry in the checkpoint, checkpoint keys absent from `state` are always silently ignored.
    dcp.load(state, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=not strict))
    for name, value in state.items():
        if is_dtensor(value) and value.placements != original_state[name].placements:
            state[name] = value.redistribute(placements=original_state[name].placements)
    set_model_state_dict(model, state)


def load_model_checkpoint_distributed(model, checkpoint_dir: str | os.PathLike, strict: bool = True) -> None:
    """
    Load local safetensors weights into an initialized model, preserving its current mesh and placements.
    """
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.7` with `torch.distributed` available.")

    safe_index_file = os.path.join(checkpoint_dir, SAFE_WEIGHTS_INDEX_NAME)
    safe_weights_file = os.path.join(checkpoint_dir, SAFE_WEIGHTS_NAME)

    for dcp_dir in (checkpoint_dir, os.path.join(checkpoint_dir, "sharded")):
        if is_sharded_checkpoint(dcp_dir):
            _load_sharded_checkpoint_in_distributed_model(model, dcp_dir, strict=strict)
            return

    checkpoint_files = []
    if os.path.isfile(safe_index_file):
        with open(safe_index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        for shard_file in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(checkpoint_dir, shard_file)
            if not os.path.isfile(shard_path):
                raise ValueError(f"Shard file {shard_path} not found in {checkpoint_dir}.")
            checkpoint_files.append(shard_path)
    elif os.path.isfile(safe_weights_file):
        checkpoint_files.append(safe_weights_file)

    if checkpoint_files:
        _load_consolidated_checkpoint_in_distributed_model(model, checkpoint_files, strict=strict)
    else:
        raise ValueError(f"No distributed, sharded, or safetensors checkpoint found in {checkpoint_dir}.")
