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

from ..utils import (
    is_torch_available,
    logging,
)
from .utils import (
    _check_distributed_checkpointing_available,
    _distributed_barrier,
    _get_torch_distributed_rank,
    is_dtensor,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from torch.utils._pytree import tree_map

if _check_distributed_checkpointing_available(raise_if_not=False):
    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader, HuggingFaceStorageWriter
    from torch.distributed.tensor import Shard, distribute_tensor
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

        The DTensor hooks expose the local storage as a single contiguous region, which would be saved
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
    _check_distributed_checkpointing_available()

    def prepare(value):
        if is_dtensor(value) and any(isinstance(p, _StridedShard) for p in value.placements):
            placements = tuple(Shard(p.dim) if isinstance(p, _StridedShard) else p for p in value.placements)
            return value.redistribute(placements=placements)
        return value

    # tree_map allows to map a function on arbitrarily nested structures.
    return tree_map(prepare, state_dict)


def save_model_checkpoint_distributed(model, checkpoint_dir: str) -> None:
    """Save rank-local model shards as safetensors with DCP for resuming training.

    Load with `load_checkpoint_in_distributed_model` or local `from_pretrained`.
    Use `save_pretrained(distributed_checkpoint=False)` for a regular checkpoint with weight conversion.
    All ranks must call this function.
    """
    _check_distributed_checkpointing_available()

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
        enable_consolidation=False,
    )
    dcp.save(state_dict, storage_writer=writer)

    # All ranks wait until the checkpoint is ready for loading.
    _distributed_barrier()


def is_sharded_checkpoint(checkpoint_dir: str | os.PathLike) -> bool:
    """Return True if the checkpoint directory contains sharded safetensors files."""
    pattern = r"^shard-[0-9]{5}-model-[0-9]{5}-of-[0-9]{5}\.safetensors$"
    if not os.path.isdir(checkpoint_dir):
        return False
    return any(re.match(pattern, name) for name in os.listdir(checkpoint_dir))


def load_checkpoint_in_distributed_model(model, checkpoint_dir: str | os.PathLike, strict: bool = True) -> None:
    """Load local safetensors into a materialized model, preserving its current mesh and placements.

    DCP reads both rank-local shards and full tensors, including multi-file checkpoints. The checkpoint must
    use the model's parameter names and shapes; weight conversions are handled by `from_pretrained`.
    With `strict=True`, missing model keys raise an error. Extra checkpoint keys are ignored.
    """
    _check_distributed_checkpointing_available()

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


def save_optimizer_distributed(model, optimizer, checkpoint_dir: str, *, consolidate: bool = False) -> None:
    """Save optimizer state via DCP, optionally also writing `optimizer.pt`.

    Native DCP files are retained in `checkpoint_dir` in both cases. Consolidation
    materializes the full optimizer state in rank 0's CPU memory. All ranks must call.
    """
    _check_distributed_checkpointing_available()

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    # Flatten optimizer state and group options into fully qualified name entries.
    # A mesh change can alter which parameters are DTensors and therefore how optimizer groups are constructed, fully
    # qualified name keys make the checkpoint independent of grouping.
    options = StateDictOptions(flatten_optimizer_state_dict=True)
    optimizer_state_dict = _prepare_state_dict_for_dcp(get_optimizer_state_dict(model, optimizer, options=options))
    dcp.save({"optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir)
    if consolidate:
        if _get_torch_distributed_rank() == 0:
            from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

            dcp_to_torch_save(checkpoint_dir, os.path.join(checkpoint_dir, "optimizer.pt"))
        _distributed_barrier()


def load_optimizer_distributed(model, optimizer, checkpoint_dir_or_file: str) -> None:
    """Load optimizer state from a DCP directory or a consolidated `optimizer.pt` file.

    Passing a directory uses the retained DCP shards. Passing the file loads the
    full optimizer state on each rank's CPU before distributing it into the current
    layout. Prefer the directory when memory is limited. All ranks must call.
    """
    _check_distributed_checkpointing_available()

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
    checkpoint_state_dict = _prepare_state_dict_for_dcp(optimizer_state_dict)
    if os.path.isfile(checkpoint_dir_or_file):
        loaded_state = torch.load(checkpoint_dir_or_file, map_location="cpu", weights_only=True)["optimizer"]
        missing_keys = checkpoint_state_dict.keys() - loaded_state.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in optimizer checkpoint: {sorted(missing_keys)}")
        for key, target in checkpoint_state_dict.items():
            value = loaded_state[key]
            if isinstance(target, torch.Tensor) and (
                not isinstance(value, torch.Tensor) or value.shape != target.shape
            ):
                raise ValueError(f"Optimizer checkpoint tensor {key!r} must have shape {tuple(target.shape)}.")
        for key, target in checkpoint_state_dict.items():
            value = loaded_state[key]
            if is_dtensor(target):
                value = distribute_tensor(value.to(target.device), target.device_mesh, target.placements)
            elif isinstance(target, torch.Tensor):
                value = value.to(target.device)
            checkpoint_state_dict[key] = value
    else:
        dcp.load({"optimizer": checkpoint_state_dict}, checkpoint_id=checkpoint_dir_or_file)

    # tree_map allows to map a function on arbitrarily nested structures.
    optimizer_state_dict = tree_map(
        lambda loaded, original: loaded.redistribute(placements=original.placements)
        if is_dtensor(original) and loaded.placements != original.placements
        else loaded,
        checkpoint_state_dict,
        optimizer_state_dict,
    )
    set_optimizer_state_dict(model, optimizer, optimizer_state_dict)
