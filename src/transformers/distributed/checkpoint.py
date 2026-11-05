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

from ..utils import (
    is_torch_available,
    is_torch_greater_or_equal,
    logging,
)
from .utils import (
    _check_distributed_checkpointing_available,
    _distributed_barrier,
    is_dtensor,
)


if TYPE_CHECKING:
    from ..quantizers import HfQuantizer


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
    from torch.utils._pytree import tree_map

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
