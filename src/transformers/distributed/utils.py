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
import warnings
from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING, TypeGuard

import safetensors.torch

from ..utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    is_torch_available,
    is_torch_distributed_available,
    is_torch_greater_or_equal,
    logging,
)
from .sharding_utils import DtensorShardOperation, _dtensor_from_local_like


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor

    from .configuration_utils import DistributedConfig


if is_torch_available():
    import torch
    from torch.utils._pytree import tree_map


def _check_distributed_checkpointing_available(raise_if_not: bool = True) -> bool:
    if not is_torch_distributed_available() or not is_torch_greater_or_equal("2.7"):
        if raise_if_not:
            raise OSError("Distributed checkpointing requires `torch>=2.7` with `torch.distributed` available.")
        return False
    return True


if _check_distributed_checkpointing_available(raise_if_not=False):
    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader, HuggingFaceStorageWriter
    from torch.distributed.tensor import Shard, distribute_tensor
    from torch.distributed.tensor.placement_types import _StridedShard


def _is_torch_distributed_initialized() -> bool:
    if not is_torch_distributed_available():
        return False
    return torch.distributed.is_initialized()


def is_dtensor(obj: object) -> TypeGuard[DTensor]:
    if not is_torch_distributed_available():
        return False
    from torch.distributed.tensor import DTensor

    return isinstance(obj, DTensor)


def _get_torch_distributed_rank() -> int:
    if not _is_torch_distributed_initialized():
        return 0
    return torch.distributed.get_rank()


def _get_torch_distributed_world_size() -> int:
    if not _is_torch_distributed_initialized():
        return 1
    return torch.distributed.get_world_size()


def is_local_dist_rank_0() -> bool:
    return _is_torch_distributed_initialized() and int(os.environ.get("LOCAL_RANK", "-1")) == 0


def _ensure_torch_distributed(device_type: str | None = None):
    """Initialize torch.distributed if not already initialized.

    If `device_type` is not given, it is detected from the current accelerator.
    """
    if not torch.distributed.is_initialized():
        if device_type is None:
            device_type = torch._C._get_accelerator().type
        if device_type == "mps":
            logger.warning_once(
                "PyTorch's built-in DeviceMesh/DTensor stack does not support an MPS mesh. Falling back to CPU."
            )
            device_type = "cpu"
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            backend_map = {
                "cuda": "nccl",
                "cpu": "gloo",
                "xpu": "xccl",
                "hpu": "hccl",
                "neuron": "neuron",
                "tpu": "tpu_dist",
            }
            backend = backend_map.get(device_type)

            # Bind the accelerator before init so the process group is created with a
            # device_id, otherwise collectives like barrier() warn (and may spin up an
            # extra NCCL comm) about the missing device binding.
            device_id = None
            if device_type != "cpu":
                getattr(torch, device_type).set_device(local_rank)
                device_id = torch.device(device_type, local_rank)
            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                device_id=device_id,
                # Sharded loading takes tens of minutes with high rank skew; the default 10-minute watchdog is too short
                timeout=timedelta(hours=2),
            )
        except Exception as e:
            raise OSError(
                "We tried to initialize torch.distributed for you, but it failed. Make "
                "sure you init torch distributed in your script to use distributed training."
            ) from e


def _distributed_barrier():
    """Barrier bound to the current accelerator device.

    Passing `device_ids` is required when the process group was initialized without a
    `device_id`; with it, the call is a no-op compared to plain `barrier()`. Safe to call
    when torch.distributed has not been initialized — returns immediately.
    """
    if not _is_torch_distributed_initialized():
        return
    device_type = torch._C._get_accelerator().type
    if device_type != "cpu":
        torch.distributed.barrier(device_ids=[getattr(torch, device_type).current_device()])
    else:
        torch.distributed.barrier()


# Retained for the legacy transformers.integrations.tensor_parallel API.
def initialize_tensor_parallelism(
    tp_plan: str | dict[str, str] | None, tp_size: int | None = None, device_mesh=None, device_map=None
):
    r"""
    Sets up the device mesh and initialized the backend for tensor parallelism.
    This function is called when the model is loaded and the TP plan is set to 'auto'.
    """
    warnings.warn(
        "`initialize_tensor_parallelism` is deprecated and will be removed in a future release. "
        "Use `initialize_distributed_mesh` with a `DistributedConfig` instead.",
        FutureWarning,
        stacklevel=2,
    )
    if tp_size is not None and tp_plan is None:
        raise ValueError("tp_plan has to be set when tp_size is passed.")
    if tp_plan is not None and device_map is not None:
        raise ValueError("`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization.")
    if device_mesh is None:
        if not is_torch_greater_or_equal("2.5"):
            raise OSError("Tensor parallel is only supported for `torch>=2.5`.")

        # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
        device_type = torch._C._get_accelerator().type
        if device_type == "mps":
            logger.warning_once(
                "PyTorch's built-in DeviceMesh/DTensor stack does not support an MPS mesh. Falling back to CPU."
            )
            device_type = "cpu"
        current_device = getattr(torch, device_type)

        if device_type != "cpu":
            current_device.set_device(int(os.environ["LOCAL_RANK"]))
            index = current_device.current_device()
            tp_device = torch.device(device_type, index)
            device_map = tp_device
        else:
            tp_device = torch.device(device_type)
            device_map = device_type or {}

        device_mesh = torch.distributed.init_device_mesh(tp_device.type, (tp_size,))
    else:
        if device_mesh.ndim > 1:
            if "tp" not in device_mesh.mesh_dim_names:
                raise ValueError(
                    "When using `tp_plan` and n-d `device_mesh`, it must contain a 'tp' dimension. "
                    "Please provide a valid `device_mesh`."
                )
            device_mesh = device_mesh["tp"]
        device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")

    return device_map, device_mesh


def initialize_fully_sharded_data_parallelism(distributed_config: DistributedConfig):
    warnings.warn(
        "`initialize_fully_sharded_data_parallelism` is deprecated and will be removed in a future release. "
        "Use `initialize_distributed_mesh` with a `DistributedConfig` instead.",
        FutureWarning,
        stacklevel=2,
    )
    # `fully_shard` itself only needs torch>=2.6, but distributed checkpoint save/load
    # (DCP + HuggingFaceStorageWriter) needs 2.7, so that is the effective requirement.
    if distributed_config.fsdp_size > 1 and not is_torch_greater_or_equal("2.7"):
        raise OSError("FSDP2 requires `torch>=2.7` (distributed checkpoint save/load).")

    device_type = torch._C._get_accelerator().type

    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        getattr(torch, device_type).set_device(local_rank)
        device_map = torch.device(device_type, local_rank)
    else:
        device_map = torch.device(device_type)

    fsdp_size = distributed_config.fsdp_size

    dims, names = [], []
    if fsdp_size > 1:
        dims.append(fsdp_size)
        names.append("fsdp")

    # Build the N-dimensional device mesh
    mesh = torch.distributed.init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
    # If N > 1, create a flattened sub-mesh so all-reduces across the world mesh ae done in one collective
    if len(dims) > 1:
        mesh._flatten("_".join(names))

    return device_map, mesh


def initialize_distributed_mesh(
    distributed_config: DistributedConfig,
):
    """Create a device mesh containing every configured parallel dimension."""
    mesh_shape = []
    mesh_dim_names = []

    if distributed_config.pp_size > 1:
        mesh_shape.append(distributed_config.pp_size)
        mesh_dim_names.append("pp")
    if distributed_config.fsdp_size > 1:
        mesh_shape.append(distributed_config.fsdp_size)
        mesh_dim_names.append("fsdp")
    if distributed_config.tp_size > 1:
        mesh_shape.append(distributed_config.tp_size)
        mesh_dim_names.append("tp")

    if not mesh_shape:
        return None, None

    device_type = torch._C._get_accelerator().type
    if distributed_config.tp_size > 1 and device_type == "mps":
        raise RuntimeError("Tensor parallelism is not supported on MPS devices.")

    _ensure_torch_distributed(device_type)
    world_size = torch.distributed.get_world_size()
    expected_world_size = distributed_config.pp_size * distributed_config.fsdp_size * distributed_config.tp_size
    if expected_world_size != world_size:
        raise RuntimeError(
            f"The parallel mesh requires {expected_world_size} processes, but world_size is {world_size}."
        )

    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        getattr(torch, device_type).set_device(local_rank)
        device_map = torch.device(device_type, local_rank)
    else:
        device_map = torch.device(device_type)

    device_mesh = torch.distributed.init_device_mesh(
        device_type,
        tuple(mesh_shape),
        mesh_dim_names=tuple(mesh_dim_names),
    )
    # A flattened sub-mesh, so an all-reduce over every rank is one collective instead of one per dimension.
    if len(mesh_dim_names) > 1:
        device_mesh._flatten("_".join(mesh_dim_names))
    return device_map, device_mesh


def gather_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather FSDP-sharded params to full plain CPU tensors.

    Only rank 0 accumulates the result; other ranks return ``{}``.
    """
    _check_distributed_checkpointing_available()

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict = get_model_state_dict(model, options=options)
    if _get_torch_distributed_rank() == 0:
        return full_state_dict
    return {}


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


def is_sharded_checkpoint(checkpoint_dir: str | os.PathLike) -> bool:
    """Return True if the checkpoint directory contains sharded safetensors files."""
    pattern = r"^shard-[0-9]{5}-model-[0-9]{5}-of-[0-9]{5}\.safetensors$"
    if not os.path.isdir(checkpoint_dir):
        return False
    return any(re.match(pattern, name) for name in os.listdir(checkpoint_dir))


def save_model_checkpoint_distributed(model, checkpoint_dir: str, *, consolidate: bool = True) -> None:
    """Save rank-local model shards as safetensors with DCP, optionally consolidating them.

    With `consolidate=True`, rank-local files are kept in `sharded/` and complete weights
    are written at the root. Otherwise, load the rank-local files with
    `load_distributed_checkpoint`; they are not `from_pretrained` checkpoints.
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
        enable_consolidation=consolidate,
    )
    dcp.save(state_dict, storage_writer=writer)

    # All ranks wait until consolidated weights are ready for loading.
    _distributed_barrier()


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


def load_checkpoint_in_distributed_model(model, checkpoint_dir: str | os.PathLike, strict: bool = True) -> None:
    """
    Load local safetensors weights into an initialized model, preserving its current mesh and placements.
    """
    _check_distributed_checkpointing_available()

    safe_index_file = os.path.join(checkpoint_dir, SAFE_WEIGHTS_INDEX_NAME)
    safe_weights_file = os.path.join(checkpoint_dir, SAFE_WEIGHTS_NAME)

    if is_sharded_checkpoint(checkpoint_dir):
        _load_sharded_checkpoint_in_distributed_model(model, checkpoint_dir, strict=strict)
    elif is_sharded_checkpoint(os.path.join(checkpoint_dir, "sharded")):
        _load_sharded_checkpoint_in_distributed_model(model, os.path.join(checkpoint_dir, "sharded"), strict=strict)
    elif os.path.isfile(safe_index_file):
        with open(safe_index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_paths = []
        for shard_file in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(checkpoint_dir, shard_file)
            if not os.path.isfile(shard_path):
                raise ValueError(f"Shard file {shard_path} not found in {checkpoint_dir}.")
            shard_paths.append(shard_path)
        _load_consolidated_checkpoint_in_distributed_model(model, shard_paths, strict=strict)
    elif os.path.isfile(safe_weights_file):
        _load_consolidated_checkpoint_in_distributed_model(model, safe_weights_file, strict=strict)
    else:
        raise ValueError(f"No distributed, sharded, or safetensors checkpoint found in {checkpoint_dir}.")


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

    # Key group options by parameter name so regrouping after a mesh change remains loadable.
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


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """
    Equivalent to torch.nn.utils.clip_grad_norm_ but supports a mixture of ordinary and DTensors parameters.
    """
    from torch.nn.utils import clip_grads_with_norm_, get_total_norm

    parameters = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    norm_type = float(norm_type)
    max_norm = float(max_norm)
    params_by_mesh = defaultdict(list)
    for param in parameters:
        if param.grad is not None:
            params_by_mesh[param.grad.device_mesh if is_dtensor(param.grad) else None].append(param)

    if len(params_by_mesh) <= 1 and max_norm != float("inf"):
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)
        return total_norm.full_tensor() if is_dtensor(total_norm) else total_norm

    if not params_by_mesh:
        return torch.tensor(0.0)

    norms = []
    for params in params_by_mesh.values():
        norm = get_total_norm([param.grad for param in params], norm_type, foreach=foreach)
        norms.append(norm.full_tensor() if is_dtensor(norm) else norm)
    stacked_norms = torch.stack([norm.to(norms[0].device) for norm in norms])

    # For order zero, each group norm counts nonzero tensor norms, combine those counts by summing.
    total_norm = stacked_norms.sum() if norm_type == 0 else torch.linalg.vector_norm(stacked_norms, norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )

    if max_norm != float("inf"):
        for params in params_by_mesh.values():
            clip_grads_with_norm_(params, max_norm, total_norm, foreach)
    return total_norm
