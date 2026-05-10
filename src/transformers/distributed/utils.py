# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from ..utils import is_torch_available, is_torch_greater_or_equal, strtobool


if TYPE_CHECKING:
    import torch.nn as nn

    from .configuration_utils import DistributedConfig

if is_torch_available():
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.distributed.tensor.placement_types import _StridedShard


def is_fsdp_enabled() -> bool:
    """Check if FSDP is active via Accelerate (env var based) — covers FSDP1 only."""
    if not is_torch_available():
        return False

    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def is_fsdp_managed_module(module: nn.Module) -> bool:
    """Check if a module is managed by FSDP (1 or 2)."""
    if not is_torch_available():
        return False
    if not torch.distributed.is_available():
        return False

    # FSDP2: attribute set by apply_fsdp2()
    if getattr(module, "_is_fsdp_managed_module", False):
        return True
    # FSDP1: wrapped by FullyShardedDataParallel
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(module, FullyShardedDataParallel)


def _ensure_torch_distributed(device_type: str):
    """Initialize torch.distributed if not already initialized."""
    if not torch.distributed.is_initialized():
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}
            backend = backend_map.get(device_type)

            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
            current_device = getattr(torch, device_type)
            if device_type != "cpu":
                current_device.set_device(local_rank)
        except Exception as e:
            raise OSError(
                "We tried to initialize torch.distributed for you, but it failed. Make "
                "sure you init torch distributed in your script to use distributed training."
            ) from e


def init_device_mesh(distributed_config: DistributedConfig) -> torch.distributed.device_mesh.DeviceMesh:
    if not is_torch_greater_or_equal("2.5"):
        raise OSError("Distributed training with DistributedConfig requires `torch>=2.5`.")

    device_type = torch._C._get_accelerator().type
    _ensure_torch_distributed(device_type)

    world_size = torch.distributed.get_world_size()
    if device_type != "cpu":
        getattr(torch, device_type).set_device(int(os.environ.get("LOCAL_RANK", 0)))

    tp_size = distributed_config.tp_size
    fsdp_size = distributed_config.fsdp_size

    assert world_size == tp_size * fsdp_size, (
        f"world_size ({world_size}) must be equal to tp_size ({tp_size}) * fsdp_size ({fsdp_size})"
    )

    dims, names = [], []
    if fsdp_size > 1:
        dims.append(fsdp_size)
        names.append("fsdp")
    if tp_size > 1:
        dims.append(tp_size)
        names.append("tp")

    # Build from a 1D world mesh via _unflatten so that PyTorch can flatten
    # sub-dimensions back when needed (e.g. for single all_reduce across
    # [fsdp, tp] during grad norm computation instead of 2 sequential ones).
    world_mesh = torch.distributed.init_device_mesh(device_type, (world_size,), mesh_dim_names=("world",))
    mesh = world_mesh._unflatten(0, tuple(dims), tuple(names))

    # Pre-create flattened sub-mesh for multi-dimensional meshes so DTensor
    # can use a single collective instead of sequential per-dimension ones.
    if len(dims) > 1:
        mesh._flatten("_".join(names))

    return mesh


def _to_cpu_fresh(tensor: torch.Tensor) -> torch.Tensor:
    """Plain tensor → contiguous CPU tensor with fresh storage for safetensors."""
    if tensor.device.type == "meta":
        return tensor
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to(device="cpu")
    out = torch.empty(t.shape, dtype=t.dtype, device="cpu")
    out.copy_(t)
    return out.contiguous()


def gather_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather all sharded params to full plain tensors for saving.

    Handles FSDP unshard and TP DTensor gather.
    Streams one parameter at a time to avoid holding all full tensors on GPU.
    Only rank 0 accumulates the result; other ranks return ``{}``.
    """
    tp_size = model.tp_size
    is_rank0 = torch.distributed.get_rank() == 0

    # Get state dict — FSDP unshard if needed (returns DTensors, not full tensors)
    if getattr(model, "_is_fsdp_managed_module", False):
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        state_dict = get_model_state_dict(model)
    else:
        state_dict = model.state_dict()

    # No TP — materialize on rank 0 only
    if tp_size is None:
        if is_rank0:
            return {k: _to_cpu_fresh(v) for k, v in state_dict.items()}
        return {}

    # Stream: gather one param at a time, only rank 0 keeps the CPU copy
    result = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, DTensor):
            # All ranks participate in the collective, only rank 0 keeps the result
            with torch.no_grad():
                full = _replicate_dtensor(tensor).to_local()
            if is_rank0:
                result[key] = _to_cpu_fresh(full)
            del full
        elif is_rank0:
            result[key] = _to_cpu_fresh(tensor)

    return result


def _replicate_dtensor(tensor: DTensor) -> DTensor:
    """All-gather a DTensor to fully Replicate, handling ``_StridedShard``.

    PyTorch's ``redistribute()`` does not support ``_StridedShard`` as a source::

        _StridedShard -> redistribute() -> Replicate      ❌ AssertionError
        _StridedShard -> redistribute() -> Shard           ❌ NotImplementedError
        Shard         -> redistribute() -> Replicate      ✅ works
        Replicate     -> redistribute() -> Shard           ✅ works
        Replicate     -> redistribute() -> _StridedShard  ✅ works

    So we bypass ``redistribute`` and call each placement's low-level
    ``_to_replicate_tensor`` (manual all-gather + interleaved reorder).

    We process mesh dims **right-to-left** (innermost first).  Under TP+FSDP
    the 2D mesh is ``(fsdp, tp)`` and both dims can shard the same tensor dim::

        placements = (_StridedShard(dim=0), Shard(dim=0))
        local shape = [64, 1024]   (global [256, 1024], fsdp=2, tp=2)

    Right-to-left means TP is gathered first (local grows to [128, 1024]),
    then FSDP (grows to [256, 1024]).  Each step must pass the correct
    intermediate logical shape — the global shape divided by the mesh sizes
    of dims not yet gathered (to the left).
    """
    mesh = tensor.device_mesh
    replicate_all = tuple(Replicate() for _ in range(mesh.ndim))
    with torch.no_grad():
        if any(isinstance(p, _StridedShard) for p in tensor.placements):
            local = tensor._local_tensor
            placements = tensor.placements
            for i in reversed(range(mesh.ndim)):
                p = placements[i]
                if p.is_replicate():
                    continue
                # Compute the logical shape seen at this step: dims to the left
                # (not yet gathered) still divide their tensor dimension.
                logical_shape = list(tensor.shape)
                for j in range(i):
                    pj = placements[j]
                    if not pj.is_replicate():
                        logical_shape[pj.dim] //= mesh.size(j)
                local = p._to_replicate_tensor(local, mesh, i, logical_shape)
            return DTensor.from_local(local, mesh, replicate_all, run_check=False)

        return tensor.redistribute(placements=replicate_all)


def convert_strided_to_shard(state_dict: dict) -> dict[str, tuple]:
    # Convert _StridedShard DTensors in a state dict to plain Shard for DCP compatibility.
    placement_map: dict[str, tuple] = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            nested = convert_strided_to_shard(value)
            for nk, nv in nested.items():
                placement_map[f"{key}.{nk}"] = nv
        elif isinstance(value, DTensor) and any(isinstance(p, _StridedShard) for p in value.placements):
            placement_map[key] = tuple(value.placements)
            shard_placements = tuple(Shard(p.dim) if isinstance(p, _StridedShard) else p for p in value.placements)
            state_dict[key] = _replicate_dtensor(value).redistribute(placements=shard_placements)
    return placement_map


def restore_strided_from_shard(state_dict: dict, placement_map: dict[str, tuple]) -> None:
    # Restore _StridedShard placements after dcp.load.
    def _resolve(d, dotted_key):
        parts = dotted_key.split(".", 1)
        if len(parts) == 2 and parts[0] in d and isinstance(d[parts[0]], dict):
            return _resolve(d[parts[0]], parts[1])
        return d, dotted_key

    for key, original_placements in placement_map.items():
        container, leaf_key = _resolve(state_dict, key)
        if leaf_key in container and isinstance(container[leaf_key], DTensor):
            container[leaf_key] = _replicate_dtensor(container[leaf_key]).redistribute(placements=original_placements)


def save_optimizer(optimizer, checkpoint_dir: str) -> None:
    # Save optimizer state via DCP, handling _StridedShard placements transparently.
    osd = optimizer.state_dict()
    placement_map = convert_strided_to_shard(osd)
    dcp.save({"optimizer": osd}, checkpoint_id=checkpoint_dir)
    if placement_map and torch.distributed.get_rank() == 0:
        torch.save(placement_map, os.path.join(checkpoint_dir, "placement_map.pt"))


def load_optimizer(optimizer, checkpoint_dir: str) -> None:
    # Load optimizer state via DCP, restoring _StridedShard placements transparently.
    osd = optimizer.state_dict()
    dcp.load({"optimizer": osd}, checkpoint_id=checkpoint_dir)
    pmap_path = os.path.join(checkpoint_dir, "placement_map.pt")
    if os.path.exists(pmap_path):
        placement_map = torch.load(pmap_path, weights_only=False)
        restore_strided_from_shard(osd, placement_map)
    optimizer.load_state_dict(osd)
