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

    from ..integrations.tensor_parallel import convert_strided_to_shard, restore_strided_from_shard


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
