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

from ..utils import is_torch_available, is_torch_greater_or_equal
from .fsdp import apply_fully_shard_data_parallel
from .sharding_utils import (
    _find_strided_shard_placement_from_fused_params,
    _replicate_dtensor,
    fuse_optimizer_state,
    get_fusion_metadata,
    unfuse_optimizer_state,
)
from .tensor_parallel import apply_tensor_parallel


if TYPE_CHECKING:
    import torch.nn as nn

    from .configuration_utils import DistributedConfig

if is_torch_available():
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageWriter
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.tensor import DTensor


def _ensure_torch_distributed(device_type: str):
    """Initialize torch.distributed if not already initialized."""
    if not torch.distributed.is_initialized():
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            backend_map = {"cuda": "nccl", "cpu": "gloo", "xpu": "xccl", "hpu": "hccl"}
            backend = backend_map.get(device_type)

            # Bind the accelerator before init so the process group is created with a
            # device_id, otherwise collectives like barrier() warn (and may spin up an
            # extra NCCL comm) about the missing device binding.
            device_id = None
            if device_type != "cpu":
                getattr(torch, device_type).set_device(local_rank)
                device_id = torch.device(device_type, local_rank)
            torch.distributed.init_process_group(
                backend=backend, rank=rank, world_size=world_size, device_id=device_id
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
    if not torch.distributed.is_initialized():
        return
    device_type = torch._C._get_accelerator().type
    if device_type != "cpu":
        torch.distributed.barrier(device_ids=[getattr(torch, device_type).current_device()])
    else:
        torch.distributed.barrier()


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


def distribute_model(model, distributed_config: DistributedConfig, device_mesh) -> nn.Module:
    """Apply TP and/or FSDP2 to `model` based on the mesh dims in `device_mesh`."""
    model.config.distributed_config = distributed_config
    model.device_mesh = device_mesh
    mesh_dim_names = device_mesh.mesh_dim_names or ()
    if "tp" in mesh_dim_names:
        tp_mesh = device_mesh["tp"] if device_mesh.ndim > 1 else device_mesh
        model = apply_tensor_parallel(model, tp_mesh, distributed_config.tp_plan)
    if "fsdp" in mesh_dim_names:
        fsdp_mesh = device_mesh["fsdp"] if device_mesh.ndim > 1 else device_mesh
        model = apply_fully_shard_data_parallel(model, fsdp_mesh, distributed_config.fsdp_plan)
    return model


@torch.no_grad()
def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0):
    """Grad-norm clip that works when params live on different DTensor meshes.

    ``torch.nn.utils.get_total_norm`` stacks per-grad norms; that fails when grads
    live on different meshes (e.g. TP-wrapped params on the (fsdp, tp) mesh and
    FSDP-only params on the (fsdp,) sub-mesh). We sidestep it by replicating each
    DTensor grad to a plain local tensor, computing the norm over those, and
    scaling the original DTensor grads in place — the placement of the original
    grads doesn't matter for the per-element clip.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    local_grads = [_replicate_dtensor(g).to_local() if isinstance(g, DTensor) else g for g in grads]
    total_norm = torch.nn.utils.get_total_norm(local_grads, norm_type=norm_type)
    torch.nn.utils.clip_grads_with_norm_(grads, max_norm=max_norm, total_norm=total_norm)
    return total_norm


def gather_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather all sharded params to full plain tensors for saving.

    Handles FSDP unshard and TP DTensor gather.
    Streams one parameter at a time to avoid holding all full tensors on GPU.
    Only rank 0 accumulates the result; other ranks return ``{}``.
    """
    is_rank0 = torch.distributed.get_rank() == 0
    state_dict = get_model_state_dict(model)

    result = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, DTensor):
            if is_rank0:
                result[key] = tensor.detach().to(device="cpu", copy=True).contiguous()
            continue

        with torch.no_grad():
            full = _replicate_dtensor(tensor).to_local()
            if is_rank0:
                result[key] = full.detach().to(device="cpu", copy=True).contiguous()
            del full
    return result


def save_model_checkpoint(model, checkpoint_dir: str) -> None:
    """Save model parameters as standard HF-format sharded safetensors using
    DCP + HuggingFaceStorageWriter with consolidation enabled.

    Every rank first writes its own shard in parallel under
    `<checkpoint_dir>/sharded/`, then a consolidation pass reads those shards
    and emits HF-compatible `model-*-of-N.safetensors` (+ index) at
    `<checkpoint_dir>/`. The result is a directory `from_pretrained` reads
    through its normal path — no special flag needed at load time.

    DTensors carrying an uncomposed `_StridedShard` placement (e.g. fused
    gate||up MoE weights) are replicated to a full tensor on every rank
    before the save, otherwise DCP cannot encode that placement.
    """
    state_dict = get_model_state_dict(model)
    for key, value in list(state_dict.items()):
        if (
            isinstance(value, DTensor)
            and _find_strided_shard_placement_from_fused_params(value.placements) is not None
        ):
            state_dict[key] = _replicate_dtensor(value)

    dcp.save(
        state_dict,
        storage_writer=HuggingFaceStorageWriter(
            path=checkpoint_dir,
            save_distributed=True,
            enable_consolidation=True,
        ),
    )
    # Wait for rank 0 to finish writing the HF safetensors so other
    # ranks don't return (and hit `from_pretrained`) before the files exist.
    _distributed_barrier()


def save_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Save optimizer state via DCP.

    Params whose DTensors carry a lonely `_StridedShard` placement (e.g. Mixtral
    `gate_up_proj`) are locally split into plain-`Shard` pieces at the boundary
    so DCP only ever sees DTensors it can encode as one contiguous chunk per
    rank.
    """
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
    fusion_metadata = get_fusion_metadata(optimizer_state_dict)
    unfuse_optimizer_state(optimizer_state_dict, fusion_metadata)
    dcp.save({"optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir)


def load_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Load optimizer state via DCP.

    Symmetric to `save_optimizer_distributed`: build the unfused template, let DCP fill
    it, then merge fused params back to their original `_StridedShard` form
    before handing the state_dict back to the optimizer.
    """
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
    fusion_metadata = get_fusion_metadata(optimizer_state_dict)
    unfuse_optimizer_state(optimizer_state_dict, fusion_metadata)
    dcp.load({"optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir)
    fuse_optimizer_state(optimizer_state_dict, fusion_metadata)
    set_optimizer_state_dict(model, optimizer, optimizer_state_dict)
