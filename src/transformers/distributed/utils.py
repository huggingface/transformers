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
from collections import defaultdict
from typing import TYPE_CHECKING

from ..utils import is_torch_available, is_torch_greater_or_equal, logging
from .fsdp import apply_fully_shard_data_parallel
from .sharding_utils import (
    _find_strided_shard_placement_from_fused_params,
    _replicate_dtensor,
    fuse_optimizer_state,
    get_fusion_metadata,
    unfuse_optimizer_state,
)
from .tensor_parallel import apply_tensor_parallel


logger = logging.get_logger(__name__)


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

    # Build the N-dimensional device mesh
    mesh = torch.distributed.init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
    # If N > 1, create a flattened sub-mesh so all-reduces across the world mesh ae done in one collective
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
    """Mesh-aware grad-norm clip with O(1) extra memory per parameter.

    Per-rank partials (`|g_local|^p .sum()`) are accumulated into buckets keyed
    by `(mesh, set-of-non-replicate-dim-indices)`, so the cross-rank reduction
    is one collective per bucket rather than one per parameter. A typical
    FSDP2+TP model has 2–4 distinct signatures across hundreds of params, so
    this collapses O(N_params) NCCL launches into O(N_buckets).
    """
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)

    norm_p = float(norm_type)
    device = parameters[0].grad.device

    # Bucket key for plain (non-DTensor) grads
    LOCAL_KEY = (None, ())
    buckets: dict[tuple, torch.Tensor] = defaultdict(lambda: torch.zeros((), dtype=torch.float32, device=device))

    # Bucket the params based on key=(mesh, placements) so we now we can reduce on these dims
    for p in parameters:
        g = p.grad
        if isinstance(g, DTensor):
            local = g.to_local()
            reduce_dims = tuple(dim for dim, placement in enumerate(g.placements) if not placement.is_replicate())
            key = (g.device_mesh, reduce_dims) if reduce_dims else LOCAL_KEY
        else:
            local = g
            key = LOCAL_KEY
        partial = local.detach().to(torch.float32).abs().pow_(norm_p).sum()
        buckets[key].add_(partial)

    accum = torch.zeros((), dtype=torch.float32, device=device)
    for (mesh, reduce_dims), bucket_sum in buckets.items():
        if mesh is not None:
            for dim_idx in reduce_dims:
                group = mesh.get_group(dim_idx) if mesh.ndim > 1 else mesh.get_group()
                torch.distributed.all_reduce(bucket_sum, op=torch.distributed.ReduceOp.SUM, group=group)
        accum.add_(bucket_sum)

    total_norm = accum.pow_(1.0 / norm_p)
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for p in parameters:
        g = p.grad
        if isinstance(g, DTensor):
            g._local_tensor.mul_(clip_coef)
        else:
            g.mul_(clip_coef)
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


def save_model_checkpoint_distributed(model, checkpoint_dir: str) -> None:
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


def has_mixed_tensor_and_dtensor_params(params) -> bool:
    has_dtensor = False
    has_tensor = False
    for param in params:
        if isinstance(param, DTensor):
            has_dtensor = True
        elif isinstance(param, torch.Tensor):
            has_tensor = True

        if has_dtensor and has_tensor:
            return True
    return False


def maybe_disable_foreach_and_fused_for_mixed_dtensor_groups(optimizer) -> None:
    """
    When get_optimizer_state_dict() or set_optimizer_state_dict() runs on an optimizer with no state yet,
    PyTorch first materializes that state by doing a no-op step() with zero gradients. If an optimizer
    group mixes regular tensors and DTensors, the batched foreach/fused optimizer kernels cannot process
    that mixed group, so we turn those kernels off for such groups before distributed optimizer save/
    load.
    """
    for i, param_group in enumerate(optimizer.param_groups):
        if has_mixed_tensor_and_dtensor_params(param_group.get("params", ())):
            logger.warning_once(
                f"Param group {i} mixes regular tensors and DTensors; disabling foreach/fused "
                "optimizer kernels for that group so distributed optimizer save/load can materialize state."
            )
            param_group["foreach"] = False
            if "fused" in param_group:
                param_group["fused"] = False


def save_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Save optimizer state via DCP.

    Params whose DTensors carry a lonely `_StridedShard` placement (e.g. Mixtral
    `gate_up_proj`) are locally split into plain-`Shard` pieces at the boundary
    so DCP only ever sees DTensors it can encode as one contiguous chunk per
    rank.
    """
    maybe_disable_foreach_and_fused_for_mixed_dtensor_groups(optimizer)
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
    maybe_disable_foreach_and_fused_for_mixed_dtensor_groups(optimizer)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
    fusion_metadata = get_fusion_metadata(optimizer_state_dict)
    unfuse_optimizer_state(optimizer_state_dict, fusion_metadata)
    dcp.load({"optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir)
    fuse_optimizer_state(optimizer_state_dict, fusion_metadata)
    set_optimizer_state_dict(model, optimizer, optimizer_state_dict)
    maybe_disable_foreach_and_fused_for_mixed_dtensor_groups(optimizer)
