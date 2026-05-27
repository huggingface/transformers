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
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.tensor import DTensor, Partial, Replicate

    if is_torch_greater_or_equal("2.7"):
        from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageWriter


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

    if distributed_config.fsdp_size > 1 and not is_torch_greater_or_equal("2.6"):
        raise OSError("FSDP2 requires `torch>=2.6`.")

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

    Each grad's local `|g|^p` sum is accumulated into a bucket keyed by its
    *reduction* signature: every non-replicate placement is canonicalized to
    `Partial()`, so grads that differ only in which tensor dim was sharded
    (e.g. `Shard(0)` vs `Shard(1)` on the same mesh dim) share one bucket and
    one collective. `full_tensor()` then issues exactly the right all-reduce(s)
    per bucket. A typical FSDP2+TP model has 2–4 distinct signatures across
    hundreds of params, so this collapses O(N_params) NCCL launches into
    O(N_buckets).
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    norm_p = float(norm_type)
    device = grads[0].device

    # Phase 1 — bucket by reduction signature, not by raw placements. Two grads
    # land in the same bucket iff they need the same all-reduces on the same mesh
    # dims. Plain (non-DTensor) grads share key=None (no reduction needed).
    norm_buckets = defaultdict(lambda: torch.zeros((), dtype=torch.float32, device=device))
    for g in grads:
        if isinstance(g, DTensor):
            local = g.to_local()
            reduce_placements = tuple(Replicate() if p.is_replicate() else Partial() for p in g.placements)
            key = (g.device_mesh, reduce_placements)
        else:
            local, key = g, None
        norm_buckets[key] += local.detach().float().abs().pow_(norm_p).sum()

    # Phase 2 — one collective per bucket; the key already encodes the right placements.
    total = torch.zeros((), dtype=torch.float32, device=device)
    for key, bucket_sum in norm_buckets.items():
        if key is not None:
            mesh, reduce_placements = key
            bucket_sum = DTensor.from_local(bucket_sum, mesh, reduce_placements).full_tensor()
        total += bucket_sum

    # Phase 3 — global norm, then scale every grad's local shard in place.
    total_norm = total.pow(1.0 / norm_p)
    clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for g in grads:
        (g._local_tensor if isinstance(g, DTensor) else g).mul_(clip_coef)
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
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpoint saving requires `torch>=2.7`.")

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
