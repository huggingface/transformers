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
import warnings
from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING, TypeGuard

from ..utils import is_torch_available, is_torch_distributed_available, is_torch_greater_or_equal, logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from torch.distributed.tensor import DTensor

    from .configuration_utils import DistributedConfig


if is_torch_available():
    import torch


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
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpointing requires `torch>=2.7`.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict = get_model_state_dict(model, options=options)
    if _get_torch_distributed_rank() == 0:
        return full_state_dict
    return {}


def _prepare_state_dict_for_dcp(state_dict):
    """Replace disjoint DTensor shards with contiguous shards in the checkpoint view."""
    from torch.distributed.tensor import Shard
    from torch.distributed.tensor.placement_types import _StridedShard
    from torch.utils._pytree import tree_map

    def prepare(value):
        if is_dtensor(value) and any(isinstance(p, _StridedShard) for p in value.placements):
            placements = tuple(Shard(p.dim) if isinstance(p, _StridedShard) else p for p in value.placements)
            return value.redistribute(placements=placements)
        return value

    return tree_map(prepare, state_dict)


def save_model_checkpoint_distributed(
    model, checkpoint_dir: str, *, checkpoint_format: str = "safetensors", consolidate: bool = True
) -> None:
    """Save rank-local model shards with DCP, optionally consolidating them.

        - `checkpoint_format` selects safetensors or native Torch DCP storage.
        - With `consolidate=True`, rank-local files are kept in `sharded/` and complete
          weights are written at the root.

    Torch consolidation materializes the full state dict in rank 0's CPU memory.
    Without consolidation, load the root directory with DCP and the matching
    storage reader, these rank-local files are not `from_pretrained` checkpoints.
    """
    if checkpoint_format not in ("safetensors", "torch"):
        raise ValueError("`checkpoint_format` must be 'safetensors' or 'torch'.")
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpointing requires `torch>=2.7`.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    # DCP describes each DTensor as one rectangular chunk, which cannot represent packed shards.
    # We redistribute any strided shards to contiguous shards so DCP can write them out.
    # Sub-optimal compared to a future DCP that can write strided shards directly, but works for now.
    state_dict = _prepare_state_dict_for_dcp(get_model_state_dict(model))
    if checkpoint_format == "safetensors":
        from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageWriter

        writer = HuggingFaceStorageWriter(
            path=checkpoint_dir,
            save_distributed=True,
            enable_consolidation=consolidate,
        )
    else:
        shard_dir = os.path.join(checkpoint_dir, "sharded") if consolidate else checkpoint_dir
        writer = dcp.FileSystemWriter(shard_dir)

    dcp.save(state_dict, storage_writer=writer)

    if checkpoint_format == "torch" and consolidate and _get_torch_distributed_rank() == 0:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

        from ..utils import WEIGHTS_NAME

        dcp_to_torch_save(shard_dir, os.path.join(checkpoint_dir, WEIGHTS_NAME))
    # All ranks wait until consolidated weights are ready for loading.
    _distributed_barrier()


def save_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Save optimizer state via DCP."""
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpointing requires `torch>=2.7`.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    # Key group options by parameter name so regrouping after a mesh change remains loadable.
    options = StateDictOptions(flatten_optimizer_state_dict=True)
    optimizer_state_dict = _prepare_state_dict_for_dcp(get_optimizer_state_dict(model, optimizer, options=options))
    dcp.save({"optimizer": optimizer_state_dict}, checkpoint_id=checkpoint_dir)


def load_optimizer_distributed(model, optimizer, checkpoint_dir: str) -> None:
    """Load optimizer state via DCP."""
    if not is_torch_greater_or_equal("2.7"):
        raise OSError("Distributed checkpointing requires `torch>=2.7`.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.utils._pytree import tree_map

    options = StateDictOptions(flatten_optimizer_state_dict=True)
    optimizer_state_dict = get_optimizer_state_dict(model, optimizer, options=options)
    checkpoint_state_dict = _prepare_state_dict_for_dcp(optimizer_state_dict)
    dcp.load({"optimizer": checkpoint_state_dict}, checkpoint_id=checkpoint_dir)
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
