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
from typing import TYPE_CHECKING

from ..utils import is_torch_available, is_torch_greater_or_equal

if TYPE_CHECKING:
    from .configuration_utils import DistributedConfig

if is_torch_available():
    import torch

    _torch_distributed_available = torch.distributed.is_available()
else:
    _torch_distributed_available = False


def _is_torch_distributed_initialized() -> bool:
    if not _torch_distributed_available:
        return False
    return hasattr(torch.distributed, "is_initialized") and torch.distributed.is_initialized()


def _get_torch_distributed_rank() -> int:
    if not _is_torch_distributed_initialized():
        return 0
    return torch.distributed.get_rank()


def _get_torch_distributed_world_size() -> int:
    if not _is_torch_distributed_initialized() or not hasattr(torch.distributed, "get_world_size"):
        return 1
    return torch.distributed.get_world_size()


def is_local_dist_rank_0() -> bool:
    return _is_torch_distributed_initialized() and int(os.environ.get("LOCAL_RANK", "-1")) == 0


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
    if not _is_torch_distributed_initialized():
        return
    device_type = torch._C._get_accelerator().type
    if device_type != "cpu":
        torch.distributed.barrier(device_ids=[getattr(torch, device_type).current_device()])
    else:
        torch.distributed.barrier()


def initialize_fully_sharded_data_parallelism(distributed_config: DistributedConfig):
    if not is_torch_greater_or_equal("2.5"):
        raise OSError("Distributed training with DistributedConfig requires `torch>=2.5`.")

    if distributed_config.fsdp_size > 1 and not is_torch_greater_or_equal("2.6"):
        raise OSError("FSDP2 requires `torch>=2.6`.")

    device_type = torch._C._get_accelerator().type
    _ensure_torch_distributed(device_type)

    world_size = torch.distributed.get_world_size()
    if device_type != "cpu":
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        getattr(torch, device_type).set_device(local_rank)
        device_map = torch.device(device_type, local_rank)
    else:
        device_map = torch.device(device_type)

    fsdp_size = distributed_config.fsdp_size

    assert world_size == fsdp_size, f"world_size ({world_size}) must be equal to fsdp_size ({fsdp_size})"

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
