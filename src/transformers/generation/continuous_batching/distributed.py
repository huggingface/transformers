# Copyright 2026 The HuggingFace Inc. team
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
import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any, TypeVar

import torch
import torch.distributed as _dist

from .requests import logger


# torch marks `torch.distributed` members as possibly-missing, which leads to type check errors. To avoid them, we mark
# the module as `Any` (same as `DeviceMeshLike` in `_typing.py`)
dist: Any = _dist


if TYPE_CHECKING or torch.distributed.is_available():  # prevents runtime import errors when distributed is off
    from torch.distributed.device_mesh import DeviceMesh
else:
    DeviceMesh = object  # only used for type checking, so this is ok


T = TypeVar("T")


class DistributedHelper:
    """A helper class to handle distributed-related operations. Notably, it does not crash when distributed is off."""

    def __init__(self, device_mesh: DeviceMesh | None, cpu_group_timeout: float | None) -> None:
        self.dist_on = dist.is_available() and dist.is_initialized()
        self.device_mesh = device_mesh

        # Check validity of the device mesh
        self.check_device_mesh_for_cb(self.device_mesh)
        # Extract a non-trivial TP mesh if it exists
        tp_mesh = self.extract_tp_mesh(self.device_mesh)
        if tp_mesh is not None and not self.dist_on:
            raise ValueError(f"Distributed is off but received {device_mesh = }.")

        # These attributes depend on the global dist state
        self.global_rank = dist.get_rank() if self.dist_on else 0
        self.world_size = dist.get_world_size() if self.dist_on else 1

        # These attributes depend on the TP state
        if tp_mesh is not None:
            self.tp_size = tp_mesh.size()
            self.tp_group = tp_mesh.get_group()
            self.tp_root_global_rank = dist.get_global_rank(self.tp_group, 0)
            self.tp_local_rank = tp_mesh.get_local_rank()
            # If TP is on, we create a dedicated CPU group, with an eventual timeout
            tp_ranks = dist.get_process_group_ranks(self.tp_group)
            timeout = None if cpu_group_timeout is None else timedelta(seconds=cpu_group_timeout)
            self.cpu_comm_group = dist.new_group(ranks=tp_ranks, backend="gloo", timeout=timeout)
        else:
            self.tp_size = 1
            self.tp_group = None
            self.tp_root_global_rank = 0
            self.tp_local_rank = 0
            self.cpu_comm_group = None

        # The TP driver owns the request queue and scheduler decisions for its TP group. Single-process runs are
        # their own driver.
        self.is_tp_driver = self.infer_if_tp_driver()

        # These attributes depend on the DP state
        self.dp_rank = self.global_rank // self.tp_size
        self.dp_size = self.world_size // self.tp_size

        # Accumulator to CPU integer comm
        self._cpu_int_acc = torch.tensor([0, 0], dtype=torch.int64, device="cpu")

    @staticmethod
    def check_device_mesh_for_cb(device_mesh: DeviceMesh | None) -> None:
        """Checks the validity of the device mesh for continuous batching."""
        # No device mesh = no distributed = life is good
        if device_mesh is None:
            return None
        # If there are no named dims, we assume it is a TP mesh  # TODO (remi): this might change after distrib rework
        if device_mesh.mesh_dim_names is None:
            return None
        # FSDP is not compatible with continuous batching, so we raise an error if it is used
        if "fsdp" in device_mesh.mesh_dim_names and device_mesh["fsdp"].size() > 1:
            raise ValueError(f"FSDP is not compatible with continuous batching but got {device_mesh = }.")

    @staticmethod
    def extract_tp_mesh(device_mesh: DeviceMesh | None) -> DeviceMesh | None:
        """Extracts the TP mesh from the device mesh if it exists and is non-trivial."""
        if device_mesh is None:
            return None
        # Case: device mesh with no named dims => assumed TP mesh
        if device_mesh.mesh_dim_names is None:
            return device_mesh if device_mesh.size() > 1 else None
        # Case: device mesh with named dims => extract the TP mesh
        if "tp" in device_mesh.mesh_dim_names and device_mesh["tp"].size() > 1:
            return device_mesh["tp"]
        return None

    def infer_if_tp_driver(self) -> bool:
        return self.tp_local_rank == 0

    def destroy_cpu_comm_group(self) -> None:
        """Destroys the CPU comm group."""
        if self.cpu_comm_group is not None:
            dist.destroy_process_group(self.cpu_comm_group)
            self.cpu_comm_group = None

    def tp_broadcast_from_rank_0(self, value: torch.Tensor) -> torch.Tensor:
        """Inside each TP group, broadcasts the given value from rank 0 to all other ranks."""
        if self.tp_size > 1:
            dist.broadcast(value, src=self.tp_root_global_rank, async_op=False, group=self.tp_group)
        return value

    def tp_all_reduce_state(self, payload_size: int, stop_status: int) -> tuple[int, int]:
        """Broadcasts two information: 1. the size of the payload held by the TP driver (all other rank broadcast 0) and
        2. the requested stop status (all to all). These information are broadcasted through a MAX-reduce operation."""
        if self.tp_size > 1:
            self._cpu_int_acc[0] = payload_size
            self._cpu_int_acc[1] = stop_status
            dist.all_reduce(self._cpu_int_acc, op=dist.ReduceOp.MAX, async_op=False, group=self.cpu_comm_group)
            payload_size, stop_status = self._cpu_int_acc.tolist()
        return payload_size, stop_status

    def tp_all_reduce_min(self, value: torch.Tensor) -> torch.Tensor:
        """Inside each TP group, all-reduces a tensor with the MIN op. No-op when TP is off."""
        if self.tp_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MIN, group=self.tp_group)
        return value

    def tp_broadcast_object_from_rank_0(self, obj: T) -> T:
        """Inside each TP group, broadcasts an arbitrary picklable Python object from TP-rank 0 to all other ranks.
        Used to keep request ingress and cancellations consistent across TP workers without requiring all ranks to
        receive the same external request stream. Uses a dedicated CPU (gloo) `cpu_comm_group` for broadcast."""
        if self.tp_size <= 1:
            return obj
        holder = [obj] if self.is_tp_driver else [None]
        dist.broadcast_object_list(
            holder, src=self.tp_root_global_rank, group=self.cpu_comm_group, device=torch.device("cpu")
        )
        return holder[0]

    def maybe_warn_nccl_graph_mixing(self) -> None:
        """Throws a warning if TP is on and NCCL's graph mixing support was supposed to be disabled but isn't. That can
        happen if the distributed group is created before graph mixing is disabled. Typically, if the model is
        initialized before the ContinuousBatchingConfig is created."""
        tp_on = self.tp_size > 1
        graph_mixing_not_disabled = os.environ.get("NCCL_GRAPH_MIXING_SUPPORT") != "0"
        if tp_on and graph_mixing_not_disabled:
            logger.warning(
                "NCCL_GRAPH_MIXING_SUPPORT was not set to '0' before init_process_group: performance will be harmed. "
                "Construct your `ContinuousBatchingConfig(...)` BEFORE calling `from_pretrained(tp_plan='auto')`, or "
                "set NCCL_GRAPH_MIXING_SUPPORT=0 in the launch environment."
            )

    def set_tp_seed(self, seed: int | None, model_device: torch.device) -> None:
        # Get an integer seed for the TP group
        if seed is None:
            tp_seed_tensor = torch.randint(0, 2**32 - 1, (1,), dtype=torch.int64, device=model_device)
        else:
            tp_seed_tensor = torch.tensor(seed, dtype=torch.int64, device=model_device)
        # Broadcast the seed to all ranks from rank 0 and memoize it
        tp_seed_tensor = self.tp_broadcast_from_rank_0(tp_seed_tensor)
        tp_seed = tp_seed_tensor.item()
        if self.global_rank == 0 and seed is None:
            logger.info(f"Found no user-specified seed in the config. Setting the config seed to: {tp_seed}.")
        # Set the seed while accounting for DP replicas
        torch.manual_seed(tp_seed + self.dp_rank)
