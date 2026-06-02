# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import os
from copy import deepcopy
from math import prod
from typing import Any, Dict, Optional, OrderedDict, Union
from warnings import warn

import torch

from projects.huggingface.transformers.models.esmfold2.distributed.utils import (
    LayoutMap,
    LayoutRightMap,
)

# grid_group_sizes objects must have
#   (1) .values() attribute, (2) .items() attribute
_GridGroupSizesType = OrderedDict[str, Union[int, tuple[int, ...]]]


class DistributedManager:
    """Borg-style singleton class for managing distributed state.

    Manages the device mesh, process groups, and subgroups for 2D context
    parallelism.  Initialized with e.g.::

        DistributedManager.initialize(OrderedDict([("dp", 1), ("cp", (N, N))]))

    to create a 2D CP grid of size N×N.
    """

    # Borg-style shared state. Every instance's ``__dict__`` is rebound to
    # ``_state`` in ``__new__``; one-time defaults are seeded via ``setdefault``
    # (rather than ``instance._foo = ...`` which pyright flags as "Cannot
    # assign to attribute" on a class without those names declared).
    _state: dict = {}
    _DEFAULT_STATE: dict = {
        "_initialized": False,
        "_has_dist": False,
        "_rank": 0,
        "_world_size": 1,
        "_local_rank": 0,
        "_device": torch.device("cpu"),
        "_backend": None,
        "_device_mesh": None,
        "_layout_device_mesh": None,
        "_has_subgroups": False,
        "_device_mesh_subgroups": None,
        "_layout_device_mesh_subgroups": None,
        "_group": {},
        "_group_rank": {},
        "_group_ranks": {},
        "_subgroups": {},
        "_subgroups_rank": {},
        "_subgroups_ranks": {},
        "_layout_subgroups": {},
        "_method_init": None,
    }

    def __new__(cls):
        instance = super().__new__(cls)
        instance.__dict__ = cls._state
        for key, default in cls._DEFAULT_STATE.items():
            cls._state.setdefault(key, default)
        return instance

    @classmethod
    def methods_init_available(cls) -> set[str]:
        return {"ENV", "SLURM"}

    @classmethod
    def backend_for_device(cls) -> Dict[str, Optional[str]]:
        return {
            "cuda": "nccl" if torch.distributed.is_nccl_available() else None,
            "cpu": "gloo" if torch.distributed.is_gloo_available() else None,
        }

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._state.get("_initialized", False)

    def __init__(self):
        if not self._initialized:
            raise RuntimeError(
                "A DistributedManager instance is being instantiated before "
                "the singleton class is initialized. "
                "Please call DistributedManager.initialize() first."
            )
        super().__init__()

    def __getattr__(self, name: str) -> Any:
        key_state = f"_{name}"
        has_key_shared_state = key_state in self.__dict__
        has_key = name in self.__dict__
        if has_key_shared_state:
            return self.__dict__[key_state]
        elif has_key:
            return self.__dict__[name]
        else:
            raise AttributeError(f'Attribute "{name}" or "_{name}" not found.')

    def __str__(self):
        return (
            f"Initialized process {self.rank} of {self.world_size} using "
            f"method '{self.method_init}'. Device set to {str(self.device)}. "
            f"Backend is {self.backend}"
        )

    @staticmethod
    def _setup(
        grid_group_sizes: Optional[_GridGroupSizesType] = None,
        device_type: str = "cuda",
        backend: Optional[str] = None,
        rank: int = -1,
        node_rank: int = -1,
        world_size: int = -1,
        local_rank: Optional[int] = None,
        addr: str = "localhost",
        port: str = "29500",
        method_init: str = "ENV",
        **kwargs_init_pg,
    ):
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"Input device type {device_type} but torch.cuda is not available"
            )

        if world_size != -1 and grid_group_sizes is not None:
            total_size = 1
            assert hasattr(grid_group_sizes, "values")
            for value in grid_group_sizes.values():
                if isinstance(value, tuple) and all(isinstance(v, int) for v in value):
                    total_size *= prod(value)
                elif isinstance(value, int):
                    total_size *= value
                else:
                    raise RuntimeError(
                        f"Values in grid_group_sizes must be either int or tuple[int, ...], got {type(value)}"
                    )
            if world_size != total_size:
                raise RuntimeError(
                    f"Non-default world_size {world_size} != product of grid_group_sizes values ({total_size})"
                )

        backend_for_device = DistributedManager.backend_for_device()

        if backend_for_device["cpu"] is None and backend_for_device["cuda"] is None:
            raise RuntimeError(
                f"No backend available for the supported device types: {backend_for_device.keys()}"
            )

        if device_type not in backend_for_device:
            raise RuntimeError(
                f"Invalid input device type {device_type}: only supports {backend_for_device.keys()}"
            )

        if backend is None:
            backend = backend_for_device[device_type]
        elif backend != backend_for_device[device_type]:
            raise RuntimeError(
                f"Invalid input backend {backend} for input device type {device_type}"
            )

        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        DistributedManager._state["_initialized"] = True
        manager = DistributedManager()

        manager._has_dist = torch.distributed.is_available()  # type: ignore[assignment]
        manager._rank = rank  # type: ignore[assignment]
        manager._world_size = world_size  # type: ignore[assignment]
        manager._node_rank = node_rank  # type: ignore[assignment]

        if device_type == "cuda":
            if (
                manager.world_size > torch.cuda.device_count()
                and manager.world_size % torch.cuda.device_count()
            ):
                warn("world_size is not a multiple of torch.cuda.device_count()")
            if local_rank is None:
                manager._local_rank = manager.rank % torch.cuda.device_count()  # type: ignore[assignment]
            else:
                manager._local_rank = local_rank  # type: ignore[assignment]
            manager._device = torch.device(f"cuda:{manager.local_rank}")  # type: ignore[assignment]
        else:
            if local_rank is not None:
                manager._local_rank = local_rank  # type: ignore[assignment]
            manager._device = torch.device("cpu")  # type: ignore[assignment]

        if not manager.has_dist:
            warn("DistributedManager initialized without torch.distributed package")
            return

        if manager.device.type == "cuda":
            torch.cuda.set_device(manager.device)
            torch.cuda.device(manager.device)
            torch.cuda.empty_cache()

        manager._backend = backend  # type: ignore[assignment]

        if manager.device.type == "cuda" and backend == "nccl":
            try:
                torch.distributed.init_process_group(
                    manager.backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                    device_id=manager.device,
                    **kwargs_init_pg,
                )
            except TypeError:
                torch.distributed.init_process_group(
                    manager.backend,
                    rank=manager.rank,
                    world_size=manager.world_size,
                    **kwargs_init_pg,
                )
        else:
            torch.distributed.init_process_group(
                manager.backend,
                rank=manager.rank,
                world_size=manager.world_size,
                **kwargs_init_pg,
            )

        manager._group["world"] = torch.distributed.group.WORLD
        manager._group_rank["world"] = manager.rank
        manager._group_ranks["world"] = torch.distributed.get_process_group_ranks(
            manager.group["world"]
        )
        manager._method_init = method_init  # type: ignore[assignment]

        if grid_group_sizes is not None:
            DistributedManager.create_grid_group(grid_group_sizes)

    @staticmethod
    def _create_device_mesh_and_groups(
        name: list[str], shape: list[int], suffix_mesh: Optional[str] = None
    ) -> None:
        if not DistributedManager.is_initialized():
            raise RuntimeError("DistributedManager is not initialized")
        if (
            not DistributedManager._state["_has_dist"]
            or not torch.distributed.is_available()
        ):
            raise RuntimeError(
                "_create_device_mesh_and_groups requires torch.distributed"
            )
        if (
            DistributedManager._state["_method_init"] is None
            or DistributedManager._state["_method_init"]
            not in DistributedManager.methods_init_available()
        ):
            raise RuntimeError(
                f"Invalid DistributedManager method_init {DistributedManager._state['_method_init']}"
            )
        if (
            DistributedManager._state["_backend"] is None
            or DistributedManager._state["_backend"]
            not in DistributedManager.backend_for_device().values()
        ):
            raise RuntimeError(
                f"Invalid DistributedManager backend {DistributedManager._state['_backend']}"
            )
        if (
            DistributedManager._state["_device"] is None
            or DistributedManager._state["_device"].type
            not in DistributedManager.backend_for_device().keys()
        ):
            raise RuntimeError(
                f"Invalid DistributedManager device type {DistributedManager._state['_device'].type}"
            )

        world_size_expected = prod(shape)
        if world_size_expected != DistributedManager._state["_world_size"]:
            raise RuntimeError(
                f"world_size {DistributedManager._state['_world_size']} does not match "
                f"expected {world_size_expected} from shape {shape}"
            )

        device_type = DistributedManager._state["_device"].type
        name_mesh = (
            f"_device_mesh_{suffix_mesh}" if suffix_mesh is not None else "_device_mesh"
        )
        layout = LayoutRightMap(tuple(shape))
        DistributedManager._state[f"_layout{name_mesh}"] = layout

        grid2rank = torch.as_strided(
            torch.arange(world_size_expected), size=layout.shape, stride=layout.strides
        )
        DistributedManager._state[name_mesh] = torch.distributed.device_mesh.DeviceMesh(
            device_type, grid2rank, mesh_dim_names=tuple(name)
        )

        for i_group in range(len(name)):
            name_group = name[i_group]
            if name_group in DistributedManager._state["_group"]:
                continue
            DistributedManager._state["_group"][name_group] = DistributedManager._state[
                name_mesh
            ].get_group(name_group)
            DistributedManager._state["_group_rank"][name_group] = (
                torch.distributed.get_group_rank(
                    DistributedManager._state["_group"][name_group],
                    DistributedManager._state["_rank"],
                )
            )
            DistributedManager._state["_group_ranks"][name_group] = (
                torch.distributed.get_process_group_ranks(
                    DistributedManager._state["_group"][name_group]
                )
            )

    @staticmethod
    def create_grid_group(grid_group_sizes: _GridGroupSizesType) -> None:
        """Create a grid group for 2D context parallelism.

        Example::

            from collections import OrderedDict

            DistributedManager.initialize(OrderedDict([("dp", 1), ("cp", (2, 2))]))
        """
        shape_groups = []
        name_groups = []
        shape_subgroups = []
        name_subgroups = []
        group2subgroup = {}
        group2subgroup_axes = {}
        assert hasattr(grid_group_sizes, "items")
        for k, v in grid_group_sizes.items():
            if isinstance(v, tuple) and all(isinstance(v_i, int) for v_i in v):
                shape_groups.append(prod(v))
                name_groups.append(k)
                shape_subgroups.extend(v)
                names_this_subgroup = [f"{k}_axis_{i}" for i in range(len(v))]
                name_subgroups.extend(names_this_subgroup)
                group2subgroup[k] = names_this_subgroup
                group2subgroup_axes[k] = list(
                    range(len(name_subgroups) - len(v), len(name_subgroups))
                )
            elif isinstance(v, int):
                shape_groups.append(v)
                name_groups.append(k)
                shape_subgroups.append(v)
                name_subgroups.append(k)
            else:
                raise RuntimeError(
                    f"Values in grid_group_sizes must be int or tuple[int, ...], got {type(v)}"
                )

        DistributedManager._create_device_mesh_and_groups(name_groups, shape_groups)
        if (name_groups == name_subgroups) != (shape_groups == shape_subgroups):
            raise RuntimeError("Inconsistent group and subgroup settings")

        DistributedManager._state["_has_subgroups"] = name_groups != name_subgroups
        if DistributedManager._state["_has_subgroups"]:
            if len(group2subgroup) == 0:
                raise RuntimeError(
                    "group2subgroup is empty while _has_subgroups is True"
                )
            DistributedManager._create_device_mesh_and_groups(
                name_subgroups, shape_subgroups, suffix_mesh="subgroups"
            )
            layout = DistributedManager._state["_layout_device_mesh_subgroups"]
            coords = DistributedManager._state[
                "_device_mesh_subgroups"
            ].get_coordinate()
            for name_group, subgroup_names in group2subgroup.items():
                DistributedManager._state["_subgroups"][name_group] = [
                    DistributedManager._state["_group"][n] for n in subgroup_names
                ]
                DistributedManager._state["_subgroups_ranks"][name_group] = [
                    DistributedManager._state["_group_ranks"][n] for n in subgroup_names
                ]
                DistributedManager._state["_subgroups_rank"][name_group] = [
                    DistributedManager._state["_group_rank"][n] for n in subgroup_names
                ]
                axes_subgroup = group2subgroup_axes[name_group]
                slices = deepcopy(coords)
                for axis in axes_subgroup:
                    slices[axis] = slice(None)
                layout_subgroup = layout[tuple(slices)]
                DistributedManager._state["_layout_subgroups"][name_group] = LayoutMap(
                    layout_subgroup.strides, layout_subgroup.shape, offset=0
                )

    @staticmethod
    def create_group(name: str, ranks: list[int], **kwargs_dist_ng) -> None:
        DistributedManager._state["_group"][name] = torch.distributed.new_group(
            ranks=ranks, **kwargs_dist_ng
        )
        DistributedManager._state["_group_ranks"][name] = ranks
        DistributedManager._state["_group_rank"][name] = (
            torch.distributed.get_group_rank(
                DistributedManager._state["_group"][name],
                DistributedManager._state["_rank"],
            )
        )

    @staticmethod
    def _initialize_env(*args, **kwargs):
        if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
            raise RuntimeError(
                "environment variables RANK and WORLD_SIZE must be set for env:// initialization"
            )
        rank = os.environ.get("RANK")
        world_size = os.environ.get("WORLD_SIZE")
        local_rank = os.environ.get("LOCAL_RANK")
        group_rank = os.environ.get("GROUP_RANK", 0)
        node_rank = int(os.environ.get("NODE_RANK", group_rank))
        try:
            rank = int(rank)  # type: ignore[arg-type]
            world_size = int(world_size)  # type: ignore[arg-type]
            if local_rank is not None:
                local_rank = int(local_rank)
        except TypeError:
            raise RuntimeError(
                "environment variables RANK, LOCAL_RANK and WORLD_SIZE must be integers"
            )
        DistributedManager._setup(
            *args,
            rank=rank,
            node_rank=node_rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=os.environ.get("MASTER_ADDR"),  # type: ignore[arg-type]
            port=os.environ.get("MASTER_PORT"),  # type: ignore[arg-type]
            method_init="ENV",
            **kwargs,
        )

    @staticmethod
    def _initialize_slurm(*args, **kwargs):
        keys = (
            "SLURM_PROCID",
            "SLURM_NPROCS",
            "SLURM_LOCALID",
            "SLURM_LAUNCH_NODE_IPADDR",
        )
        if not all(k in os.environ for k in keys):
            raise RuntimeError(
                f"environment variables {keys} must be set for SLURM initialization"
            )
        rank = os.environ.get("SLURM_PROCID")
        node_rank = int(os.environ.get("SLURM_NODEID", 0))
        world_size = os.environ.get("SLURM_NPROCS")
        local_rank = os.environ.get("SLURM_LOCALID")
        addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")
        try:
            rank = int(rank)  # type: ignore[arg-type]
            world_size = int(world_size)  # type: ignore[arg-type]
            if local_rank is not None:
                local_rank = int(local_rank)
        except TypeError:
            raise RuntimeError(
                "environment variables SLURM_{PROCID,NPROCS,LOCALID} must be integers"
            )
        DistributedManager._setup(
            *args,
            rank=rank,
            node_rank=node_rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,  # type: ignore[arg-type]
            method_init="SLURM",
            **kwargs,
        )

    @staticmethod
    def initialize(
        grid_group_sizes: Optional[OrderedDict[str, int | tuple[int, ...]]] = None,
        device_type: str = "cuda",
        backend: Optional[str] = None,
        **kwargs_init_pg,
    ):
        """Initialize the DistributedManager singleton.

        Parameters
        ----------
        grid_group_sizes:
            E.g. ``OrderedDict([("dp", 1), ("cp", (2, 2))])`` for a 2×2 CP grid.
        device_type:
            "cuda" (default) or "cpu".
        backend:
            Defaults to nccl for cuda, gloo for cpu.
        """
        if DistributedManager.is_initialized():
            warn("DistributedManager is already initialized. Skip initialize()")
            return
        if backend == "nccl":
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        method_init = os.getenv("ESMCFOLD_DISTRIBUTED_INIT_METHOD")
        if (
            method_init is not None
            and method_init not in DistributedManager.methods_init_available()
        ):
            raise ValueError(
                f"Unknown ESMCFOLD_DISTRIBUTED_INIT_METHOD={method_init}. "
                f"Allowed: {DistributedManager.methods_init_available()}"
            )
        if method_init is None:
            try:
                DistributedManager._initialize_env(
                    grid_group_sizes,
                    device_type=device_type,
                    backend=backend,
                    **kwargs_init_pg,
                )
            except RuntimeError as except_env:
                try:
                    DistributedManager._initialize_slurm(
                        grid_group_sizes,
                        device_type=device_type,
                        backend=backend,
                        **kwargs_init_pg,
                    )
                except RuntimeError as except_slurm:
                    warn(
                        "Could not initialize DistributedManager with env:// nor slurm.\n"
                        f"Error env://: {except_env}\n"
                        f"Error slurm: {except_slurm}\n"
                        "Will default initialize DistributedManager"
                    )
                    DistributedManager._state["_initialized"] = True
        elif method_init == "ENV":
            DistributedManager._initialize_env(
                grid_group_sizes,
                device_type=device_type,
                backend=backend,
                **kwargs_init_pg,
            )
        elif method_init == "SLURM":
            DistributedManager._initialize_slurm(
                grid_group_sizes,
                device_type=device_type,
                backend=backend,
                **kwargs_init_pg,
            )

    @staticmethod
    def cleanup():
        if DistributedManager._state.get("_group", {}) != {}:
            if torch.distributed.is_initialized():
                if (
                    DistributedManager._state["_device"].type == "cuda"
                    and torch.cuda.is_available()
                ):
                    torch.distributed.barrier(
                        device_ids=[DistributedManager._state["_local_rank"]]
                    )
                else:
                    torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            else:
                DistributedManager._state = {}
