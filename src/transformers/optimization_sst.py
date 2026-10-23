# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
"""Optimizer-independent Subspace Tuning, including sharded parameter writeback."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.optim import Optimizer


SST_STATE_NAME = "sst_state.pt"


@dataclass
class _SSTBasis:
    left: torch.Tensor
    singular_values: torch.Tensor
    right: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return self.left.shape[0], self.right.shape[0]


def _canonical_name(name: str) -> str:
    for prefix in ("_fsdp_wrapped_module.", "_orig_mod.", "_checkpoint_wrapped_module."):
        name = name.replace(prefix, "")
    return name


def _matrix_shape(parameter: torch.Tensor) -> tuple[int, ...]:
    return tuple(getattr(parameter, "ds_shape", parameter.shape))


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return dtype
    return torch.complex64 if dtype.is_complex else torch.float32


def _optimizer_defaults(optimizer) -> dict[str, Any]:
    """Some engine proxies expose groups/state but keep defaults on the underlying optimizer."""
    defaults = getattr(optimizer, "defaults", None)
    while defaults is None and hasattr(optimizer, "optimizer"):
        optimizer = optimizer.optimizer
        defaults = getattr(optimizer, "defaults", None)
    if defaults is None:
        raise TypeError("SST could not find the inner optimizer's defaults.")
    return defaults


def _unit_parameters(module: nn.Module, prefix: str = "", root: bool = True) -> Iterator[tuple[str, nn.Parameter]]:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if not root and isinstance(module, FSDP):
        return
    for name, parameter in module.named_parameters(recurse=False):
        yield _canonical_name(f"{prefix}{name}"), parameter
    for name, child in module.named_children():
        yield from _unit_parameters(child, f"{prefix}{name}.", root=False)


class SST(Optimizer):
    r"""
    Wrap an optimizer and project its applied updates outside principal weight subspaces.

    The inner optimizer receives the original gradients and retains its own momentum, variance, weight decay, and
    parameter groups. After its step, SST restores each selected matrix to
    `U diag(s) Vᴴ + (I - UUᴴ) W (I - VVᴴ)`. Here `(U, s, V)` are the protected singular triples of the initial weight,
    or of the current weight at the latest explicit or scheduled refresh. The basis is fixed between refreshes.

    Args:
        optimizer (`torch.optim.Optimizer`):
            The optimizer to wrap. Accelerate's prepared optimizer wrappers are also supported.
        model (`torch.nn.Module`, *optional*):
            The model owning the parameters. Required for module-name selection, legacy FSDP flat parameters, and
            DeepSpeed's fp32 master-parameter access. Bind the prepared/sharded model when using those backends.
        rank_fraction (`float`, *optional*, defaults to 0.99):
            Fraction of principal singular directions to protect. The rank is rounded down and clamped to
            `[1, min(rows, columns) - 1]`, leaving a nonempty update subspace.
        target_modules (`str` or `Sequence[str]`, *optional*):
            Module suffixes or full-match regular expressions. `"all-linear"` selects `torch.nn.Linear` weights.
            By default, all trainable matrices owned by the inner optimizer with both dimensions at least two are
            selected. Biases and vectors are not projected.
        initialize (`bool`, *optional*, defaults to `True`):
            Build the initial basis immediately. Set to `False` when restoring a checkpoint, then call `load_state_dict`
            or `load_sst_state_dict` before stepping. A missing basis is never silently rebuilt from trained weights.
        refresh_every (`int`, *optional*):
            Recompute the basis after this many successful, projected optimizer updates. `None` keeps the initial
            basis fixed. Accumulation and AMP-overflow skips do not advance the refresh clock. A manual `refresh()`
            restarts the interval. Checkpoints restore the saved schedule, update count, and active basis.

    Example:

    ```python
    >>> import torch
    >>> from transformers import SST
    >>> model = torch.nn.Linear(8, 8)
    >>> inner_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    >>> optimizer = SST(inner_optimizer, model=model, rank_fraction=0.5)
    >>> model(torch.ones(2, 8)).sum().backward()
    >>> optimizer.step()
    >>> optimizer.zero_grad()
    ```

    FSDP1 gathers and writes back one wrapping unit at a time. FSDP2/DTensor and DeepSpeed assemble one selected matrix
    at a time and write its projected value back to the original shards/master parameters. These operations are
    collective: every participating rank must call `step` and any manual `refresh` in the same order, with the same
    refresh schedule. Refreshed bases are synchronized within the parameter's owning process groups, not across
    independent submeshes. Exact projection needs fp32/fp64 optimizer parameters;
    writing directly to bf16/fp16 parameters introduces rounding error.
    """

    _step_supports_amp_scaling = False

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        model: nn.Module | None = None,
        rank_fraction: float = 0.99,
        target_modules: str | Sequence[str] | None = None,
        initialize: bool = True,
        refresh_every: int | None = None,
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("SST requires a torch.optim.Optimizer or a prepared Accelerate optimizer.")
        if isinstance(optimizer, SST):
            raise ValueError("This optimizer is already wrapped by SST.")
        if not math.isfinite(rank_fraction) or not 0.0 < rank_fraction < 1.0:
            raise ValueError("rank_fraction must be finite and strictly between 0 and 1.")
        self._validate_refresh_every(refresh_every)
        defaults = _optimizer_defaults(optimizer)
        if defaults.get("differentiable", False) or any(
            group.get("differentiable", False) for group in optimizer.param_groups
        ):
            raise ValueError("SST's post-step projection does not support differentiable=True.")
        if target_modules is not None and model is None:
            raise ValueError("Pass model= when selecting SST target modules by name.")
        if target_modules is not None and not isinstance(target_modules, (str, Sequence)):
            raise TypeError("target_modules must be a string or a sequence of strings.")
        patterns = (target_modules,) if isinstance(target_modules, str) else target_modules or ()
        if any(not isinstance(pattern, str) for pattern in patterns):
            raise TypeError("Every SST target pattern must be a string.")
        for pattern in patterns:
            re.compile(pattern)

        self.optimizer = optimizer
        self.model = model
        self.rank_fraction = float(rank_fraction)
        self.target_modules = target_modules
        self.refresh_every = refresh_every
        self.num_updates = 0
        self.last_refresh_step = 0
        self.num_refreshes = 0
        self._basis: dict[str, _SSTBasis] = {}
        self._warned_low_precision: set[str] = set()
        self._initializing = True
        super().__init__(optimizer.param_groups, defaults)
        self._initializing = False
        self._sync_optimizer_state()
        if initialize:
            self.initialize()

    def __getattr__(self, name: str):
        optimizer = self.__dict__.get("optimizer")
        if optimizer is None:
            raise AttributeError(name)
        return getattr(optimizer, name)

    def __getstate__(self):
        return {
            **super().__getstate__(),
            "optimizer": self.optimizer,
            "model": self.model,
            "rank_fraction": self.rank_fraction,
            "target_modules": self.target_modules,
            "refresh_every": self.refresh_every,
            "num_updates": self.num_updates,
            "last_refresh_step": self.last_refresh_step,
            "num_refreshes": self.num_refreshes,
            "_basis": self._basis,
            "_warned_low_precision": self._warned_low_precision,
            "_initializing": False,
        }

    def __setstate__(self, state):
        super().__setstate__(state)
        self._sync_optimizer_state()

    def _sync_optimizer_state(self) -> None:
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.defaults = _optimizer_defaults(self.optimizer)

    @property
    def is_initialized(self) -> bool:
        """Whether the active basis has been initialized or restored."""
        return bool(self._basis)

    @staticmethod
    def _validate_refresh_every(value) -> None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError("refresh_every must be a positive integer or None.")

    def _checkpoint_topology(self, target_names=None):
        if not torch.distributed.is_available():
            return None
        from torch.distributed import get_process_group_ranks, get_rank, get_world_size, is_initialized

        if not is_initialized():
            return None
        world_size = get_world_size()
        groups = {}
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.tensor import DTensor

        units = (
            []
            if self.model is None
            else [(name, module) for name, module in self.model.named_modules() if isinstance(module, FSDP)]
        )
        for name, unit in units:
            if get_world_size(unit.process_group) != world_size:
                groups[f"fsdp:{_canonical_name(name)}"] = get_process_group_ranks(unit.process_group)
        if not units:
            for name, parameter in self._parameters():
                if target_names is not None:
                    if name not in target_names:
                        continue
                elif self._basis:
                    if name not in self._basis:
                        continue
                elif not self._selected(name, parameter, self.target_modules):
                    continue
                if isinstance(parameter, DTensor):
                    mesh = parameter.device_mesh.mesh
                    if mesh.numel() != world_size:
                        groups[name] = {"shape": list(mesh.shape), "ranks": mesh.flatten().tolist()}
                else:
                    group = getattr(parameter, "ds_process_group", getattr(parameter, "_dp_group", None))
                    if group is not None and get_world_size(group) != world_size:
                        groups[name] = get_process_group_ranks(group)
        if groups:
            return {"world_size": world_size, "rank": get_rank(), "groups": groups}
        return None

    @property
    def requires_per_rank_state(self) -> bool:
        """Whether submesh-owned targets require rank-specific active-basis checkpoints."""
        return self._checkpoint_topology() is not None

    def _parameters(self, *, writeback: bool = False) -> Iterator[tuple[str, nn.Parameter]]:
        owned = {id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]}
        if self.model is None:
            seen = set()
            index = 0
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if id(parameter) not in seen:
                        seen.add(id(parameter))
                        yield f"parameter_{index}", parameter
                        index += 1
            return

        inner = self.optimizer
        while inner is not None:
            # ZeRO keeps original owned parameters here; a rank's local fp32 mapping can be None.
            for attribute in ("bit16_groups", "fp16_groups"):
                for group in getattr(inner, attribute, ()):
                    owned.update(id(parameter) for parameter in group)
            inner = getattr(inner, "optimizer", None)

        model = self.model
        while isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
            model = model.module
        model = cast(nn.Module, model)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        units = [(name, module) for name, module in model.named_modules() if isinstance(module, FSDP)]
        if units:
            for prefix, unit in units:
                if not any(id(parameter) in owned for _, parameter in _unit_parameters(unit)):
                    continue
                with FSDP.summon_full_params(unit, recurse=False, writeback=writeback):
                    parameters = list(_unit_parameters(unit))
                    uses_original_parameters = any(id(parameter) in owned for _, parameter in parameters)
                    seen = set()
                    for name, parameter in parameters:
                        if id(parameter) in seen:
                            continue
                        seen.add(id(parameter))
                        if not uses_original_parameters or id(parameter) in owned:
                            yield _canonical_name(f"{prefix}.{name}" if prefix else name), parameter
            return

        for name, parameter in model.named_parameters():
            if id(parameter) in owned:
                yield _canonical_name(name), parameter

    def _selected(self, name: str, parameter: nn.Parameter, targets=None) -> bool:
        shape = _matrix_shape(parameter)
        if not parameter.requires_grad or len(shape) != 2 or min(shape) < 2:
            return False
        if targets is None:
            return True
        module_name = name.removesuffix(".weight")
        if isinstance(targets, str) and targets.replace("_", "-") == "all-linear":
            model = self.model
            while isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
                model = model.module
            model = cast(nn.Module, model)
            modules = {_canonical_name(key): module for key, module in model.named_modules()}
            return name.endswith("weight") and isinstance(modules.get(module_name), nn.Linear)
        patterns = (targets,) if isinstance(targets, str) else targets
        return any(
            module_name == pattern or module_name.endswith(f".{pattern}") or re.fullmatch(pattern, module_name)
            for pattern in patterns
        )

    @staticmethod
    def _read_matrix(parameter: nn.Parameter) -> torch.Tensor:
        if hasattr(parameter, "ds_id") or hasattr(parameter, "_hp_mapping"):
            from deepspeed.utils import safe_get_full_fp32_param

            matrix = safe_get_full_fp32_param(parameter)
            if matrix is None:
                raise RuntimeError("DeepSpeed did not return an SST target's full fp32 optimizer parameter.")
            return matrix
        from torch.distributed.tensor import DTensor

        if isinstance(parameter, DTensor):
            matrix = parameter.detach()
            if matrix.device.type != matrix.device_mesh.device_type:
                matrix = matrix.to(matrix.device_mesh.device_type)
            return matrix.full_tensor()
        return parameter.detach()

    @staticmethod
    def _write_matrix(parameter: nn.Parameter, matrix: torch.Tensor) -> None:
        if hasattr(parameter, "ds_id") or hasattr(parameter, "_hp_mapping"):
            from deepspeed.utils import safe_set_full_fp32_param

            safe_set_full_fp32_param(parameter, matrix)
            if hasattr(parameter, "ds_id"):
                from deepspeed import zero

                with zero.GatheredParameters([parameter], modifier_rank=0):
                    parameter.copy_(matrix.to(device=parameter.device, dtype=parameter.dtype))
            else:
                parameter.copy_(matrix.to(device=parameter.device, dtype=parameter.dtype))
            return
        from torch.distributed.tensor import DTensor, distribute_tensor

        if isinstance(parameter, DTensor):
            shard = distribute_tensor(
                matrix.to(dtype=parameter.dtype),
                device_mesh=parameter.device_mesh,
                placements=parameter.placements,
            )
            parameter.to_local().copy_(shard.to_local())
        else:
            parameter.copy_(matrix.to(device=parameter.device, dtype=parameter.dtype))

    def _basis_groups(self, name: str, parameter: nn.Parameter) -> list[Any]:
        if not torch.distributed.is_available():
            return []
        from torch.distributed import get_world_size, is_initialized
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.tensor import DTensor

        if not is_initialized():
            return []
        if isinstance(parameter, DTensor):
            groups = [parameter.device_mesh.get_group(dim) for dim in range(parameter.device_mesh.ndim)]
        else:
            group = getattr(parameter, "ds_process_group", None) or getattr(parameter, "_dp_group", None)
            groups = [group] if group is not None else []
            model = self.model
            while isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
                if not groups and isinstance(model, nn.parallel.DistributedDataParallel):
                    groups = [model.process_group]
                model = model.module
            if not groups and model is not None:
                model = cast(nn.Module, model)
                units = [
                    (_canonical_name(prefix), unit)
                    for prefix, unit in model.named_modules()
                    if isinstance(unit, FSDP)
                    and (not _canonical_name(prefix) or name.startswith(_canonical_name(prefix) + "."))
                ]
                if units:
                    _, unit = max(units, key=lambda item: len(item[0]))
                    groups = [unit.process_group]
                    replica_group = getattr(unit, "_inter_node_pg", None)
                    if replica_group is not None:
                        groups.append(replica_group)
        seen = set()
        unique = []
        for group in groups:
            if get_world_size(group) > 1 and id(group) not in seen:
                unique.append(group)
                seen.add(id(group))
        return unique

    @staticmethod
    def _reduce_basis_tensor(tensor: torch.Tensor, group, operation: str) -> None:
        from torch.distributed import ReduceOp, all_reduce, get_backend

        device = tensor.device
        backend = get_backend(group)
        if backend == "nccl" and device.type != "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        elif backend == "gloo":
            device = torch.device("cpu")
        value = tensor.to(device).resolve_conj().contiguous()
        all_reduce(
            torch.view_as_real(value) if value.is_complex() else value,
            op=getattr(ReduceOp, operation),
            group=group,
        )
        if value is not tensor:
            tensor.copy_(value.to(tensor.device))

    def _make_basis(self, name: str, parameter: nn.Parameter) -> _SSTBasis:
        matrix = self._read_matrix(parameter)
        if matrix.is_meta:
            raise ValueError("SST cannot initialize a basis from a meta parameter. Materialize the model first.")
        matrix = matrix.to(dtype=_work_dtype(matrix.dtype))
        rank = min(max(1, math.floor(min(matrix.shape) * self.rank_fraction)), min(matrix.shape) - 1)
        groups = self._basis_groups(name, parameter)
        if groups:
            from torch.distributed import get_rank

            source = torch.tensor(get_rank(), dtype=torch.int64, device=matrix.device)
            for group in groups:
                self._reduce_basis_tensor(source, group, "MIN")
            compute = get_rank() == source.item()
        else:
            compute = True
        left = values = right = None
        error = None
        try:
            if compute:
                left, values, vh = torch.linalg.svd(matrix, full_matrices=False)
                left = left[:, :rank].contiguous()
                values = values[:rank].contiguous()
                right = vh[:rank].mH.resolve_conj().contiguous()
                if not all(torch.isfinite(tensor).all() for tensor in (left, values, right)):
                    raise ValueError("SVD returned a nonfinite basis.")
            else:
                left = matrix.new_zeros((matrix.shape[0], rank))
                values = matrix.real.new_zeros(rank)
                right = matrix.new_zeros((matrix.shape[1], rank))
        except Exception as exception:
            error = exception
        failed = torch.tensor(int(error is not None), dtype=torch.int64, device=matrix.device)
        for group in groups:
            self._reduce_basis_tensor(failed, group, "MAX")
        if failed.item():
            raise RuntimeError(
                f"Could not compute SST basis for {name!r}; the active basis was not replaced."
            ) from error
        assert left is not None and values is not None and right is not None
        # Only the source contributes; successive mesh dimensions distribute its basis to every owner.
        for group in groups:
            for tensor in (left, values, right):
                self._reduce_basis_tensor(tensor, group, "SUM")
        return _SSTBasis(left.cpu(), values.cpu(), right.cpu())

    def _build_bases(self, *, refresh: bool) -> dict[str, _SSTBasis]:
        bases = {} if refresh else self._basis.copy()
        for name, parameter in self._parameters():
            if refresh:
                if name not in self._basis:
                    continue
                if _matrix_shape(parameter) != self._basis[name].shape:
                    raise ValueError(f"SST target {name!r} changed shape after initialization.")
            elif name in bases or not self._selected(name, parameter, self.target_modules):
                continue
            bases[name] = self._make_basis(name, parameter)
        if not bases:
            raise ValueError("No trainable SST target matrices matched the optimizer and target_modules.")
        if refresh and set(bases) != set(self._basis):
            raise ValueError(f"SST optimizer lost target parameters: {sorted(set(self._basis) - set(bases))}.")
        return bases

    @torch.no_grad()
    def initialize(self) -> None:
        """Initialize only new targets; do not refresh existing bases."""
        self._basis = self._build_bases(refresh=False)

    @torch.no_grad()
    def refresh(self) -> None:
        """Re-anchor to current weights without changing parameters or inner state; all owning ranks must call."""
        if not self._basis:
            raise RuntimeError("Initialize or restore SST's basis before refreshing.")
        self._basis = self._build_bases(refresh=True)
        self.last_refresh_step = self.num_updates
        self.num_refreshes += 1

    @torch.no_grad()
    def project(self) -> None:
        """Project the applied update without modifying gradients or inner optimizer state."""
        if not self._basis:
            raise RuntimeError("SST's active basis is not initialized. Restore its checkpoint or call initialize().")
        visited = set()
        for name, parameter in self._parameters(writeback=True):
            if name not in self._basis:
                continue
            basis = self._basis[name]
            if _matrix_shape(parameter) != basis.shape:
                raise ValueError(f"SST target {name!r} changed shape after initialization.")
            matrix = self._read_matrix(parameter)
            if matrix.dtype in (torch.float16, torch.bfloat16) and name not in self._warned_low_precision:
                warnings.warn(
                    f"SST is writing {name} directly in {matrix.dtype}. Use fp32 optimizer parameters or master "
                    "weights for exact projection; low-precision parameter writeback is approximate.",
                    stacklevel=2,
                )
                self._warned_low_precision.add(name)
            matrix = matrix.to(dtype=basis.left.dtype)
            left = basis.left.to(matrix.device)
            right = basis.right.to(matrix.device)
            singular_values = basis.singular_values.to(matrix.device)
            free = matrix - left @ (left.mH @ matrix)
            free = free - (free @ right) @ right.mH
            projected = (left * singular_values.unsqueeze(0)) @ right.mH + free
            self._write_matrix(parameter, projected)
            visited.add(name)
        if visited != set(self._basis):
            raise ValueError(f"SST optimizer lost target parameters: {sorted(set(self._basis) - visited)}.")

    def step(self, closure=None, **kwargs):
        """Run the inner optimizer, then project unless its mixed-precision step was skipped."""
        if not self._basis:
            raise RuntimeError("Restore SST's active basis or call initialize() before stepping.")
        result = self.optimizer.step(**kwargs) if closure is None else self.optimizer.step(closure=closure, **kwargs)
        self._sync_optimizer_state()
        gradient_state = getattr(self.optimizer, "gradient_state", None)
        accumulating = gradient_state is not None and not gradient_state.sync_gradients
        if not accumulating and not getattr(self.optimizer, "step_was_skipped", False):
            self.project()
            self.num_updates += 1
            if self.refresh_every is not None and self.num_updates - self.last_refresh_step >= self.refresh_every:
                self.refresh()
        return result

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        if self._initializing:
            return super().add_param_group(param_group)
        self.optimizer.add_param_group(param_group)
        self._sync_optimizer_state()
        self.initialize()

    def sst_state_dict(self) -> dict[str, Any]:
        """Return the active basis and refresh clock separately for engines that transform optimizer state."""
        if not self._basis:
            raise RuntimeError("SST's active basis has not been initialized or restored.")
        return {
            "version": 2,
            "topology": self._checkpoint_topology(),
            "rank_fraction": self.rank_fraction,
            "target_modules": self.target_modules,
            "refresh_every": self.refresh_every,
            "num_updates": self.num_updates,
            "last_refresh_step": self.last_refresh_step,
            "num_refreshes": self.num_refreshes,
            "protected_weights": [
                {"name": name, "left": basis.left, "singular_values": basis.singular_values, "right": basis.right}
                for name, basis in self._basis.items()
            ],
        }

    def load_sst_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the saved basis and refresh clock without an SVD or parameter/state projection."""
        version = state_dict.get("version", 1)
        if version not in (1, 2):
            raise ValueError("Unsupported SST checkpoint version.")
        if version == 2 and "refresh_every" not in state_dict:
            raise ValueError("SST checkpoint refresh_every is missing.")
        refresh_every = state_dict.get("refresh_every") if version == 2 else None
        self._validate_refresh_every(refresh_every)
        counters = {
            key: state_dict.get(key) if version == 2 else 0
            for key in ("num_updates", "last_refresh_step", "num_refreshes")
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters.values()):
            raise ValueError("SST checkpoint refresh counters must be nonnegative integers.")
        if counters["last_refresh_step"] > counters["num_updates"] or (
            counters["num_refreshes"] == 0 and counters["last_refresh_step"] != 0
        ):
            raise ValueError("Inconsistent SST checkpoint refresh counters.")
        fraction = float(state_dict["rank_fraction"])
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("Invalid SST checkpoint rank_fraction.")
        targets = state_dict.get("target_modules", self.target_modules)
        bases = {}
        for item in state_dict["protected_weights"]:
            name = item["name"]
            if name in bases:
                raise ValueError(f"Duplicate SST checkpoint target {name!r}.")
            left = item["left"].detach().contiguous().cpu()
            right = item["right"].detach().contiguous().cpu()
            values = item["singular_values"].detach().contiguous().cpu()
            if left.ndim != 2 or right.ndim != 2 or values.ndim != 1:
                raise ValueError(f"Invalid SST basis dimensions for {name!r}.")
            if (
                left.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128)
                or right.dtype != left.dtype
                or values.dtype != left.real.dtype
            ):
                raise ValueError(f"Invalid SST basis dtypes for {name!r}.")
            if left.shape[1] != values.numel() or right.shape[1] != values.numel():
                raise ValueError(f"Inconsistent SST basis rank for {name!r}.")
            if not 0 < values.numel() < min(left.shape[0], right.shape[0]):
                raise ValueError(f"Invalid SST protected rank for {name!r}.")
            if not all(torch.isfinite(tensor).all() for tensor in (left, right, values)):
                raise ValueError(f"Nonfinite SST basis for {name!r}.")
            if (values < 0).any():
                raise ValueError(f"Negative SST singular values for {name!r}.")
            bases[name] = _SSTBasis(left, values, right)
        current = {
            name: _matrix_shape(parameter)
            for name, parameter in self._parameters()
            if self._selected(name, parameter, targets)
        }
        if not bases or set(current) != set(bases):
            raise ValueError(f"SST checkpoint targets do not match: current={sorted(current)}, saved={sorted(bases)}.")
        for name, basis in bases.items():
            if current[name] != basis.shape:
                raise ValueError(f"SST checkpoint basis shape does not match {name!r}.")
        if state_dict.get("topology") != self._checkpoint_topology(target_names=set(bases)):
            raise ValueError(
                "SST's per-rank basis checkpoint requires matching distributed rank and submesh topology."
            )
        self.rank_fraction = fraction
        self.target_modules = targets
        self.refresh_every = refresh_every
        self.num_updates = counters["num_updates"]
        self.last_refresh_step = counters["last_refresh_step"]
        self.num_refreshes = counters["num_refreshes"]
        self._basis = bases
        self._sync_optimizer_state()

    def state_dict(self) -> dict[str, Any]:
        for hook in self._optimizer_state_dict_pre_hooks.values():
            hook(self)
        state = dict(self.optimizer.state_dict())
        if "sst" in state:
            raise ValueError("The inner optimizer already uses the reserved SST checkpoint key.")
        state["sst"] = self.sst_state_dict()
        for hook in self._optimizer_state_dict_post_hooks.values():
            result = hook(self, state)
            if result is not None:
                state = result
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = state_dict.copy()
        for hook in self._optimizer_load_state_dict_pre_hooks.values():
            result = hook(self, state)
            if result is not None:
                state = result
        if "sst" not in state:
            raise ValueError("SST's active basis is missing from this optimizer checkpoint.")
        self.load_sst_state_dict(state.pop("sst"))
        self.optimizer.load_state_dict(state)
        self._sync_optimizer_state()
        for hook in self._optimizer_load_state_dict_post_hooks.values():
            hook(self)
