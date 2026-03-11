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

import math
import pickle
from contextlib import contextmanager
from typing import Iterable, Sequence

from ..utils import logging
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.autograd.function import Function


logger = logging.get_logger(__name__)


class PipelineParallelSkipped(nn.Module):
    """Placeholder module used on ranks that do not own a stage.

    It simply returns the first positional argument or the value of
    ``hidden_states`` if provided. This keeps the module graph valid while
    bypassing the computation on non-local pipeline stages.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        if "hidden_states" in kwargs:
            return kwargs["hidden_states"]
        return args[0] if len(args) > 0 else None


@contextmanager
def _silent_non_primary(local_rank: int):
    """Silence stdout/stderr on non-primary ranks to reduce clutter."""

    if local_rank == 0:
        yield
    else:
        import os
        import sys

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        try:
            yield
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout, sys.stderr = old_stdout, old_stderr


def initialize_pipeline_parallelism(pp_plan: dict[str, tuple[list[str], list[str]]] | None, pp_size: int | None = None, device_mesh=None, device_map=None):
    """Infer pipeline parallel rank/size and device mapping.

    This mirrors ``initialize_tensor_parallelism`` but only sets PP-related
    attributes. When a multi-dimensional ``device_mesh`` is provided, the
    ``"pp"`` dimension is used; otherwise ``WORLD_SIZE``/``RANK`` env vars are
    consulted.
    """

    if not is_torch_available():
        raise ImportError("PyTorch is required for pipeline parallelism")

    if device_mesh is not None:
        # If the mesh is multi-dim, use the "pp" named dimension when present.
        if device_mesh.ndim > 1 and "pp" in (device_mesh.mesh_dim_names or {}):
            pp_mesh = device_mesh["pp"]
        else:
            pp_mesh = device_mesh
        pp_size = pp_mesh.size()
        pp_rank = pp_mesh.get_local_rank()
        return device_map, pp_mesh, pp_size, pp_rank

    # Fallback to env-derived initialization (single-dimension pipeline only)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for pipeline parallelism")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before using pipeline parallelism")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    pp_size = pp_size or world_size
    if world_size != pp_size:
        raise ValueError("When no device_mesh is provided, world_size must equal pp_size")
    return device_map, None, pp_size, rank


def _expand_module_name(model: nn.Module, name: str) -> list[str]:
    """Expand ModuleList/Sequential entries into fully qualified names."""

    try:
        module = model.get_submodule(name)
    except Exception:
        return [name]

    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        return [f"{name}.{idx}" for idx, _ in enumerate(module)]
    return [name]


def flatten_pp_units(model: nn.Module, pp_plan: dict[str, tuple[list[str], list[str]]]) -> list[tuple[str, nn.Module, list[str], list[str]]]:
    """Return ordered pipeline units expanded from ``pp_plan`` keys."""

    units = []
    for key, (inputs, outputs) in pp_plan.items():
        for expanded in _expand_module_name(model, key):
            try:
                mod = model.get_submodule(expanded)
            except Exception:
                continue
            units.append((expanded, mod, inputs, outputs))
    return units


def partition_units(units: list[tuple[str, nn.Module, list[str], list[str]]], pp_size: int) -> list[list[str]]:
    """Evenly partition units into ``pp_size`` contiguous chunks."""

    if pp_size <= 0:
        raise ValueError("pp_size must be positive")
    chunk = math.ceil(len(units) / pp_size) if len(units) > 0 else 1
    partitions = []
    for idx in range(pp_size):
        start = idx * chunk
        stop = min((idx + 1) * chunk, len(units))
        partitions.append([name for name, _, _, _ in units[start:stop]])
    return partitions


def _group_ranks(group):
    """Return (local_rank, global_rank, world_size) for a process group or the default group."""

    local_rank = dist.get_rank(group) if group is not None else dist.get_rank()
    global_rank = dist.get_global_rank(group, local_rank) if group is not None else local_rank
    world_size = dist.get_world_size(group) if group is not None else dist.get_world_size()
    return local_rank, global_rank, world_size


def _use_cuda_stream(tensors: Sequence[torch.Tensor]) -> bool:
    return any(t.is_cuda for t in tensors) and torch.cuda.is_available()


def _global_to_group_rank(group, global_rank: int) -> int:
    return dist.get_group_rank(group, global_rank) if group is not None else global_rank


def send_tensors(tensors: Sequence[torch.Tensor], dst: int, group=None, use_cuda_stream: bool = True) -> None:
    """Send tensors with shape/dtype metadata to ``dst``."""

    if len(tensors) == 0:
        return
    meta = [(tuple(t.shape), t.dtype) for t in tensors]
    meta_bytes = pickle.dumps(meta)
    meta_tensor = torch.tensor(list(meta_bytes), dtype=torch.uint8)
    meta_len = torch.tensor([meta_tensor.numel()], dtype=torch.int64)
    dist.send(meta_len, dst=dst, group=group)
    dist.send(meta_tensor, dst=dst, group=group)

    stream = torch.cuda.Stream() if use_cuda_stream and _use_cuda_stream(tensors) else None
    handles = []
    if stream is not None:
        with torch.cuda.stream(stream):
            for tensor in tensors:
                handles.append(dist.isend(tensor, dst=dst, group=group))
        torch.cuda.current_stream().wait_stream(stream)
    else:
        for tensor in tensors:
            handles.append(dist.isend(tensor, dst=dst, group=group))
    for h in handles:
        h.wait()


def recv_tensors(count: int, src: int, device, group=None, use_cuda_stream: bool = True) -> list[torch.Tensor]:
    """Receive ``count`` tensors from ``src`` using metadata first."""

    meta_len = torch.empty(1, dtype=torch.int64)
    dist.recv(meta_len, src=src, group=group)
    meta_tensor = torch.empty(int(meta_len.item()), dtype=torch.uint8)
    dist.recv(meta_tensor, src=src, group=group)
    meta = pickle.loads(bytes(meta_tensor.tolist()))
    tensors: list[torch.Tensor] = []
    stream = torch.cuda.Stream() if use_cuda_stream and torch.cuda.is_available() and device is not None and torch.device(device).type == "cuda" else None
    handles = []
    for shape, dtype in meta:
        buffer = torch.empty(shape, device=device, dtype=dtype)
        tensors.append(buffer)
        if stream is not None:
            with torch.cuda.stream(stream):
                handles.append(dist.irecv(buffer, src=src, group=group))
        else:
            handles.append(dist.irecv(buffer, src=src, group=group))
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    for h in handles:
        h.wait()
    return tensors


def send_grads(tensors: Sequence[torch.Tensor], dst: int, group=None, use_cuda_stream: bool = True) -> None:
    """Send gradients to previous stage."""

    filtered = [t for t in tensors if t is not None]
    if len(filtered) == 0:
        return
    send_tensors(filtered, dst, group=group, use_cuda_stream=use_cuda_stream)


def recv_grads(count: int, src: int, device, group=None, use_cuda_stream: bool = True) -> list[torch.Tensor]:
    """Receive gradients from next stage."""

    return recv_tensors(count, src, device, group=group, use_cuda_stream=use_cuda_stream)


class _PipelineRecvFn(Function):
    @staticmethod
    def forward(ctx, count: int, src: int, device, group, use_cuda_stream: bool):
        ctx.src = src
        ctx.group = group
        ctx.use_cuda_stream = use_cuda_stream
        ctx.device = device
        tensors = recv_tensors(count, src=src, device=device, group=group, use_cuda_stream=use_cuda_stream)
        if count == 1:
            return tensors[0]
        return tuple(tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = [g for g in grad_outputs]
        send_grads(grads, dst=ctx.src, group=ctx.group, use_cuda_stream=ctx.use_cuda_stream)
        # None gradients for non-tensor args
        return (None, None, None, None, None)


class _PipelineSendFn(Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dst: int, group, use_cuda_stream: bool):
        ctx.dst = dst
        ctx.group = group
        ctx.use_cuda_stream = use_cuda_stream
        send_tensors([tensor], dst=dst, group=group, use_cuda_stream=use_cuda_stream)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grads = recv_grads(1, src=ctx.dst, device=grad_output.device, group=ctx.group, use_cuda_stream=ctx.use_cuda_stream)
        return grads[0], None, None, None


def _pipeline_coordinates(device_mesh):
    """Return (pp_rank, pp_size, tp_rank) tuple for the current process."""

    if device_mesh is None:
        return dist.get_rank(), dist.get_world_size(), 0
    mesh_names = device_mesh.mesh_dim_names or {}
    coord = device_mesh.get_coordinate()
    if coord is None:
        raise RuntimeError("Device mesh coordinate is undefined")
    if "pp" in mesh_names:
        pp_dim = mesh_names.index("pp") if isinstance(mesh_names, tuple) else list(mesh_names).index("pp")
    else:
        pp_dim = 0
    if device_mesh.ndim > 1 and "tp" in mesh_names:
        tp_dim = list(mesh_names).index("tp") if not isinstance(mesh_names, dict) else mesh_names.index("tp")
        tp_rank = coord[tp_dim]
    else:
        tp_rank = 0
    return coord[pp_dim], device_mesh.size(pp_dim) if device_mesh is not None else dist.get_world_size(), tp_rank


def add_pipeline_parallel_hooks(
    model: nn.Module,
    pp_plan: dict[str, tuple[list[str], list[str]]],
    device_mesh=None,
    pp_size: int | None = None,
    use_cuda_stream: bool = True,
):
    """Attach send/recv hooks according to ``pp_plan``.

    This mirrors tensor-parallel's hook-based API: modules are prepared once,
    and normal ``forward`` calls will automatically exchange activations
    between pipeline stages.
    """

    if pp_plan is None or len(pp_plan) == 0:
        return model
    if not is_torch_available():
        raise ImportError("PyTorch is required for pipeline parallelism")
    if not dist.is_available() or not dist.is_initialized():
        logger.warning("torch.distributed is not initialized; running without pipeline parallelism")
        return model

    _, pp_mesh, pp_size, pp_rank = initialize_pipeline_parallelism(pp_plan, pp_size, device_mesh=device_mesh)
    pp_group = pp_mesh.get_group() if pp_mesh is not None else None
    model._pp_size = pp_size
    model._pp_rank = pp_rank
    model._pp_group = pp_group

    units = flatten_pp_units(model, pp_plan)
    partitions = partition_units(units, pp_size)
    local_unit_names = set(partitions[pp_rank]) if pp_rank < len(partitions) else set()

    # Replace non-local modules with skipped placeholders to avoid computation
    for name, module, _, _ in units:
        if name not in local_unit_names:
            parent_path, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_path) if parent_path != "" else model
            setattr(parent, child_name, PipelineParallelSkipped())

    # Identify first/last local modules to add comm hooks
    local_units_ordered = [u for u in units if u[0] in local_unit_names]
    if len(local_units_ordered) == 0:
        return model
    first_name, first_mod, first_inputs, _ = local_units_ordered[0]
    last_name, last_mod, _, last_outputs = local_units_ordered[-1]

    runtime_key = "_hf_pp_runtime"

    def _init_runtime(module, args, kwargs):
        # args/kwargs are from model forward; seed activation dict with kwargs
        runtime = {"activations": {}, "device": args[0].device if len(args) > 0 and hasattr(args[0], "device") else None}
        runtime["activations"].update(kwargs)
        setattr(model, runtime_key, runtime)
        return None

    def _cleanup_runtime(module, args, kwargs, output):
        if hasattr(model, runtime_key):
            delattr(model, runtime_key)
        return output

    model.register_forward_pre_hook(_init_runtime, with_kwargs=True)
    model.register_forward_hook(_cleanup_runtime, with_kwargs=True)

    def _recv_pre_hook(mod, args, kwargs):
        runtime = getattr(model, runtime_key)
        local_pp_rank, global_pp_rank, _ = _group_ranks(pp_group)
        if pp_rank > 0:
            src_local = local_pp_rank - 1
            src_global = dist.get_global_rank(pp_group, src_local) if pp_group is not None else global_pp_rank - 1
            with _silent_non_primary(pp_rank):
                tensors = [
                    _PipelineRecvFn.apply(
                        1,
                        src_global,
                        runtime.get("device"),
                        pp_group,
                        use_cuda_stream,
                    )
                    for _ in first_inputs
                ]
                for name, tensor in zip(first_inputs, tensors):
                    runtime["activations"][name] = tensor
            return tuple(tensors), {}
        else:
            return args, kwargs
        # Build kwargs for this module from runtime activations
        new_kwargs = {name: runtime["activations"].get(name) for name in first_inputs if name in runtime["activations"]}
        return (), new_kwargs

    def _send_post_hook(mod, args, kwargs, output):
        runtime = getattr(model, runtime_key)
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        to_send = []
        for name in last_outputs:
            # Map sequentially to module outputs
            idx = last_outputs.index(name)
            if idx < len(outputs):
                tensor = outputs[idx]
            else:
                tensor = outputs[0]
            runtime["activations"][name] = tensor
            to_send.append(tensor)

        if pp_rank < pp_size - 1:
            local_pp_rank, global_pp_rank, _ = _group_ranks(pp_group)
            dst_local = local_pp_rank + 1
            dst_global = dist.get_global_rank(pp_group, dst_local) if pp_group is not None else global_pp_rank + 1
            to_send = [_PipelineSendFn.apply(tensor, dst_global, pp_group, use_cuda_stream) for tensor in to_send]
        return output

    first_mod.register_forward_pre_hook(_recv_pre_hook, with_kwargs=True)
    last_mod.register_forward_hook(_send_post_hook, with_kwargs=True)

    # For all local modules, store outputs into activation dict for downstream
    def _store_outputs(mod, args, kwargs, output, output_names):
        runtime = getattr(model, runtime_key)
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        for idx, name in enumerate(output_names):
            if idx < len(outputs):
                runtime["activations"][name] = outputs[idx]
                # For first module when not pipeline root, ensure gradients propagate
                if pp_rank > 0 and outputs[idx].requires_grad and name in first_inputs:
                    outputs[idx].register_hook(lambda grad, name=name: grad)
        return output

    for name, module, _, out_names in local_units_ordered:
        module.register_forward_hook(lambda m, a, kw, o, out_names=out_names: _store_outputs(m, a, kw, o, out_names), with_kwargs=True)

    return model


__all__ = [
    "PipelineParallelSkipped",
    "initialize_pipeline_parallelism",
    "add_pipeline_parallel_hooks",
    "flatten_pp_units",
    "partition_units",
]
