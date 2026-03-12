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

import functools
import math
from typing import Sequence

from ..utils import logging
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.autograd.function import Function


logger = logging.get_logger(__name__)


# Stable dtype ↔ integer mapping used in tensor metadata encoding (no pickle).
_DTYPE_TO_ID: dict = {}
_ID_TO_DTYPE: dict = {}

def _build_dtype_maps():
    import torch
    pairs = [
        (torch.float32, 0), (torch.float16, 1), (torch.bfloat16, 2),
        (torch.float64, 3), (torch.int64, 4), (torch.int32, 5),
        (torch.int16, 6), (torch.int8, 7), (torch.uint8, 8), (torch.bool, 9),
    ]
    for dtype, idx in pairs:
        _DTYPE_TO_ID[dtype] = idx
        _ID_TO_DTYPE[idx] = dtype

def _encode_tensor_meta(tensors) -> "torch.Tensor":
    """Encode shapes/dtypes into a flat int64 tensor — no pickle."""
    if not _DTYPE_TO_ID:
        _build_dtype_maps()
    # Format: [N, ndim_0, d0_0, ..., dtype_id_0, ndim_1, ...]
    import torch
    values = [len(tensors)]
    for t in tensors:
        values.append(len(t.shape))
        values.extend(t.shape)
        values.append(_DTYPE_TO_ID[t.dtype])
    return torch.tensor(values, dtype=torch.int64)


def _decode_tensor_meta(meta_tensor) -> list:
    """Decode shapes/dtypes from a flat int64 tensor."""
    if not _ID_TO_DTYPE:
        _build_dtype_maps()
    data = meta_tensor.tolist()
    n, idx, result = data[0], 1, []
    for _ in range(n):
        ndim = data[idx]; idx += 1
        shape = tuple(data[idx : idx + ndim]); idx += ndim
        dtype = _ID_TO_DTYPE[data[idx]]; idx += 1
        result.append((shape, dtype))
    return result


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
    meta_tensor = _encode_tensor_meta(tensors)
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
    meta_tensor = torch.empty(int(meta_len.item()), dtype=torch.int64)
    dist.recv(meta_tensor, src=src, group=group)
    meta = _decode_tensor_meta(meta_tensor)
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
    """Receive N tensors from the previous stage; send N grads back on backward."""

    @staticmethod
    def forward(ctx, src: int, device, group, use_cuda_stream: bool, n: int):
        ctx.src = src
        ctx.group = group
        ctx.use_cuda_stream = use_cuda_stream
        ctx.n = n
        return tuple(recv_tensors(n, src=src, device=device, group=group, use_cuda_stream=use_cuda_stream))

    @staticmethod
    def backward(ctx, *grad_outputs):
        send_grads(list(grad_outputs), dst=ctx.src, group=ctx.group, use_cuda_stream=ctx.use_cuda_stream)
        return (None, None, None, None, None)


class _PipelineSendFn(Function):
    """Send N tensors to the next stage; receive N grads back on backward."""

    @staticmethod
    def forward(ctx, dst: int, group, use_cuda_stream: bool, *tensors: torch.Tensor):
        ctx.dst = dst
        ctx.group = group
        ctx.use_cuda_stream = use_cuda_stream
        ctx.device = tensors[0].device if tensors else None
        ctx.n = len(tensors)
        send_tensors(tensors, dst=dst, group=group, use_cuda_stream=use_cuda_stream)
        return tensors

    @staticmethod
    def backward(ctx, *grad_outputs):
        grads = recv_grads(ctx.n, src=ctx.dst, device=ctx.device, group=ctx.group, use_cuda_stream=ctx.use_cuda_stream)
        return (None, None, None) + tuple(grads)


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


def _recv_pre_hook(mod, args, kwargs, *, pp_rank, pp_group, n_inputs, use_cuda_stream):
    """Receive activations from the previous pipeline stage."""
    if pp_rank == 0:
        return args, kwargs
    device = args[0].device if args and hasattr(args[0], "device") else next(mod.parameters()).device
    local_rank = dist.get_rank(pp_group) if pp_group is not None else dist.get_rank()
    src = dist.get_global_rank(pp_group, local_rank - 1) if pp_group is not None else local_rank - 1
    tensors = _PipelineRecvFn.apply(src, device, pp_group, use_cuda_stream, n_inputs)
    return tensors, {}


def _send_post_hook(mod, args, kwargs, output, *, pp_rank, pp_size, pp_group, n_outputs, use_cuda_stream):
    """Send activations to the next pipeline stage."""
    if pp_rank >= pp_size - 1:
        return output
    outputs = output if isinstance(output, (tuple, list)) else (output,)
    local_rank = dist.get_rank(pp_group) if pp_group is not None else dist.get_rank()
    dst = dist.get_global_rank(pp_group, local_rank + 1) if pp_group is not None else local_rank + 1
    to_send = tuple(outputs[min(i, len(outputs) - 1)] for i in range(n_outputs))
    _PipelineSendFn.apply(dst, pp_group, use_cuda_stream, *to_send)
    return output


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

    local_idx = 0
    for name, module, input_names, output_names in units:
        if name not in local_unit_names:
            parent_path, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, child_name, PipelineParallelSkipped())
            continue
        if local_idx == 0:
            module.register_forward_pre_hook(
                functools.partial(
                    _recv_pre_hook,
                    pp_rank=pp_rank, pp_group=pp_group,
                    n_inputs=len(input_names), use_cuda_stream=use_cuda_stream,
                ),
                with_kwargs=True,
            )
        if local_idx == len(local_unit_names) - 1:
            module.register_forward_hook(
                functools.partial(
                    _send_post_hook,
                    pp_rank=pp_rank, pp_size=pp_size, pp_group=pp_group,
                    n_outputs=len(output_names), use_cuda_stream=use_cuda_stream,
                ),
                with_kwargs=True,
            )

    return model


__all__ = [
    "PipelineParallelSkipped",
    "initialize_pipeline_parallelism",
    "add_pipeline_parallel_hooks",
    "flatten_pp_units",
    "partition_units",
]
