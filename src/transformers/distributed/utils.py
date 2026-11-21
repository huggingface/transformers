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
import os
import warnings
from datetime import timedelta
from typing import TYPE_CHECKING, TypeGuard

from ..utils import (
    is_torch_available,
    is_torch_distributed_available,
    is_torch_greater_or_equal,
    logging,
)


logger = logging.get_logger(__name__)

# What one seek costs, expressed as the number of bytes a sequential read gets through in the same
# time. Skipping the other ranks' shards trades bytes for seeks, so this is what decides whether the
# trade pays. Measured at 12 MiB on a cross-region Lustre mount (50 ms per seek, 0.24 GiB/s per
# stream); a local NVMe is far below that, where the effect is only to fall back more readily.
_SEEK_COST_BYTES = 12 * 2**20

# How much a rank may warm by asking the kernel to read ahead, which costs almost nothing but is
# only advice: past a few GiB the kernel drops most of it, the pages are not there when the loader
# asks for them, and the load pays for the miss. Beyond this, warming reads the bytes itself.
# Measured: 1.3 and 4.4 GiB of readahead land, 7.5, 8 and 29.7 GiB do not.
_READAHEAD_MAX_TOTAL_BYTES = 5 * 2**30


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor

    from .configuration_utils import DistributedConfig


if is_torch_available():
    import torch


def _check_distributed_checkpointing_available() -> bool:
    # `torch.distributed.checkpoint.hf_storage` (HuggingFaceStorageReader/Writer) first ships in torch 2.8.
    return is_torch_distributed_available() and is_torch_greater_or_equal("2.8")


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


def _warning_once_on_rank_zero(message: str) -> None:
    """Log a warning once per process, on rank 0 only."""
    if _get_torch_distributed_rank() == 0:
        logger.warning_once(message)


def _get_torch_distributed_world_size() -> int:
    if not _is_torch_distributed_initialized():
        return 1
    return torch.distributed.get_world_size()


def _merge(spans: list[tuple[int, int]], path: str) -> list[tuple[str, int, int]]:
    """Sort byte ranges of one file and join the ones that touch or overlap."""
    merged = []
    for start, end in sorted(spans):
        # Bridging a gap costs its bytes and saves a seek, so bridge when the gap is the cheaper one.
        if merged and start <= merged[-1][2] + _SEEK_COST_BYTES:
            merged[-1] = (path, merged[-1][1], max(merged[-1][2], end))
        else:
            merged.append((path, start, end))
    return merged


def _rank_byte_spans(checkpoint_files: list[str], meta_state_dict: dict) -> tuple[list, list]:
    """Byte spans of the checkpoint this rank will read, merged per file.

    A rank slices every sharded parameter down to its own shard, so it reads `1 / world` of those
    bytes and all of the rest. Returns `(own, common)`, both lists of `(path, start, end)`: the common
    spans come out identical on every rank, so local ranks can share them out between them.
    """
    import json
    import re
    import struct

    owned_experts = _owned_expert_range(meta_state_dict)
    own, common = [], []
    for path in checkpoint_files:
        with open(path, "rb") as f:
            header_length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_length))
        base = 8 + header_length
        mine, whole = [], []
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = (base + offset for offset in meta["data_offsets"])
            param = meta_state_dict.get(name)
            expert = re.search(r"\.experts\.(\d+)\.", name)
            # A checkpoint that stores one tensor per expert names them in a way the packed parameter
            # does not match, so they never resolve above. The rank still only wants the experts it
            # owns, and consecutive experts sit next to each other on disk, so keeping those and
            # dropping the rest leaves a few long runs.
            if expert and owned_experts:
                if owned_experts[0] <= int(expert.group(1)) < owned_experts[1]:
                    mine.append((start, end))
                continue
            # Sharding on dim 0 is the only kind that keeps a rank's share contiguous on disk, and only
            # when the checkpoint stores the parameter whole rather than one piece per expert. Slice
            # those; read everything else in full, which covers what the rank needs and then some.
            rows = meta["shape"][0] if meta["shape"] else 0
            owned = _dim_0_range(param) if is_dtensor(param) and rows == param.shape[0] else None
            if owned:
                row_bytes = (end - start) // rows
                start, end = start + owned[0] * row_bytes, start + owned[1] * row_bytes
                mine.append((start, end))
            else:
                whole.append((start, end))
        own += _merge(mine, path)
        common += _merge(whole, path)
    return own, common


def _dim_0_range(param: DTensor) -> tuple[int, int] | None:
    """The `[start, end)` rows of dim 0 this rank holds, or `None` if that is not well defined.

    It is not well defined when a mesh dim splits some other dimension, when the split is strided so
    a rank's rows are not one run, or when this rank is not a member of the parameter's mesh at all,
    which happens to the ranks outside an expert-parallel group.
    """
    from .sharding_utils import DtensorShardOperation

    shards = [placement for placement in param.placements if placement.is_shard()]
    contiguous_on_dim_0 = bool(shards) and all(
        placement.dim in (0, -param.ndim) and not getattr(placement, "split_factor", 0) for placement in shards
    )
    if not contiguous_on_dim_0 or param.device_mesh.get_coordinate() is None:
        return None
    operation = DtensorShardOperation(param)
    return operation._axis0_offset, operation._axis0_offset + operation._axis0_local_size


def _owned_expert_range(meta_state_dict: dict) -> tuple[int, int] | None:
    """The `[start, end)` experts this rank holds, or `None` if it cannot be determined.

    Every expert parameter is sharded the same way, so one of them answers for all of them.
    """
    for name, param in meta_state_dict.items():
        if ".experts." in name and is_dtensor(param) and param.ndim == 3:
            return _dim_0_range(param)
    return None


def _all_ranks_agree(value: bool) -> bool:
    """`value` on every rank, once every rank has answered."""
    if not _is_torch_distributed_initialized():
        return value
    device_type = torch._C._get_accelerator().type
    index = None if device_type == "cpu" else getattr(torch, device_type).current_device()
    agreed = torch.tensor([value], dtype=torch.uint8, device=torch.device(device_type, index))
    torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN)
    return bool(agreed.item())


def prefetch_checkpoint_shards(checkpoint_files: list[str], meta_state_dict: dict | None = None) -> None:
    """Warm the page cache for the checkpoint shards before the per-tensor loading pass, opt-in via
    `HF_SHARD_PREFETCH=<read threads per rank>`.

    The per-tensor read pattern of sharded loading reads a network filesystem at well under 1 GiB/s
    while large sequential reads sustain many times that. The reads run in background threads, ahead of
    the loading pass, which starts right away and finds more and more of the checkpoint in the page
    cache. Given `meta_state_dict`, a rank warms the byte spans of its own shard and shares the rest out
    with the other ranks on the node, which on a model sharded across several nodes leaves each node
    warming a fraction of the checkpoint. Otherwise local ranks split the shard list between them and
    warm it whole (every node needs the full checkpoint cached, since every rank slices tensors from all
    shards).
    """
    prefetch_threads = int(os.environ.get("HF_SHARD_PREFETCH", "0"))
    if not checkpoint_files or not prefetch_threads:
        return
    from concurrent.futures import ThreadPoolExecutor

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

    def _warm(job, bufsize=16 * 2**20, readahead=False):
        path, start, end = job
        if readahead:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, start, end - start, os.POSIX_FADV_WILLNEED)
            finally:
                os.close(fd)
            return
        with open(path, "rb", buffering=0) as f:
            f.seek(start)
            left = end - start
            while left:
                chunk = f.read(min(bufsize, left))
                if not chunk:
                    return
                left -= len(chunk)

    whole = [(path, 0, os.path.getsize(path)) for path in checkpoint_files][local_rank::local_world]
    own, common = ([], []) if meta_state_dict is None else _rank_byte_spans(checkpoint_files, meta_state_dict)
    jobs = own + common[local_rank::local_world]
    # Reading only this rank's shard saves bytes and costs seeks. Price the seeks in bytes and keep
    # whichever plan reads less. Both sides are the node's read divided by its ranks, so the
    # comparison holds even when there are fewer shards than local ranks and this rank was dealt none.
    cost = sum(end - start for _, start, end in jobs) + len(jobs) * _SEEK_COST_BYTES
    take_spans = bool(jobs) and cost < sum(os.path.getsize(path) for path in checkpoint_files) / local_world
    # The two plans divide the checkpoint up differently, so ranks that disagree leave parts of it
    # cold: take the spans only where every rank does.
    if not _all_ranks_agree(take_spans):
        jobs = whole

    # Handing the whole plan to the kernel is nearly free, but only while it is small enough to be
    # honoured; a rank warming tens of GiB has to read them, or the pages will not be there.
    readahead = sum(end - start for _, start, end in jobs) <= _READAHEAD_MAX_TOTAL_BYTES
    pool = ThreadPoolExecutor(max_workers=prefetch_threads)
    for job in jobs:
        pool.submit(_warm, job, readahead=readahead)
    pool.shutdown(wait=False)


def is_local_dist_rank_0() -> bool:
    return _is_torch_distributed_initialized() and int(os.environ.get("LOCAL_RANK", "-1")) == 0


DISTRIBUTED_BACKEND_MAP = {
    "cuda": "nccl",
    "cpu": "gloo",
    "xpu": "xccl",
    "hpu": "hccl",
    "neuron": "neuron",
    "tpu": "tpu_dist",
}


def get_distributed_backend(device_type: str) -> str | None:
    """Return the `torch.distributed` backend for `device_type`, or `None` if it has no dedicated one."""
    dist_backend = DISTRIBUTED_BACKEND_MAP.get(device_type)
    if dist_backend is None:
        raise ValueError(f"No distributed backend found for device type '{device_type}'")
    return dist_backend


def _bind_local_device(device_type: str, local_rank: int) -> torch.device:
    """Bind this process to its accelerator and return the device (no binding on CPU)."""
    if device_type == "cpu":
        return torch.device(device_type)
    # torch_tpu gives each process a single chip, always at index 0
    device_index = torch.accelerator.current_device_index() if device_type == "tpu" else local_rank
    torch.accelerator.set_device_index(device_index)
    return torch.device(device_type, device_index)


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

            backend = get_distributed_backend(device_type)

            # Bind the accelerator before init so the process group is created with a
            # device_id, otherwise collectives like barrier() warn (and may spin up an
            # extra NCCL comm) about the missing device binding.
            device = _bind_local_device(device_type, local_rank)
            # Only pass device_id to backends that support communicator splitting, for now we limit
            # it to NCCL: a bound device makes DeviceMesh create its subgroups with `split_group`,
            # which fails on backends without split support (e.g. Neuron).
            device_id = device if backend == "nccl" else None
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
        torch.distributed.barrier(device_ids=[torch.accelerator.current_device_index()])
    else:
        torch.distributed.barrier()


class TransformersDeviceMesh:
    """
    Holds the device meshes used by a model.

    dense layers and experts are sharded differently, so they need different views of the same ranks.

        dense  : (pp, dp_replicate, fsdp, tp)     attention, dense MLPs, embeddings, lm_heads
        expert : (pp, dp_replicate, efsdp, ep)    experts

    Both views cover the same world, so pp * dp_replicate * fsdp * tp == pp * dp_replicate * efsdp * ep.
    efsdp is not something you pick, it is whatever is left once ep is fixed. The relationship is as follow:
    efsdp = fsdp * tp / ep. It is the FSDP axis for expert weights same role `fsdp` plays for the dense params.
    There is no etp (expert tensor parallel) axis: with EP on, experts are never tensor-sharded. (for now)
    For experts, EP plays the role TP plays for dense layers (weights stay sharded during compute),
    and efsdp plays the role of fsdp (weights all-gathered before compute).

    If one were ever added, the relationship would become pp * efsdp * ep * etp == pp * fsdp * tp and efsdp would shrink by etp
    (efsdp = fsdp * tp / (ep * etp))

    When ep_size == tp_size, efsdp == fsdp (given the relationship efsdp = fsdp * tp / ep), thus same axis, same size and same groups of ranks.
    In that case experts could reuse the dense mesh's fsdp axis.
    When ep_size != tp_size, efsdp != fsdp, so we can't reuse the same mesh as they don't have the same groups of ranks
    This explains why experts need their own efsdp axis.

    Regarding ep value, one can decide to default the value to node width (8 on most machines) so that all-to-all never leaves the node.
    That has several implications on efsdp value given your setup:
    - On a single node, ep == fsdp * tp thus efsdp = 1, the axis does nothing.
    - On several nodes, we still keep ep at node width, since all-to-all across nodes is expensive.
    However, each node then holds a full copy of the expert group and efsdp is the number of copies, which is where FSDP happens for the experts
    i.e: 2 nodes x 8 GPUs -> efsdp = 16 / 8 = 2, one EP group per node, two copies, sharded over efsdp.
    """

    def __init__(self, dense_mesh: DeviceMesh, expert_mesh: DeviceMesh):
        self._dense_mesh = dense_mesh
        self._expert_mesh = expert_mesh

    def get_mesh(self, dims: str | tuple[str, ...]) -> DeviceMesh:
        """Select expert axes for `ep`/`efsdp`, otherwise dense axes; DeviceMesh handles slicing."""
        dims = (dims,) if isinstance(dims, str) else dims
        mesh = self._expert_mesh if "ep" in dims or "efsdp" in dims else self._dense_mesh
        return mesh[dims]


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
        # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
        device_type = torch._C._get_accelerator().type
        if device_type == "mps":
            logger.warning_once(
                "PyTorch's built-in DeviceMesh/DTensor stack does not support an MPS mesh. Falling back to CPU."
            )
            device_type = "cpu"
        if device_type != "cpu":
            tp_device = _bind_local_device(device_type, int(os.environ.get("LOCAL_RANK", 0)))
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
        device_map = _bind_local_device(device_mesh.device_type, int(os.environ.get("LOCAL_RANK", 0)))

    return device_map, device_mesh


def initialize_fully_sharded_data_parallelism(distributed_config: DistributedConfig):
    warnings.warn(
        "`initialize_fully_sharded_data_parallelism` is deprecated and will be removed in a future release. "
        "Use `initialize_distributed_mesh` with a `DistributedConfig` instead.",
        FutureWarning,
        stacklevel=2,
    )
    # `fully_shard` itself only needs torch>=2.6, but distributed checkpoint save/load
    # (DCP + HuggingFaceStorageWriter) needs 2.8, so that is the effective requirement.
    if distributed_config.fsdp_size > 1 and not is_torch_greater_or_equal("2.8"):
        raise OSError("FSDP2 requires `torch>=2.8` (distributed checkpoint save/load).")

    device_type = torch._C._get_accelerator().type

    device_map = _bind_local_device(device_type, int(os.environ.get("LOCAL_RANK", 0)))

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
) -> tuple[torch.device | None, TransformersDeviceMesh | None]:
    """Create a device mesh containing every configured parallel dimension."""
    mesh_shape = (
        distributed_config.pp_size,
        distributed_config.dp_replicate_size,
        distributed_config.fsdp_size,
        distributed_config.tp_size,
    )
    if mesh_shape == (1, 1, 1, 1):
        return None, None

    device_type = torch._C._get_accelerator().type
    if distributed_config.tp_size > 1 and device_type == "mps":
        raise RuntimeError("Tensor parallelism is not supported on MPS devices.")

    _ensure_torch_distributed(device_type)
    world_size = torch.distributed.get_world_size()
    expected_world_size = math.prod(mesh_shape)
    if expected_world_size != world_size:
        raise RuntimeError(
            f"The parallel mesh requires {expected_world_size} processes, but world_size is {world_size}."
        )

    device_map = _bind_local_device(device_type, int(os.environ.get("LOCAL_RANK", 0)))

    dense_mesh = torch.distributed.init_device_mesh(
        device_type,
        mesh_shape,
        mesh_dim_names=("pp", "dp_replicate", "fsdp", "tp"),
    )
    expert_mesh = torch.distributed.init_device_mesh(
        device_type,
        (
            distributed_config.pp_size,
            distributed_config.dp_replicate_size,
            distributed_config.efsdp_size,
            distributed_config.ep_size,
        ),
        mesh_dim_names=("pp", "dp_replicate", "efsdp", "ep"),
    )
    return device_map, TransformersDeviceMesh(dense_mesh, expert_mesh)


def gather_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather FSDP-sharded params to full plain CPU tensors.

    Only rank 0 accumulates the result; other ranks return ``{}``.
    """
    if not _check_distributed_checkpointing_available():
        raise OSError("Distributed checkpointing requires `torch>=2.8` with `torch.distributed` available.")

    # Import here because otherwise it emits a warning every time it's imported on some hardware - this keeps the warning from
    # being emitted if the function is not used
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict = get_model_state_dict(model, options=options)
    if _get_torch_distributed_rank() == 0:
        return full_state_dict
    return {}
