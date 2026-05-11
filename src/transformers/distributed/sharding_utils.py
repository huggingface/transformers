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

import math
from typing import TYPE_CHECKING

from ..utils import is_torch_available


if TYPE_CHECKING:
    import torch
    from torch.distributed.tensor import DTensor

if is_torch_available():
    import torch
    from torch.distributed._functional_collectives import wait_tensor
    from torch.distributed.tensor import DTensor, Replicate
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    from torch.distributed.tensor.placement_types import Shard, _StridedShard


class DtensorShardOperation:
    def __init__(self, param: DTensor):
        self.device_mesh = param.device_mesh
        self.placements = tuple(param.placements)
        self.param_ndim = param.ndim
        local_shape, offsets = compute_local_shape_and_global_offset(param.shape, self.device_mesh, self.placements)
        # Where this rank's slice starts along axis 0, and how many indices
        # it covers. Example: param of shape [8, in, out] with Shard(0) on
        # 2 ranks gives:
        #   rank 0 → _axis0_offset=0, _axis0_local_size=4  (owns experts 0..3)
        #   rank 1 → _axis0_offset=4, _axis0_local_size=4  (owns experts 4..7)
        # When the checkpoint stores one tensor per expert, shard_tensor
        # checks whether tensor_idx falls in this rank's range to decide
        # whether to keep the piece or drop it.
        self._axis0_offset = offsets[0]
        self._axis0_local_size = local_shape[0]

    def shard_tensor(
        self, source: torch.Tensor, tensor_idx: int | None = None, device=None, dtype=None
    ) -> torch.Tensor | None:
        """Slice source down to this rank's shard.

        The checkpoint can store the parameter in two layouts. Take a stack
        of N MoE experts of shape [in, out] as a running example — the
        param shape is [N, in, out]:

        - Single tensor (tensor_idx is None): the checkpoint holds one
          [N, in, out] tensor, so source.shape == param.shape. Every
          sharded dim is sliced here, including axis 0.

        - One tensor per piece (tensor_idx given): the checkpoint holds N
          separate [in, out] tensors, one per expert. shard_tensor is
          called once per expert; on each call source is the [in, out]
          tensor for expert number tensor_idx (so 0 <= tensor_idx < N).
          Note: source has one fewer dim than the param: the axis-0
          index lives in tensor_idx, not in source.shape.
          If this rank does not own tensor_idx along axis 0, return None
          and the piece is discarded. Otherwise slice only the inner
          dims; the caller (MergeModulelist / Concatenate) collects the
          kept pieces and stacks them back along axis 0 to rebuild the
          full [N, in, out] param.
        """
        source_shape = list(source.shape) if isinstance(source, torch.Tensor) else source.get_shape()
        placements = [(md, p) for md, p in enumerate(self.placements) if hasattr(p, "dim")]

        # Dense path
        if tensor_idx is None:
            if not placements:
                return source[...].to(device=device, dtype=dtype)
            has_strided = any(not p.is_shard() for _, p in placements)
            intervals = [[(0, size)] for size in source_shape]
            for mesh_dim, placement in placements:  # [i.e: (0, Shard(0)), (1, Shard(-1))]
                sub_mesh = self._get_sub_mesh(mesh_dim)
                rank, world_size = sub_mesh.get_local_rank(), sub_mesh.size()
                source_dim = self._norm_dim(placement.dim)
                if not placement.is_shard():
                    intervals[source_dim] = self._strided_intervals(
                        intervals[source_dim], rank, world_size, placement.split_factor
                    )
                else:
                    intervals[source_dim] = self._contiguous_intervals(intervals[source_dim], rank, world_size)
            # Only _StridedShard can produce multi-interval dims that need cat.
            if has_strided:
                return self._slice_and_cat(source, intervals, device, dtype)
            slices = tuple(slice(*(pieces[0] if pieces else (0, 0))) for pieces in intervals)
            return source[slices].to(device=device, dtype=dtype)

        # MoE path: drop the piece if this rank does not own tensor_idx
        # along axis 0. Once shard_tensor has been called for all N pieces,
        # the caller (MergeModulelist) stacks the kept slices along axis 0 to
        # form this rank's local shard of the param.
        shards_leading_axis = any(self._norm_dim(p.dim) == 0 for _, p in placements)
        owns_index = self._axis0_offset <= tensor_idx < self._axis0_offset + self._axis0_local_size
        if shards_leading_axis and not owns_index:
            return None

        # Inner dims use only _contiguous_intervals (one piece per dim), so a
        # single slice suffices
        inner_placements = [(md, p) for md, p in placements if self._norm_dim(p.dim) != 0]
        if not inner_placements:
            return source[...].to(device=device, dtype=dtype)
        slice_per_dim: list[tuple[int, int]] = [(0, size) for size in source_shape]
        # placement.dim is indexed in param space (e.g. axis 2 of [N, in, out]).
        # source is in source space (e.g. axis 1 of [in, out]), so we translate
        # from one to the other by stripping the leading axis.
        for mesh_dim, placement in inner_placements:
            sub_mesh = self._get_sub_mesh(mesh_dim)
            rank, world_size = sub_mesh.get_local_rank(), sub_mesh.size()
            param_dim = self._norm_dim(placement.dim)
            source_dim = param_dim - 1
            pieces = self._contiguous_intervals([slice_per_dim[source_dim]], rank, world_size)
            slice_per_dim[source_dim] = pieces[0] if pieces else (0, 0)
        return source[tuple(slice(s, e) for s, e in slice_per_dim)].to(device=device, dtype=dtype)

    def _strided_intervals(
        self, intervals: list[tuple[int, int]], rank: int, world_size: int, split_factor: int
    ) -> list[tuple[int, int]]:
        narrowed = []
        for start, end in intervals:
            group_size = math.ceil((end - start) / split_factor)
            for group_idx in range(split_factor):
                group_start = start + group_idx * group_size
                group_end = min(group_start + group_size, end)
                if group_end <= group_start:
                    continue
                size, offset = Shard.local_shard_size_and_offset(group_end - group_start, world_size, rank)
                if size > 0:
                    narrowed.append((group_start + offset, group_start + offset + size))
        return narrowed

    def _contiguous_intervals(
        self, intervals: list[tuple[int, int]], rank: int, world_size: int
    ) -> list[tuple[int, int]]:
        total = sum(end - start for start, end in intervals)
        my_size, my_offset = Shard.local_shard_size_and_offset(total, world_size, rank)
        if my_size == 0:
            return []

        out: list[tuple[int, int]] = []
        flat_pos = 0
        slice_end = my_offset + my_size
        for start, end in intervals:
            length = end - start
            interval_end_flat = flat_pos + length
            if interval_end_flat <= my_offset:  # entirely before my slice
                flat_pos = interval_end_flat
                continue
            if flat_pos >= slice_end:  # entirely after my slice
                break
            sub_start = max(0, my_offset - flat_pos)
            sub_end = min(length, slice_end - flat_pos)
            out.append((start + sub_start, start + sub_end))
            flat_pos = interval_end_flat
        return out

    def _slice_and_cat(self, source, intervals, device, dtype):
        multi_interval_dim: int | None = None
        slices: list[slice] = []
        for source_dim, pieces in enumerate(intervals):
            if len(pieces) == 1:
                start, end = pieces[0]
                slices.append(slice(start, end))
                continue
            if multi_interval_dim is not None:
                raise ValueError("Shard-on-read only supports disjoint ranges on a single checkpoint dimension.")
            multi_interval_dim = source_dim
            slices.append(slice(None))  # placeholder, filled per-piece below

        # Fast path: every dim is one contiguous interval, read in a single slice.
        if multi_interval_dim is None:
            return source[tuple(slices)].to(device=device, dtype=dtype)

        # Multi-interval dim: read each piece separately, then concatenate.
        pieces_read = []
        for start, end in intervals[multi_interval_dim]:
            piece_slices = list(slices)
            piece_slices[multi_interval_dim] = slice(start, end)
            pieces_read.append(source[tuple(piece_slices)])
        return torch.cat(pieces_read, dim=multi_interval_dim).to(device=device, dtype=dtype)

    def _get_sub_mesh(self, mesh_dim: int):
        if self.device_mesh.ndim == 1:
            return self.device_mesh
        return self.device_mesh[self.device_mesh.mesh_dim_names[mesh_dim]]

    def _norm_dim(self, dim: int) -> int:
        # if dim is negative, it should be normalized to the last axis
        return dim if dim >= 0 else self.param_ndim + dim


def _replicate_dtensor(tensor: DTensor) -> DTensor:
    """All-gather a DTensor to fully Replicate, handling ``_StridedShard``.

    PyTorch's ``redistribute()`` does not support ``_StridedShard`` as a source::

        _StridedShard -> redistribute() -> Replicate      ❌ AssertionError
        _StridedShard -> redistribute() -> Shard           ❌ NotImplementedError
        Shard         -> redistribute() -> Replicate      ✅ works
        Replicate     -> redistribute() -> Shard           ✅ works
        Replicate     -> redistribute() -> _StridedShard  ✅ works

    So we bypass ``redistribute`` and call each placement's low-level
    ``_to_replicate_tensor`` (manual all-gather + interleaved reorder).

    We process mesh dims **right-to-left** (innermost first).  Under TP+FSDP
    the 2D mesh is ``(fsdp, tp)`` and both dims can shard the same tensor dim::

        placements = (_StridedShard(dim=0), Shard(dim=0))
        local shape = [64, 1024]   (global [256, 1024], fsdp=2, tp=2)

    Right-to-left means TP is gathered first (local grows to [128, 1024]),
    then FSDP (grows to [256, 1024]).  Each step must pass the correct
    intermediate logical shape — the global shape divided by the mesh sizes
    of dims not yet gathered (to the left).
    """
    mesh = tensor.device_mesh
    replicate_all = tuple(Replicate() for _ in range(mesh.ndim))
    with torch.no_grad():
        if any(isinstance(p, _StridedShard) for p in tensor.placements):
            local = tensor._local_tensor
            placements = tensor.placements
            for i in reversed(range(mesh.ndim)):
                p = placements[i]
                if p.is_replicate():
                    continue
                # Compute the logical shape seen at this step: dims to the left
                # (not yet gathered) still divide their tensor dimension.
                logical_shape = list(tensor.shape)
                for j in range(i):
                    pj = placements[j]
                    if not pj.is_replicate():
                        logical_shape[pj.dim] //= mesh.size(j)
                local = p._to_replicate_tensor(local, mesh, i, logical_shape)
                # Drain the async functional collective so downstream storage
                # queries (e.g. tied-weight dedup) don't hit "invalid python storage".
                local = wait_tensor(local)
            return DTensor.from_local(local, mesh, replicate_all, run_check=False)

        return tensor.redistribute(placements=replicate_all)


def convert_strided_to_shard(state_dict: dict) -> dict[str, tuple]:
    # Convert _StridedShard DTensors in a state dict to plain Shard for DCP compatibility.
    placement_map: dict[str, tuple] = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            nested = convert_strided_to_shard(value)
            for nk, nv in nested.items():
                placement_map[f"{key}.{nk}"] = nv
        elif isinstance(value, DTensor) and any(isinstance(p, _StridedShard) for p in value.placements):
            placement_map[key] = tuple(value.placements)
            shard_placements = tuple(Shard(p.dim) if isinstance(p, _StridedShard) else p for p in value.placements)
            state_dict[key] = _replicate_dtensor(value).redistribute(placements=shard_placements)
    return placement_map


def restore_strided_from_shard(state_dict: dict, placement_map: dict[str, tuple]) -> None:
    # Restore _StridedShard placements after dcp.load.
    def _resolve(d, dotted_key):
        parts = dotted_key.split(".", 1)
        if len(parts) == 2 and parts[0] in d and isinstance(d[parts[0]], dict):
            return _resolve(d[parts[0]], parts[1])
        return d, dotted_key

    for key, original_placements in placement_map.items():
        container, leaf_key = _resolve(state_dict, key)
        if leaf_key in container and isinstance(container[leaf_key], DTensor):
            container[leaf_key] = _replicate_dtensor(container[leaf_key]).redistribute(placements=original_placements)
