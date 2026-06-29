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
from typing import TYPE_CHECKING

from ..utils import is_torch_available


if TYPE_CHECKING:
    import torch
    from torch.distributed.tensor import DTensor

if is_torch_available():
    import torch
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    from torch.distributed.tensor.placement_types import Shard

    # torch < 2.10 names as an underscore before `local_shard_size_and_offset`: alias it the non-underscored version
    if not hasattr(Shard, "local_shard_size_and_offset") and hasattr(Shard, "_local_shard_size_and_offset"):
        Shard.local_shard_size_and_offset = Shard._local_shard_size_and_offset


class DtensorShardOperation:
    """Shard-on-read: slice a full checkpoint tensor down to this rank's local
    DTensor shard, for any combination of placements on a 1-D or n-D mesh.

    Placements primer
    -----------------
    Each mesh dim carries one placement describing how it slices the tensor:

    | Placement                | Local data on each rank of the mesh dim         |
    |--------------------------|-------------------------------------------------|
    | Replicate                | full tensor (no slicing)                        |
    | Shard(d)                 | contiguous chunk of dim d (rows r*c .. (r+1)*c) |
    | _StridedShard(d, sf=N)   | one chunk from each of N groups along dim d,   |
    |                          | concatenated together (interleaved layout)      |

    Different scenarios of different placements
    ------------------------------------------
    Placement tuples are ordered outermost-first; for a 2-D (fsdp, tp) mesh
    the tuple is (fsdp_placement, tp_placement).

    | Scenario                                          | Placements                                  |
    |---------------------------------------------------|---------------------------------------------|
    | TP-only, non-fused (e.g. q_proj/k_proj/v_proj)    | [Shard(d)]                                   |
    | TP-only, fused gate/up                            | [_StridedShard(d, sf=2)]                     |
    | TP + FSDP, same tensor dim (contiguous TP case)   | [Shard(d), Shard(d)]                         |
    | TP + FSDP, same tensor dim (fused/interleaved TP) | [_StridedShard(d, sf=tp_size), Shard(d)]     |
    | TP + FSDP, different dims                         | [Shard(d1), Shard(d2)]                       |

    Mesh dimensions are listed outermost-first. For a 2-D (fsdp=F, tp=T) mesh,
    rank index = fsdp_idx * T + tp_idx.

    Loading (this class)
    --------------------
    During `from_pretrained`, each rank reads the full checkpoint tensors,
    then calls `shard_tensor(source, tensor_idx=...)` to keep only its local
    DTensor shard. The class encapsulates the placements + mesh so the
    slicing logic doesn't have to be repeated at every call site.

    Two checkpoint layouts are supported. Running example: an MoE weight
    stack of param shape [N_experts, in, out]:

    | Checkpoint layout                  | tensor_idx | source.shape  | Returns                     |
    |------------------------------------|------------|---------------|-----------------------------|
    | One stacked tensor                 | None       | [N, in, out]  | This rank's slice along     |
    |                                    |            |               | every sharded dim.          |
    | N per-expert tensors (called once  | 0..N-1     | [in, out]     | Inner-dim slice if this     |
    | per expert by the caller)          |            |               | rank owns expert `i`,       |
    |                                    |            |               | else None (caller drops it).|
    """

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
        dim_placements = [
            (mesh_dim, placement) for mesh_dim, placement in enumerate(self.placements) if hasattr(placement, "dim")
        ]

        # Dense path
        if tensor_idx is None:
            if not dim_placements:
                return source[...].to(device=device, dtype=dtype)

            # Determine for each tensor dimension, which type of sharding operations to apply (_StridedShard or Shard) and which rank to apply it to.
            # i.e: dim 0 -> [ Strided(rank0, size=2, sf=2), Shard(rank1, size=2) ]
            # i.e: dim 1 -> [ Shard(rank0, size=2)]
            planned_ops_by_dim = [[] for _ in source_shape]
            for mesh_dim, placement in dim_placements:
                sub_mesh = self._get_sub_mesh(mesh_dim)
                rank, world_size = sub_mesh.get_local_rank(), sub_mesh.size()
                dim_idx = self._normalize_param_dim(placement.dim)
                planned_ops_by_dim[dim_idx].append((placement, rank, world_size))

            # Apply the sharding operations to each tensor dimension.
            intervals_by_dim = [[(0, size)] for size in source_shape]
            for dim_idx, planned_ops in enumerate(planned_ops_by_dim):
                intervals = intervals_by_dim[dim_idx]
                for placement, rank, world_size in planned_ops:
                    if placement.is_shard():
                        intervals = self._apply_contiguous_shard(intervals, rank, world_size)
                    else:
                        intervals = self._apply_strided_shard(intervals, rank, world_size, placement.split_factor)
                intervals_by_dim[dim_idx] = intervals

            has_strided_shard = any(not placement.is_shard() for _, placement in dim_placements)
            if has_strided_shard:
                # Multi-interval dim: read each piece separately, then concatenate.
                return self._slice_and_cat(source, intervals_by_dim, device, dtype)
            else:
                slice_parts = []
                for intervals in intervals_by_dim:
                    start, end = intervals[0] if intervals else (0, 0)
                    slice_parts.append(slice(start, end))

                return source[tuple(slice_parts)].to(device=device, dtype=dtype)

        # MoE path: drop the piece if this rank does not own tensor_idx
        # along axis 0. Once shard_tensor has been called for all N pieces,
        # the caller (MergeModulelist) stacks the kept slices along axis 0 to
        # form this rank's local shard of the param.
        shards_axis0 = any(self._normalize_param_dim(placement.dim) == 0 for _, placement in dim_placements)
        owns_tensor_idx = self._axis0_offset <= tensor_idx < self._axis0_offset + self._axis0_local_size
        if shards_axis0 and not owns_tensor_idx:
            return None

        # Inner dims use only _apply_contiguous_shard (one piece per dim), so a
        # single slice suffices
        inner_dim_placements = [
            (mesh_dim, placement)
            for mesh_dim, placement in dim_placements
            if self._normalize_param_dim(placement.dim) != 0
        ]
        if not inner_dim_placements:
            return source[...].to(device=device, dtype=dtype)
        source_ranges_by_dim: list[tuple[int, int]] = [(0, size) for size in source_shape]
        # placement.dim is indexed in param space (e.g. axis 2 of [N, in, out]).
        # source is in source space (e.g. axis 1 of [in, out]), so we translate
        # from one to the other by stripping the leading axis.
        for mesh_dim, placement in inner_dim_placements:
            sub_mesh = self._get_sub_mesh(mesh_dim)
            rank, world_size = sub_mesh.get_local_rank(), sub_mesh.size()
            param_dim = self._normalize_param_dim(placement.dim)
            source_dim = param_dim - 1
            interval_pieces = self._apply_contiguous_shard([source_ranges_by_dim[source_dim]], rank, world_size)
            source_ranges_by_dim[source_dim] = interval_pieces[0] if interval_pieces else (0, 0)
        return source[tuple(slice(s, e) for s, e in source_ranges_by_dim)].to(device=device, dtype=dtype)

    def _apply_strided_shard(
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

    def _apply_contiguous_shard(
        self, intervals: list[tuple[int, int]], rank: int, world_size: int
    ) -> list[tuple[int, int]]:
        # We apply contiguous sharding to a list of intervals. Two cases:
        # 1. The intervals is a single interval -> we return the local interval for this rank.
        # 2. The intervals are disjoint (i.e: 2D TP + FSDP, same tensor dim (fused/interleaved TP = [StridedShard(d, sf=tp_size), Shard(d)])
        # -> find overlapping local intervals across the disjoint intervals.

        # Compute rank's local length and start offset when the total length is partitioned across world_size ranks.
        flat_total_len = sum(end - start for start, end in intervals)
        local_flat_len, local_flat_start = Shard.local_shard_size_and_offset(flat_total_len, world_size, rank)
        local_flat_end = local_flat_start + local_flat_len

        if local_flat_len == 0:
            return []

        # Single-interval case.
        if len(intervals) == 1:
            source_start, _ = intervals[0]
            return [(source_start + local_flat_start, source_start + local_flat_end)]

        # Disjoint intervals case.
        # 1) Build flat mapping from source intervals.
        # Example: intervals=[(0,3), (10,15)] -> flat segments: (0,3,0) and (3,8,10)
        # meaning flat [0,3) maps to source [0,3), and flat [3,8) maps to source [10,15).
        flat_segments = []  # (interval_flat_start, interval_flat_end, source_start)
        idx = 0
        for source_start, source_end in intervals:
            interval_len = source_end - source_start
            if interval_len <= 0:
                continue
            flat_segments.append((idx, idx + interval_len, source_start))
            idx += interval_len

        # 2) Intersect this rank's flat span with each flat segment, then map overlap
        # back to source coordinates.
        # Example: local flat span [2,6) with segments above gives:
        #   overlap with [0,3) => [2,3) -> source (2,3)
        #   overlap with [3,8) => [3,6) -> source (10,13)
        # result: local_intervals = [(2,3), (10,13)]
        local_intervals = []
        for interval_flat_start, interval_flat_end, source_start in flat_segments:
            overlap_flat_start = max(interval_flat_start, local_flat_start)
            overlap_flat_end = min(interval_flat_end, local_flat_end)
            if overlap_flat_start < overlap_flat_end:
                source_overlap_start = source_start + (overlap_flat_start - interval_flat_start)
                source_overlap_end = source_start + (overlap_flat_end - interval_flat_start)
                local_intervals.append((source_overlap_start, source_overlap_end))

        return local_intervals

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

    def _normalize_param_dim(self, dim: int) -> int:
        # if dim is negative, it should be normalized to the last axis
        return dim if dim >= 0 else self.param_ndim + dim


def _dtensor_from_local_like(local_tensor: torch.Tensor, ref: DTensor) -> DTensor:
    """Wrap `local_tensor` as a DTensor that mirrors `ref`'s mesh, placements,
    global shape, and stride."""
    return DTensor.from_local(
        local_tensor.contiguous(),
        ref.device_mesh,
        ref.placements,
        run_check=False,
        shape=ref.shape,
        stride=tuple(ref.stride()),
    )
