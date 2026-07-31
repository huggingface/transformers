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
    """Shard-on-read: slice a full disk tensor down to this rank's local
    DTensor shard, for any combination of placements on a 1-D or n-D mesh.  It's on
    read because instructions are made so the cpu only fetches on disk the parts we want.

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

    Loading (this class)
    --------------------
    During from_pretrained, each rank looks up every tensor key, but does not load the weight bytes yet (safetensors get_slice).
    When a weight is needed, shard_tensor indexes that slice to read only this rank's local shard through Dtensor logic which provides
    placement logics

    Depending on how the checkpoint was saved, the weight loader either gives us one big tensor or many small ones:

    1. One stacked tensor: the checkpoint has one key with all experts together, shaped [num_experts, in, out].
       The weight loader passes it straight through; we slice out this rank's piece.

    2. One tensor per expert:  the checkpoint has a separate key for each expert (expert 0, expert 1, …), each shaped [in, out].
       The weight loader feeds them in one at a time. If this rank doesn't own a given expert,
       we skip it. Later, `MergeModulelist` will stack the owned expert we kept to create the rank's local shard
    """

    def __init__(self, param: DTensor):
        self.device_mesh = param.device_mesh
        self.placements = tuple(param.placements)
        self.param_ndim = param.ndim
        local_shape, offsets = compute_local_shape_and_global_offset(param.shape, self.device_mesh, self.placements)
        # Axis-0 range owned by this rank (used to filter per-expert pieces)
        # [_axis0_offset, _axis0_offset + _axis0_local_size)
        self._axis0_offset = offsets[0]
        self._axis0_local_size = local_shape[0]

    def shard_tensor(
        self, source: torch.Tensor, tensor_idx: int | None = None, device=None, dtype=None
    ) -> torch.Tensor | None:
        """Return this rank's local shard of a checkpoint tensor.

        Two layouts (example param shape [N, in, out]):

        - tensor_idx is None: one stacked [N, in, out] tensor;
          slice every sharded dim (including axis 0).
        - tensor_idx given: one [in, out] tensor per expert;
          return None if this rank does not own that expert, else slice
          inner dims only. Surviving pieces are stacked by MergeModulelist
          into this rank's local [n_local, in, out] shard.
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

            # prepare the slices to fetch on disk for each tensor dimension.
            intervals_by_dim = [[(0, size)] for size in source_shape]
            for dim_idx, planned_ops in enumerate(planned_ops_by_dim):
                intervals = intervals_by_dim[dim_idx]
                for placement, rank, world_size in planned_ops:
                    if placement.is_shard():
                        intervals = self._compute_contiguous_slice(intervals, rank, world_size)
                    else:
                        intervals = self._compute_strided_slice(intervals, rank, world_size, placement.split_factor)
                intervals_by_dim[dim_idx] = intervals

            has_strided_shard = any(not placement.is_shard() for _, placement in dim_placements)
            # finally fetch from the disk only the slices
            # finally fetch from the disk only the slices
            if has_strided_shard:
                # Multi-interval dim: read each piece separately, then concatenate.
                return self._slice_and_cat(source, intervals_by_dim, device, dtype)
            else:
                slice_parts = []
                for intervals in intervals_by_dim:
                    start, end = intervals[0] if len(intervals) > 0 else (0, 0)
                    slice_parts.append(slice(start, end))

                return source[tuple(slice_parts)].to(device=device, dtype=dtype)

        # MoE path
        # tensor_idx identifies the axis-0 piece in param space (not in source.shape).
        normalized_dim_placements = [
            (mesh_dim, placement, self._normalize_param_dim(placement.dim)) for mesh_dim, placement in dim_placements
        ]

        # if this rank owns expert `tensor_idx` along axis 0, we need to slice the inner dimensions, else we drop it
        has_axis0_shard = any(param_dim == 0 for _, _, param_dim in normalized_dim_placements)
        owns_tensor_idx = self._axis0_offset <= tensor_idx < self._axis0_offset + self._axis0_local_size
        if has_axis0_shard and not owns_tensor_idx:
            return None

        # `param_dim` indexes the full parameter layout [N, in, out] (expert axis first).
        # In per-expert loading, leading axis is absent ([in, out]).
        # Therefore we need to shift the dimensions by 1 to the left so that sharding operations can be applied to the inner dimensions.
        #   param [N, in, out]  ↔  source [in, out]
        #   Shard(param dim 1)  →  slice source dim 0
        #   Shard(param dim 2)  →  slice source dim 1
        planned_ops_by_source_dim = [[] for _ in source_shape]
        for mesh_dim, _, param_dim in normalized_dim_placements:
            if param_dim > 0:
                source_dim = param_dim - 1
                sub_mesh = self._get_sub_mesh(mesh_dim)
                rank, world_size = sub_mesh.get_local_rank(), sub_mesh.size()
                planned_ops_by_source_dim[source_dim].append((rank, world_size))

        intervals_by_source_dim = [[(0, size)] for size in source_shape]
        for source_dim, planned_ops in enumerate(planned_ops_by_source_dim):
            intervals = intervals_by_source_dim[source_dim]
            for rank, world_size in planned_ops:
                intervals = self._compute_contiguous_slice(intervals, rank, world_size)
            intervals_by_source_dim[source_dim] = intervals

        slice_parts = []
        for intervals in intervals_by_source_dim:
            start, end = intervals[0] if intervals else (0, 0)
            slice_parts.append(slice(start, end))

        return source[tuple(slice_parts)].to(device=device, dtype=dtype)

    def _compute_strided_slice(
        self, intervals: list[tuple[int, int]], rank: int, world_size: int, split_factor: int
    ) -> list[tuple[int, int]]:
        local_intervals = []

        for interval_start, interval_end in intervals:
            # For each interval, we break it into split_factor consecutive groups.
            group_width = math.ceil((interval_end - interval_start) / split_factor)

            for group_idx in range(split_factor):
                # Compute this group's boundaries in source coordinates.
                group_start = interval_start + group_idx * group_width
                group_end = min(group_start + group_width, interval_end)
                group_len = group_end - group_start

                if group_len > 0:
                    # Inside each group, we do a normal contiguous shard across world_size ranks.
                    local_shard_size, local_shard_offset = Shard.local_shard_size_and_offset(
                        group_len, world_size, rank
                    )
                    if local_shard_size > 0:
                        # Convert group-local offset back to global source coordinates.
                        shard_start = group_start + local_shard_offset
                        local_intervals.append((shard_start, shard_start + local_shard_size))

        return local_intervals

    def _compute_contiguous_slice(
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
            if interval_len > 0:
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

    def _slice_and_cat(
        self,
        source: torch.Tensor,
        intervals: list[list[tuple[int, int]]],
        device: torch.device | str | int | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        multi_interval_dims = [dim_idx for dim_idx, dim_intervals in enumerate(intervals) if len(dim_intervals) > 1]
        if len(multi_interval_dims) > 1:
            # NOTE(3outeille): not sure yet which scenario will have StridedShard
            # placements on both row and column. Thus, delay implementing this for now.
            raise ValueError("Current shard-on-read only supports disjoint ranges on a single checkpoint dimension.")
        concat_dim = multi_interval_dims[0] if multi_interval_dims else None

        base_slices = []
        for dim_idx, dim_intervals in enumerate(intervals):
            if dim_idx == concat_dim:
                # Disconnected intervals on this dim — placeholder; filled per interval below.
                base_slices.append(slice(None))
            else:
                # Single contiguous slice on this dim.
                start, end = dim_intervals[0]
                base_slices.append(slice(start, end))

        # Fast path: every dim is one contiguous interval, read in a single slice.
        if concat_dim is None:
            return source[tuple(base_slices)].to(device=device, dtype=dtype)

        # Multi-interval dim: keep base slices fixed and vary concat_dim only.
        base_slices_tuple = tuple(base_slices)
        interval_tensors = []
        for interval_start, interval_end in intervals[concat_dim]:
            interval_slices = (
                *base_slices_tuple[:concat_dim],
                slice(interval_start, interval_end),
                *base_slices_tuple[concat_dim + 1 :],
            )
            interval_tensors.append(source[interval_slices])

        return torch.cat(interval_tensors, dim=concat_dim).to(device=device, dtype=dtype)

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
