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

    | Scenario                                          | Placements                            |
    |---------------------------------------------------|---------------------------------------|
    | TP-only, non-fused (e.g. q_proj/k_proj/v_proj)    | [Shard(d)]                            |
    | TP-only, fused gate/up                            | [_StridedShard(d, sf=2)]              |
    | TP + FSDP, same tensor dim                        | [_StridedShard(d, sf=tp_size), Shard(d)] |
    | TP + FSDP, different dims                         | [Shard(d1), Shard(d2)]                |

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
    """All-gather a DTensor to fully Replicate, handling _StridedShard.

    PyTorch's redistribute() does not support _StridedShard as a source:
        _StridedShard -> redistribute() -> Replicate      ❌ AssertionError
        _StridedShard -> redistribute() -> Shard          ❌ NotImplementedError
        Shard         -> redistribute() -> Replicate      ✅ works
        Replicate     -> redistribute() -> Shard          ✅ works
        Replicate     -> redistribute() -> _StridedShard  ✅ works

    During reconstruction, we walk placements right-to-left (innermost mesh dim first),
    invoke each one's low-level _to_replicate_tensor, and wait for the async collective to finish
    at each step. Example — global [256, 1024], mesh (fsdp=2, tp=2),
    placements (_StridedShard(0), Shard(0)), local [64, 1024]:

        i=1 (tp,   Shard(0)):         [64, 1024] -> [128, 1024]
        i=0 (fsdp, _StridedShard(0)): [128, 1024] -> [256, 1024]
    """
    mesh = tensor.device_mesh
    placements = tensor.placements
    replicate_all = tuple(Replicate() for _ in range(mesh.ndim))

    if not any(isinstance(p, _StridedShard) for p in placements):
        return tensor.redistribute(placements=replicate_all)

    with torch.no_grad():
        local = tensor._local_tensor
        for i in reversed(range(mesh.ndim)):
            p = placements[i]
            if p.is_replicate():
                continue
            logical_shape = list(tensor.shape)
            for j, pj in enumerate(placements[:i]):
                if not pj.is_replicate():
                    size, _ = Shard.local_shard_size_and_offset(
                        logical_shape[pj.dim], mesh.size(j), mesh.get_local_rank(j)
                    )
                    logical_shape[pj.dim] = size
            local = p._to_replicate_tensor(local, mesh, i, logical_shape)
            local = wait_tensor(local)
        return DTensor.from_local(local, mesh, replicate_all, run_check=False)


def _find_strided_shard_placement_from_fused_params(placements):
    for i, p in enumerate(placements):
        if not isinstance(p, _StridedShard):
            continue
        # We want to find the first _StridedShard placement that is not composed with another placement on the same dim.
        # Because that means it's a fused parameter. Meanwhile if it's acting on the same dim, that means it comes from composing parallelism together
        has_partner_on_same_dim = any(
            j != i and getattr(other, "dim", None) == p.dim for j, other in enumerate(placements)
        )
        if not has_partner_on_same_dim:
            return p
    return None


def _split_fused_dtensor(dt, dim, n_pieces):
    local_pieces = list(dt._local_tensor.chunk(n_pieces, dim=dim))
    if len(local_pieces) != n_pieces:
        raise RuntimeError(
            f"Cannot split DTensor of shape {tuple(dt.shape)} into {n_pieces} pieces along dim {dim}: "
            f"got {len(local_pieces)} chunks instead."
        )
    new_placements = tuple(
        Shard(p.dim) if (isinstance(p, _StridedShard) and p.dim == dim) else p for p in dt.placements
    )
    return [
        DTensor.from_local(lp.contiguous(), dt.device_mesh, new_placements, run_check=False) for lp in local_pieces
    ]


def _merge_unfused_dtensors(pieces, dim, target_placements):
    local_cat = torch.cat([p._local_tensor for p in pieces], dim=dim).contiguous()
    return DTensor.from_local(local_cat, pieces[0].device_mesh, target_placements, run_check=False)


def get_fusion_metadata(optimizer_state_dict):
    """Inspect the optimizer state dict and return metadata describing every
    fused param that needs to be split for DCP. Does NOT mutate
    `optimizer_state_dict`. Pair it with `unfuse_optimizer_state` to apply
    the split and `fuse_optimizer_state` to undo it on load.

    Args:
        optimizer_state_dict: as returned by `get_optimizer_state_dict(...)`. For
            a Mixtral on a (fsdp=2, tp=2) mesh with AdamW and a single MoE layer,
            it looks like:

                {
                    "state": {
                        "model.layers.0.mlp.experts.gate_up_proj": {
                            "exp_avg":    DTensor(shape=(4, 16, 8),
                                                  placements=(Shard(0), _StridedShard(1, sf=2))),
                            "exp_avg_sq": DTensor(shape=(4, 16, 8),
                                                  placements=(Shard(0), _StridedShard(1, sf=2))),
                            "step":       tensor(7.0),
                        },
                        "model.layers.0.mlp.experts.down_proj": {
                            "exp_avg":    DTensor(shape=(4, 8, 16),
                                                  placements=(Shard(0), Shard(2))),
                            "step":       tensor(7.0),
                        },
                    },
                    "param_groups": [{"lr": 1e-4, ...}],
                }

    Returns:
            fusion_metadata = {
                "model.layers.0.mlp.experts.gate_up_proj": {
                    "chunk_dim": 1,
                    "placements": (Shard(0), _StridedShard(1, split_factor=2)),
                    "unfused_keys": [
                        "model.layers.0.mlp.experts.gate_up_proj.0",
                        "model.layers.0.mlp.experts.gate_up_proj.1",
                    ],
                },
            }
    """
    optimizer_state = optimizer_state_dict.get("state", {})
    fusion_metadata: dict[str, dict] = {}

    for param_name, optimizer_fields in optimizer_state.items():
        dtensor = next((v for v in optimizer_fields.values() if isinstance(v, DTensor)), None)
        if dtensor is None:
            continue
        strided_shard_placement = _find_strided_shard_placement_from_fused_params(dtensor.placements)
        if strided_shard_placement is None:
            continue
        fusion_metadata[param_name] = {
            "chunk_dim": strided_shard_placement.dim,
            "placements": tuple(dtensor.placements),
            "unfused_keys": [f"{param_name}.{i}" for i in range(strided_shard_placement.split_factor)],
        }

    return fusion_metadata


def unfuse_optimizer_state(optimizer_state_dict, fusion_metadata):
    """Apply `fusion_metadata` to split each fused param into its `sf` plain-Shard
    pieces in place. After the call, `optimizer_state_dict["state"]` has
    `<param>.0` .. `<param>.{sf-1}` keys in place of each original fused key;
    non-fused params (absent from `fusion_metadata`) are untouched.

    Args:
        optimizer_state_dict: same shape as the input to `get_fusion_metadata`.
        fusion_metadata: as returned by `get_fusion_metadata(optimizer_state_dict)`.
    """
    optimizer_state = optimizer_state_dict.get("state", {})

    for param_name, metadata in fusion_metadata.items():
        optimizer_fields = optimizer_state[param_name]

        for unfused_key in metadata["unfused_keys"]:
            optimizer_state[unfused_key] = {}

        for optim_param_name, value in optimizer_fields.items():
            if isinstance(value, DTensor):
                chunks = _split_fused_dtensor(value, metadata["chunk_dim"], len(metadata["unfused_keys"]))
            else:
                chunks = [value] * len(metadata["unfused_keys"])  # scalar — replicate
            for unfused_key, chunk in zip(metadata["unfused_keys"], chunks):
                optimizer_state[unfused_key][optim_param_name] = chunk

        del optimizer_state[param_name]


def fuse_optimizer_state(optimizer_state_dict, fusion_metadata):
    """Fuse the optimizer state dict back in place using the fusion info.

    Inverse of `unfuse_optimizer_state`: concatenates each set of per-piece
    sub-dicts along `chunk_dim`, rewraps each DTensor field with the original
    `_StridedShard` placement, then replaces the piece keys with the fused key
    in `optimizer_state_dict["state"]`.

    Args:
        optimizer_state_dict: the unfused state dict (e.g. just filled by
            `dcp.load`). For our Mixtral example, before this call it looks like:

                {
                    "state": {
                        "model.layers.0.mlp.experts.gate_up_proj.0": {
                            "exp_avg":    DTensor(shape=(4, 8, 8),
                                                  placements=(Shard(0), Shard(1))),
                            "exp_avg_sq": DTensor(shape=(4, 8, 8),
                                                  placements=(Shard(0), Shard(1))),
                            "step":       tensor(7.0),
                        },
                        "model.layers.0.mlp.experts.gate_up_proj.1": {
                            "exp_avg":    DTensor(shape=(4, 8, 8),
                                                  placements=(Shard(0), Shard(1))),
                            "exp_avg_sq": DTensor(shape=(4, 8, 8),
                                                  placements=(Shard(0), Shard(1))),
                            "step":       tensor(7.0),
                        },
                        "model.layers.0.mlp.experts.down_proj": {
                            "exp_avg":    DTensor(shape=(4, 8, 16),
                                                  placements=(Shard(0), Shard(2))),
                            "step":       tensor(7.0),
                        },
                    },
                    "param_groups": [{"lr": 1e-4, ...}],
                }

            After the call the `.0`/`.1` piece keys are gone and the original
            fused key is restored with the `_StridedShard` placement.

        fusion_metadata: as returned by `get_fusion_metadata`.
    """
    optimizer_state = optimizer_state_dict.get("state", {})

    for param_name, metadata in fusion_metadata.items():
        first_piece = optimizer_state[metadata["unfused_keys"][0]]

        merged_fields = {}
        for optim_param_name, value in first_piece.items():
            chunks = [optimizer_state[unfused_key][optim_param_name] for unfused_key in metadata["unfused_keys"]]
            if isinstance(value, DTensor):
                merged_fields[optim_param_name] = _merge_unfused_dtensors(
                    chunks, metadata["chunk_dim"], metadata["placements"]
                )
            else:
                merged_fields[optim_param_name] = value  # scalar — pick any copy

        optimizer_state[param_name] = merged_fields

        for unfused_key in metadata["unfused_keys"]:
            del optimizer_state[unfused_key]
