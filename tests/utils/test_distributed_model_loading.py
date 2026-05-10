# Copyright 2025 HuggingFace Inc.
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
import unittest

import torch
from torch.distributed.tensor.placement_types import Replicate, Shard, _StridedShard

from transformers.distributed.model_loading import DtensorShardOperation


class FakeMesh:
    """Fake multi-dimensional device mesh for testing DtensorShardOperation."""

    def __init__(self, shape, rank, dim_names=None):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.mesh_dim_names = dim_names or tuple(f"dim{i}" for i in range(self.ndim))
        # Compute nD coordinate (row-major: last dim changes fastest)
        self._coord = []
        r = rank
        for s in reversed(self.shape):
            self._coord.insert(0, r % s)
            r //= s

    def get_local_rank(self):
        return self._coord[0]

    def get_coordinate(self):
        return tuple(self._coord)

    def size(self):
        result = 1
        for s in self.shape:
            result *= s
        return result

    def _is_current_rank_part_of_mesh(self):
        return True

    def _sym_get_coordinate(self, dim):
        return self._coord[dim]

    def __getitem__(self, name):
        idx = self.mesh_dim_names.index(name)
        return FakeMesh(
            shape=(self.shape[idx],),
            rank=self._coord[idx],
            dim_names=(name,),
        )


def _make_dtensor_shard_op(mesh, placements, param_shape, local_shape):
    """Build a DtensorShardOperation without requiring a real DTensor / distributed init.

    The axis-0 ownership cache is computed by mimicking
    ``compute_local_shape_and_global_offset`` for the leading dim only:
    locate the mesh dim that shards param dim 0 (if any) and use its local rank.
    """
    op = object.__new__(DtensorShardOperation)
    op.device_mesh = mesh
    op.placements = tuple(placements)
    op.param_ndim = len(param_shape)
    op._axis0_offset = 0
    op._axis0_local_size = local_shape[0]
    for mesh_dim, p in enumerate(placements):
        if hasattr(p, "dim") and (p.dim % len(param_shape)) == 0:
            sub = mesh[mesh.mesh_dim_names[mesh_dim]] if mesh.ndim > 1 else mesh
            op._axis0_offset = sub.get_local_rank() * local_shape[0]
            break
    return op


class TestDtensorShardOperation(unittest.TestCase):
    """Unit tests for DtensorShardOperation.

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

    def test_no_shard_placements_returns_full_copy(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: tensor,  # rank 0 — no shards, full copy
            1: tensor,  # rank 1 — no shards, full copy
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Replicate()], param_shape=(4, 4), local_shape=(4, 4))
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_1D_shard(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: tensor[:2],  # rank 0 — first half
            1: tensor[2:],  # rank 1 — second half
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(4, 4), local_shape=(2, 4))
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_1D_strided_shard(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: tensor[[0, 2]],  # first piece of each group — rows {0, 2}
            1: tensor[[1, 3]],  # second piece of each group — rows {1, 3}
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(
                mesh, [_StridedShard(dim=0, split_factor=2)], param_shape=(4, 4), local_shape=(2, 4)
            )
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_2D_shard_different_dims(self):
        tensor = torch.arange(64).reshape(8, 8).float()
        expected = {
            0: tensor[:4, :4],  # top-left
            1: tensor[:4, 4:],  # top-right
            2: tensor[4:, :4],  # bottom-left
            3: tensor[4:, 4:],  # bottom-right
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0), Shard(1)], param_shape=(8, 8), local_shape=(4, 4))
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_2D_shard_same_dim(self):
        tensor = torch.arange(64).reshape(8, 8).float()
        expected = {
            0: tensor[:2],  # rows 0-1
            1: tensor[2:4],  # rows 2-3
            2: tensor[4:6],  # rows 4-5
            3: tensor[6:8],  # rows 6-7
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0), Shard(0)], param_shape=(8, 8), local_shape=(2, 8))
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_2D_strided_shard_same_dim(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: tensor[[0]],  # row 0
            1: tensor[[2]],  # row 2
            2: tensor[[1]],  # row 1
            3: tensor[[3]],  # row 3
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(
                mesh,
                [_StridedShard(dim=0, split_factor=2), Shard(0)],
                param_shape=(4, 4),
                local_shape=(1, 4),
            )
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_2D_strided_shard_different_dims(self):
        tensor = torch.arange(16).reshape(4, 4).float()
        expected = {
            0: torch.cat([tensor[:2, 0:1], tensor[:2, 2:3]], dim=1),  # top rows, cols {0, 2}
            1: torch.cat([tensor[:2, 1:2], tensor[:2, 3:4]], dim=1),  # top rows, cols {1, 3}
            2: torch.cat([tensor[2:, 0:1], tensor[2:, 2:3]], dim=1),  # bottom rows, cols {0, 2}
            3: torch.cat([tensor[2:, 1:2], tensor[2:, 3:4]], dim=1),  # bottom rows, cols {1, 3}
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(
                mesh,
                [Shard(0), _StridedShard(dim=1, split_factor=2)],
                param_shape=(4, 4),
                local_shape=(2, 2),
            )
            torch.testing.assert_close(op.shard_tensor(tensor), expected[rank], msg=f"rank {rank}")

    def test_moe_1D_shard_filters_by_axis0_ownership(self):
        source = torch.ones(2, 2)
        expected = {
            0: {
                0: source,  # first owned
                1: source,  # last owned
                2: None,  # first not-owned (upper boundary, exclusive)
                3: None,  # not owned
            },
            1: {
                0: None,  # not owned
                1: None,  # last not-owned (just below offset)
                2: source,  # first owned (lower boundary, inclusive)
                3: source,  # last owned
            },
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(4, 2, 2), local_shape=(2, 2, 2))
            for tensor_idx, exp in expected[rank].items():
                with self.subTest(rank=rank, tensor_idx=tensor_idx):
                    shard = op.shard_tensor(source, tensor_idx=tensor_idx)
                    if exp is None:
                        self.assertIsNone(shard)
                    else:
                        torch.testing.assert_close(shard, exp)

    def test_moe_1D_strided_shard_on_inner_dim_degrades_to_contiguous(self):
        source = torch.arange(8).reshape(4, 2).float()
        expected = {
            0: source[:2],  # rank 0 — first half (strided silently degraded to contiguous)
            1: source[2:],  # rank 1 — second half
        }
        for rank in range(2):
            mesh = FakeMesh(shape=(2,), rank=rank)
            op = _make_dtensor_shard_op(
                mesh,
                [_StridedShard(dim=1, split_factor=2)],
                param_shape=(8, 8, 2),
                local_shape=(8, 4, 2),
            )
            torch.testing.assert_close(op.shard_tensor(source, tensor_idx=0), expected[rank], msg=f"rank {rank}")

    def test_moe_2D_shard_on_axis0_and_inner_dim_slices_inner(self):
        source = torch.arange(8).reshape(4, 2).float()
        expected = {
            0: source[:2],  # owned; inner Shard(1) → source dim 0 first half
            1: source[2:],  # owned; inner Shard(1) → source dim 0 second half
            2: None,  # not owned (axis-0 ownership filter)
            3: None,  # not owned
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(0), Shard(1)], param_shape=(4, 4, 2), local_shape=(2, 2, 2))
            shard = op.shard_tensor(source, tensor_idx=1)
            if expected[rank] is None:
                self.assertIsNone(shard, msg=f"rank {rank}")
            else:
                torch.testing.assert_close(shard, expected[rank], msg=f"rank {rank}")

    def test_moe_2D_shard_with_negative_dim_indices(self):
        source = torch.arange(8).reshape(4, 2).float()
        expected = {
            0: source[:, :1],  # owned; inner Shard(-1) → source dim 1, first half
            1: source[:, 1:],  # owned; inner Shard(-1) → source dim 1, second half
            2: None,  # not owned
            3: None,  # not owned
        }
        for rank in range(4):
            mesh = FakeMesh(shape=(2, 2), rank=rank)
            op = _make_dtensor_shard_op(mesh, [Shard(-3), Shard(-1)], param_shape=(4, 4, 2), local_shape=(2, 4, 1))
            shard = op.shard_tensor(source, tensor_idx=1)
            if expected[rank] is None:
                self.assertIsNone(shard, msg=f"rank {rank}")
            else:
                torch.testing.assert_close(shard, expected[rank], msg=f"rank {rank}")

    def test_strided_intervals(self):
        # Direct tests for _strided_intervals(intervals, rank, world_size, split_factor).
        # Keys: (input_interval, rank, world_size, split_factor) -> expected output list.
        mesh = FakeMesh(shape=(2,), rank=0)
        op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(8,), local_shape=(4,))
        expected = {
            # Even (0, 8) sf=2 ws=2 -> groups (0,4) (4,8); each rank takes half of each
            ((0, 8), 0, 2, 2): [(0, 2), (4, 6)],
            ((0, 8), 1, 2, 2): [(2, 4), (6, 8)],
            # Uneven (0, 7) sf=2 -> group 0 = (0,4), group 1 = (4,7) -> size 3
            ((0, 7), 0, 2, 2): [(0, 2), (4, 6)],  # rank 0 -> half of each group
            ((0, 7), 1, 2, 2): [(2, 4), (6, 7)],  # rank 1's piece in group 1 is 1 elem wide
            # split_factor=1 collapses to a single group -> contiguous behavior
            ((0, 4), 0, 2, 1): [(0, 2)],
            # split_factor=4 on size-2: groups (0,1), (1,2), (2,2), (3,2) -> last 2 empty -> skipped
            ((0, 2), 0, 2, 4): [(0, 1), (1, 2)],
        }
        for (interval, rank, ws, sf), exp in expected.items():
            with self.subTest(interval=interval, rank=rank, ws=ws, sf=sf):
                self.assertEqual(op._strided_intervals([interval], rank=rank, world_size=ws, split_factor=sf), exp)

    def test_contiguous_intervals(self):
        # Direct tests for _contiguous_intervals(intervals, rank, world_size).
        # Keys: (input_intervals, rank, world_size) -> expected output list.
        mesh = FakeMesh(shape=(2,), rank=0)
        op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(8,), local_shape=(4,))
        expected = {
            # Single interval, even split -> rank 0 -> first 4 elems, rank 1 -> last 4 elems
            (((0, 8),), 0, 2): [(0, 4)],
            (((0, 8),), 1, 2): [(4, 8)],
            # Uneven: size 5 / 2 ranks -> rank 0 -> first 3 elems, rank 1 -> last 2 elems
            (((0, 5),), 0, 2): [(0, 3)],
            (((0, 5),), 1, 2): [(3, 5)],
            # Empty: size 3 / 4 ranks -> rank 3 -> nothing
            (((0, 3),), 3, 4): [],
            # Multi-input intervals (8 elems): rank -> takes its slice from whichever interval(s) cover it
            (((0, 4), (10, 14)), 0, 2): [(0, 4)],  # rank 0 -> first 4 elems = first interval entirely
            (((0, 4), (10, 14)), 1, 2): [(10, 14)],  # rank 1 -> last 4 elems = second interval entirely
            (((0, 4), (10, 14)), 2, 4): [(10, 12)],  # ws=4, rank 2 -> cuts mid-input-interval
        }
        for (intervals, rank, ws), exp in expected.items():
            with self.subTest(intervals=intervals, rank=rank, ws=ws):
                self.assertEqual(op._contiguous_intervals(list(intervals), rank=rank, world_size=ws), exp)

    def test_slice_and_cat(self):
        # Direct tests for _slice_and_cat(source, intervals, device, dtype).
        tensor = torch.arange(64).reshape(8, 8).float()
        mesh = FakeMesh(shape=(2,), rank=0)
        op = _make_dtensor_shard_op(mesh, [Shard(0)], param_shape=(8, 8), local_shape=(4, 8))
        expected = {
            # Fast path: every dim is single-interval -> one slice read, no cat
            "fast_path": ([[(0, 4)], [(0, 8)]], tensor[:4, :]),
            # Cat along dim 1: two disjoint col ranges -> read separately and concat
            "cat_dim1": ([[(0, 4)], [(0, 2), (4, 6)]], torch.cat([tensor[:4, :2], tensor[:4, 4:6]], dim=1)),
            # Cat along dim 0: two disjoint row ranges -> read separately and concat
            "cat_dim0": ([[(0, 2), (4, 6)], [(0, 8)]], torch.cat([tensor[:2, :], tensor[4:6, :]], dim=0)),
        }
        for case, (intervals, exp) in expected.items():
            with self.subTest(case=case):
                torch.testing.assert_close(op._slice_and_cat(tensor, intervals, None, None), exp)

        # Reject: two dims with disjoint ranges -> would require an outer-product of reads.
        with self.assertRaises(ValueError):
            op._slice_and_cat(tensor, [[(0, 2), (4, 6)], [(0, 2), (4, 6)]], None, None)

        result = op._slice_and_cat(tensor, [[(0, 4)], [(0, 8)]], None, torch.float16)
        self.assertEqual(result.dtype, torch.float16)


if __name__ == "__main__":
    unittest.main()
