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

"""Which bytes of a checkpoint a rank warms before loading it."""

import os
import socket
import struct
import tempfile
import unittest
from unittest.mock import patch

from transformers import is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from safetensors.torch import save_file
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from transformers.distributed.utils import _merge, _rank_byte_spans


EXPERTS, HIDDEN, LAYERS = 8, 64, 2
SPANS = "transformers.distributed.utils._SEEK_COST_BYTES"


def write_checkpoints(directory):
    """The two layouts an MoE checkpoint is saved in, holding the same weights."""
    packed = os.path.join(directory, "packed.safetensors")
    per_expert = os.path.join(directory, "per_expert.safetensors")
    save_file(
        {f"model.layers.{i}.mlp.experts.gate_up_proj": torch.zeros(EXPERTS, HIDDEN, HIDDEN) for i in range(LAYERS)},
        packed,
    )
    save_file(
        {
            f"model.layers.{i}.mlp.experts.{e}.gate_up_proj": torch.zeros(HIDDEN, HIDDEN)
            for i in range(LAYERS)
            for e in range(EXPERTS)
        },
        per_expert,
    )
    return packed, per_expert


def data_start(path):
    """Where the tensor data begins: the 8 byte header length, then the header."""
    with open(path, "rb") as f:
        return 8 + struct.unpack("<Q", f.read(8))[0]


def expert_bytes():
    return HIDDEN * HIDDEN * torch.finfo(torch.float32).bits // 8


def total(spans):
    return sum(end - start for _, start, end in spans)


@require_torch
class MergeTest(unittest.TestCase):
    def test_sorts_and_joins_overlapping_ranges(self):
        with patch(SPANS, 0):
            self.assertEqual(_merge([(30, 40), (0, 10), (5, 20)], "f"), [("f", 0, 20), ("f", 30, 40)])

    def test_joins_ranges_that_touch(self):
        with patch(SPANS, 0):
            self.assertEqual(_merge([(0, 10), (10, 20)], "f"), [("f", 0, 20)])

    def test_bridges_a_gap_cheaper_than_a_seek(self):
        with patch(SPANS, 100):
            self.assertEqual(_merge([(0, 10), (60, 70)], "f"), [("f", 0, 70)])

    def test_keeps_a_gap_dearer_than_a_seek(self):
        with patch(SPANS, 100):
            self.assertEqual(_merge([(0, 10), (600, 700)], "f"), [("f", 0, 10), ("f", 600, 700)])


@require_torch
class SingleRankSpansTest(unittest.TestCase):
    def test_an_unsharded_state_dict_reads_every_tensor_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            packed, _ = write_checkpoints(directory)
            state_dict = {
                f"model.layers.{i}.mlp.experts.gate_up_proj": torch.zeros(EXPERTS, HIDDEN, HIDDEN, device="meta")
                for i in range(LAYERS)
            }
            own, common = _rank_byte_spans([packed], state_dict)
            self.assertEqual(own, [])
            # One span over the whole data section, which starts after the header.
            self.assertEqual(common, [(packed, data_start(packed), os.path.getsize(packed))])
            self.assertEqual(total(common), LAYERS * EXPERTS * expert_bytes())

    def test_spans_stay_inside_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            packed, per_expert = write_checkpoints(directory)
            for path in (packed, per_expert):
                own, common = _rank_byte_spans([path], {})
                for _, start, end in own + common:
                    self.assertGreaterEqual(start, 0)
                    self.assertLessEqual(end, os.path.getsize(path))


def _spans_worker(rank, world, port, directory, seek_cost):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        # The model always holds experts packed; only the checkpoint layout differs. Expert
        # parallelism shards them over the mesh on dim 0, the expert dimension.
        state_dict = {
            f"model.layers.{i}.mlp.experts.gate_up_proj": distribute_tensor(
                torch.zeros(EXPERTS, HIDDEN, HIDDEN), mesh, [Shard(0)]
            )
            for i in range(LAYERS)
        }
        share = LAYERS * (EXPERTS // world) * expert_bytes()
        with patch(SPANS, seek_cost):
            for name in ("packed", "per_expert"):
                path = os.path.join(directory, f"{name}.safetensors")
                own, common = _rank_byte_spans([path], state_dict)
                assert common == [], f"{name}: rank {rank} kept unsharded spans {common}"
                assert total(own) == share, f"{name}: rank {rank} warms {total(own)}, expected {share}"
                # Its experts are consecutive on disk, so they come out as one run per layer.
                assert len(own) == LAYERS, f"{name}: rank {rank} got {len(own)} spans, expected {LAYERS}"
                assert min(start for _, start, _ in own) >= data_start(path), (
                    f"{name}: rank {rank} warms bytes that fall inside the header"
                )
    finally:
        dist.destroy_process_group()


def _bridging_worker(rank, world, port, directory, seek_cost):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        state_dict = {
            f"model.layers.{i}.mlp.experts.gate_up_proj": distribute_tensor(
                torch.zeros(EXPERTS, HIDDEN, HIDDEN), mesh, [Shard(0)]
            )
            for i in range(LAYERS)
        }
        path = os.path.join(directory, "per_expert.safetensors")
        with patch(SPANS, seek_cost):
            own, _ = _rank_byte_spans([path], state_dict)
        # Every gap here is far cheaper than a seek, so the runs collapse into one read.
        assert len(own) == 1, f"rank {rank} got {len(own)} spans, expected 1"
        assert total(own) > LAYERS * (EXPERTS // world) * expert_bytes()
    finally:
        dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@require_torch
class TwoRankSpansTest(unittest.TestCase):
    def run_workers(self, worker, seek_cost):
        with tempfile.TemporaryDirectory() as directory:
            write_checkpoints(directory)
            mp.spawn(worker, args=(2, free_port(), directory, seek_cost), nprocs=2, join=True)

    def test_each_rank_warms_only_the_experts_it_owns(self):
        # Both layouts, priced so no gap is bridged: what is left is exactly this rank's share.
        self.run_workers(_spans_worker, 0)

    def test_cheap_gaps_between_owned_experts_are_bridged(self):
        self.run_workers(_bridging_worker, 12 * 2**20)
