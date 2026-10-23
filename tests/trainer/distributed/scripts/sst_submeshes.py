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
"""Validate independent SST bases and checkpoint ownership on distinct two-rank meshes."""

import argparse
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from transformers import SST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--backend")
    parser.add_argument("--optim")
    parser.add_argument("--precision")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("replica", "shard"))
    submesh = mesh["shard"]
    replica = mesh.get_coordinate()[0]
    matrix = torch.arange(24, device="cuda", dtype=torch.float32).reshape(6, 4) / 100
    matrix += torch.eye(6, 4, device="cuda") * (2 + replica)
    parameter = torch.nn.Parameter(distribute_tensor(matrix, submesh, [Shard(0)]))
    optimizer = SST(torch.optim.SGD([parameter], lr=0.1, momentum=0.9), rank_fraction=0.5)
    assert optimizer.requires_per_rank_state
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    basis = optimizer.sst_state_dict()
    full = parameter.detach().full_tensor()
    entry = basis["protected_weights"][0]
    left, right = entry["left"].cuda(), entry["right"].cuda()
    delta = full - (left * entry["singular_values"].cuda().unsqueeze(0)) @ right.mH
    assert (left.mH @ delta).abs().max() < 3e-6
    assert (delta @ right).abs().max() < 3e-6
    state = copy.deepcopy(optimizer.state_dict())
    with patch("torch.linalg.svd", side_effect=AssertionError("Resume cannot reanchor")):
        restored = SST(torch.optim.SGD([parameter], lr=1.0, momentum=0.9), initialize=False)
        restored.load_state_dict(state)
    assert restored.sst_state_dict()["topology"] == basis["topology"]
    if dist.get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    torch.save(basis, args.output_dir / f"basis-{dist.get_rank()}.pt")
    dist.barrier()
    other = torch.load(
        args.output_dir / f"basis-{(dist.get_rank() + 1) % dist.get_world_size()}.pt",
        map_location="cpu",
        weights_only=True,
    )
    try:
        restored.load_sst_state_dict(other)
    except ValueError as error:
        assert "topology" in str(error)
    else:
        raise AssertionError("A rank must not load another rank's submesh basis.")
    for first, second in zip(basis["protected_weights"], restored.sst_state_dict()["protected_weights"]):
        for key in ("left", "singular_values", "right"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    if dist.get_rank() == 0:
        result = {
            "world_size": dist.get_world_size(),
            "distinct_two_rank_submeshes": True,
            "fixed_basis_restored_without_svd": True,
            "incorrect_rank_basis_rejected": True,
            "protected_subspaces_preserved": True,
        }
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
