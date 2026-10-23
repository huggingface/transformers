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
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from transformers import SST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--backend")
    parser.add_argument("--optim")
    parser.add_argument("--precision")
    parser.add_argument("--refresh_every", type=int)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    mesh = DeviceMesh("cuda", [[3, 0], [1, 2]], mesh_dim_names=("replica", "shard"))
    submesh = mesh["shard"]
    replica = mesh.get_coordinate()[0]
    matrix = torch.arange(24, device="cuda", dtype=torch.float32).reshape(6, 4) / 100
    matrix += torch.eye(6, 4, device="cuda") * (2 + replica)
    parameter = torch.nn.Parameter(distribute_tensor(matrix, submesh, [Shard(0)]))
    optimizer = SST(
        torch.optim.SGD([parameter], lr=0.1, momentum=0.9),
        rank_fraction=0.5,
        refresh_every=args.refresh_every,
    )
    assert optimizer.requires_per_rank_state
    torch.testing.assert_close(parameter.detach().full_tensor(), matrix, rtol=0, atol=0)
    initial_values = optimizer.sst_state_dict()["protected_weights"][0]["singular_values"].cuda()
    owners = submesh.mesh.flatten().tolist()
    all_values = [torch.empty_like(initial_values) for _ in range(dist.get_world_size())]
    dist.all_gather(all_values, initial_values)
    for rank, values in enumerate(all_values):
        if rank in owners:
            torch.testing.assert_close(initial_values, values, rtol=0, atol=0)
        else:
            assert (initial_values - values).abs().max().item() > 0.5
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    basis = optimizer.sst_state_dict()
    full = parameter.detach().full_tensor()
    entry = basis["protected_weights"][0]
    left, right = entry["left"].cuda(), entry["right"].cuda()
    delta = full - (left * entry["singular_values"].cuda().unsqueeze(0)) @ right.mH
    leakage = max((left.mH @ delta).abs().max().item(), (delta @ right).abs().max().item())
    # A freshly computed fp32 SVD's residual scales with the leading singular value.
    relative_leakage = leakage / max(1.0, entry["singular_values"].max().item())
    assert relative_leakage < 3e-6, (leakage, relative_leakage)
    state = copy.deepcopy(optimizer.state_dict())
    saved_full = full.clone()
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    final_full = parameter.detach().full_tensor()
    final_state = copy.deepcopy(optimizer.sst_state_dict())
    replacement = torch.nn.Parameter(distribute_tensor(saved_full, submesh, [Shard(0)]))
    with patch("torch.linalg.svd", side_effect=AssertionError("Resume cannot reanchor")):
        restored = SST(torch.optim.SGD([replacement], lr=1.0, momentum=0.9), initialize=False)
        restored.load_state_dict(state)
    assert restored.sst_state_dict()["topology"] == basis["topology"]
    for _ in range(3):
        replacement.grad = torch.ones_like(replacement)
        restored.step()
    torch.testing.assert_close(final_full, replacement.detach().full_tensor(), rtol=0, atol=0)
    assert restored.num_updates == 6
    assert restored.num_refreshes == (6 // args.refresh_every if args.refresh_every else 0)
    for first, second in zip(final_state["protected_weights"], restored.sst_state_dict()["protected_weights"]):
        for key in ("left", "singular_values", "right"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
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
    for first, second in zip(final_state["protected_weights"], restored.sst_state_dict()["protected_weights"]):
        for key in ("left", "singular_values", "right"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    source = min(submesh.mesh.flatten().tolist())
    failure = (
        patch("torch.linalg.svd", side_effect=RuntimeError("Injected source-rank SVD failure"))
        if dist.get_rank() == source
        else nullcontext()
    )
    with failure:
        try:
            restored.refresh()
        except RuntimeError as error:
            assert "active basis was not replaced" in str(error)
        else:
            raise AssertionError("Every owner must receive the source rank's refresh failure.")
    assert restored.num_refreshes == final_state["num_refreshes"]
    for first, second in zip(final_state["protected_weights"], restored.sst_state_dict()["protected_weights"]):
        for key in ("left", "singular_values", "right"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    if dist.get_rank() == 0:
        result = {
            "world_size": dist.get_world_size(),
            "distinct_two_rank_submeshes": True,
            "bases_identical_within_owners_and_distinct_between_submeshes": True,
            "active_basis_restored_without_extra_svd": True,
            "refresh_every": args.refresh_every,
            "refresh_clock_restored": True,
            "permuted_mesh_rank_order": True,
            "source_rank_failure_propagated_without_replacing_bases": True,
            "incorrect_rank_basis_rejected": True,
            "protected_subspaces_preserved": True,
            "maximum_relative_protected_leakage": relative_leakage,
        }
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
