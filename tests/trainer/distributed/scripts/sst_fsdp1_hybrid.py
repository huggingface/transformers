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
"""Validate SST refresh across both axes of a permuted FSDP1 hybrid mesh."""

import argparse
import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from sst import ObservedSST, assert_same_state
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--backend")
    parser.add_argument("--optim")
    parser.add_argument("--precision")
    parser.add_argument("--refresh_every", type=int)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=device)
    mesh = DeviceMesh("cuda", [[3, 0], [1, 2]], mesh_dim_names=("replica", "shard"))

    def make_model():
        torch.manual_seed(17)
        return FSDP(
            torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8)).to(device),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh,
            device_id=device,
            use_orig_params=True,
        )

    def update(model, optimizer):
        optimizer.zero_grad()
        model(torch.arange(16, device=device).float().reshape(2, 8) / 16).square().mean().backward()
        optimizer.step()

    original_svd = torch.linalg.svd

    def source_only_svd(*values, **kwargs):
        assert dist.get_rank() == 0, "Only the owning hybrid mesh's elected source may compute a basis."
        return original_svd(*values, **kwargs)

    with patch("torch.linalg.svd", side_effect=source_only_svd):
        model = make_model()
        optimizer = ObservedSST(
            torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9),
            model=model,
            rank_fraction=0.5,
            refresh_every=args.refresh_every,
        )
        assert optimizer.requires_per_rank_state
        for _ in range(3):
            update(model, optimizer)
        saved_model = copy.deepcopy(model.state_dict())
        saved_inner = copy.deepcopy(FSDP.optim_state_dict(model, optimizer.optimizer))
        saved_basis = copy.deepcopy(optimizer.sst_state_dict())
        for _ in range(3):
            update(model, optimizer)
        final_model = copy.deepcopy(model.state_dict())
        final_inner = copy.deepcopy(FSDP.optim_state_dict(model, optimizer.optimizer))
        final_basis = copy.deepcopy(optimizer.sst_state_dict())

        replacement = make_model()
        replacement.load_state_dict(saved_model)
        restored_inner = torch.optim.SGD(replacement.parameters(), lr=1.0, momentum=0.9)
        restored_inner.load_state_dict(FSDP.optim_state_dict_to_load(replacement, restored_inner, saved_inner))
        restored = ObservedSST(restored_inner, model=replacement, initialize=False)
        restored.load_sst_state_dict(saved_basis)
        for _ in range(3):
            update(replacement, restored)
        assert_same_state(final_model, replacement.state_dict())
        assert_same_state(final_inner, FSDP.optim_state_dict(replacement, restored.optimizer))
        assert_same_state(final_basis, restored.sst_state_dict())
        assert restored.num_updates == 6
        assert restored.num_refreshes == (6 // args.refresh_every if args.refresh_every else 0)
        before_refresh = copy.deepcopy(replacement.state_dict())
        restored.refresh()
        assert_same_state(before_refresh, replacement.state_dict())
        assert restored.num_refreshes == final_basis["num_refreshes"] + 1

    before_failure = copy.deepcopy(restored.sst_state_dict())
    failure = (
        patch("torch.linalg.svd", side_effect=RuntimeError("Injected hybrid source-rank failure"))
        if dist.get_rank() == 0
        else nullcontext()
    )
    with failure:
        try:
            restored.refresh()
        except RuntimeError as error:
            assert "active basis was not replaced" in str(error)
        else:
            raise AssertionError("A source-rank failure must reach both the shard and replica groups.")
    assert_same_state(before_failure, restored.sst_state_dict())
    assert optimizer.maximum_leakage < 2e-6
    assert optimizer.maximum_replica_difference == 0
    assert restored.maximum_leakage < 2e-6
    assert restored.maximum_replica_difference == 0
    if dist.get_rank() == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "world_size": dist.get_world_size(),
            "fsdp1_hybrid_shard": True,
            "permuted_mesh_rank_order": True,
            "refresh_every": args.refresh_every,
            "refresh_clock_restored": True,
            "single_source_for_both_mesh_axes": True,
            "refresh_preserves_weights_and_inner_state": True,
            "source_rank_failure_propagated_without_replacing_bases": True,
            "maximum_protected_leakage": max(optimizer.maximum_leakage, restored.maximum_leakage),
            "maximum_replica_difference": 0,
        }
        (args.output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
