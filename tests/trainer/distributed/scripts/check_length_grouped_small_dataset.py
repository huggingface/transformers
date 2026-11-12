# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Exercise length-grouped padding through real CPU DDP optimizer steps."""

import copy
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from transformers.trainer_pt_utils import DistributedLengthGroupedSampler


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        torch.manual_seed(17)
        model = nn.Linear(4, 1)
        reference = copy.deepcopy(model)
        model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        inputs = torch.tensor([[1.0, -0.5, 0.25, 2.0]])
        targets = torch.tensor([[0.75]])
        dataset = TensorDataset(inputs, targets)
        sampler = DistributedLengthGroupedSampler(1, lengths=[4], seed=17)
        loader = DataLoader(dataset, sampler=sampler, batch_size=1)
        indices = []
        steps = 0
        for epoch in range(2):
            sampler.set_epoch(epoch)
            indices.append(list(sampler))
            for batch_inputs, batch_targets in loader:
                optimizer.zero_grad()
                (model(batch_inputs) - batch_targets).square().mean().backward()
                optimizer.step()
                steps += 1
            reference_optimizer.zero_grad()
            (reference(inputs) - targets).square().mean().backward()
            reference_optimizer.step()
            for actual, expected in zip(model.module.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

        values = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, values)
        for other in gathered:
            torch.testing.assert_close(values, other, rtol=0, atol=0)
        report = {"steps": steps, "indices": indices, "parameters": values.tolist()}
        (Path(sys.argv[1]) / f"rank-{dist.get_rank()}.json").write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
