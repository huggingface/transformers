# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""
Worker script for dataloader worker seed divergence tests.

Verifies that dataloader workers get different random seeds across GPUs,
so that each rank sees different random augmentations.

Run via torchrun or accelerate launch.
"""

import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.testing_utils import torch_device


def gather_from_all_gpus(tensor, world_size):
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    return gather_list


class DummyDataset(Dataset):
    def __init__(self):
        self.length = 64

    def __len__(self):
        return self.length

    def __getitem__(self, i) -> int:
        x = random.random()
        y = np.random.random()
        z = torch.rand([]).item()
        return {"x": torch.tensor([x, y, z])}


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        local_tensor = torch.tensor(x, device=torch_device)
        gathered = gather_from_all_gpus(local_tensor, dist.get_world_size())
        assert not all(torch.allclose(t, gathered[0]) for t in gathered[1:])
        y = self.fc(x)
        return (y.mean(), y)


def run_distributed_training(training_args):
    set_seed(42)
    model = DummyModel()
    dataset = DummyDataset()
    training_args.max_steps = 3
    # dataloader_num_workers must be > 0 to enable worker_init_fn
    training_args.dataloader_num_workers = 2
    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]
    run_distributed_training(training_args)
