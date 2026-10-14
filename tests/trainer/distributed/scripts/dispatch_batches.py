# Copyright 2020 The HuggingFace Team. All rights reserved.
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
Worker script for dispatch_batches=False with a finite iterable dataset.

Verifies that training completes successfully when ``dispatch_batches``
is disabled.

Run via torchrun or accelerate launch.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset

from transformers import HfArgumentParser, Trainer, TrainingArguments


class RegressionModel(nn.Module):
    def __init__(self, a=0, b=0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a).float())
        self.b = nn.Parameter(torch.tensor(b).float())
        self.config = None

    def forward(self, input_x, labels=None, **kwargs):
        y = input_x * self.a + self.b
        if labels is None:
            return (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y)


class RegressionDataset:
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        np.random.seed(seed)
        self.label_names = ["labels"] if label_names is None else label_names
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.ys = [a * self.x + b + np.random.normal(scale=0.1, size=(length,)) for _ in self.label_names]
        self.ys = [y.astype(np.float32) for y in self.ys]

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        result = {name: y[i] for name, y in zip(self.label_names, self.ys)}
        result["input_x"] = self.x[i]
        return result


class FiniteIterableDataset(IterableDataset):
    def __init__(self, a=2, b=3, length=64, seed=42, label_names=None):
        self.dataset = RegressionDataset(a=a, b=b, length=length, seed=seed, label_names=label_names)
        self.current_sample = 0

    def __iter__(self):
        while self.current_sample < len(self.dataset):
            yield self.dataset[self.current_sample]
            self.current_sample += 1


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]

    training_args.per_device_train_batch_size = 1
    training_args.max_steps = 1
    training_args.accelerator_config.dispatch_batches = False

    train_dataset = FiniteIterableDataset(label_names=["labels", "extra"], length=1)
    model = RegressionModel()

    trainer = Trainer(model, training_args, train_dataset=train_dataset)
    trainer.train()
