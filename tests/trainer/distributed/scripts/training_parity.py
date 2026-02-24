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
Worker script for training parity tests.

Runs a short training loop with a simple in-memory model and saves
per-step losses to ``<output_dir>/losses.json`` so the test harness
can compare results across different launchers (torchrun vs accelerate).

Run via torchrun or accelerate launch.
"""

import json
import os

import numpy as np
import torch.nn as nn

from transformers import HfArgumentParser, Trainer, TrainerCallback, TrainingArguments, set_seed


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.config = None

    def forward(self, input_x, labels=None, **kwargs):
        y = self.fc(input_x.unsqueeze(-1)).squeeze(-1)
        if labels is None:
            return (y,)
        loss = nn.functional.mse_loss(y, labels)
        return (loss, y)


class RegressionDataset:
    def __init__(self, length=64, seed=42):
        np.random.seed(seed)
        self.length = length
        self.x = np.random.normal(size=(length,)).astype(np.float32)
        self.y = (2.0 * self.x + 3.0 + np.random.normal(scale=0.1, size=(length,))).astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return {"input_x": self.x[i], "labels": self.y[i]}


class StoreLossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments,))
    training_args = parser.parse_args_into_dataclasses()[0]

    set_seed(42)

    loss_callback = StoreLossCallback()

    training_args.logging_steps = 1
    training_args.max_steps = 5
    training_args.learning_rate = 1e-2
    training_args.disable_tqdm = True
    training_args.dataloader_drop_last = True

    trainer = Trainer(
        model=SimpleModel(),
        args=training_args,
        train_dataset=RegressionDataset(),
        callbacks=[loss_callback],
    )
    trainer.train()

    if training_args.local_process_index == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "losses.json"), "w") as f:
            json.dump(loss_callback.losses, f)
