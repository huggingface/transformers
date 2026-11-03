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

"""
Worker script: with gradient accumulation, the gradients of a model sharded at load time by `DistributedConfig` are
only reduced on the last micro-batch. FSDP2 only sets the sharded `.grad` of a parameter once it reduced it.

Launched via ``torchrun`` from ``test_trainer_distributed_transformers.py``.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset

from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import get_distributed_backend
from transformers.testing_utils import torch_device


class GradientReductionCallback(TrainerCallback):
    def __init__(self):
        self.reduced = []

    def on_substep_end(self, args, state, control, model=None, **kwargs):
        self.reduced.append(any(p.grad is not None for p in model.parameters()))

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        self.reduced.append(all(p.grad is not None for p in model.parameters()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dp_replicate_size", type=int, default=1)
    script_args = parser.parse_args()

    if torch_device != "cpu":
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=get_distributed_backend(torch_device), timeout=timedelta(seconds=60))
    try:
        distributed_config = DistributedConfig(
            dp_replicate_size=script_args.dp_replicate_size,
            fsdp_size=dist.get_world_size() // script_args.dp_replicate_size,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "trl-internal-testing/tiny-Qwen3ForCausalLM", distributed_config=distributed_config
        )
        input_ids = [[i + 10] * 8 for i in range(32)]
        args = TrainingArguments(
            output_dir=script_args.output_dir,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            save_strategy="no",
            report_to="none",
        )
        callback = GradientReductionCallback()
        dataset = Dataset.from_dict({"input_ids": input_ids, "labels": input_ids})
        Trainer(model=model, args=args, train_dataset=dataset, callbacks=[callback]).train()

        # 3 micro-batches without reduction, then the reduced gradients of the optimizer step, twice
        expected = [False, False, False, True] * 2
        assert callback.reduced == expected, f"rank {dist.get_rank()}: {callback.reduced} != {expected}"
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
