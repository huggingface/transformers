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
Worker script: training after an evaluation updates every parameter of a model sharded at load time by
`DistributedConfig`. The evaluation leaves the final norm and the LM head, sharded with `reshard_after_forward=False`,
holding their unsharded parameters.

Launched via ``torchrun`` from ``test_trainer_distributed_transformers.py``.
"""

import argparse
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import get_distributed_backend
from transformers.testing_utils import torch_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    script_args = parser.parse_args()

    dist.init_process_group(backend=get_distributed_backend(torch_device), timeout=timedelta(seconds=60))
    model = AutoModelForCausalLM.from_pretrained(
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        distributed_config=DistributedConfig(fsdp_size=dist.get_world_size()),
    )
    input_ids = [list(range(i + 10, i + 18)) for i in range(32)]
    dataset = Dataset.from_dict({"input_ids": input_ids, "labels": input_ids})
    args = TrainingArguments(
        output_dir=script_args.output_dir,
        max_steps=2,
        per_device_train_batch_size=2,
        learning_rate=0.1,
        save_strategy="no",
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset, eval_dataset=dataset)
    previous_params = {name: param.full_tensor().clone() for name, param in trainer.model.named_parameters()}

    trainer.evaluate()
    trainer.train()

    for name, param in trainer.model.named_parameters():
        assert not torch.equal(param.full_tensor(), previous_params[name]), f"{name} has not changed"
    trainer.end()


if __name__ == "__main__":
    main()
