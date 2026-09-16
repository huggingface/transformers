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
Worker script for the expert-parallel `Trainer` tests: train a tiny MoE for a few steps under one layout.

Launched via ``torchrun`` from ``test_trainer_distributed_expert_parallel.py``. Every layout consumes the same
global batch per step, so the logged losses and gradient norms and the saved weights must match the single-process
``reference`` run.
"""

import argparse
import json
import os

import torch
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.distributed import DistributedConfig


GLOBAL_BATCH_SIZE = 8
SEQ_LEN = 16
NUM_STEPS = 4

# `world_size=4` layouts. `dp` is the number of distinct batches per step; each rank trains on
# `GLOBAL_BATCH_SIZE // dp` samples. The masked plan is Qwen3 MoE's dispatch plan with the two forward rules
# overridden; dispatch uses the model's default plan.
MASKED_PLAN = {"model.layers.*.mlp.gate": "ep_router", "model.layers.*.mlp.experts": "moe_tp_experts"}
LAYOUTS = {
    "reference": {"world_size": 1, "dp": 1, "config": None},
    "masked": {
        "world_size": 4,
        "dp": 2,
        "config": {"tp_size": 2, "fsdp_size": 2, "ep_size": 2, "ep_plan": MASKED_PLAN},
    },
    "dispatch": {"world_size": 4, "dp": 4, "config": {"tp_size": 1, "fsdp_size": 4, "ep_size": 2}},
    "dispatch_tp": {"world_size": 4, "dp": 2, "config": {"tp_size": 2, "fsdp_size": 2, "ep_size": 4}},
}


class TokenDataset(Dataset):
    def __init__(self, vocab_size: int):
        generator = torch.Generator().manual_seed(0)
        self.input_ids = torch.randint(0, vocab_size, (NUM_STEPS * GLOBAL_BATCH_SIZE, SEQ_LEN), generator=generator)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "labels": self.input_ids[index].clone()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=LAYOUTS, required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    layout = LAYOUTS[args.layout]
    assert int(os.environ.get("WORLD_SIZE", 1)) == layout["world_size"], args.layout

    distributed_config = DistributedConfig(**layout["config"]) if layout["config"] else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, distributed_config=distributed_config
    )
    if distributed_config is None:
        model = model.to("cuda")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "trainer"),
        per_device_train_batch_size=GLOBAL_BATCH_SIZE // layout["dp"],
        max_steps=NUM_STEPS,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=0,
        data_seed=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        average_tokens_across_devices=True,
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=TokenDataset(model.config.vocab_size))
    trainer.train()
    # Gathers the sharded weights; every rank takes part, the main process writes.
    trainer.save_model(os.path.join(args.output_dir, "model"))

    if trainer.is_world_process_zero():
        steps = [log for log in trainer.state.log_history if "loss" in log]
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump({"loss": [s["loss"] for s in steps], "grad_norm": [s["grad_norm"] for s in steps]}, f)


if __name__ == "__main__":
    main()
