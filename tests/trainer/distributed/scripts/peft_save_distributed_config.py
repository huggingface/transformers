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
Worker script: `Trainer` saves the LoRA adapter of a model sharded at load time by `DistributedConfig`, and the saved
adapter is the trained one.

Launched via ``torchrun`` from ``test_trainer_distributed_transformers.py``.
"""

import argparse
import os

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.distributed import DistributedConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="trl-internal-testing/tiny-Qwen3ForCausalLM")
    parser.add_argument("--bias", default="none")
    script_args = parser.parse_args()
    output_dir = script_args.output_dir

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model,
        distributed_config=DistributedConfig(fsdp_size=int(os.environ["WORLD_SIZE"])),
    )
    peft_config = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], bias=script_args.bias)
    model = get_peft_model(model, peft_config)
    input_ids = [[1, 2, 3, 4, 5, 6, 7, 8]] * 4
    dataset = Dataset.from_dict({"input_ids": input_ids, "labels": input_ids})
    args = TrainingArguments(
        output_dir=output_dir,
        max_steps=1,
        save_steps=1,
        save_only_model=True,
        per_device_train_batch_size=1,
        report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=dataset).train()

    # Gathering a sharded parameter (e.g. a trained bias of the base model) is collective: every rank does it
    live = {
        name.replace(".default", ""): (param.full_tensor() if isinstance(param, DTensor) else param).detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if dist.get_rank() == 0:
        saved = load_file(os.path.join(output_dir, "checkpoint-1", "adapter_model.safetensors"))
        assert saved.keys() == live.keys(), f"saved {sorted(saved)} != trained {sorted(live)}"
        for name, param in live.items():
            torch.testing.assert_close(saved[name], param, rtol=0, atol=0)


if __name__ == "__main__":
    main()
