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
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import get_distributed_backend
from transformers.testing_utils import torch_device
from transformers.utils import ADAPTER_CONFIG_NAME, ADAPTER_SAFE_WEIGHTS_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="trl-internal-testing/tiny-Qwen3ForCausalLM")
    parser.add_argument("--bias", default="none")
    parser.add_argument("--parallelism", choices=["fsdp", "tp"], required=True)
    script_args = parser.parse_args()
    output_dir = script_args.output_dir

    if torch_device != "cpu":
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=get_distributed_backend(torch_device), timeout=timedelta(seconds=60))
    try:
        save_and_check(script_args, output_dir)
    finally:
        dist.destroy_process_group()


def save_and_check(script_args, output_dir):
    # Load the sharded model
    distributed_config = DistributedConfig(**{f"{script_args.parallelism}_size": dist.get_world_size()})
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model,
        distributed_config=distributed_config,
    )
    peft_config = LoraConfig(
        r=4,
        target_modules=["q_proj", "v_proj"],
        bias=script_args.bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Create a dummy dataset and train for one step to have a trained adapter
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
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    assert trainer.is_distributed_loading_by_transformers
    trainer.train()
    checkpoint_dir = os.path.join(output_dir, "checkpoint-1")

    # PEFT selects the checkpoint entries and gathers sharded weights collectively on every rank.
    live = {name: param.detach().cpu() for name, param in get_peft_model_state_dict(model).items()}
    dist.barrier()
    if dist.get_rank() == 0:
        assert os.path.isfile(os.path.join(checkpoint_dir, ADAPTER_CONFIG_NAME))

        # Check that the checkpoint contains exactly the expected keys.
        saved = load_file(os.path.join(checkpoint_dir, ADAPTER_SAFE_WEIGHTS_NAME))
        assert saved.keys() == live.keys(), f"saved {sorted(saved)} != trained {sorted(live)}"

        # Check that loading the checkpoint reconstructs the same adapter weights.
        reloaded = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(script_args.model), checkpoint_dir)
        torch.testing.assert_close(get_peft_model_state_dict(reloaded), live, rtol=0, atol=0)
    dist.barrier()


if __name__ == "__main__":
    main()
