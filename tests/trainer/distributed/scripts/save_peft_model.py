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
"""Save a tensor-parallel PEFT model through Trainer and verify its adapter checkpoint."""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.distributed import DistributedConfig
from transformers.distributed.utils import get_distributed_backend
from transformers.testing_utils import torch_device
from transformers.utils import ADAPTER_CONFIG_NAME, ADAPTER_SAFE_WEIGHTS_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if torch_device != "cpu":
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    backend = get_distributed_backend(torch_device)
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=60))
    try:
        distributed_config = DistributedConfig(tp_size=dist.get_world_size())
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            distributed_config=distributed_config,
        )
        lora_config = LoraConfig(r=4, target_modules=["q_proj", "o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)
        adapter_weights = [param for name, param in model.named_parameters() if "lora_" in name]

        assert any(
            isinstance(param, DTensor) and param.to_local().shape != param.shape for param in adapter_weights
        ), "The test must exercise sharded adapter weights"

        with torch.no_grad():
            for param in adapter_weights:
                if isinstance(param, DTensor):
                    param.to_local().normal_()
                else:
                    param.normal_()

        expected = get_peft_model_state_dict(model)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(args.output_dir, save_only_model=True, report_to=[]),
        )
        assert trainer.is_distributed_loading_by_transformers
        trainer.save_model()
        dist.barrier()

        if dist.get_rank() == 0:
            assert os.path.isfile(os.path.join(args.output_dir, ADAPTER_CONFIG_NAME))
            checkpoint = load_file(os.path.join(args.output_dir, ADAPTER_SAFE_WEIGHTS_NAME))
            torch.testing.assert_close(checkpoint, expected, rtol=0, atol=0)
            reloaded = PeftModel.from_pretrained(
                AutoModelForCausalLM.from_pretrained(args.model_name), args.output_dir
            )
            torch.testing.assert_close(get_peft_model_state_dict(reloaded), expected, rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
