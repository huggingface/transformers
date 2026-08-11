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

"""Dumps distributed environment info to a JSON file for verification.

This script creates a Trainer (which initializes the accelerator) and writes
each worker's env vars, TrainingArguments fields, and accelerator state to
``<output_dir>/env_rank<N>.json``.

Accepts all TrainingArguments flags (e.g. ``--deepspeed``, ``--fsdp``) so the
Trainer sets up the correct framework regardless of launcher.

Works with any launcher (torchrun, accelerate launch with DDP/FSDP/DeepSpeed).
"""

import json
import os

from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments


def main():
    parser = HfArgumentParser((TrainingArguments,))
    (args,) = parser.parse_args_into_dataclasses()
    args.disable_tqdm = True

    model_name = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    model = AutoModelForCausalLM.from_pretrained(model_name)

    trainer = Trainer(model=model, args=args)
    accelerator = trainer.accelerator

    env_info = {
        # Raw env vars set by torchrun / accelerate
        "env_world_size": os.environ.get("WORLD_SIZE"),
        "env_rank": os.environ.get("RANK"),
        "env_local_rank": os.environ.get("LOCAL_RANK"),
        "env_master_addr": os.environ.get("MASTER_ADDR"),
        "env_master_port": os.environ.get("MASTER_PORT"),
        # TrainingArguments-derived values
        "args_local_rank": args.local_rank,
        "args_world_size": args.world_size,
        "args_process_index": args.process_index,
        "args_local_process_index": args.local_process_index,
        "args_parallel_mode": str(args.parallel_mode),
        "args_n_gpu": args.n_gpu,
        # Accelerator state
        "accelerator_num_processes": accelerator.num_processes,
        "accelerator_process_index": accelerator.process_index,
        "accelerator_local_process_index": accelerator.local_process_index,
        "accelerator_is_main_process": accelerator.is_main_process,
        "accelerator_is_local_main_process": accelerator.is_local_main_process,
        "accelerator_use_distributed": accelerator.use_distributed,
        "accelerator_distributed_type": str(accelerator.distributed_type),
        "accelerator_device": str(accelerator.device),
        # Trainer-level flags (these gate framework-specific code paths)
        "trainer_is_fsdp_enabled": trainer.is_fsdp_enabled,
        "trainer_is_deepspeed_enabled": trainer.is_deepspeed_enabled,
    }

    # FSDP plugin info
    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if fsdp_plugin is not None:
        env_info["fsdp_version"] = getattr(fsdp_plugin, "fsdp_version", None)
        env_info["fsdp_sharding_strategy"] = str(getattr(fsdp_plugin, "sharding_strategy", None))
        env_info["fsdp_cpu_offload"] = str(getattr(fsdp_plugin, "cpu_offload", None))
        env_info["fsdp_auto_wrap_policy"] = str(getattr(fsdp_plugin, "auto_wrap_policy", None))

    # DeepSpeed plugin info
    deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if deepspeed_plugin is not None:
        env_info["deepspeed_zero_stage"] = deepspeed_plugin.zero_stage
        env_info["deepspeed_offload_optimizer_device"] = str(deepspeed_plugin.offload_optimizer_device)
        env_info["deepspeed_offload_param_device"] = str(deepspeed_plugin.offload_param_device)

    output_file = os.path.join(args.output_dir, f"env_rank{args.process_index}.json")
    with open(output_file, "w") as f:
        json.dump(env_info, f)


if __name__ == "__main__":
    main()
