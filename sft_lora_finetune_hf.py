# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import os
import socket

import torch
from datasets import load_dataset
from accelerate import ParallelismConfig
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from peft import get_peft_model
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser, get_peft_config


def main(script_args, training_args, model_args):
    # ---------------------------------------------------------------------------
    # Tensor Parallelism
    # ---------------------------------------------------------------------------
    tp_size = int(os.environ.get("TP_SIZE", "1"))

    tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
    }

    kwargs = {}
    if tp_size > 1:
        training_args.parallelism_config = ParallelismConfig(tp_size=tp_size)
        kwargs["tp_plan"] = tp_plan
        kwargs["tp_size"] = tp_size

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=True,
    )
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        revision=model_args.model_revision,
        trust_remote_code=True,
        # attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **kwargs,
    )

    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        model = get_peft_model(model, peft_config)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    if int(os.environ.get("COMPILE", "0")):
        print("Compiling model")
        model = torch.compile(model, backend="neuron")

    model.train()

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    if training_args.max_steps < 0:
        print("max_steps not set, defaulting to 100 for profiling")
        training_args.max_steps = 100
    main(script_args, training_args, model_args)
