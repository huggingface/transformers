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

"""
Minimalist SFT example with LoRA and Tensor Parallelism. Works on any device that supports TP
(Trainium, CUDA, ...). The companion shell script sets the Neuron-specific environment variables.

# LoRA + TP=2 (Neuron)
```
bash examples/scripts/sft_neuron.sh
```

# Or launch directly on any device
```
torchrun --nproc_per_node=2 examples/scripts/sft_neuron.py \
    --model_name_or_path Qwen/Qwen3.5-9B \
    --dataset_name trl-lib/Capybara \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen3.5-9B-SFT-LoRA
```
"""

import os

import torch
from datasets import load_dataset
from accelerate import ParallelismConfig
from transformers import AutoModelForCausalLM, AutoConfig

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

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    config.attention_dropout = 0.0
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.dtype,
        **kwargs,
    )

    if int(os.environ.get("COMPILE", "0")):
        model = torch.compile(model, backend="neuron")

    # ---------------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------------
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # ---------------------------------------------------------------------------
    # Trainer
    # ---------------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args)
