<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# N-D parallelism

N-D parallelism combines multiple parallelism strategies across a mesh of N devices. For each dimension (data, tensor, sequence, etc.), a GPU is assigned a coordinate in the mesh and is responsible for its portion of the job. Stacking parallelism strategies addresses scaling problems that no single strategy solves.

The 3D parallelism diagram below (DP x TP x SP) assumes there are 8 GPUs. This means 2 DP groups process different mini-batches, each model replica is split across 2 GPUs at the tensor level, and each sequence is split across 2 GPUs. GPUs exchange partial results with collectives like all-reduce or all-gather.

```text
         ◄──────────── data parallel (DP=2) ─────────────►
                 same weights · different data

              replica 0             replica 1
           ┌─────────┬─────────┐  ┌─────────┬─────────┐
seq[0:S/2] │ ▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓ │  │ ▓▓▓▓▓▓▓ │ ▓▓▓▓▓▓▓ │ ─┐
           │  GPU 0  │  GPU 1  │  │  GPU 4  │  GPU 5  │  │
           ├─────────┼─────────┤  ├─────────┼─────────┤  │ SP=2
seq[S/2:S] │ ░░░░░░░ │ ░░░░░░░ │  │ ░░░░░░░ │ ░░░░░░░ │  │
           │  GPU 2  │  GPU 3  │  │  GPU 6  │  GPU 7  │ ─┘
           └─────────┴─────────┘  └─────────┴─────────┘
                └─ TP=2 ─┘              └─ TP=2 ─┘

```

## TP + SP

```py
from accelerate.utils import DeepSpeedSequenceParallelConfig, ParallelismConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto",
    dtype="auto",
)

# Trainer auto-detects tp_size from the model.
# 8 GPUs: tp=2 (from model) * sp=2 * dp_replicate=2 = 8
parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=2,
    dp_replicate_size=2,
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length_is_variable=True,
        sp_attn_implementation="flash_attention_2",
    ),
)

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="ds_config.json",
    parallelism_config=parallelism_config,
    per_device_train_batch_size=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
)
trainer.train()
```

```shell
torchrun --nproc-per-node 8 train.py
```

## TP + DP (ZeRO 3)

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    tp_plan="auto",
    dtype="auto",
)

training_args = TrainingArguments(
    output_dir="./output",
    fsdp="full_shard auto_wrap",
    fsdp_config={"version": 2},
    per_device_train_batch_size=1,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset,
)
trainer.train()
```

```shell
torchrun --nproc-per-node 8 train.py
```

## Next steps

- Read the [5D Parallelism in a Nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell) chapter from The Ultra-Scale Playbook for more details about how the different parallelism strategies interact with each other.