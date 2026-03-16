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

N-D parallelism combines multiple parallelism strategies across a mesh of N devices. GPUs are organized into a mesh where each axis corresponds to a parallelism dimension (data, tensor, sequence, etc.). Each GPU in the mesh handles its slice of the work along each dimension.

The 3D parallelism diagram below (DP x TP x SP) uses 8 GPUs. Two DP groups process different mini-batches. Within each group, each model replica is split across 2 GPUs at the tensor level, and each sequence is split across 2 GPUs. GPUs exchange partial results with collectives like all-reduce or all-gather.

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

## Choosing a strategy

No single strategy solves every scaling problem. Stack them to address multiple bottlenecks at once.

- Start with DP (FSDP or ZeRO) if a single layer fits on one GPU. Start with ZeRO-1 (least communication overhead) and move to ZeRO-2 or ZeRO-3 if you run out of memory. Add offloading if a model still doesn't fit in memory.
- If a single layer doesn't fit on one GPU, add TP within a node to shrink per-GPU layer size. Use DP across the remaining GPUs.
- If sequences are too long to fit in memory, add SP.

Generally, TP should be used *within* a node to utilize fast interconnect and DP should be used *across* nodes because it tolerates a slower network.

The examples below show some of the ways you can compose the strategies.

<hfoptions id="parallelism-combo">
<hfoption id="TP + SP (large layers and long sequences)">

TP splits each layer across GPUs within a node while SP splits the sequence across GPUs. Use this combination when a model's layers are too large for a single GPU *and* sequences are too long to fit in memory. Don't use it if layers already fit on one GPU because the additional collective communication from both TP and SP adds overhead.

```py
from accelerate.utils import DeepSpeedSequenceParallelConfig, ParallelismConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_id = "MiniMaxAI/MiniMax-M2"
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

</hfoption>
<hfoption id="TP + FSDP (large layers across nodes)">

TP splits layers across GPUs within a node and FSDP `full_shard` shards parameters, gradients, and optimizer states across the remaining GPUs (equivalent to ZeRO-3). Use this combination when a single layer doesn't fit on one GPU and you need to scale across multiple nodes. Don't use it if layers fit on one GPU because FSDP alone has lower communication overhead and is simpler to configure.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_id = "MiniMaxAI/MiniMax-M2"
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

</hfoption>
<hfoption id="DP + SP (long sequences)">

DP replicates the model across GPUs to process different mini-batches and SP splits long sequences. Use this combination when each layer fits on a single GPU but sequences are too long. TP is not required. Don't use it if sequences fit in memory with standard DP because SP adds communication overhead for sequence chunk exchanges.

```py
from accelerate.utils import DeepSpeedSequenceParallelConfig, ParallelismConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

model_id = "MiniMaxAI/MiniMax-M2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
)

# 8 GPUs: dp_replicate=2 * dp_shard=1 * sp=4 = 8
parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,
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

</hfoption>
</hfoptions>

## Next steps

- Read the [5D Parallelism in a Nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell) chapter from The Ultra-Scale Playbook for more details about how the different parallelism strategies interact with each other.