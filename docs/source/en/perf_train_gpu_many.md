<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
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

## Composing strategies

Stack strategies when one doesn't solve your bottleneck on its own. Each one you add costs more collective communication, so reach for a second or third only after the first is exhausted. See [Choosing a strategy](./distributed_overview) to work out which one you need first.

The examples below run on 8 GPUs and show three combinations worth knowing. Each is configured through [`Trainer`], which is the path that supports stacking today.

<hfoptions id="parallelism-combo">
<hfoption id="TP + SP (large layers and long sequences)">

TP splits each layer across GPUs within a node while SP splits the sequence across GPUs. Use this combination when a model's layers are too large for a single GPU *and* sequences are too long to fit in memory. Don't use it if layers already fit on one GPU because the additional collective communication from both TP and SP adds overhead.

```py
from accelerate.utils import DeepSpeedSequenceParallelConfig, ParallelismConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.distributed import DistributedConfig

model_id = "MiniMaxAI/MiniMax-M2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(tp_size=2),
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
    processing_class=tokenizer,
    train_dataset=train_dataset,
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
from transformers.distributed import DistributedConfig

model_id = "MiniMaxAI/MiniMax-M2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Trainer auto-detects tp_size from the model.
# 8 GPUs: tp=2 (from model) * dp_shard=4
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(tp_size=2),
    dtype="auto",
)

training_args = TrainingArguments(
    output_dir="./output",
    fsdp=True,
    fsdp_config={"version": 2},
    per_device_train_batch_size=1,
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
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
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
```

```shell
torchrun --nproc-per-node 8 train.py
```

</hfoption>
</hfoptions>

## Next steps

- See [Choosing a strategy](./distributed_overview) if you aren't sure which strategies you need.
- See [Ulysses sequence parallelism](./deepspeed_alst) for the sequence parallel fields used above.
- See [Debugging](./debugging) for diagnosing mesh and communication errors.
- Read the [5D Parallelism in a Nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell) chapter from The Ultra-Scale Playbook for more details about how the different parallelism strategies interact with each other.