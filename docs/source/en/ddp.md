<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DDP

[DistributedDataParallel (DDP)](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html) maintains a full copy of a model on each GPU. Each GPU processes a non-overlapping shard of data with a forward and backward pass. Before the optimizer step, an all-reduce averages gradients across all GPUs. The all-reduce runs on the final micro-batch. [`Trainer`] skips the all-reduce on intermediate gradient accumulation steps, keeping all GPUs in sync after every update. Use DDP when your model fits on a single GPU.

```text
                         ┌─────────────────┐
                         │  training data  │
                         └────────┬────────┘
               ┌──────────────────┼──────────────────┐
               │ shard 0          │ shard 1          │ shard 2
               ▼                  ▼                  ▼
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │   model     │    │   model     │    │   model     │
        │  (copy 0)   │    │  (copy 1)   │    │  (copy 2)   │
        │   GPU 0     │    │   GPU 1     │    │   GPU 2     │
        └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
               │ grads            │ grads            │ grads
               └──────────────────┼──────────────────┘
                               all-reduce
                          (average gradients)
               ┌──────────────────┼──────────────────┐
               ▼                  ▼                  ▼
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │  optimizer  │    │  optimizer  │    │  optimizer  │
        │    step     │    │    step     │    │    step     │
        └─────────────┘    └─────────────┘    └─────────────┘
          (identical)        (identical)        (identical)
```

DDP activates automatically when you launch with a multi-process launcher like [Accelerate](./accelerate).

```cli
# 4 GPUs on one machine
accelerate launch --num_processes 4 train.py
```

## Configure DDP

Pass these [`TrainingArguments`] to control DDP behavior.

- [`~TrainingArguments.gradient_accumulation_steps`] determines when to perform the all-reduce. For example, with `gradient_accumulation_steps=4`, the all-reduce runs every 4 backward passes. This is a general [`TrainingArguments`] setting that interacts with DDP.
- [`ddp_find_unused_parameters`] searches the full graph at the *start* of the backward pass for parameters that won't receive a gradient and marks them as ready so they don't block the all-reduce. Don't use with [`~TrainingArguments.gradient_checkpointing`] because gradient checkpointing discards intermediate activations and recomputes them on the fly.
- [`ddp_bucket_cap_mb`] is the bucket size for batching gradients into a single all-reduce during the backward pass. A larger bucket means fewer all-reduce calls and less launch overhead.
- [`ddp_broadcast_buffers`] synchronizes model buffers (such as BatchNorm running statistics) from rank 0 to all other ranks at the start of every forward pass. Disable if your model only uses LayerNorm. Don't use with [`~TrainingArguments.gradient_checkpointing`].
- [`ddp_backend`] sets the communication backend. Use `"nccl"` for NVIDIA GPUs (default and fastest), `"gloo"` for CPU training or debugging, and `"xccl"`, `"hccl"`, or `"cncl"` for other hardware.
- [`~TrainingArguments.ddp_timeout`] sets the time limit for all processes and operations (all-reduce, broadcast) to complete. If a process hangs, like when loading a large model slowly, the timeout raises an error instead of blocking indefinitely.

```py
from transformers import TrainingArguments

args = TrainingArguments(
    ...,
    gradient_accumulation_steps=4,
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25,
    ddp_broadcast_buffers=True,
    ddp_timeout=1800,
)
```

## Next steps

- See [FSDP](./fsdp) for training models too large to fit on a single GPU.
- See [DeepSpeed](./deepspeed) for ZeRO optimization and offloading.
- Read the [Data Parallelism](https://nanotron-ultrascale-playbook.static.hf.space/index.html#data_parallelism) chapter from The Ultra-Scale Playbook for more information about how DDP works.
