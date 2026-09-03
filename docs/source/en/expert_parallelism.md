<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Expert parallelism

[Expert parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism) is a parallelism strategy for [mixture-of-experts (MoE) models](https://huggingface.co/blog/moe). Each expert's feedforward layer lives on a different hardware accelerator. A router dispatches tokens to the appropriate experts and gathers the results. This approach scales models to far larger parameter counts without increasing computation cost because each token activates only a few experts.

## DistributedConfig

Enable expert parallelism with the [`DistributedConfig`] class and the `enable_expert_parallel` argument.

```py
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    enable_expert_parallel=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    distributed_config=distributed_config,
)
```

> [!TIP]
> Expert parallelism automatically enables [tensor parallelism](./perf_infer_gpu_multi) for attention layers.

This argument switches to the `ep_plan` (expert parallel plan) defined in each MoE model's config file. The [`GroupedGemmParallel`] class splits expert weights so each device loads only its local experts. The `ep_router` routes tokens to experts and an all-reduce operation combines their outputs.

Launch your inference script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and specify how many devices to use. The number of devices must evenly divide the total number of experts.

```zsh
torchrun --nproc-per-node 8 your_script.py
```

## Combining with FSDP2

Expert parallelism only shards the experts. Everything else (attention, embeddings, norms) and its optimizer state is replicated on every expert-parallel rank, which is what limits the model size you can train. Set `fsdp_size` together with `tp_size` to add [FSDP2](./fsdp) on a second mesh dimension.

```py
distributed_config = DistributedConfig(
    tp_size=4,  # expert parallel size
    fsdp_size=2,  # data parallel shards
    enable_expert_parallel=True,
)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", distributed_config=distributed_config)
```

The model is loaded on a 2-D `(fsdp, tp)` device mesh, and `tp_size * fsdp_size` must equal the number of processes. The expert parallel plan shards the experts across `tp`, then FSDP2 shards every parameter, experts included, across `fsdp` and owns their gradient reduction. Each `fsdp` rank trains on its own part of the batch. Nothing else changes: train with the [`Trainer`] as usual (it computes the gradient norm across the two meshes and steps the optimizer per parameter), and `save_model` gathers the sharded weights and writes a regular checkpoint.

On 8 GPUs, full fine-tuning of Qwen3-30B-A3B in bf16 at sequence length 2048:

| configuration | tokens/s/GPU | peak memory/GPU |
|---|---|---|
| `tp_size=8` | 3485 | 38.6 GB |
| `tp_size=4, fsdp_size=2` | 2900 | 34.2 GB |
| `tp_size=2, fsdp_size=4` | 2830 | 32.3 GB |

> [!WARNING]
> Resuming from a checkpoint is not supported yet for models sharded at load time, so the [`Trainer`] only accepts `save_only_model=True` or `save_strategy="no"` for them.

[[autodoc]] DistributedConfig
