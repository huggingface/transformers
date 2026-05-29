<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Expert parallelism

[Expert parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=expert_parallelism) is a parallelism strategy for [mixture-of-experts (MoE) models](https://huggingface.co/blog/moe). Each expert's feedforward layer lives on a different hardware accelerator. A router dispatches tokens to the appropriate experts and gathers the results. This approach scales models to far larger parameter counts without increasing computation cost because each token activates only a few experts.

## DistributedConfig

> [!WARNING]
> The [`DistributedConfig`] API is experimental and its usage may change in the future.

Enable expert parallelism with the [`DistributedConfig`] class and the `enable_expert_parallel` argument.

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(enable_expert_parallel=True)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    dtype="auto",
    distributed_config=distributed_config,
)
```

> [!TIP]
> Expert parallelism automatically enables [tensor parallelism](./perf_infer_gpu_multi) for attention layers.

This argument switches to the `ep_plan` (expert parallel plan) defined in each MoE model's config file. The plan is declared at parameter granularity: the `grouped_gemm` style shards each expert weight (`gate_up_proj`, `down_proj`) along the expert dimension so every device loads only its local experts, the `ep_router` style routes tokens to those local experts, and `moe_experts_allreduce` combines the per-device outputs with an all-reduce. The same decomposition is used for tensor parallelism (`moe_tp_gate_up_colwise` / `moe_tp_down_rowwise` shard the expert weights within each expert instead of across experts), so weight sharding always lives in the config at parameter level while the experts module only declares forward communication.

Launch your inference script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and specify how many devices to use. The number of devices must evenly divide the total number of experts.

```zsh
torchrun --nproc-per-node 8 your_script.py
```
