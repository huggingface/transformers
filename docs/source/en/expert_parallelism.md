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

Enable expert parallelism with the [`DistributedConfig`] class and the `ep_size` argument. The current all-reduce implementation requires `ep_size=tp_size`, so every rank in an expert group receives the same tokens.

```py
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),
    ep_size=int(os.environ["WORLD_SIZE"]),
)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    distributed_config=distributed_config,
)
```

Each MoE model defines two plans in its config: `base_model_tp_plan` for the dense modules and `base_model_ep_plan` for the experts. They are exposed on the loaded model as `model.tp_plan` and `model.ep_plan`. With `tp_size > 1` and `ep_size > 1`, both apply: the [tensor parallel](./perf_infer_gpu_multi) plan shards attention and the dense MLPs, and the expert parallel plan shards the experts. EP rules take precedence over TP rules for the same modules, so expert weights are sharded once, by the EP plan. In the EP plan, the [`GroupedGemmParallel`] style splits the expert weights along the expert dimension so each rank loads only its local experts, and `ep_router` masks the experts that live on other ranks before an all-reduce combines the expert outputs.

`tp_plan` is applied only when `tp_size > 1`, and `ep_plan` only when `ep_size > 1`. With TP enabled and EP disabled, the full TP plan applies, expert rules included.

> [!TIP]
> `enable_expert_parallel=True` is a deprecated alias for `ep_size=tp_size`, used only when `ep_size` is omitted, and emits a `FutureWarning`.

Launch your inference script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and specify how many devices to use. The number of devices must evenly divide the total number of experts.

```zsh
torchrun --nproc-per-node 8 your_script.py
```

### Overriding the plans

Pass `tp_plan={...}` or `ep_plan={...}` to [`DistributedConfig`] to override individual rules of the predefined plans. Unspecified rules are kept, and the merged plans are stored on the model. Each key must match a module, a parameter, or an existing plan entry; otherwise loading raises a `ValueError` before anything is sharded. Use the full path as seen from the loaded model, so `model.layers.*` for a causal LM and `layers.*` for its base model.

```py
distributed_config = DistributedConfig(
    tp_size=4,
    ep_size=4,
    tp_plan={"model.layers.*.self_attn.q_proj": "colwise_rep"},
    ep_plan={"model.layers.*.mlp.experts.down_proj": "grouped_gemm"},
)
```

Providing a plan does not infer parallel sizes: set `tp_size` and `ep_size` explicitly.

## Combining with FSDP2

Tensor and expert parallelism shard the weights across `tp`, but the optimizer state and the modules without a rule are still replicated on every rank of the group, which limits how large a model you can train. Add [FSDP2](./fsdp) on a second mesh dimension with `fsdp_size`, and keep `ep_size=tp_size` for the expert parallel width.

```py
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(
    tp_size=4,
    ep_size=4,  # expert parallel size, must match tp_size
    fsdp_size=2,  # data parallel shards
)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", distributed_config=distributed_config)
```

Load the model as usual, then train with [`Trainer`]. It takes the gradient norm across both meshes and gives each mesh its own optimizer param group. [`~Trainer.save_model`] gathers sharded weights into a regular checkpoint. This requires `accelerate>=1.12` so the `Trainer` can mirror `tp_size` and `fsdp_size` into [`~Accelerate.ParallelismConfig`].

The table below compares EP-only training with 2D EP+FSDP2 on 8xH100 GPUs. The workload is full fine-tuning of Qwen3-30B-A3B in bf16 at sequence length 2048. More FSDP shards cut peak memory, and tokens/s drop some because FSDP2 all-gathers and reduce-scatters the experts across `fsdp`.

| configuration | tokens/s/GPU | peak memory/GPU |
|---|---|---|
| `tp_size=8` | 3485 | 38.6 GB |
| `tp_size=4, fsdp_size=2` | 2900 | 34.2 GB |
| `tp_size=2, fsdp_size=4` | 2830 | 32.3 GB |

> [!WARNING]
> Resuming from a checkpoint is not supported yet for models sharded at load time, so the [`Trainer`] only accepts `save_only_model=True` or `save_strategy="no"` for them.

## API reference

[[autodoc]] DistributedConfig
