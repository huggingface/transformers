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

# Tensor parallelism

Tensor parallelism (TP) splits weight matrices column-wise or row-wise across GPUs. Each GPU holds a shard, computes a partial result, and synchronizes with an all-reduce to produce the full output.

TP relies on frequent cross-GPU communication. It works best on hardware with fast intra-node links such as NVLink.

```text
    ┌─────────────────────────────┐
    │       X  (replicated)       │
    └────┬──────────┬─────────┬───┘
         │          │         │
    ┌────▼───┐ ┌────▼───┐ ┌───▼────┐
    │ ▓▓▓ W₀ │ │ ░░░ W₁ │ │ ███ W₂ │
    │  X@W₀  │ │  X@W₁  │ │  X@W₂  │
    └────┬───┘ └────┬───┘ └───┬────┘
         └──────────┼─────────┘
               Y₀+Y₁+Y₂
    ┌────────────────────────────┐
    │          Y (full)          │
    └────────────────────────────┘
```

Transformers supports TP for architectures whose config defines `base_model_tp_plan`. Check that field first to see whether a model supports native TP.

```py
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
print(config.base_model_tp_plan is not None)
print(config.base_model_tp_plan)
```

If a model supports TP, pass a [`DistributedConfig`] with `tp_size` to [`~PreTrainedModel.from_pretrained`]. Transformers initializes the device mesh and shards the supported layers for you.

> [!WARNING]
> Don't use `device_map` with a [`DistributedConfig`]. The two conflict at the weight-loading level. `device_map` places whole modules on specific GPUs, while TP shards those same parameters across all GPUs.

```py
import torch

from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=4),
)
```

[`Trainer`] reads `tp_size` from the model and creates a [`~accelerate.parallelism_config.ParallelismConfig`] automatically.

Launch training on one node with 4 GPUs.

```shell
torchrun --nproc-per-node 4 train_tp.py
```

## ParallelismConfig

Pass [`~accelerate.parallelism_config.ParallelismConfig`] explicitly when combining TP with other parallelism techniques like [FSDP](./fsdp) under [`Trainer`]. See [N-D parallelism](./perf_train_gpu_many) for the strategies worth stacking.

```py
import torch

from accelerate import ParallelismConfig
from transformers import AutoModelForCausalLM, TrainingArguments
from transformers.distributed import DistributedConfig

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=2),
)

# 8 GPUs: tp=2 (from the model) * dp_shard=4
parallelism_config = ParallelismConfig(tp_size=2, dp_shard_size=4)

args = TrainingArguments(
    ...,
    parallelism_config=parallelism_config,
)
```

`tp_size` must match the model's, otherwise [`Trainer`] overwrites it with the value the weights were actually sharded with.

## Next steps

- See [DistributedConfig](./distributed_config) to shard with tensor parallelism at load time through [`~PreTrainedModel.from_pretrained`], without [`Trainer`].
- Read the [Tensor Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=tensor_parallelism) chapter from The Ultra-Scale Playbook for more details about how it works.
- Read the [tensor parallelism inference guide](./perf_infer_gpu_multi) to learn more about partitioning strategies, manual TP plans, and implementation details.
