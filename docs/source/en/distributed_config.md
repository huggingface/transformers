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

# DistributedConfig

[`DistributedConfig`] shards a model across GPUs directly through [`~PreTrainedModel.from_pretrained`]. It supports [FSDP2](./fsdp), [tensor parallelism](./tensor_parallelism), and [N-D parallelism](./perf_train_gpu_many).

Pass a [`DistributedConfig`] to [`~PreTrainedModel.from_pretrained`] and Transformers builds the device mesh and shards the supported layers for you.

The fields below control how the model is sharded.

| field | description |
|---|---|
| `tp_size` | Number of devices for tensor parallelism. Defaults to 1 when only `fsdp_size` is set. |
| `tp_plan` | Tensor parallel sharding plan. Leave as `None` to use the model's default plan. |
| `fsdp_size` | Number of devices for FSDP2. Defaults to 1 when only `tp_size` is set. |
| `fsdp_cpu_offload` | Offload parameters and gradients to CPU to save GPU memory. Defaults to `False`. |
| `fsdp_mixed_precision` | Compute in `bfloat16` and reduce gradients in `float32`. Defaults to `False`. |
| `enable_expert_parallel` | Shard mixture-of-experts layers across devices. See [Expert parallelism](./expert_parallelism). |

The product of `tp_size` and `fsdp_size` must equal the number of devices you launch with.

## FSDP2

[FSDP2](./fsdp) shards parameters, gradients, and optimizer states across GPUs. Set `fsdp_size` to the number of devices to shard across.

```py
import torch
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(fsdp_size=4)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    distributed_config=distributed_config,
)
```

Transformers wraps each layer according to the model's `base_model_fsdp_plan`. Check whether a model declares one before sharding.

```py
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
print(config.base_model_fsdp_plan)
```

The plan maps modules to a sharding strategy. `free_full_weight` reshards a module after the forward pass to save memory, and `keep_full_weight` keeps it gathered to avoid a second all-gather during the backward pass.

```py
{
    "embed_tokens": "free_full_weight",
    "layers.*": "free_full_weight",
    "norm": "keep_full_weight",
}
```

Set `fsdp_mixed_precision=True` to compute in `bfloat16` while reducing gradients in `float32`, and set `fsdp_cpu_offload=True` to move parameters and gradients to CPU when they aren't in use.

```py
distributed_config = DistributedConfig(
    fsdp_size=4,
    fsdp_mixed_precision=True,
    fsdp_cpu_offload=True,
)
```

## Tensor parallelism

[Tensor parallelism](./tensor_parallelism) splits weight matrices across GPUs. Set `tp_size` to shard the model's supported layers.

```py
import torch
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(tp_size=4)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    distributed_config=distributed_config,
)
```

Transformers shards according to the model's `base_model_tp_plan`. Pass `tp_plan` to override the layout, for example `{"model.layers.*.self_attn.q_proj": "colwise"}`.

## N-D parallelism

Combine FSDP2 and tensor parallelism by setting both sizes. The example below runs on 4 GPUs, sharding each tensor-parallel group of 2 GPUs with FSDP2 across the remaining 2.

```py
import torch
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

distributed_config = DistributedConfig(tp_size=2, fsdp_size=2)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    distributed_config=distributed_config,
)
```

## Launch

Launch your script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and set `--nproc-per-node` to the total number of devices, equal to `tp_size * fsdp_size`.

```shell
torchrun --nproc-per-node 4 train.py
```

## Next steps

- See [FSDP2](./fsdp) for sharded training.
- See [Tensor parallelism](./tensor_parallelism) for more details on partitioning strategies and manual plans.
- See [Expert parallelism](./expert_parallelism) for sharding mixture-of-experts models.
- See [N-D parallelism](./perf_train_gpu_many) for combining parallelism strategies.
- Read [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) for a deeper look at how these strategies work.
