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

[`DistributedConfig`] shards a model across GPUs directly through [`~PreTrainedModel.from_pretrained`]. It supports [tensor parallelism](./tensor_parallelism), [FSDP2](./fsdp), and [expert parallelism](./expert_parallelism).

Use this for a custom training loop or inference, where you shard the model at load time instead of through [`Trainer`]. If you're training with [`Trainer`], configure FSDP2 through [Accelerate](./accelerate) instead.

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

The product of `tp_size` and `fsdp_size` must equal the number of devices you launch with. Set one of them at a time. Setting both above 1 raises a `ValueError` because combining FSDP2 and tensor parallelism in a single mesh isn't supported yet. To stack parallelism strategies today, train with [`Trainer`] and see [N-D parallelism](./perf_train_gpu_many).

[`DistributedConfig`] is mutually exclusive with `device_map`. `device_map` places whole modules on specific GPUs, while a distributed config shards those same parameters across GPUs.

## Tensor parallelism

[Tensor parallelism](./tensor_parallelism) splits weight matrices across GPUs. Set `tp_size` to shard the model's supported layers.

```py
import torch
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(tp_size=4)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    distributed_config=distributed_config,
)
```

Transformers shards according to the model's `base_model_tp_plan`. Set `tp_plan` to a dict to override the layout, and read the resolved plan back from `model.tp_plan` after loading.

```py
distributed_config = DistributedConfig(
    tp_size=4,
    tp_plan={"model.layers.*.self_attn.q_proj": "colwise"},
)
```

See [Tensor parallelism for inference](./perf_infer_gpu_multi) for the full set of partitioning strategies.

## FSDP2

[FSDP2](./fsdp) shards parameters, gradients, and optimizer states across GPUs. Set `fsdp_size` to the number of devices to shard across. Requires torch>=2.7, which is what the distributed checkpoint save and load paths need.

```py
import torch
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(fsdp_size=4)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    distributed_config=distributed_config,
)
```

Transformers wraps each module according to the model's FSDP plan. The plan maps module names to a sharding strategy, where `free_full_weight` reshards a module after the forward pass to save memory and `keep_full_weight` keeps it gathered to avoid a second all-gather during the backward pass. Every config inherits the default base model plan below, and a model overrides it with `base_model_fsdp_plan`.

```py
{
    "embed_tokens": "free_full_weight",
    "layers.*": "free_full_weight",
    "norm": "keep_full_weight",
}
```

Task-specific classes add their own entries on top, such as `{"lm_head": "keep_full_weight"}` for causal language models. Read the merged plan from the model after loading to see what was actually applied.

```py
print(model.fsdp_plan)
```

A model without an FSDP plan raises a `ValueError` at load time rather than silently sharding nothing.

Set `fsdp_mixed_precision=True` to compute in `bfloat16` while reducing gradients in `float32`, and set `fsdp_cpu_offload=True` to move parameters and gradients to CPU when they aren't in use.

```py
distributed_config = DistributedConfig(
    fsdp_size=4,
    fsdp_mixed_precision=True,
    fsdp_cpu_offload=True,
)
```

## Save a sharded model

[`~PreTrainedModel.save_pretrained`] writes a single Hugging Face checkpoint by default. Each rank sends its shards to rank 0, which gathers the full weights on CPU and writes the safetensors files. Call it from every rank so the non-writing ranks wait at the barrier instead of racing ahead.

```py
model.save_pretrained("./checkpoint")
```

Gathering a large model onto one CPU is slow and can run out of host memory. Set `distributed_checkpoint=True` to take the [distributed checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) path instead, where every rank writes its own shard in parallel and a consolidation pass merges them into standard `model-*-of-N.safetensors` files.

```py
model.save_pretrained("./checkpoint", distributed_checkpoint=True)
```

The result is an ordinary checkpoint directory that is reloaded with [`~PreTrainedModel.from_pretrained`]. This path only works for FSDP2-sharded models loaded with a [`DistributedConfig`], and it requires torch>=2.7.

To resume training, save and load the optimizer state alongside the model with `save_optimizer_distributed` and `load_optimizer_distributed`.

```py
from transformers.distributed import load_optimizer_distributed, save_optimizer_distributed

save_optimizer_distributed(model, optimizer, "./checkpoint/optimizer")
load_optimizer_distributed(model, optimizer, "./checkpoint/optimizer")
```

## Launch

Launch your script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and set `--nproc-per-node` to the number of devices you configured.

```shell
torchrun --nproc-per-node 4 train.py
```

## Next steps

- See [Distributed](./main_classes/distributed) for the [`DistributedConfig`] API reference.
- See [Tensor parallelism](./tensor_parallelism) for how weight sharding works and how to combine it with [`Trainer`].
- See [FSDP2](./fsdp) for sharded training through [`Trainer`] and Accelerate.
- See [Expert parallelism](./expert_parallelism) for sharding mixture-of-experts models.
- See [N-D parallelism](./perf_train_gpu_many) for stacking parallelism strategies.
- Read [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) for a deeper look at how these strategies work.
