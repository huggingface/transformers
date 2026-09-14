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

Enable expert parallelism with the [`DistributedConfig`] class and the `ep_size` argument. For all-reduce inference, set `ep_size=tp_size` so every rank in an expert group receives the same tokens.

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

This configuration uses the `ep_plan` (expert parallel plan) defined in each MoE model's config file. Attention sharding depends on that plan; expert parallelism does not automatically apply the model's tensor parallel plan. The [`GroupedGemmParallel`] class splits expert weights so each device loads only its local experts. The `ep_router` routes tokens to experts and an all-reduce operation combines their outputs.

Launch your inference script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html) and specify how many devices to use. The number of devices must evenly divide the total number of experts.

```zsh
torchrun --nproc-per-node 8 your_script.py
```

## Token dispatch

With all-reduce, every expert parallel rank runs the whole batch, keeps only the experts it owns, and all-reduces expert outputs after every MoE layer. Use `tp_size=1` and set `ep_size` independently to send each token to the rank that owns its experts. Each rank then trains on its own batch shard. The default `experts_dispatch="auto"` selects all-to-all for this topology.

```py
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(
    tp_size=1,
    fsdp_size=8,
    ep_size=4,
)
```

Each rank trains on its own part of the batch. At every MoE layer it routes its tokens, sends each (token, expert) pair to the rank that owns the expert with an all-to-all, runs its local experts, and gets the results back with a second all-to-all. Only the routed tokens travel.

With `tp_size=1`, `ep_size` must divide `fsdp_size` and the number of experts. When only `ep_size` is supplied, `fsdp_size` defaults to `WORLD_SIZE`.

For the rest of the model:

- The parameters outside the experts are sharded with [FSDP2](./fsdp) across the `fsdp` mesh, and FSDP2 reduces their gradients.
- Experts are sharded across `ep` and additionally across `efsdp`, whose size is `fsdp_size // ep_size`. With `efsdp_size=1` they are outside FSDP2, so `fsdp_mixed_precision` and `fsdp_cpu_offload` do not apply to them.
- The [`Trainer`] uses ordinary data-parallel batching for training and evaluation and counts tokens across all ranks.

The legacy `enable_expert_parallel=True` spelling is deprecated. A dispatch configuration with `tp_size=4, fsdp_size=2, enable_expert_parallel=True` is translated to `tp_size=1, fsdp_size=8, ep_size=4`, preserving its expert groups and independent batches per rank. Legacy all-reduce configurations retain their original `tp_size` and `fsdp_size`.

## Token dispatch with trunk tensor parallelism

Set `tp_size > 1` with explicit `ep_size` and `experts_dispatch="all-to-all"` to apply the model's tensor parallel plan to the trunk and its expert parallel plan to the routed experts. For example, on eight processes:

```py
distributed_config = DistributedConfig(
    tp_size=2,
    fsdp_size=4,
    ep_size=4,
    experts_dispatch="all-to-all",
)
```

Each pair of TP ranks receives the same batch. Attention and other dense modules are sharded according to the TP plan. At each MoE layer, TP ranks dispatch disjoint slices of the tokens, then combine their results into a replicated output. EP groups span four ranks, and each expert is additionally FSDP-sharded across `efsdp_size = fsdp_size * tp_size // ep_size = 2` ranks. The trunk's FSDP group spans four ranks.

`ep_size` must be a multiple of `tp_size`, divide `fsdp_size * tp_size`, and divide the number of experts. Token slices may be uneven or empty, including during single-token decoding. The model's usual TP constraints, such as attention-head divisibility, still apply.

The [`Trainer`] shares batches within each TP group and counts each group's tokens once. The effective global batch size is `per_device_train_batch_size * fsdp_size * gradient_accumulation_steps`. Sequence parallelism is not required for this path. With `experts_dispatch="auto"`, `ep_size=tp_size` still selects legacy all-reduce; request `"all-to-all"` explicitly to use trunk TP in that case.

## Combining with FSDP2

Without token dispatch, expert-only plans leave attention, embeddings, and norms replicated on every expert-parallel rank. Add [FSDP2](./fsdp) with `fsdp_size`, and set `ep_size=tp_size` for all-reduce.

```py
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(
    tp_size=4,  # expert parallel size
    fsdp_size=2,  # data parallel shards
    ep_size=4,
)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", distributed_config=distributed_config)
```

Both dispatchers use one mesh builder that prepares a dense view `(pp, fsdp, tp)` and an expert view `(pp, efsdp, ep)`, retaining size-one axes. Mesh sizes depend on the parallelism degrees, not the dispatcher. In these examples `pp_size=1`, so `tp_size * fsdp_size` and `ep_size * efsdp_size` both equal the number of processes. Token dispatch uses the expert view for expert ownership and FSDP sharding. Legacy all-reduce keeps using the dense view: its expert parallel plan shards the experts across `tp` (the same ranks as `ep`), then FSDP2 shards every parameter, experts included, across `fsdp` and owns their gradient reduction. Each `fsdp` rank trains on its own part of the batch. The all-to-all examples above show how to choose EP independently, with or without trunk TP.

Load the model as usual, then train with [`Trainer`]. It takes the gradient norm across both meshes and gives each mesh its own optimizer param group. [`~Trainer.save_model`] gathers sharded weights into a regular checkpoint. This requires `accelerate>=1.12` so the `Trainer` can mirror `tp_size` and `fsdp_size` into [`~Accelerate.ParallelismConfig`].

The table below compares EP-only training with 2D EP+FSDP2 on 8xH100 GPUs. The workload is full fine-tuning of Qwen3-30B-A3B in bf16 at sequence length 2048. More FSDP shards cut peak memory, and tokens/s drop some because FSDP2 all-gathers and reduce-scatters the experts across `fsdp`.

| configuration | tokens/s/GPU | peak memory/GPU |
|---|---|---|
| `tp_size=8` | 3485 | 38.6 GB |
| `tp_size=4, fsdp_size=2` | 2900 | 34.2 GB |
| `tp_size=2, fsdp_size=4` | 2830 | 32.3 GB |

> [!WARNING]
> Resuming from a checkpoint is not supported yet for models sharded at load time, so the [`Trainer`] only accepts `save_only_model=True` or `save_strategy="no"` for them.

[[autodoc]] DistributedConfig
