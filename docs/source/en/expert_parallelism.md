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

Enable expert parallelism with the [`DistributedConfig`] class and the `ep_size` argument. Most models route with masking and all-reduce, which requires `ep_size=tp_size` so every rank in an expert group receives the same tokens. Models whose plan uses [token dispatch](#token-dispatch), such as Qwen3 MoE, can set `ep_size` independently of `tp_size`.

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

The expert forward rule in `ep_plan` selects how tokens reach the experts:

| rule | mechanism | layout |
| :--- | :--- | :--- |
| `"moe_tp_experts"` with `"ep_router"` on the router | masking and all-reduce: every rank runs its local experts on the whole batch, the router masks the others, and an all-reduce combines the outputs | `ep_size=tp_size` |
| `"ep_dispatch_experts"` | [token dispatch](#token-dispatch): each rank keeps its own tokens and only exchanges the routed (token, expert) pairs with two all-to-all collectives | `ep_size` a multiple of `tp_size` that divides `fsdp_size * tp_size` |

> [!TIP]
> `enable_expert_parallel=True` is a deprecated alias for `ep_size=tp_size`, used only when `ep_size` is omitted, and emits a `FutureWarning`.

Launch your inference script with [torchrun](https://pytorch.org/docs/stable/elastic/run.html). The number of processes must equal `tp_size * fsdp_size * pp_size`, and `ep_size` must evenly divide the number of experts.

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

Qwen3 MoE defaults to `"ep_dispatch_experts"`. To use masking and all-reduce instead, set `ep_size=tp_size` and override both the router and the expert forward rules:

```py
distributed_config = DistributedConfig(
    tp_size=4,
    ep_size=4,
    ep_plan={
        "model.layers.*.mlp.gate": "ep_router",
        "model.layers.*.mlp.experts": "moe_tp_experts",
    },
)
```

Conversely, override the expert forward rule of a model whose plan uses masking with `"ep_dispatch_experts"` to use token dispatch. The router rule is then ignored, since dispatch needs the global expert ids to find each expert's owner.

## Token dispatch

With token dispatch, each rank trains on its own part of the batch. At every MoE layer, a rank routes its tokens, sends each (token, expert) pair to the rank that owns the expert with an all-to-all, runs its local experts on what it receives, gets the results back with a second all-to-all and combines them with the routing weights. Only the routed activations and expert outputs travel, and no rank computes experts for tokens it does not own.

```py
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(
    tp_size=1,
    fsdp_size=8,
    ep_size=4,
)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", distributed_config=distributed_config)
```

With `tp_size=1`, `ep_size` must divide `fsdp_size` and the number of experts, and attention is not bound by `num_key_value_heads`. For the rest of the model:

- The parameters outside the experts are sharded with [FSDP2](./fsdp) across `fsdp`, which reduces their gradients.
- The experts are sharded across `ep` and, when `efsdp_size = fsdp_size * tp_size // ep_size` is larger than one, additionally FSDP-sharded across `efsdp`. They are always FSDP-wrapped, so `fsdp_mixed_precision` and `fsdp_cpu_offload` apply to them too and [`~PreTrainedModel.save_pretrained`] gathers them like any other parameter.
- An expert parallel group holds `ep_size / tp_size` batches, so an expert's gradient is a sum over that many batches. The `efsdp` reduction divides by `fsdp_size` instead of its group size, which gives the same per-batch average FSDP2 takes for the dense modules.
- Every local expert also processes one zero pad row per layer. A rank whose experts received no tokens still joins the reverse all-to-all and the expert gradient reduction.

### How the sizes combine

- `tp_size * fsdp_size` is the number of processes. `ep_size` adds none: it regroups the same ranks for the expert weights only.
- `ep_size` cuts the expert list into `ep_size` blocks. Each rank computes `num_experts / ep_size` experts, and `ep_size` consecutive ranks hold one complete set. That set of ranks is the group the all-to-all runs in.
- `efsdp_size = fsdp_size * tp_size / ep_size` is how many complete copies of the expert set exist. Ranks at the same position in different copies shard those experts for memory and average their gradients, like FSDP does for the dense modules.
- The batch a rank holds depends on `fsdp` only. Consecutive ranks form a TP group and get the same batch; the `fsdp_size` groups get different batches.

Eight processes, `tp_size=2, fsdp_size=4`, eight experts:

```text
rank         0      1       2      3       4      5       6      7
batch       [====B0====]   [====B1====]   [====B2====]   [====B3====]     one batch per TP pair
tp           0      1       0      1       0      1       0      1

ep_size=2   E0-3   E4-7    E0-3   E4-7    E0-3   E4-7    E0-3   E4-7     group = a TP pair,   efsdp_size=4
ep_size=4   E0E1   E2E3    E4E5   E6E7    E0E1   E2E3    E4E5   E6E7     group = two pairs,   efsdp_size=2
ep_size=8   E0     E1      E2     E3      E4     E5      E6     E7       group = all ranks,   efsdp_size=1
```

Two numbers follow from the picture:

- Inside an EP group, each token exists `tp_size` times, once per rank of the pair that holds its batch. This does not depend on `ep_size`.
- An EP group holds `ep_size / tp_size` different batches. This is the count an expert's gradient sums over.

### With tensor parallelism

Set `tp_size > 1` to shard the dense modules with the TP plan while the experts use dispatch. On eight processes:

```py
distributed_config = DistributedConfig(
    tp_size=2,
    fsdp_size=4,
    ep_size=4,
)
```

Each pair of TP ranks receives the same batch, because tensor parallelism replicates the activations inside the pair. If both ranks dispatched all of their tokens, the rank owning an expert would receive every token twice, compute it twice, and its weight gradient would double. Experts are whole on one rank, so the duplicate cannot be split by weights, and the owner is usually another rank, so it cannot be resolved by ownership as masking does. The pair therefore splits the rows: each TP rank dispatches a disjoint `1 / tp_size` of the tokens, results come back to the rank that sent them, and an all-reduce over the pair of the zero-padded halves restores the replicated output the next layer expects. The split is by `tp_size`, not `ep_size`, since only the ranks that hold a batch can send it. Expert groups span four ranks and each expert is FSDP-sharded across `efsdp_size = 2` ranks, while the trunk's FSDP group spans four ranks. The model's usual TP constraints, such as attention-head divisibility, still apply to the dense modules. Token slices may be uneven or empty, including during single-token decoding.

Token dispatch cannot be combined yet with pipeline parallelism yet; use `pp_size=1` (not tested yet)
These configurations each use eight GPUs:

| Configuration | Result |
| :--- | :--- |
| `DistributedConfig(tp_size=4, fsdp_size=2, ep_size=4)` | Dispatch with TP groups of four, each slicing its batch in four (Qwen3 MoE default plan); unless you specify a ep_plan to use the legacy masked EP |
| `DistributedConfig(tp_size=1, fsdp_size=8, ep_size=4)` | Dispatch with an independent batch on each rank, no slicing, and experts FSDP-sharded across pairs of ranks. |
| `DistributedConfig(tp_size=2, fsdp_size=4, ep_size=4)` | Dispatch with a TP pair per batch, each pair slicing its batch in two; two batches per expert group. |
| `DistributedConfig(tp_size=8, ep_size=8)` | Dispatch with every rank sharing one batch, sliced in eight, or masking and all-reduce for a masked plan. |

The [`Trainer`] trains these layouts as loaded: it leaves the placement and gradient reduction to the model's own FSDP2 and expert-parallel wrappers instead of wrapping it again, gives each device mesh its own optimizer param group and gradient-norm term, and counts tokens once per TP group. The effective global batch size is `per_device_train_batch_size * fsdp_size * gradient_accumulation_steps`, whichever the layout. [`~Trainer.save_model`] gathers the sharded weights into a regular checkpoint.

## Combining with FSDP2

Tensor and expert parallelism shard the weights across `tp`, but the optimizer state and the modules without a rule are still replicated on every rank of the group, which limits how large a model you can train. Add [FSDP2](./fsdp) on a second mesh dimension with `fsdp_size`. With masking and all-reduce, keep `ep_size=tp_size` and pass `ep_plan={"layers.*.mlp.gate": "ep_router", "layers.*.mlp.experts": "moe_tp_experts"}``.

```py
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

distributed_config = DistributedConfig(
    tp_size=4,
    ep_size=4,  # expert parallel size, must match tp_size with masking and all-reduce
    fsdp_size=2,  # data parallel shards
    ep_plan={
        "model.layers.*.mlp.gate": "ep_router",
        "model.layers.*.mlp.experts": "moe_tp_experts",
    },
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
