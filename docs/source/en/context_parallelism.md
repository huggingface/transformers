<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Context parallelism

[Context parallelism](https://arxiv.org/abs/2309.14509) (CP) shards the **sequence** dimension of every per-token activation across a group of accelerators. Each rank holds the full set of model weights but only `N_local = N_total // cp_world` tokens. Point-wise ops (norms, projections, RoPE, MLP, residual adds) run locally on the shard with no communication. The single global op is attention, which would normally require seeing the whole sequence — Transformers' CP implementation uses **Ulysses-style** attention, which swaps the head axis for the sequence axis via two `all_to_all`s around the SDPA call. The result is mathematically identical to single-GPU attention, and the communication cost scales as `O(B * S * d / cp_world)` per layer.

CP is complementary to tensor parallelism (TP) and expert parallelism (EP):

| What it shards | TP | EP | CP |
|---|---|---|---|
| Weights | yes (column- or row-wise) | yes (per-expert) | no |
| Activations | hidden dim | expert dim | sequence dim |
| Main benefit | larger models per GPU | larger MoE | longer sequences per GPU |

Use CP when the attention scratchpad (`Q @ K^T` is `B * H * N * N`) is what's pushing you out of memory, or when you want to use idle GPUs to roughly halve your step time at long sequences.

## DistributedConfig

> [!WARNING]
> The [`DistributedConfig`] API is experimental and its usage may change in the future.

Enable context parallelism with the `enable_context_parallel` and `cp_world_size` arguments. The CP group is built from the first `cp_world_size` ranks in the default world process group.

```py
import torch
from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig

cfg = DistributedConfig(enable_context_parallel=True, cp_world_size=2)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-20b",
    dtype="bfloat16",
    distributed_config=cfg,
)
```

Launch the script with `torchrun`:

```bash
torchrun --nproc-per-node 2 your_script.py
```

You can compose CP with expert parallelism. With 4 GPUs and `ep_world_size = cp_world_size = 2`, you get a 2-D mesh where every rank holds half of the experts *and* half of the sequence.

```py
cfg = DistributedConfig(
    enable_expert_parallel=True,
    enable_context_parallel=True,
    cp_world_size=2,
)
```

## Post-load API

For more direct control (custom process groups, 2-D meshes, manual launch outside `from_pretrained`), use `apply_context_parallel` directly:

```py
from transformers.integrations.context_parallel import apply_context_parallel

apply_context_parallel(model, cp_world_size=2)
```

This walks the model's `_cp_plan` (a class-level dict that names which attention modules are eligible) and stashes the CP process group on each. The model's `_attn_implementation` is set to `"context_parallel_ulysses"` so the registered Ulysses attention function is dispatched at every attention call.

## Supported models

A model opts in by declaring a `_cp_plan` class attribute, in the same spirit as `_tp_plan`. Currently enabled:

- `GptOssForCausalLM` (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`)
- `Qwen3MoeForCausalLM` (`Qwen/Qwen3-MoE-*`)

Adding CP support to another model is a one-line change:

```py
class MyModelForCausalLM(MyModelPreTrainedModel, GenerationMixin):
    _cp_plan = {"model.layers.*.self_attn": "context_parallel_ulysses"}
```

Sinks (per-head attention sink parameters, used by GPT-OSS) and sliding-window attention are handled automatically: the registered Ulysses function reads `module.sinks` and `module.sliding_window` if present.

## Inputs and outputs

CP expects pre-sharded sequence inputs. Every rank in the CP group must hold a contiguous slice of the input sequence of length `N_total // cp_world`. The simplest way to produce this is to split your `input_ids` and `position_ids` tensors along the sequence axis after the tokenizer call:

```py
import torch.distributed as dist

batch = tokenizer(text, return_tensors="pt", padding="max_length", max_length=N_total)
rank = dist.get_rank()
cp_world = cfg.cp_world_size
n_local = N_total // cp_world
local_batch = {
    k: v[:, rank * n_local : (rank + 1) * n_local] for k, v in batch.items()
}
```

Outputs (logits) are also seq-sharded; gather them with `dist.all_gather` if you need the full sequence on every rank.

## Limitations

- **Strategy**: only Ulysses (head-axis `all_to_all`) is implemented. Ring Attention is a planned follow-up.
- **Dropout**: attention dropout is rejected — independent per-rank drops would desynchronise the softmax. Disable dropout for CP training, or pre-broadcast a shared mask.
- **Explicit attention masks**: only the implicit causal mask (plus optional sliding window) is supported. Custom additive masks would need to be sharded the same way as the seq axis; not in v1.
- **Head divisibility**: `num_attention_heads` and `num_key_value_heads` must both be divisible by `cp_world_size`. For GPT-OSS (64 Q-heads, 8 KV-heads), this allows `cp_world_size ∈ {1, 2, 4, 8}`.

## References

- Jacobs et al., *DeepSpeed-Ulysses: System Optimizations for Training Extreme Long Sequence Transformer Models* — [arXiv:2309.14509](https://arxiv.org/abs/2309.14509)
- Reference implementation + parity tests + benchmarks: [AlexWortega/transformers-cp-research](https://github.com/AlexWortega/transformers-cp-research)
