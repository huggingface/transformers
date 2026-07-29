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

# Choosing a strategy

Distributed training splits work across accelerators, and each strategy targets a different bottleneck. Pick the one that matches what's actually running out, then stack more only when a single strategy isn't enough.

| Blocker | Strategy | Guide |
|---|---|---|
| Model fits on one GPU, but training is slow | Data parallelism (DDP) | [DDP](./ddp) |
| Model, gradients, or optimizer states don't fit on one GPU | FSDP2 or DeepSpeed ZeRO | [FSDP2](./fsdp), [DeepSpeed ZeRO](./deepspeed) |
| A single layer doesn't fit on one GPU | Tensor parallelism | [Tensor parallelism](./tensor_parallelism) |
| Sequences are too long to fit in memory | Sequence parallelism | [Ulysses sequence parallelism](./deepspeed_alst) |
| A mixture-of-experts model is too large | Expert parallelism | [Expert parallelism](./expert_parallelism) |
| One strategy isn't enough on its own | Stack several | [N-D parallelism](./perf_train_gpu_many) |

1. Start with data parallelism if a single layer fits on one GPU. With DeepSpeed, begin at ZeRO-1 for the least communication overhead and move to ZeRO-2 or ZeRO-3 as you run out of memory. Add offloading if the model still doesn't fit.
2. If a single layer doesn't fit on one GPU, add tensor parallelism within a node to shrink the per-GPU layer size, and use data parallelism across the remaining GPUs.
3. If sequences are too long to fit in memory, add sequence parallelism.

Keep tensor parallelism *within* a node so it uses the fast interconnect, and data parallelism *across* nodes because it tolerates a slower network.

## Two ways to shard

Transformers has two entry points for sharding.

| | [`Trainer`] and Accelerate | [`~distributed.DistributedConfig`] |
|---|---|---|
| When | Training through [`Trainer`] | Custom training loop, or inference |
| Where you configure it | [`TrainingArguments`] or an [Accelerate config file](./accelerate#accelerate-config-file) | [`~PreTrainedModel.from_pretrained`] |
| When sharding happens | After the model loads | At load time, so the full model never lands on one device |
| Strategies | DDP, FSDP2, DeepSpeed ZeRO, TP, sequence parallelism | FSDP2, TP, expert parallelism |

See [Accelerate](./accelerate) for the [`Trainer`] path and [DistributedConfig](./distributed_config) for the load-time path.

## Next steps

- See [Accelerator selection](./accelerator_selection) to control which devices PyTorch uses before you start.
- See [N-D parallelism](./perf_train_gpu_many) for combining strategies once one isn't enough.
- See [Debugging](./debugging) for diagnosing communication failures and distributed error states.
- Read [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) for a deeper look at how these strategies work, especially the [5D Parallelism in a Nutshell](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=5d_parallelism_in_a_nutshell) chapter.
