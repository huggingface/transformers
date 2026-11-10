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

# Context parallelism

Context parallelism (CP) splits every sequence across the ranks of a cp group: each rank keeps one contiguous slice of it. Activation memory per GPU shrinks with `cp_size`, so longer sequences fit on the same GPUs. Attention still needs the whole sequence, which the attention style of the plan gives it back.

```text
 full sequence (8 tokens)                cp_size = 2
┌───┬───┬───┬───┬───┬───┬───┬───┐        ┌───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │  ───►  │ 0 │ 1 │ 2 │ 3 │  rank 0
└───┴───┴───┴───┴───┴───┴───┴───┘        ├───┼───┼───┼───┤
                                          │ 4 │ 5 │ 6 │ 7 │  rank 1
                                          └───┴───┴───┴───┘
```

## Usage

Set `cp_size` to the number of ranks that split a sequence. Every rank of a cp group receives the same full batch, and the model splits it. The `world_size // cp_size` groups run different batches.

```py
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

CP_SIZE = 4
rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

distributed_config = DistributedConfig(fsdp_size=world_size // CP_SIZE, cp_size=CP_SIZE)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    distributed_config=distributed_config,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# the ranks of a cp group take the same rows, so the sampler has one replica per group
sampler = DistributedSampler(dataset, num_replicas=world_size // CP_SIZE, rank=rank // CP_SIZE)
loader = DataLoader(dataset, batch_size=1, sampler=sampler)

for batch in loader:
    batch = {name: value.to(model.device) for name, value in batch.items()}
    loss = model(**batch).loss  # the loss of the full sequences, the same on every rank of the group
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```shell
torchrun --nproc-per-node 8 train_cp.py
```

- `loss` is the loss of the full sequences. The `logits` are only this rank's slice.
- A sequence whose length `cp_size` does not divide is padded at its end. The padding gets labels `-100`, and no real token attends to it.
- [FSDP2](./fsdp) shards the parameters over `fsdp_size * cp_size` ranks.
- Training only: `past_key_values` raise, so `generate` does not work. Requires `torch>=2.10`.

## The context parallel plan

`PreTrainedConfig.base_model_cp_plan` holds the default plan, and `DistributedConfig(cp_plan=...)` overrides its rules, like `tp_plan` does. The default plan pairs two kinds of styles: one on the decoder layers, one on their attention modules.

| style | module | what it does |
| :--- | :--- | :--- |
| `"cp_split_sequence"` | the decoder layers (`layers.*`) | run on this rank's slice: the first layer cuts `hidden_states`, and every layer cuts `position_embeddings` |
| `"cp_ulysses"` | their attention modules (`layers.*.self_attn`) | run the attention function over the whole sequence, with Ulysses |

The model builds the embeddings, the rotary tables and the attention mask on the whole sequence, before the first decoder layer cuts it. So every mask the model supports works unchanged: packing with `position_ids`, a padding `attention_mask`, sliding windows and chunked attention.

A model whose layers have other names declares its own `base_model_cp_plan`. A model that context parallelism cannot run sets it to `None`.

## Attention styles

The attention style says how the attention modules see the whole sequence. `DistributedConfig(cp_plan={"model.layers.*.self_attn": ...})` picks another one.

| style | how it works | limits |
| :--- | :--- | :--- |
| `"cp_ulysses"` (default) | an all-to-all gives each rank the whole sequence for `num_heads / cp_size` heads, and a second one goes back | `cp_size * tp_size` must divide the attention heads and the key/value heads; `eager` and `flex_attention` are refused |

To add one, subclass `ContextParallelAttention` and add it to `ALL_CP_STYLES` under its name. `check` raises a `ValueError` for a module it cannot run. `attention` runs `attention_function` on this rank's slice of `query`, `key` and `value`, with the mask the model built for the whole sequence, and returns this rank's output and the attention weights.

```py
from transformers.distributed import DistributedConfig
from transformers.distributed.context_parallel import ALL_CP_STYLES, ContextParallelAttention


class MyAttention(ContextParallelAttention):
    def check(self, module, cp_mesh):
        ...

    def attention(self, cp_mesh, attention_function, module, query, key, value, attention_mask, **kwargs):
        ...


ALL_CP_STYLES["cp_mine"] = MyAttention()
distributed_config = DistributedConfig(cp_size=4, cp_plan={"model.layers.*.self_attn": "cp_mine"})
```

## Trainer

[`Trainer`] trains a model loaded with `DistributedConfig(cp_size=N)` like a model on a single device: its dataloader gives the same batch to every rank of a cp group, and the loss and the gradients match a single-device run with the same total batch size. Evaluation only supports the loss (`prediction_loss_only=True`), because a rank only holds the logits of its slice.

## Combining with the other parallelisms

- [Tensor parallelism](./tensor_parallelism): each rank runs Ulysses on its `num_heads / tp_size` heads. The ranks of a tp group hold the same batch and the same slice.
- [Expert parallelism](./expert_parallelism) with token dispatch (`"ep_dispatch_experts"`): each rank dispatches the tokens of its own slice. `ep_size` must divide `fsdp_size * cp_size * tp_size`.
- Pipeline parallelism is not supported.

## What is not supported

- Attention that does not go through the [attention interface](./attention_interface) (`_supports_attention_backend = False`).
- Layers that mix tokens outside softmax attention: linear attention, recurrent layers, convolutions, sparse indexers.
- Heads other than a causal LM.
- Layers that take another full-sequence input than `hidden_states` and `position_embeddings`.

Ring attention is not implemented. The `"sequence_parallel"` rule of a tensor parallel plan is a different thing: it splits the activations of the normalization layers between the TP ranks.
