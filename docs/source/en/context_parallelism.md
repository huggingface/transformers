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

Context parallelism (CP) splits every sequence across the ranks of a cp group, and each rank keeps one contiguous slice of it. Activation memory per GPU shrinks with `cp_size`, which is what lets long sequences fit on the same GPUs. Attention still has to compare every token with the earlier tokens of the whole sequence, so `cp_implementation` decides how the ranks get the full sequence back for each attention call.

## Usage

Set `cp_size` to the number of ranks that split a sequence, and `fsdp_size` to the number of data parallel groups, which run different batches. Each rank still receives the full sequences of its batch: a forward pre-hook keeps this rank's slice of `input_ids`, `position_ids` and `labels`. The ranks of a cp group must receive the same batch, because together they hold one copy of it. There are `world_size // (cp_size * tp_size)` distinct batches, and the data parallel rank of a process is `rank // (cp_size * tp_size)`: the `cp_size` ranks of a group split every sequence, and the `tp_size` ranks of a [tensor parallel](./tensor_parallelism) group hold the same slice of it.

```py
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig

CP_SIZE = 4
rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

# cp_size ranks split every sequence, fsdp_size groups of them run different batches
distributed_config = DistributedConfig(fsdp_size=world_size // CP_SIZE, cp_size=CP_SIZE)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    distributed_config=distributed_config,
    attn_implementation="flash_attention_2",
    dtype=torch.bfloat16,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# `dataset` yields packed rows of `input_ids`, `labels` and `position_ids`, with `position_ids` restarting at each
# document. The ranks of a cp group take the same rows, so the sampler has one replica per group.
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

- `loss` is the loss of the full sequences, the same on every rank of the group, so the backward pass and the optimizer step are the usual ones. The `logits` of the output are only this rank's slice of the sequence, and the logits of the padding positions at the end of the last rank's slice are not meaningful.
- A sequence length that is not a multiple of `cp_size` is padded at the end. No real token attends to the padding, its labels are `-100`, and its `position_ids` continue those of the last document. `num_items_in_batch` counts the real tokens only.
- `attn_implementation` must be `sdpa` or a flash kernel, as listed in [Choosing an implementation](#choosing-an-implementation).
- A 2D `attention_mask` pads the batch, with `1` for a real token and `0` for padding, as in an unsharded run. The mask of the whole sequence is rebuilt before every attention call, so no token attends to the padding. A materialized 4D mask raises: the ranks hold different slices of the same sequence, so a mask that was already computed cannot be sharded.
- Pack documents into full rows and pass `position_ids` that restart at each document, as in the example. Attention then stops at the document boundaries.
- Sliding-window layers work with both kinds of kernel: `sdpa` gets a mask that stops at the window, and flash kernels apply the `sliding_window` they are given themselves. `chunked_attention` layers need `sdpa`.
- With `sdpa`, the mask of the whole sequence is materialized, which takes `batch_size x seq_len x seq_len` booleans in every layer and on every rank. Prefer a flash kernel for long sequences.
- Requires `torch>=2.10`. Training only: `past_key_values` raise and `use_cache` is turned off, so `generate` is out.
- [FSDP2](./fsdp) shards the parameters over `fsdp_size * cp_size` ranks, because the ranks of a cp group hold parts of the same batch. `DistributedConfig(cp_size=8)` alone also works: with `fsdp_size=1` the parameters are sharded across the cp group.
- [Tensor parallelism](./tensor_parallelism) splits the heads of every attention layer first, so each rank runs Ulysses on `num_heads / tp_size` heads and splits those across the cp group. The ranks of a tp group hold the same batch and the same slice, so they run the same tokens and only their parameters differ. `DistributedConfig(tp_size=2, cp_size=2)` on 4 ranks works when `tp_size * cp_size` divides the attention heads and the key/value heads.
- [Expert parallelism](./expert_parallelism) works with token dispatch (`"ep_dispatch_experts"`, the default of most MoE models): each rank dispatches the tokens of its own slice. `ep_size` must be a multiple of `tp_size` and divide `fsdp_size * cp_size * tp_size`. The all-reduce expert plans (`ep_router` with `moe_tp_experts`, and `megamoe_experts`) still need `ep_size=tp_size`.

## Choosing an implementation

| `cp_implementation` | mechanism | limits |
| :--- | :--- | :--- |
| `"ulysses"` (default) | two all-to-all exchanges around every attention layer: the first gives each rank the whole sequence for `num_heads / (cp_size * tp_size)` heads, the second returns to the sequence slices | `cp_size * tp_size` must divide the attention heads and the key/value heads; attention must be `sdpa`, `flash_attention_2`, `flash_attention_3` or a hub flash kernel, so `eager`, `flex_attention` and paged attention are refused; `chunked_attention` layers need `sdpa` |

Subclass `CPImplementation` and register it in `cp_implementations` to add your own. `query`, `key` and `value` are this rank's slices of the sequence, `self.position_ids` holds the positions of the full sequence for the current forward, `self.padding_mask` holds its 2D padding mask when some tokens are padding, and the returned output must be the slice of the sequence this rank holds.

```py
from transformers.distributed import DistributedConfig
from transformers.distributed.context_parallel import CPImplementation, cp_implementations


class MyCP(CPImplementation):
    def check(self, model):
        ...  # raise a ValueError when `model` cannot run with this implementation

    def attention(self, attention_function, module, query, key, value, attention_mask, **kwargs):
        ...  # attention over the whole sequence, returning this rank's output and the attention weights


cp_implementations["my_cp"] = MyCP
distributed_config = DistributedConfig(cp_size=4, cp_implementation="my_cp")
```

## What is not supported

- Models whose attention does not go through the [attention interface](./attention_interface) (`_supports_attention_backend = False`). Their attention would only see this rank's slice.
- Attention patterns besides causal, packed documents, padding, sliding windows and chunks, such as a custom `and_mask_function` or `or_mask_function`. The mask of the whole sequence is rebuilt from those patterns, so anything else would be silently dropped.
- Models with layers that mix tokens outside softmax attention: linear attention, Mamba and other recurrent layers, convolutions, sparse indexers. Those layers would run on this rank's slice only and drop the rest.
- Model heads other than a causal LM. Their loss would come from the shifted labels of one slice while the logits stay sharded.
- `cp_size` with `pp_size`. `DistributedConfig` refuses the combination.
- [`Trainer`], whose dataloader hands the ranks of a cp group different batches. Write the loop yourself until it is supported.

## Naming

Accelerate's `ParallelismConfig(cp_size=...)` is ring attention with SDPA, while the `sp_size` of Accelerate and DeepSpeed is Ulysses. The `"sequence_parallel"` style of a tensor parallel plan is a different thing: it splits the activations of the normalization and dropout layers along the sequence between the TP ranks, and gathers them back for attention and the MLP. For the DeepSpeed path, see [Ulysses sequence parallelism](./deepspeed_alst).
