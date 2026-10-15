<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Pipeline parallelism for inference

Pipeline parallel inference splits a model by layer across GPUs so you can call [`~GenerationMixin.generate`] on a model that doesn't fit on one device. Each *rank* owns a contiguous slice of decoder layers. A rank is the distributed process ID (`RANK`, from `0` to `WORLD_SIZE - 1`), not a GPU; with one process per GPU they line up one to one. Hidden states flow to the next rank. The last rank broadcasts logits so every rank returns the same sequences.

## Run pipeline parallel generate

Pass [`DistributedConfig`] with `pp_size` to [`~PreTrainedModel.from_pretrained`], then call [`~GenerationMixin.generate`] on every rank. Omit `tp_size` for a 1D pipeline. In that case `pp_size` must equal `WORLD_SIZE`. The example below shards [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) across four GPUs.

```py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    distributed_config=DistributedConfig(pp_size=4),
)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

if int(os.environ["RANK"]) == 0:
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

Launch one process per GPU.

```bash
torchrun --nproc-per-node 4 generate_pp.py
```

> [!WARNING]
> A mesh whose `pp_size * tp_size * fsdp_size` does not match the process-group world size raises `RuntimeError`.

Don't pass `device_map` on this 1D path. Each rank is placed on its local GPU, and a passed `device_map` is overwritten rather than rejected.

## What each rank owns

Layers are split evenly by index, and leftover layers go to the last rank. Rank 0 always owns the token embeddings. The last rank owns the final norm and `lm_head`. When the model ties `lm_head` to the embedding table, the last rank also keeps the embeddings so `lm_head` can share those weights locally.

[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) has 24 layers and tied embeddings. With `pp_size=4` the stages look like this.

```text
tokens --> +------+   +------+   +------+   +------+ --> logits --> broadcast
           | PP 0 |-->| PP 1 |-->| PP 2 |-->| PP 3 |
           |emb+  |   |L6-11 |   |L12-17|   |L18-23|
           |L0-5  |   |      |   |      |   |+head |
           +------+   +------+   +------+   +------+
```

A 32-layer model on three ranks is uneven. Ranks 0 and 1 get 10 layers each, and rank 2 gets 12.

The load path in `core_model_loading.py` emits loading logs you can use to check that every GPU received the right weights. Set Transformers logging to `INFO` to print a `LOAD REPORT` of which checkpoint keys this rank owns and which it skipped.

```py
from transformers.utils import logging

logging.set_verbosity_info()
```

## Combine with tensor parallelism

Pass both `pp_size` and `tp_size` on [`DistributedConfig`]. The distributed mixin applies pipeline parallelism first, then tensor parallelism. `WORLD_SIZE` must equal `pp_size * tp_size`.

Layers still split evenly by index across PP stages. Within a stage, tensor parallelism shards weights across the TP group. With `pp_size=2` and `tp_size=2` the mesh looks like this.

```text
              PP 0 (L0-11)              PP 1 (L12-23)
         +--------+  +--------+    +--------+  +--------+
         |  TP 0  |--|  TP 1  | -> |  TP 0  |--|  TP 1  |
         | shard  |  | shard  |    | shard  |  | shard  |
         +--------+  +--------+    +--------+  +--------+
```

Do not pass `device_map` with PP+TP. If `tp_size > 1`, Transformers raises `ValueError`. On the 1D PP-only path, a passed `device_map` is still overwritten to the local device.

The recipe below uses four processes with `pp_size=2` and `tp_size=2` on [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). That model has 14 attention heads and 2 KV heads, so `tp_size` must divide both (every sharded dim must be divisible by `tp_size`).

```py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(pp_size=2, tp_size=2),
)
model.eval()

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)

if int(os.environ["RANK"]) == 0:
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

```bash
torchrun --nproc-per-node=4 generate_pp_tp.py
```

## Limits

This path is 1D when only `pp_size` is set. Then `pp_size` must equal `WORLD_SIZE`. It is 2D when both `pp_size` and `tp_size` are set. Then `pp_size * tp_size` must equal `WORLD_SIZE`.

- The model must have a `layers` stack plus `embed_tokens`, `norm`, and `lm_head`.
- There is no custom `pp_plan`. Layers split evenly by index.
- There is no backward pass. Only one pipeline stage computes at a time (with TP, the `tp_size` ranks in that stage run together).
- Combining FSDP with pipeline parallelism raises `ValueError`.

## Next steps

If you need a different split, take a look at these guides.

- To slice each layer across GPUs instead of stacking whole layers, see [tensor parallelism for inference](./perf_infer_gpu_multi).
- For training-oriented pipeline parallelism, see [parallelism methods](./perf_train_gpu_many#pipeline-parallelism).
