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

Pass [`DistributedConfig`] with `pp_size` equal to `WORLD_SIZE` to [`~PreTrainedModel.from_pretrained`], then call [`~GenerationMixin.generate`] on every rank. The example below shards [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) across four GPUs.

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

Launch one process per GPU. `pp_size` must equal `WORLD_SIZE`.

```bash
torchrun --nproc-per-node 4 generate_pp.py
```

> [!WARNING]
> A `pp_size` that doesn't match the process-group world size raises `RuntimeError`. Pipeline parallel loading requires `torch>=2.5`, and older installs raises `OSError`.

Don't pass `device_map` because each rank is placed on its local GPU. Passing `device_map` is overwritten, not rejected.

## What each rank owns

Layers are split evenly by index, and leftover layers go to the last rank. Rank 0 always owns the token embeddings. The last rank owns the final norm and `lm_head`. When the model ties `lm_head` to the embedding table, the last rank also keeps the embeddings so `lm_head` can share those weights locally.

[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) has 24 layers and tied embeddings. With `pp_size=4` the split looks like this. Hidden states move forward along the pipeline, and the last rank broadcasts logits to every rank before the next token.

```text
+----------------------------------------------------------+
|  tokens                                                  |
|    |                                                     |
|  rank 0  (GPU 0)   embed + layers 0–5                    |
|    | hidden                                              |
|  rank 1  (GPU 1)   layers 6–11                           |
|    | hidden                                              |
|  rank 2  (GPU 2)   layers 12–17                          |
|    | hidden                                              |
|  rank 3  (GPU 3)   layers 18–23 + norm + lm_head         |
|    | logits                                              |
|    +-- broadcast to every rank, then the next token      |
+----------------------------------------------------------+
```

A 32-layer model on three ranks is uneven. Ranks 0 and 1 get 10 layers each, and rank 2 gets 12.

Set Transformers logging to `INFO` to print a `LOAD REPORT` of which checkpoint keys this rank owns and which it skipped.

```py
from transformers.utils import logging

logging.set_verbosity_info()
```

## Limits

This path is a 1D sequential split of a decoder-only model, not a 2D mesh. `pp_size` must equal `WORLD_SIZE`, so every process is a pipeline stage.

- It does not support other model layouts. The model must expose a `layers` stack plus `embed_tokens`, `norm`, and `lm_head`.
- It has no custom pipeline plans. Layers split evenly by index.
- Don't set `pp_size` with `tp_size` or `fsdp_size`. [`~PreTrainedModel.from_pretrained`] applies tensor parallelism, then FSDP2, then pipeline parallelism, and stops at the first match, so a pairwise mix never runs this path. [`DistributedConfig`] raises only when all three are greater than 1.
- There is no backward pass. Only one rank computes at a time.

## Next steps

If you need a different split, take a look at these guides.

- To slice each layer across GPUs instead of stacking whole layers, see [tensor parallelism for inference](./perf_infer_gpu_multi).
- For training-oriented pipeline parallelism, see [parallelism methods](./perf_train_gpu_many#pipeline-parallelism).
