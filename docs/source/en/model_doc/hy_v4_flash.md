<!--Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was contributed to Hugging Face Transformers on 2026-11-24.*

# HYV4Flash

## Overview

HYV4Flash is a long-context member of the [HYV4](./hy_v4) family. It keeps the mixture-of-experts text
backbone and the vision encoder of [HYV4VL](./hy_v4_vl), and trades HYV4's uniformly global attention for
three mechanisms aimed at long sequences:

- **Mixed attention.** Three quarters of the layers attend to a 512-token sliding window with their own,
  wider head shape; the rest keep a global view. The two kinds of layers rotate with different RoPE bases,
  a short one for the local layers and a long one for the global ones.
- **A compressed indexer.** Each global layer pools every four tokens into one compressed key and scores
  queries against those, so a query recalls whole groups of tokens instead of ranking the context one token
  at a time. Attention then runs over the tokens the recalled groups stand for.
- **Engram memories.** Four layers carry a hashed n-gram memory: the bigram and trigram ending at each
  position index a learned embedding table, and the result is gated into the hidden streams before
  attention. The memories read a normalized vocabulary that ships with the checkpoint, so tokens differing
  only by case, accent or whitespace share an entry.

The routed experts run in a narrower latent space than the hidden states, which keeps the expert weights
small enough to hold 512 of them per layer.

## Usage

The examples use the released model ID `tencent/Hy4-flash`. The full model requires multiple GPUs. The
examples shard the dense layers with tensor parallelism and distribute the experts with expert parallelism,
which is more efficient for this mixture-of-experts model than tensor parallelism alone.

Save the example as `generate.py` and launch it with one process per GPU. For a single-node allocation,
set `NUM_GPUS` to the number of GPUs to use and run:

```bash
torchrun --standalone --nproc-per-node="$NUM_GPUS" generate.py
```

### Image inputs

```python
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, DistributedConfig

model_id = "tencent/Hy4-flash"
processor = AutoProcessor.from_pretrained(model_id)

distributed_config = DistributedConfig(
    tp_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the dense layers are sharded
    ep_size=int(os.environ["WORLD_SIZE"]),  # Number of devices across which the experts are distributed
)
model = AutoModelForImageTextToText.from_pretrained(model_id, distributed_config=distributed_config)

messages = [{
    "role": "user",
    "content": [
        {
            "type": "image",
            "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        },
        {"type": "text", "text": "Describe this image in one sentence."},
    ],
}]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    processor_kwargs={"images_kwargs": {"device": model.device}},
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

if torch.distributed.get_rank() == 0:
    completion = outputs[0, inputs["input_ids"].shape[1]:]

    print(processor.decode(completion, skip_special_tokens=True))
```

For text-only generation, use a message containing only a `"text"` item. Video inputs work the same way
and follow [HYV4VL](./hy_v4_vl#video-inputs), whose image and video processors this model reuses.

## Usage Tips

- The Engram memories read `input_ids`, so they only contribute where the model is given tokens. Visual
  spans carry projector output rather than tokens and get no n-gram, and neither does padding.
- Each Engram embedding table is ~32 GB in bf16. Under `device_map="auto"`, tables that do not fit on a
  GPU are left on CPU via `_no_placement_params`, and only the looked-up rows are transferred to the
  active device during forward.
- The compressed indexer only prunes once the context outgrows the recall budget (`index_topk` tokens,
  grouped into `index_topk // index_kpool` pools of `index_kpool` tokens). Below that every token is
  recalled anyway and the layers attend densely, so short prompts cost nothing extra.
- Sliding layers carry their own head count and head dimensions via `per_layer_config`. A checkpoint whose
  `layer_types` lists a sliding layer must carry weights shaped for those dimensions.

## HYV4FlashConfig

[[autodoc]] HYV4FlashConfig

## HYV4FlashTextConfig

[[autodoc]] HYV4FlashTextConfig

## HYV4FlashProcessor

[[autodoc]] HYV4FlashProcessor
    - __call__

## HYV4FlashTextModel

[[autodoc]] HYV4FlashTextModel
    - forward

## HYV4FlashModel

[[autodoc]] HYV4FlashModel
    - forward
    - get_image_features
    - get_video_features

## HYV4FlashForConditionalGeneration

[[autodoc]] HYV4FlashForConditionalGeneration
    - forward
