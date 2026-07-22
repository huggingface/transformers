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
*This model was contributed to Hugging Face Transformers on 2026-07-22.*

# LocateAnything

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

[LocateAnything](https://huggingface.co/nvidia/LocateAnything-3B) is a vision-language model for visual grounding —
object detection, referring-expression grounding, GUI element grounding, and point localization — that outputs
quantized bounding-box coordinates as structured text. It composes a [MoonViT](https://huggingface.co/moonshotai/MoonViT-SO-400M)
native-resolution vision encoder, an MLP projector, and a [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
language model.

The model is trained with Parallel Box Decoding (PBD), a block-wise multi-token prediction scheme that predicts whole
boxes in parallel. This Transformers integration implements the standard auto-regressive (`generate`) path; the
optional MagiAttention / `la_flash` MTP kernels used by the original repository for faster parallel decoding are not
included.

> [!NOTE]
> The model weights are released by NVIDIA under the [NVIDIA License](https://huggingface.co/nvidia/LocateAnything-3B/blob/main/LICENSE)
> for non-commercial (research and evaluation) use only.

## Usage

```python
import re

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, LocateAnythingForConditionalGeneration

model_id = "nvidia/LocateAnything-3B"
processor = AutoProcessor.from_pretrained(model_id)
model = LocateAnythingForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Locate all the instances that matches the following description: remote."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

generated = model.generate(**inputs, do_sample=False, max_new_tokens=256)
answer = processor.decode(generated[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=False)

# Coordinates are normalized to [0, 1000]; scale them to pixels.
width, height = image.size
boxes = [
    (int(x1) / 1000 * width, int(y1) / 1000 * height, int(x2) / 1000 * width, int(y2) / 1000 * height)
    for x1, y1, x2, y2 in re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)
]
print(boxes) # boxes (pixels): [(41.0, 73.0, 174.7, 117.1), (334.1, 76.8, 371.2, 187.2)]
```

## LocateAnythingConfig

[[autodoc]] LocateAnythingConfig

## LocateAnythingVisionConfig

[[autodoc]] LocateAnythingVisionConfig

## LocateAnythingImageProcessor

[[autodoc]] LocateAnythingImageProcessor
    - preprocess

## LocateAnythingProcessor

[[autodoc]] LocateAnythingProcessor

## LocateAnythingVisionModel

[[autodoc]] LocateAnythingVisionModel
    - forward

## LocateAnythingModel

[[autodoc]] LocateAnythingModel
    - forward

## LocateAnythingForConditionalGeneration

[[autodoc]] LocateAnythingForConditionalGeneration
    - forward
