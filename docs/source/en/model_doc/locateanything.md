<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

*This model was released on 2026-05-26 and added to Hugging Face Transformers on 2026-05-30.*

# LocateAnything

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

LocateAnything is a vision-language model for visual grounding. It localizes objects, text regions, GUI elements, and other referring expressions from natural-language prompts, returning bounding boxes or points.

The model was proposed in [LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf) by Yangzhou Liu, Han Wang, Zhe Chen, Tianheng Cheng, and collaborators.

LocateAnything combines a MoonViT vision encoder, a Qwen2.5 language model, and a multimodal projector. Its main decoding feature is Parallel Box Decoding (PBD), which predicts the coordinate tokens for a full box or point as a structured unit instead of generating every coordinate strictly token by token.

The original code can be found [here](https://github.com/NVlabs/Eagle). The model card is available [here](https://huggingface.co/nvidia/LocateAnything-3B), and the project page is available [here](https://research.nvidia.com/labs/lpr/locate-anything/).

## Usage

```python
from PIL import Image
import requests
import torch

from transformers import AutoProcessor, LocateAnythingForConditionalGeneration


model_id = "nvidia/LocateAnything-3B"
model = LocateAnythingForConditionalGeneration.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open(
    requests.get(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
        stream=True,
    ).raw
).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Locate all instances that match the following description: cat."},
        ],
    }
]

text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos = processor.process_vision_info(messages)
inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        pixel_values=inputs["pixel_values"].to(model.dtype),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws"),
        tokenizer=processor.tokenizer,
        max_new_tokens=128,
        generation_mode="slow",
        use_cache=True,
    )

print(output)
```

## LocateAnythingConfig

[[autodoc]] LocateAnythingConfig

## MoonViTConfig

[[autodoc]] MoonViTConfig

## LocateAnythingImageProcessor

[[autodoc]] LocateAnythingImageProcessor
    - preprocess

## LocateAnythingProcessor

[[autodoc]] LocateAnythingProcessor
    - __call__

## LocateAnythingForConditionalGeneration

[[autodoc]] LocateAnythingForConditionalGeneration
    - forward
    - get_image_features
    - get_placeholder_mask
