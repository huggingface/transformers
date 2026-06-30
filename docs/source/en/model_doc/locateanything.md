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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# LocateAnything

[LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) is a vision-language model from NVIDIA designed
for spatial reasoning, open-vocabulary object detection and visual grounding. It pairs a
[MoonViT](https://huggingface.co/nvidia/LocateAnything-3B) vision encoder with a
[`Qwen2`](./qwen2) language model through an MLP projector, and predicts structured bounding boxes and points.

Its key contribution is **Parallel Box Decoding (PBD)**: instead of decoding one coordinate token at a time, the model
predicts a complete box / point unit (a *block*) in parallel using a block-diffusion attention mask combined with
Multi-Token Prediction (MTP). Three decoding regimes are available:

- `"fast"` — Multi-Token Prediction only (all blocks decoded in parallel).
- `"slow"` — auto-regressive decoding only.
- `"hybrid"` — MTP with an auto-regressive fallback on uncertain boxes (the default).

## Usage example

```python
import torch
from transformers import AutoProcessor, LocateAnythingForConditionalGeneration

model_id = "nvidia/LocateAnything-3B"
processor = AutoProcessor.from_pretrained(model_id)
model = LocateAnythingForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
            {"type": "text", "text": "Detect all objects in the image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, generation_mode="hybrid", max_new_tokens=512)
output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

# Parse the structured grounding output into absolute-pixel boxes / points.
image_width, image_height = 800, 600
boxes = processor.parse_boxes(output_text[0], image_width, image_height)
print(boxes)
```

## LocateAnythingVisionConfig

[[autodoc]] LocateAnythingVisionConfig

## LocateAnythingConfig

[[autodoc]] LocateAnythingConfig

## LocateAnythingImageProcessor

[[autodoc]] LocateAnythingImageProcessor
    - preprocess

## LocateAnythingProcessor

[[autodoc]] LocateAnythingProcessor
    - __call__

## LocateAnythingVisionModel

[[autodoc]] LocateAnythingVisionModel
    - forward

## LocateAnythingModel

[[autodoc]] LocateAnythingModel
    - forward
    - get_image_features

## LocateAnythingForConditionalGeneration

[[autodoc]] LocateAnythingForConditionalGeneration
    - forward
    - generate
    - get_image_features
