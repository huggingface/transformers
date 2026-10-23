<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-10-23.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Xiaomi-Lens-0

## Overview

[Xiaomi-Lens-0](https://huggingface.co/papers/2610.23279) is a 4B multimodal model from Xiaomi built on [Qwen3.5](./qwen3_5)-4B that grounds its answers at the pixel level. Besides text, it writes a `<|mask|>` token for every region it refers to, and a lightweight mask decoder turns the hidden state of each `<|mask|>` token into a segmentation mask. The same mechanism covers referring expression comprehension and segmentation, multi-instance grounding, reasoning segmentation, embodied referring, and fine-grained perception, where masks also appear in the reasoning trace (for example one per counted object).

The mask decoder reads a single feature grid from the vision encoder, before the spatial merger. The patch features go through one self-attention layer and two transposed convolutions, and are matched against an embedding of each `<|mask|>` hidden state. A final 4x bilinear upsampling brings the masks to the resolution of the resized image fed to the vision encoder.

The original code can be found [here](https://github.com/SeerRay-Lab/Xiaomi-Lens-0).

## Usage

Grounded instances are written as `<bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>`, with box coordinates normalized to `[0, 1000]`. Generate the answer first, then run a forward pass over the full sequence to decode one mask per `<|mask|>` token. [`~XiaomiLens0Processor.post_process_grounding`] keeps the boxes and masks of the final answer, after the thinking block, and resizes them to the original image.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "SeerRay-Lab/Xiaomi-Lens-0"
model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {
                "type": "text",
                "text": 'Locate all instances in the image that match the description "cat".\nReport bbox coordinates and masks.',
            },
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(generated_ids[0, inputs.input_ids.shape[1] :]))

# Decode the masks of the generated `<|mask|>` tokens
mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids(generated_ids), device=model.device)
with torch.no_grad():
    outputs = model(
        input_ids=generated_ids,
        pixel_values=inputs.pixel_values,
        image_grid_thw=inputs.image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )
results = processor.post_process_grounding(
    generated_ids, outputs.pred_masks, inputs.image_grid_thw, target_sizes=[(480, 640)]
)
print(results[0]["boxes"], results[0]["masks"].shape)
```

Tips:

- Ask for boxes and masks explicitly with "Report bbox coordinates and masks." at the end of the instruction. Without it, the model answers in plain text.
- The mask decoder sees the image at the resolution chosen by the image processor (up to `max_pixels`). Masks are predicted at that resolution and [`~XiaomiLens0Processor.post_process_grounding`] resizes them to `target_sizes`.
- For batched inputs with padding, also pass the `attention_mask` of the full sequences to the forward pass.

## XiaomiLens0Config

[[autodoc]] XiaomiLens0Config

## XiaomiLens0TextConfig

[[autodoc]] XiaomiLens0TextConfig

## XiaomiLens0VisionConfig

[[autodoc]] XiaomiLens0VisionConfig

## XiaomiLens0MaskDecoderConfig

[[autodoc]] XiaomiLens0MaskDecoderConfig

## XiaomiLens0Processor

[[autodoc]] XiaomiLens0Processor
    - post_process_grounding

## XiaomiLens0PreTrainedModel

[[autodoc]] XiaomiLens0PreTrainedModel

## XiaomiLens0VisionModel

[[autodoc]] XiaomiLens0VisionModel
    - forward

## XiaomiLens0TextModel

[[autodoc]] XiaomiLens0TextModel
    - forward

## XiaomiLens0Model

[[autodoc]] XiaomiLens0Model
    - forward

## XiaomiLens0ForConditionalGeneration

[[autodoc]] XiaomiLens0ForConditionalGeneration
    - forward
    - get_pred_masks
