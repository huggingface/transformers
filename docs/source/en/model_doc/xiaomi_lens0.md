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
*This model was contributed to Hugging Face Transformers on 2026-10-27.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Xiaomi-Lens-0

## Overview

[Xiaomi-Lens-0](https://huggingface.co/papers/2610.23279) is a 4B multimodal language model built on [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). It adds a lightweight pixel-mask decoder and is trained with joint language and mask supervision. Each generated `<|mask|>` token identifies an image region that the decoder turns into a segmentation mask.

The model supports referring expression comprehension and segmentation, multi-instance grounding, reasoning segmentation, embodied referring, and fine-grained visual perception. It can produce masks in its final answer or during reasoning. See the [model card](https://huggingface.co/SeerRay-Lab/Xiaomi-Lens-0) for all examples and details.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/xiaomi_lens0_architecture.jpg"
alt="xiaomi lens0 architecture overview" width="600"/>

The original code can be found [here](https://github.com/SeerRay-Lab/Xiaomi-Lens-0).

<hfoptions id="usage">
<hfoption id="AutoModelForImageTextToText">

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "guarin/Xiaomi-Lens-0"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
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

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        return_dict_in_generate=True,
        output_hidden_states=[model.config.text_config.num_hidden_layers - 1],
        max_new_tokens=1024,
    )
    hidden_states = torch.cat([step[-1] for step in outputs.hidden_states], dim=1)
    mask_logits, mask_token_indices = model.decode_masks(
        input_ids=outputs.sequences,
        hidden_states=hidden_states,
        image_hidden_states=outputs.image_hidden_states,
        image_grid_thw=inputs.image_grid_thw,
    )
results = processor.post_process_grounding(
    outputs.sequences, inputs.image_grid_thw, mask_logits, mask_token_indices, target_sizes=[(480, 640)]
)
print(results[0]["boxes"]) # bounding boxes in absolute (top_left_x, top_left_y, bottom_right_x, bottom_right_y) pixel coordinates
# tensor([[344.9600,  24.9600, 640.0000, 372.0000],
#         [  8.9600,  55.2000, 316.1600, 474.2400]])
print(results[0]["masks"].shape) # binary segmentation masks
# torch.Size([2, 480, 640])
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/xiaomi_lens0_grounding.png"
alt="xiaomi lens0 grounding boxes and masks results" width="600"/>

</hfoption>
</hfoptions>

## Notes

- Boxes are written as `<bbox>[x1, y1, x2, y2]</bbox>` with coordinates normalized to `[0, 1000]`, and masks as `<mask><|mask|></mask>`. A grounded instance is written as `<bbox>[x1, y1, x2, y2]</bbox><mask><|mask|></mask>`.
- Ask for boxes and masks explicitly with "Report bbox coordinates and masks." at the end of the instruction.
- To detect boxes or masks only ask with "Report bbox coordinates." or "Report masks."
- The mask decoder sees the image at the resolution chosen by the image processor (up to `max_pixels`). Masks are predicted at that resolution and [`~XiaomiLens0Processor.post_process_grounding`] resizes them to `target_sizes`.
- Decoding and postprocessing exclude boxes and masks from thinking blocks by default. Set `ignore_thinking=False` in [`~XiaomiLens0ForConditionalGeneration.decode_masks`] to include masks and in [`~XiaomiLens0Processor.post_process_grounding`] to include boxes.

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
    - decode_masks

## XiaomiLens0GenerateOutput

[[autodoc]] XiaomiLens0GenerateOutput
