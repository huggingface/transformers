<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-09-18.*

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# LFM2-VL

## Overview

[LFM2-VL](https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models) first series of vision-language foundation models developed by [Liquid AI](https://liquid.ai/). These multimodal models are designed for low-latency and device-aware deployment. LFM2-VL extends the LFM2 family of open-weight Liquid Foundation Models (LFMs) into the vision-language space, supporting both text and image inputs with variable resolutions.

## Architecture

LFM2-VL consists of three main components: a language model backbone, a vision encoder, and a multimodal projector. LFM2-VL builds upon the LFM2 backbone, inheriting from either LFM2-1.2B (for LFM2-VL-1.6B) or LFM2-350M (for LFM2-VL-450M). For the vision tower, LFM2-VL uses SigLIP2 NaFlex encoders to convert input images into token sequences. Two variants are implemented:

* Shape-optimized (400M) for more fine-grained vision capabilities for LFM2-VL-1.6B
* Base (86M) for fast image processing for LFM2-VL-450M

The encoder processes images at their native resolution up to 512×512 pixels, efficiently handling smaller images without upscaling and supporting non-standard aspect ratios without distortion. Larger images are split into non-overlapping square patches of 512×512 each, preserving detail. In LFM2-VL-1.6B, the model also receives a thumbnail (a small, downscaled version of the original image capturing the overall scene) to enhance global context understanding and alignment. Special tokens mark each patch’s position and indicate the thumbnail’s start. The multimodal connector is a 2-layer MLP connector with pixel unshuffle to reduce image token count.

## Example

The following example shows how to generate an answer using the `AutoModelForImageTextToText` class.

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
\
# Load model and processor
model_id = "LiquidAI/LFM2-VL-1.6B"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16",
)
processor = AutoProcessor.from_pretrained(model_id)

# Load image and create conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is in this image?"},
        ],
    },
]

# Generate snswer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=64)
processor.batch_decode(outputs, skip_special_tokens=True)[0]

```

## Lfm2VlImageProcessorFast

[[autodoc]] Lfm2VlImageProcessorFast

## Lfm2VlProcessor

[[autodoc]] Lfm2VlProcessor
    - __call__

## Lfm2VlConfig

[[autodoc]] Lfm2VlConfig

## Lfm2VlModel

[[autodoc]] Lfm2VlModel
    - forward
    - get_image_features

## Lfm2VlForConditionalGeneration

[[autodoc]] Lfm2VlForConditionalGeneration
    - forward
    - get_image_features
