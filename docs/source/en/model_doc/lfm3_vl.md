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
*This model was contributed to Hugging Face Transformers on 2026-10-23.*


# LFM3-VL

## Overview

LFM3-VL is the vision-language model family built on [LFM3](lfm3) by [Liquid AI](https://liquid.ai/). It takes images
and videos at their native resolution together with text.

## Architecture

LFM3-VL consists of a vision encoder, a projector and an LFM3 language model. The vision encoder is itself a Liquid
hybrid: a linear patchifier followed by gated depthwise 2D convolution layers and bidirectional grouped-query attention
layers with 2D rotary embeddings over the patch grid of each frame. Images are split into 16x16 patches with the
dynamic-resolution preprocessing of Qwen2-VL, videos additionally pair consecutive frames and prefix each frame with its
timestamp. The projector merges every 2x2 block of patch features into one language token with a layer norm and a
two-layer MLP.

## Example

The following example shows how to generate an answer using the `AutoModelForImageTextToText` class.

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

# Load model and processor
model_id = "LiquidAI/LFM3-VL-1.9B-Preview"
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

# Generate answer
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

## Lfm3VlVideoProcessor

[[autodoc]] Lfm3VlVideoProcessor

## Lfm3VlProcessor

[[autodoc]] Lfm3VlProcessor

## Lfm3VlConfig

[[autodoc]] Lfm3VlConfig

## Lfm3VlVisionConfig

[[autodoc]] Lfm3VlVisionConfig

## Lfm3VlVisionModel

[[autodoc]] Lfm3VlVisionModel
    - forward

## Lfm3VlModel

[[autodoc]] Lfm3VlModel
    - forward

## Lfm3VlForConditionalGeneration

[[autodoc]] Lfm3VlForConditionalGeneration
    - forward
