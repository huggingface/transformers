<!--Copyright 2026 Cohere Inc. and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-08-10.*

# CohereCompass

<div class="flex flex-wrap space-x-1">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

CohereCompass is the base architecture for small, specialized (vision-)language models trained by Cohere.

## Usage examples

The following example loads an image from a URL and asks the model to describe it. Prompts can interleave text with one or more images; for text-only prompts, omit the image entries.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "CohereLabs/North-Micro-Vision-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
)

image_url = "https://cdn-uploads.huggingface.co/production/uploads/66d732effe6684fc16b12c28/Io_5OCmftsmH-n158ZtPs.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": image_url},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=128,
)

input_length = inputs["input_ids"].shape[-1]
response = processor.decode(
    outputs[0][input_length:],
    skip_special_tokens=True,
)
print(response)
```


## CohereCompassConfig

[[autodoc]] CohereCompassConfig

## CohereCompassTextConfig

[[autodoc]] CohereCompassTextConfig

## CohereCompassVisionConfig

[[autodoc]] CohereCompassVisionConfig

## CohereCompassModel

[[autodoc]] CohereCompassModel
    - forward

## CohereCompassTextModel

[[autodoc]] CohereCompassTextModel
    - forward

## CohereCompassVisionModel

[[autodoc]] CohereCompassVisionModel
    - forward

## CohereCompassForConditionalGeneration

[[autodoc]] CohereCompassForConditionalGeneration
    - forward
    - get_image_features

## CohereCompassForCausalLM

[[autodoc]] CohereCompassForCausalLM

## CohereCompassTextForSequenceClassification

[[autodoc]] CohereCompassTextForSequenceClassification
    - forward

## CohereCompassImageProcessor

[[autodoc]] CohereCompassImageProcessor
    - preprocess

## CohereCompassImageProcessorPil

[[autodoc]] CohereCompassImageProcessorPil
    - preprocess

## CohereCompassVideoProcessor

[[autodoc]] CohereCompassVideoProcessor
    - preprocess

## CohereCompassProcessor

[[autodoc]] CohereCompassProcessor
    - __call__
