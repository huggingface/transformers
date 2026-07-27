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


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-07-27.*

# Step3p7 (Step-3.7-Flash)

## Overview

Step-3.7-Flash is a vision-language model from StepFun.

## Usage example

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.image_utils import load_image


model = AutoModelForImageTextToText.from_pretrained(
    "stepfun-ai/Step-3.7-Flash", dtype=torch.bfloat16, device_map="auto",
)
processor = AutoProcessor.from_pretrained("stepfun-ai/Step-3.7-Flash")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = processor(images=[image], text=text, return_tensors="pt").to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
```

## Step3p7Config

[[autodoc]] Step3p7Config

## Step3p7VisionConfig

[[autodoc]] Step3p7VisionConfig

## Step3p7TextConfig

[[autodoc]] Step3p7TextConfig

## Step3p7ImageProcessor

[[autodoc]] Step3p7ImageProcessor

## Step3p7Processor

[[autodoc]] Step3p7Processor

## Step3p7VisionModel

[[autodoc]] Step3p7VisionModel
    - forward

## Step3p7TextModel

[[autodoc]] Step3p7TextModel
    - forward

## Step3p7Model

[[autodoc]] Step3p7Model
    - forward

## Step3p7ForConditionalGeneration

[[autodoc]] Step3p7ForConditionalGeneration
    - forward
