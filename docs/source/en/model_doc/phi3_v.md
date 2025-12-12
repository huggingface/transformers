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
*This model was released on 2024-06-18 and added to Hugging Face Transformers on 2025-11-15.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Phi-3.5 Vision

*Phi-3.5 vision is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision. The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.*


## Usage Example

The example below demonstrates how to generate text with `Phi3VForConditionalGeneration`.


```py
import torch
from transformers import Phi3VForConditionalGeneration, Phi3VProcessor

model_id = "yaswanthgali/Phi-3.5-vision-instruct"

# Prepare inputsfor generation.
messages = [
    {
        "role": "user",
        "content": [
            {'type':'image', 'url': 'http://images.cocodataset.org/val2017/000000039769.jpg'},
            {'type':"text", "text":"What do you see in this image?."}
        ]
    },
]

processor = Phi3VProcessor.from_pretrained(model_id)
model = Phi3VForConditionalGeneration.from_pretrained(model_id,     
        dtype=torch.bfloat16,
        device_map="auto")

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    generation_mode="text",
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

output = model.generate(**inputs, max_new_tokens=20,do_sample=True)
text = processor.decode(output[0], skip_special_tokens=True)
print(text)
```

## Phi3VConfig

[[autodoc]] Phi3VConfig

## Phi3VImageProcessorFast

[[autodoc]] Phi3VImageProcessorFast

## Phi3VProcessor

[[autodoc]] Phi3VProcessor

## Phi3VModel

[[autodoc]] Phi3VModel
    - forward

## Phi3VForConditionalGeneration

[[autodoc]] Phi3VForConditionalGeneration
    - forward
    - generate


