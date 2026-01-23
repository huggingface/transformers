<!--Copyright 2025 Deepseek AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-01-19.*


# DeepSeekOCR

## Overview

DeepSeekOCR is a vision-language model designed for optical character recognition (OCR) tasks. The model combines dual vision encoders (SAM and CLIP) with a language model to process both text and images for generating contextually relevant OCR outputs, including document understanding, grounding, and markdown conversion.
The model uses a modified [DeepSeek-V2](./deepseek_v2) as its text decoder.

### Usage tips

The example below demonstrates how to perform OCR with grounding on a document image using the [`AutoModel`] class.

```py
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

processor = AutoProcessor.from_pretrained("deepseek-ai/deepseek-ocr")
model = AutoModel.from_pretrained("deepseek-ai/deepseek-ocr", torch_dtype=torch.bfloat16)

image = Image.open("document.png").convert("RGB")

conversation = [
    {
        "role": "<|User|>",
        "content": [
            {"type": "image", "path": "./document.png"},
            {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    return_dict=True,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=250)

text = processor.batch_decode(generated, skip_special_tokens=False)[0]
print(text)
```

## DeepseekOcrConfig

[[autodoc]] DeepseekOcrConfig

## DeepseekOcrVisionConfig

[[autodoc]] DeepseekOcrVisionConfig

## DeepseekOcrSamConfig

[[autodoc]] DeepseekOcrSamConfig

## DeepseekOcrCLIPVisionConfig

[[autodoc]] DeepseekOcrCLIPVisionConfig

## DeepseekOcrProjectorConfig

[[autodoc]] DeepseekOcrProjectorConfig

## DeepseekOcrProcessor

[[autodoc]] DeepseekOcrProcessor

## DeepseekOcrImageProcessorFast

[[autodoc]] DeepseekOcrImageProcessorFast

## DeepseekOcrModelOutputWithPast

[[autodoc]] DeepseekOcrModelOutputWithPast

## DeepseekOcrCausalLMOutputWithPast

[[autodoc]] DeepseekOcrCausalLMOutputWithPast

## DeepseekOcrTextModel

[[autodoc]] DeepseekOcrTextModel
    - forward

## DeepseekOcrCLIPVisionModel

[[autodoc]] DeepseekOcrCLIPVisionModel
    - forward

## DeepseekOcrProjector

[[autodoc]] DeepseekOcrProjector
    - forward

## DeepseekOcrModel

[[autodoc]] DeepseekOcrModel
    - forward

## DeepseekOcrForConditionalGeneration

[[autodoc]] DeepseekOcrForConditionalGeneration
    - forward
