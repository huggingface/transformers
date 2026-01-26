<!--Copyright 2025 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-01-23.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# Molmo2

## Overview

Molmo2 is a multimodal vision-language model developed by AllenAI. It combines a Vision Transformer (ViT) for image processing with a text decoder for generating text responses. The model supports both image and video inputs, making it suitable for various vision-language tasks.

The model architecture consists of:
- A Vision Transformer (ViT) for processing images
- An adapter layer that connects vision and text modalities
- A text decoder based on transformer architecture with rotary position embeddings

## Usage example

### Image-text-to-text generation

Here's how to use Molmo2 for image-text-to-text generation:

```python
from transformers import Molmo2ForConditionalGeneration, Molmo2Processor
import torch
from PIL import Image
import requests

processor = Molmo2Processor.from_pretrained("allenai/Molmo2-8B")
model = Molmo2ForConditionalGeneration.from_pretrained(
    "allenai/Molmo2-8B",
    dtype=torch.float16,
    device_map="auto",
)

# Load an image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

# Prepare inputs
text = "Describe this image."
inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

# Generate
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_text)
```

## Molmo2Config

[[autodoc]] Molmo2Config

## Molmo2VitConfig

[[autodoc]] Molmo2VitConfig

## Molmo2AdapterConfig

[[autodoc]] Molmo2AdapterConfig

## Molmo2TextConfig

[[autodoc]] Molmo2TextConfig

## Molmo2Processor

[[autodoc]] Molmo2Processor
    - __call__

## Molmo2ImageProcessor

[[autodoc]] Molmo2ImageProcessor
    - __call__
    - preprocess

## Molmo2VideoProcessor

[[autodoc]] Molmo2VideoProcessor
    - __call__

## Molmo2Model

[[autodoc]] Molmo2Model
    - forward

## Molmo2ForConditionalGeneration

[[autodoc]] Molmo2ForConditionalGeneration
    - forward
