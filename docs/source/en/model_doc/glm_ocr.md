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
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-01-27.*

# GLM-OCR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
</div>

## Overview

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) is a multimodal OCR (Optical Character Recognition) model designed for complex document understanding from [Z.ai](https://github.com/zai-org/GLM-OCR). The model combines a CogViT visual encoder (pre-trained on large-scale image-text data), a lightweight cross-modal connector with efficient token downsampling, and a GLM-0.5B language decoder.

Key features of GLM-OCR include:
- **Lightweight**: Only 0.9B parameters while achieving state-of-the-art performance (94.62 on OmniDocBench V1.5)
- **Multi-task**: Excels at text recognition, formula recognition, table recognition, and information extraction
- **Multi-modal**: Processes document images for text, formula, and table extraction

This model was contributed by the [zai-org](https://huggingface.co/zai-org) team.
The original code can be found [here](https://github.com/zai-org/GLM-OCR).

## Usage example

### Single image inference

```python
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
import torch

model_id = "zai-org/GLM-OCR"

processor = AutoProcessor.from_pretrained(model_id)
model = GlmOcrForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"},
            {"type": "text", "text": "Text Recognition:"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Batch inference

The model supports batching multiple images for efficient processing.

```python
from transformers import AutoProcessor, GlmOcrForConditionalGeneration
import torch

model_id = "zai-org/GLM-OCR"

processor = AutoProcessor.from_pretrained(model_id)
model = GlmOcrForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

# First document
message1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"},
            {"type": "text", "text": "Text Recognition:"},
        ],
    }
]

# Second document
message2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Text Recognition:"},
        ],
    }
]

messages = [message1, message2]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    padding=True,
).to(model.device)

output = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(output, skip_special_tokens=True))
```

### Flash Attention 2

GLM-OCR supports Flash Attention 2 for faster inference. First, install the latest version of Flash Attention:

```bash
pip install -U flash-attn --no-build-isolation
```

Then load the model with one of the supported kernels of the [kernels-community](https://huggingface.co/kernels-community):

```python
from transformers import GlmOcrForConditionalGeneration
import torch

model = GlmOcrForConditionalGeneration.from_pretrained(
    "zai-org/GLM-OCR",
    dtype=torch.bfloat16,
    attn_implementation="kernels-community/flash-attn2",  # other options: kernels-community/vllm-flash-attn3, kernels-community/paged-attention
    device_map="auto",
)
```

## GlmOcrConfig

[[autodoc]] GlmOcrConfig

## GlmOcrVisionConfig

[[autodoc]] GlmOcrVisionConfig

## GlmOcrTextConfig

[[autodoc]] GlmOcrTextConfig

## GlmOcrVisionModel

[[autodoc]] GlmOcrVisionModel
- forward

## GlmOcrTextModel

[[autodoc]] GlmOcrTextModel
- forward

## GlmOcrModel

[[autodoc]] GlmOcrModel
- forward

## GlmOcrForConditionalGeneration

[[autodoc]] GlmOcrForConditionalGeneration
- forward