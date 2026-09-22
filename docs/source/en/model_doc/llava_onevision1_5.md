<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-09-29 and contributed to Hugging Face Transformers on 2026-09-22.*

# LLaVA-OneVision-1.5

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

LLaVA-OneVision-1.5 was proposed in [LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training](https://huggingface.co/papers/2509.23661) by the LLaVA-OneVision team.

LLaVA-OneVision-1.5 is a vision-language model with a Qwen-style vision encoder and a [Qwen3](qwen3) language
backbone. It supports image, multi-image, and video understanding.

*We present LLaVA-OneVision-1.5, a fully open framework for large-scale multimodal training. The framework improves
data, model, and training recipes while releasing the complete training code, datasets, and model weights.*

The original code can be found [here](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5).

## Usage example

```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=50)
print(processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## LlavaOnevision1_5Config

[[autodoc]] LlavaOnevision1_5Config

## LlavaOnevision1_5TextConfig

[[autodoc]] LlavaOnevision1_5TextConfig

## LlavaOnevision1_5VisionConfig

[[autodoc]] LlavaOnevision1_5VisionConfig

## LlavaOnevision1_5VisionModel

[[autodoc]] LlavaOnevision1_5VisionModel
    - forward

## LlavaOnevision1_5Model

[[autodoc]] LlavaOnevision1_5Model
    - forward
    - get_image_features
    - get_video_features

## LlavaOnevision1_5ForConditionalGeneration

[[autodoc]] LlavaOnevision1_5ForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features
