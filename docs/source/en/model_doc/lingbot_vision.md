<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

*This model was contributed to Hugging Face Transformers on 2026-07-21.*

# LingBot-Vision

## Overview

LingBot-Vision is a vision transformer encoder for image feature extraction. The architecture uses register tokens
(called storage tokens upstream), axial rotary position embeddings, LayerScale, and MLP or SwiGLU feed-forward layers
depending on the checkpoint size — the same building blocks as [DINOv3](./dinov3_vit), which this implementation
reuses.

This model was contributed by [IMvision12](https://huggingface.co/IMvision12).
The original code can be found [here](https://github.com/Robbyant/lingbot-vision).

## Usage Example

```python
import torch

from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image


model_id = "IMvision12/lingbot-vision-vit-giant-hf"
image_processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
inputs = image_processor(images=image, return_tensors="pt")

with torch.inference_mode():
    outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
```

## LingbotVisionConfig

[[autodoc]] LingbotVisionConfig

## LingbotVisionImageProcessor

[[autodoc]] LingbotVisionImageProcessor
    - preprocess

## LingbotVisionImageProcessorPil

[[autodoc]] LingbotVisionImageProcessorPil
    - preprocess

## LingbotVisionModel

[[autodoc]] LingbotVisionModel
    - forward

## LingbotVisionBackbone

[[autodoc]] LingbotVisionBackbone
    - forward
