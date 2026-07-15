<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LingBot-Vision

## Overview

LingBot-Vision is a vision transformer encoder for image feature extraction. The architecture uses storage tokens,
axial rotary position embeddings, LayerScale, and MLP or SwiGLU feed-forward layers depending on the checkpoint size.

The original code can be found [here](https://github.com/Robbyant/lingbot-vision).

## Usage Example

```python
import torch

from transformers import AutoModel


model = AutoModel.from_pretrained("robbyant/lingbot-vision-vit-small")
pixel_values = torch.randn(1, 3, 512, 512)

outputs = model(pixel_values)
last_hidden_state = outputs.last_hidden_state
```

## LingbotVisionConfig

[[autodoc]] LingbotVisionConfig

## LingbotVisionModel

[[autodoc]] LingbotVisionModel
    - forward

## LingbotVisionForImageClassification

[[autodoc]] LingbotVisionForImageClassification
    - forward

## LingbotVisionBackbone

[[autodoc]] LingbotVisionBackbone
    - forward
