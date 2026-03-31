<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-03-31.*

# SAM3-LiteText

## Overview

SAM3-LiteText is a lightweight variant of [SAM3](sam3) that replaces the CLIP text encoder with a compact MobileCLIP-S0 text encoder. This reduces the text encoder parameters by up to 88% while maintaining the full SAM3 vision and segmentation capabilities.

The model was introduced in the [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) repository by Simon Zeng.

Key differences from SAM3:
- **Text encoder**: MobileCLIP-S0 (RepMixer + Transformer) instead of CLIP (512 hidden dim, 4 transformer layers + 2 RepMixer blocks, context length 16)
- **All other components** (ViT-H backbone, FPN, geometry encoder, DETR encoder/decoder, mask decoder) are identical to SAM3

## Usage

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("Simon7108528/EfficientSAM3", device_map="auto")
processor = AutoProcessor.from_pretrained("Simon7108528/EfficientSAM3")

inputs = processor(images=image, text="cat", return_tensors="pt")
outputs = model(**inputs)
```

## Sam3LiteTextConfig

[[autodoc]] Sam3LiteTextConfig

## Sam3LiteTextMobileCLIPConfig

[[autodoc]] Sam3LiteTextMobileCLIPConfig

## Sam3LiteTextViTConfig

[[autodoc]] Sam3LiteTextViTConfig

## Sam3LiteTextModel

[[autodoc]] Sam3LiteTextModel
    - forward
    - get_text_features
    - get_vision_features

## Sam3LiteTextViTModel

[[autodoc]] Sam3LiteTextViTModel
    - forward

## Sam3LiteTextImageProcessor

[[autodoc]] Sam3LiteTextImageProcessor

## Sam3LiteTextProcessor

[[autodoc]] Sam3LiteTextProcessor
