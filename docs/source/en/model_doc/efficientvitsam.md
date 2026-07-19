<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not
be rendered properly in your Markdown viewer.
-->
*This model was contributed to Hugging Face Transformers on 2026-07-19.*

# EfficientViT-SAM

## Overview

EfficientViT-SAM is a family of efficient Segment Anything models that uses an EfficientViT image encoder for
promptable image segmentation. Given an image and optional point or box prompts, the model predicts segmentation
masks and their quality scores.

```python
import requests
import torch
from PIL import Image

from transformers import EfficientViTSamModel, EfficientViTSamProcessor


model = EfficientViTSamModel.from_pretrained("mit-han-lab/efficientvit-sam-l1")
processor = EfficientViTSamProcessor.from_pretrained("mit-han-lab/efficientvit-sam-l1")

image = Image.open(requests.get("https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png", stream=True).raw)
inputs = processor(image, input_points=[[[450, 600]]], return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
```

## EfficientViTSamConfig

[[autodoc]] EfficientViTSamConfig

## EfficientViTSamVisionConfig

[[autodoc]] EfficientViTSamVisionConfig

## EfficientViTSamMaskDecoderConfig

[[autodoc]] EfficientViTSamMaskDecoderConfig

## EfficientViTSamPromptEncoderConfig

[[autodoc]] EfficientViTSamPromptEncoderConfig

## EfficientViTSamProcessor

[[autodoc]] EfficientViTSamProcessor
    - __call__

## EfficientViTSamImageProcessor

[[autodoc]] EfficientViTSamImageProcessor
    - preprocess

## EfficientViTSamImageProcessorPil

[[autodoc]] EfficientViTSamImageProcessorPil
    - preprocess

## EfficientViTSamVisionEncoderOutput

[[autodoc]] EfficientViTSamVisionEncoderOutput

## EfficientViTSamImageSegmentationOutput

[[autodoc]] EfficientViTSamImageSegmentationOutput

## EfficientViTSamPositionalEmbedding

[[autodoc]] EfficientViTSamPositionalEmbedding

## EfficientViTSamVisionModel

[[autodoc]] EfficientViTSamVisionModel
    - forward

## EfficientViTSamModel

[[autodoc]] EfficientViTSamModel
    - forward

## NestedList

[[autodoc]] NestedList
