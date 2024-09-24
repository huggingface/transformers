<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TimmWrapper

## Overview

Helper class to enable loading timm models to be used with the transformers library and its autoclasses.

```python
import torch
from urllib.request import urlopen
from PIL import Image
from transformers import AutoConfig, AutoModelForImageClassification, AutoImageProcessor

checkpoint = "timm/resnet50.a1_in1k"
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

image_processor = AutoImageProcessor.from_pretrained(checkpoint)
inputs = image_processor(img, return_tensors="pt")
model = AutoModelForImageClassification.from_pretrained(checkpoint)

with torch.no_grad():
    logits = model(**inputs).logits

top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1) * 100, k=5)
```

## TimmWrapperConfig

[[autodoc]] TimmWrapperConfig

## TimmWrapperImageProcessor

[[autodoc]] TimmWrapperImageProcessor
    - preprocess

## TimmWrapperModel

[[autodoc]] TimmWrapperModel
    - forward

## TimmWrapperForImageClassification

[[autodoc]] TimmWrapperForImageClassification
    - forward
