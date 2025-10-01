<!--Copyright 2024 The HuggingFace Team. All rights reserved.

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

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Helper class to enable loading timm models to be used with the transformers library and its autoclasses.

```python
>>> import torch
>>> from PIL import Image
>>> from urllib.request import urlopen
>>> from transformers import AutoModelForImageClassification, AutoImageProcessor

>>> # Load image
>>> image = Image.open(urlopen(
...     'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
... ))

>>> # Load model and image processor
>>> checkpoint = "timm/resnet50.a1_in1k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForImageClassification.from_pretrained(checkpoint).eval()

>>> # Preprocess image
>>> inputs = image_processor(image)

>>> # Forward pass
>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # Get top 5 predictions
>>> top5_probabilities, top5_class_indices = torch.topk(logits.softmax(dim=1) * 100, k=5)
```

## Resources:

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with TimmWrapper.

<PipelineTag pipeline="image-classification"/>

- [Collection of Example Notebook](https://github.com/ariG23498/timm-wrapper-examples) 🌎

> [!TIP]
> For a more detailed overview please read the [official blog post](https://huggingface.co/blog/timm-transformers) on the timm integration.

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
