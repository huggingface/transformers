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
*This model was released on 2026-03-11 and added to Hugging Face Transformers on 2026-03-11.*
# CHMv2

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Canopy Height Maps v2 (CHMv2) model was proposed in [CHMv2: Improvements in Global Canopy Height Mapping using DINOv3](https://huggingface.co/papers/2603.06382). Building on our [original high-resolution canopy height maps](https://sustainability.atmeta.com/blog/2024/04/22/using-artificial-intelligence-to-map-the-earths-forests/) released in 2024, CHMv2 delivers substantial improvements in accuracy, detail, and global consistency by leveraging DINOv3, Meta's self-supervised vision model.

You can find more information [here](http://ai.meta.com/blog/world-resources-institute-dino-canopy-height-maps-v2), and the original code [here](https://github.com/facebookresearch/dinov3).

The abstract from the paper is the following:

*Accurate canopy height information is essential for quantifying forest carbon, monitoring restoration and degradation, and assessing habitat structure, yet high-fidelity measurements from airborne laser scanning (ALS) remain unevenly available globally. Here we present CHMv2, a global, meter-resolution canopy height map derived from high-resolution optical satellite imagery using a depth-estimation model built on DINOv3 and trained against ALS canopy height models. Compared to existing products, CHMv2 substantially improves accuracy, reduces bias in tall forests, and better preserves fine-scale structure such as canopy edges and gaps. These gains are enabled by a large expansion of geographically diverse training data, automated data curation and registration, and a loss formulation and data sampling strategy tailored to canopy height distributions. We validate CHMv2 against independent ALS test sets and against tens of millions of GEDI and ICESat-2 observations, demonstrating consistent performance across major forest biomes.*

## Usage examples

Run inference on an image with the following code:

```python
from PIL import Image
import torch

from transformers import AutoModelForDepthEstimation, AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-chmv2-dpt-head")
model = AutoModelForDepthEstimation.from_pretrained("facebook/dinov3-vitl16-chmv2-dpt-head")

image = Image.open("image.tif")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

depth = processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)]
)[0]["predicted_depth"]
```

## CHMv2Config

[[autodoc]] CHMv2Config

## CHMv2ImageProcessorFast

[[autodoc]] CHMv2ImageProcessorFast
    - preprocess
    - post_process_depth_estimation

## CHMv2ForDepthEstimation

[[autodoc]] CHMv2ForDepthEstimation
    - forward
