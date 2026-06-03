<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-08-20.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MetaCLIP 2

## Overview

MetaCLIP 2 is a replication of the original CLIP model trained on 300+ languages. It achieves state-of-the-art (SOTA) results on multilingual benchmarks (e.g., XM3600, CVQA, Babel‑ImageNet), surpassing previous SOTA such as [mSigLIP](siglip) and [SigLIP‑2](siglip2). The authors show that English and non-English worlds can mutually benefit and elevate each other.

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/facebookresearch/MetaCLIP).

You can find all the MetaCLIP 2 checkpoints under the [Meta](https://huggingface.co/facebook/models?search=metaclip-2) organization.

> [!TIP]
> Click on the MetaCLIP 2 models in the right sidebar for more examples of how to apply MetaCLIP 2 to different image and language tasks.

The example below demonstrates how to calculate similarity scores between multiple text descriptions and an image with [`Pipeline`] or the [`AutoModel`] class. Usage of the MetaCLIP 2 models is identical to the CLIP models, you just need the `MetaClip2Model` class instead of `CLIPModel`.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

clip = pipeline(
   task="zero-shot-image-classification",
   model="facebook/metaclip-2-worldwide-huge-quickgelu",
   dtype=torch.bfloat16,
   device=0
)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
clip("http://images.cocodataset.org/val2017/000000039769.jpg", candidate_labels=labels)
```

</hfoption>
<hfoption id="AutoModel">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu", dtype=torch.bfloat16, attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
most_likely_idx = probs.argmax(dim=1).item()
most_likely_label = labels[most_likely_idx]
print(f"Most likely label: {most_likely_label} with probability: {probs[0][most_likely_idx].item():.3f}")
```

</hfoption>
</hfoptions>

## MetaClip2Config

[[autodoc]] MetaClip2Config
    - from_text_vision_configs

## MetaClip2TextConfig

[[autodoc]] MetaClip2TextConfig

## MetaClip2VisionConfig

[[autodoc]] MetaClip2VisionConfig

## MetaClip2Model

[[autodoc]] MetaClip2Model
    - forward
    - get_text_features
    - get_image_features

## MetaClip2TextModel

[[autodoc]] MetaClip2TextModel
    - forward

## MetaClip2TextModelWithProjection

[[autodoc]] MetaClip2TextModelWithProjection
    - forward

## MetaClip2VisionModelWithProjection

[[autodoc]] MetaClip2VisionModelWithProjection
    - forward

## MetaClip2VisionModel

[[autodoc]] MetaClip2VisionModel
    - forward

## MetaClip2ForImageClassification

[[autodoc]] MetaClip2ForImageClassification
    - forward
