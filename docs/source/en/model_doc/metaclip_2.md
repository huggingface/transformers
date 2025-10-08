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
*This model was released on 2025-07-29 and added to Hugging Face Transformers on 2025-08-20 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MetaCLIP 2

[MetaCLIP 2](https://huggingface.co/papers/2507.22062) is a CLIP-based foundation model trained from scratch on web-scale, multilingual image-text pairs to overcome limitations of previous multilingual CLIP models, including poor non-English performance and the “curse of multilinguality.” It introduces a training recipe that effectively leverages both English and non-English data without relying on translations or specialized architecture changes. In experiments, MetaCLIP 2 ViT-H/14 improves zero-shot ImageNet classification over English-only CLIP and outperforms prior multilingual models like mSigLIP. It also sets new state-of-the-art results on multilingual benchmarks such as CVQA, Babel-ImageNet, and XM3600 for image-to-text retrieval.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="zero-shot-image-classification", model="facebook/metaclip-2-worldwide-huge-quickgelu", dtype="auto"")
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg", candidate_labels=labels)
```

</hfoption>
<hfoption id="AutoModel">

```py
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu", dtype="auto")
processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
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
