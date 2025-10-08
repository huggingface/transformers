<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-10-22 and added to Hugging Face Transformers on 2021-04-01 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Vision Transformer (ViT)

[ViT](https://huggingface.co/papers/2010.11929) demonstrates that a pure Transformer architecture can excel in image classification tasks without relying on convolutional networks. By training on large datasets and transferring to various benchmarks, ViT achieves excellent results while using fewer computational resources than state-of-the-art convolutional networks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="google/vit-base-patch16-224", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## ViTConfig

[[autodoc]] ViTConfig

## ViTFeatureExtractor

[[autodoc]] ViTFeatureExtractor
    - __call__

## ViTImageProcessor

[[autodoc]] ViTImageProcessor
    - preprocess

## ViTImageProcessorFast

[[autodoc]] ViTImageProcessorFast
    - preprocess

## ViTModel

[[autodoc]] ViTModel
    - forward

## ViTForMaskedImageModeling

[[autodoc]] ViTForMaskedImageModeling
    - forward

## ViTForImageClassification

[[autodoc]] ViTForImageClassification
    - forward

