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
*This model was released on 2023-01-19 and added to Hugging Face Transformers on 2024-12-05 and contributed by [jmtzt](https://huggingface.co/jmtzt).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# I-JEPA

[I-JEPA](https://huggingface.co/papers/2301.08243) is a self-supervised learning method that predicts representations of various target blocks in an image based on a single context block. This approach emphasizes learning semantic features without using hand-crafted data transformations or focusing on pixel-level details. Key to I-JEPA's design is a masking strategy that involves selecting large-scale target blocks and using a spatially distributed context block. When integrated with Vision Transformers, I-JEPA scales effectively, enabling the training of a ViT-Huge/14 on ImageNet in under 72 hours using 16 A100 GPUs, achieving strong performance across tasks like linear classification, object counting, and depth prediction.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/ijepa_vitg16_22k", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/ijepa_vitg16_22k")
model = AutoModelForImageClassification.from_pretrained("facebook/ijepa_vitg16_22k", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## IJepaConfig

[[autodoc]] IJepaConfig

## IJepaModel

[[autodoc]] IJepaModel
    - forward

## IJepaForImageClassification

[[autodoc]] IJepaForImageClassification
    - forward

