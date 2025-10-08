<!--Copyright 2024 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2023-09-28 and added to Hugging Face Transformers on 2024-12-24 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DINOv2 with Registers

[DINOv2 with Registers](https://huggingface.co/papers/2309.16588) addresses artifacts in feature maps of Vision Transformers (ViTs) by introducing additional "register" tokens during pre-training. This solution eliminates artifacts, enhances interpretability of attention maps, and improves performance for both supervised and self-supervised models. The model achieves state-of-the-art results in self-supervised visual tasks and enables better object discovery with larger models.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov2-with-registers-base", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-with-registers-base")
model = AutoModelForImageClassification.from_pretrained("facebook/dinov2-with-registers-base", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Dinov2WithRegistersConfig

[[autodoc]] Dinov2WithRegistersConfig

## Dinov2WithRegistersModel

[[autodoc]] Dinov2WithRegistersModel
    - forward

## Dinov2WithRegistersForImageClassification

[[autodoc]] Dinov2WithRegistersForImageClassification
    - forward

