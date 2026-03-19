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
*This model was released on 2020-12-23 and added to Hugging Face Transformers on 2021-04-13.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DeiT

[DeiT](https://huggingface.co/papers/2012.12877) (Data-efficient Image Transformers) is a [Vision Transformer (ViT)](vit) trained more efficiently for image classification, requiring far less data and compute compared to the original ViT. DeiT introduces a teacher-student distillation strategy specific to transformers, using a distillation token that learns from a teacher (e.g., a ConvNet) through attention.

You can find all the original DeiT checkpoints under the [Facebook](https://huggingface.co/facebook?search_models=deit) organization.

> [!TIP]
> Click on the DeiT models in the right sidebar for more examples of how to apply DeiT to different vision tasks.

The example below demonstrates how to classify an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="facebook/deit-base-distilled-patch16-224",
    dtype=torch.float16,
    device=0
)
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(
    "facebook/deit-base-distilled-patch16-224",
    use_fast=True,
)
model = AutoModelForImageClassification.from_pretrained(
    "facebook/deit-base-distilled-patch16-224",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to(model.device)

with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```

</hfoption>
</hfoptions>

## Notes

- DeiT uses a distillation token to learn from a teacher model (typically a ResNet). The distillation token interacts with the class ([CLS]) and patch tokens through self-attention layers.
- There are two ways to fine-tune distilled models: (1) classic fine-tuning with only a prediction head on the class token ([`DeiTForImageClassification`]), or (2) fine-tuning with distillation using both a class token head and a distillation token head ([`DeiTForImageClassificationWithTeacher`]).
- All released checkpoints were pre-trained and fine-tuned on ImageNet-1k only, without external data.
- Use [`DeiTImageProcessorFast`] to resize and normalize images for the model.

## DeiTConfig

[[autodoc]] DeiTConfig

## DeiTImageProcessor

[[autodoc]] DeiTImageProcessor
    - preprocess

## DeiTImageProcessorFast

[[autodoc]] DeiTImageProcessorFast
    - preprocess

## DeiTModel

[[autodoc]] DeiTModel
    - forward

## DeiTForMaskedImageModeling

[[autodoc]] DeiTForMaskedImageModeling
    - forward

## DeiTForImageClassification

[[autodoc]] DeiTForImageClassification
    - forward

## DeiTForImageClassificationWithTeacher

[[autodoc]] DeiTForImageClassificationWithTeacher
    - forward
