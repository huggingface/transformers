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
*This model was released on 2020-10-22 and added to Hugging Face Transformers on 2021-04-01.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Vision Transformer (ViT)

[Vision Transformer (ViT)](https://huggingface.co/papers/2010.11929) is a transformer adapted for computer vision tasks. An image is split into smaller fixed-sized patches which are treated as a sequence of tokens, similar to words for NLP tasks. ViT requires less resources to pretrain compared to convolutional architectures and its performance on large datasets can be transferred to smaller downstream tasks.

You can find all the original ViT checkpoints under the [Google](https://huggingface.co/google?search_models=vit) organization.

> [!TIP]
> Click on the ViT models in the right sidebar for more examples of how to apply ViT to different computer vision tasks.

The example below demonstrates how to classify an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="google/vit-base-patch16-224",
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
    "google/vit-base-patch16-224",
    use_fast=True,
)
model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
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

- The best results are obtained with supervised pretraining, and during fine-tuning, it may be better to use images with a resolution higher than 224x224.
- Use [`ViTImageProcessorFast`] to resize (or rescale) and normalize images to the expected size.
- The patch and image resolution are reflected in the checkpoint name. For example, google/vit-base-patch16-224, is the **base-sized** architecture with a patch resolution of 16x16 and fine-tuning resolution of 224x224.

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
