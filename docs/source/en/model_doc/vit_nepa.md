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
*This model was released on 2025-12-18 and added to Hugging Face Transformers on 2026-01-12.*
*This model was released on 2025-12-18 and added to Hugging Face Transformers on .*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ViT Next-Embedding Predictive Autoregression (NEPA)

[ViT NEPA](https://huggingface.co/papers/2512.16922) is a self-supervised learning method for Vision Transformers (ViTs) that predicts the next patch embedding in an autoregressive manner. This approach enables ViTs to learn strong visual representations without relying on labeled data, making it effective for various computer vision tasks.

You can find all the original ViT NEPA checkpoints under the [SixAILab's NEPA](https://huggingface.co/collections/SixAILab/nepa) collection.

> [!TIP]
> Click on the ViTNepa models in the right sidebar for more examples of how to apply ViTNepa to different computer vision tasks.

The example below demonstrates how to classify an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="SixAILab/nepa-base-patch14-224-sft",
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
    "SixAILab/nepa-base-patch14-224-sft",
    use_fast=True,
)
model = AutoModelForImageClassification.from_pretrained(
    "SixAILab/nepa-base-patch14-224-sft",
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

- The ViT NEPA models are pretrained using self-supervised learning on large-scale datasets without labels.
- After pretraining, the models are fine-tuned on downstream tasks such as image classification.
- The models utilize standard Vision Transformer architectures with modifications to support the NEPA training objective.
- Use [`ViTImageProcessorFast`] to resize (or rescale) and normalize images to the expected size.
- The patch and image resolution are reflected in the checkpoint name. For example, [SixAILab/nepa-base-patch14-224-sft](https://huggingface.co/SixAILab/nepa-base-patch14-224-sft), is the **base-sized** architecture with a patch resolution of 16x16 and fine-tuning resolution of 224x224.

## ViTNepaConfig

[[autodoc]] ViTNepaConfig

## ViTNepaModel

[[autodoc]] ViTNepaModel
    - forward

## ViTNepaForPreTraining

[[autodoc]] ViTNepaForPreTraining
    - forward

## ViTNepaForImageClassification

[[autodoc]] ViTNepaForImageClassification
    - forward
