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
*This model was released on 2020-12-23 and added to Hugging Face Transformers on 2021-04-13 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DeiT

[DeiT](https://huggingface.co/papers/2012.12877) addresses the inefficiency of training visual transformers by developing a more data-efficient model. This model achieves competitive results on ImageNet with only internal data and minimal computational resources, training on a single computer in less than 3 days. A key innovation is the introduction of a token-based distillation strategy, which enhances the student model's learning from a teacher model, particularly when the teacher is a convolutional neural network. This approach results in top-1 accuracy of up to 85.2% on ImageNet and strong performance on other tasks.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/deit-base-distilled-patch16-224", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
model = AutoModelForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- Compared to ViT, DeiT models use a distillation token to learn from a teacher (typically a ResNet-like model). The distillation token learns through backpropagation by interacting with the class (`[CLS]`) and patch tokens through self-attention layers.
- Fine-tune distilled models in two ways: (1) Classic fine-tuning places a prediction head only on the class token without using the distillation signal, or (2) Fine-tuning with distillation places prediction heads on both the class token and distillation token.
- For fine-tuning with distillation, the `[CLS]` prediction head trains using regular cross-entropy between the prediction and ground-truth label. The distillation prediction head trains using hard distillation (cross-entropy between the distillation head prediction and teacher's predicted label). At inference, take the average prediction between both heads as the final prediction.
- Fine-tuning with distillation relies on a teacher already fine-tuned on the downstream dataset. Use [`DeiTForImageClassification`] for classic fine-tuning and [`DeiTForImageClassificationWithTeacher`] for fine-tuning with distillation.
- All released checkpoints were pre-trained and fine-tuned on ImageNet-1k only. No external data was used. This contrasts with the original ViT model, which used external data like JFT-300M dataset/ImageNet-21k for pre-training.
- DeiT authors released more efficiently trained ViT models that plug directly into [`ViTModel`] or [`ViTForImageClassification`]. Techniques like data augmentation, optimization, and regularization simulate training on a much larger dataset while only using ImageNet-1k for pre-training.

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

