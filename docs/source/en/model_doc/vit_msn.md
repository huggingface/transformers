<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-04-14 and added to Hugging Face Transformers on 2022-09-22 and contributed by [sayakpaul](https://huggingface.co/sayakpaul).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# ViTMSN

[Masked Siamese Networks (MSN)](https://huggingface.co/papers/2204.07141) is a self-supervised learning framework designed to learn image representations by matching the representation of an image with randomly masked patches to that of the original unmasked image. This approach enhances scalability, especially for Vision Transformers, by processing only the unmasked patches. MSN achieves high semantic-level representations, demonstrating competitive performance in low-shot image classification. On ImageNet-1K, the base MSN model reaches 72.4% top-1 accuracy with 5,000 annotated images and 75.7% top-1 accuracy with just 1% of the labels, establishing a new state-of-the-art in self-supervised learning.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/vit-msn-small", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
model = AutoModelForImageClassification.from_pretrained("facebook/vit-msn-small", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## Usage tips

- MSN (masked siamese networks) is a method for self-supervised pretraining of Vision Transformers (ViTs). The pretraining objective matches the prototypes assigned to unmasked views of images to those of masked views of the same images.
- The authors released only pretrained weights of the backbone (ImageNet-1k pretraining). To use these on your own image classification dataset, use the [`ViTMSNForImageClassification`] class, which initializes from [`ViTMSNModel`]. Follow this [notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) for a detailed fine-tuning tutorial.
- MSN excels in low-shot and extreme low-shot regimes. It achieves 75.7% top-1 accuracy with only 1% of ImageNet-1K labels when fine-tuned.

## ViTMSNConfig

[[autodoc]] ViTMSNConfig

## ViTMSNModel

[[autodoc]] ViTMSNModel
    - forward

## ViTMSNForImageClassification

[[autodoc]] ViTMSNForImageClassification
    - forward

