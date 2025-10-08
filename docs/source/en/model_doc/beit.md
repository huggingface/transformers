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
*This model was released on 2021-06-15 and added to Hugging Face Transformers on 2021-08-04 and contributed by [nielsr](https://huggingface.co/nielsr).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# BEiT

[BEiT: BERT Pre-Training of Image Transformers](https://huggingface.co/papers/2106.08254) introduces a self-supervised vision representation model inspired by BERT. BEiT pre-trains Vision Transformers by predicting visual tokens from masked image patches. This approach outperforms supervised pre-training methods. Experiments show that BEiT achieves competitive results on image classification and semantic segmentation, with a base-size model reaching 83.2% top-1 accuracy on ImageNet-1K, surpassing DeiT trained from scratch. A large-size BEiT model achieves 86.3% on ImageNet-1K, even outperforming a ViT-L model pre-trained on ImageNet-22K.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="microsoft/beit-base-patch16-224-pt22k", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
model = AutoModelForImageClassification.from_pretrained("microsoft/beit-base-patch16-224-pt22k", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## BEiT specific outputs

[[autodoc]] models.beit.modeling_beit.BeitModelOutputWithPooling

## BeitConfig

[[autodoc]] BeitConfig

## BeitImageProcessor

[[autodoc]] BeitImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## BeitImageProcessorFast

[[autodoc]] BeitImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation

## BeitModel

[[autodoc]] BeitModel
    - forward

## BeitForMaskedImageModeling

[[autodoc]] BeitForMaskedImageModeling
    - forward

## BeitForImageClassification

[[autodoc]] BeitForImageClassification
    - forward

## BeitForSemanticSegmentation

[[autodoc]] BeitForSemanticSegmentation
    - forward

