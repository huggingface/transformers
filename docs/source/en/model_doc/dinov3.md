<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2025-08-13 and added to Hugging Face Transformers on 2025-08-14.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# DINOv3

[DINOv3](https://huggingface.co/papers/2508.10104) is a self-supervised vision model designed to scale across large datasets and architectures without task-specific tailoring. It introduces Gram anchoring, a novel technique that prevents degradation of dense feature maps during long training, alongside post-hoc methods that improve flexibility across resolutions, model sizes, and text alignment. The model produces high-quality dense features, enabling strong performance on diverse vision tasks and surpassing both self-supervised and weakly supervised baselines. The authors also release a suite of DINOv3 models to support scalable deployment under varied computational and resource constraints.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="facebook/dinov3-vits16-pretrain-lvd1689m", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
model = AutoModelForImageClassification.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## DINOv3ViTConfig

[[autodoc]] DINOv3ViTConfig

## DINOv3ConvNextConfig

[[autodoc]] DINOv3ConvNextConfig

## DINOv3ViTModel

[[autodoc]] DINOv3ViTModel
    - forward

## DINOv3ViTBackbone    
[[autodoc]] DINOv3ViTBackbone

## DINOv3ConvNextModel

[[autodoc]] DINOv3ConvNextModel
    - forward

## DINOv3ViTImageProcessorFast

[[autodoc]] DINOv3ViTImageProcessorFast
    - preprocess

## DINOv3ConvNextBackbone

[[autodoc]] DINOv3ConvNextBackbone
    - forward
