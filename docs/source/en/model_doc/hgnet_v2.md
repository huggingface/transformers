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
*This model was released on 2024-07-01 and added to Hugging Face Transformers on 2025-04-29.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# HGNet-V2

[HGNetV2](https://github.com/PaddlePaddle/PaddleClas/blob/v2.6.0/docs/zh_CN/models/ImageNet1k/PP-HGNetV2.md) is a next-generation convolutional neural network (CNN) backbone built for optimal accuracy-latency tradeoff on NVIDIA GPUs. Building on the original[HGNet](https://github.com/PaddlePaddle/PaddleClas/blob/v2.6.0/docs/en/models/PP-HGNet_en.md), HGNetV2 delivers high accuracy at fast inference speeds and performs strongly on tasks like image classification, object detection, and segmentation, making it a practical choice for GPU-based computer vision applications.

You can find all the original HGNet V2 models under the [USTC](https://huggingface.co/ustc-community/models?search=hgnet) organization.

> [!TIP]
> This model was contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber).
> Click on the HGNet V2 models in the right sidebar for more examples of how to apply HGNet V2 to different computer vision tasks.

The example below demonstrates how to classify an image with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="image-classification",
    model="ustc-community/hgnet-v2",
    dtype=torch.float16,
    device=0
)
pipeline("http://images.cocodataset.org/val2017/000000039769.jpg")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from transformers import HGNetV2ForImageClassification, AutoImageProcessor
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

model = HGNetV2ForImageClassification.from_pretrained("ustc-community/hgnet-v2")
processor = AutoImageProcessor.from_pretrained("ustc-community/hgnet-v2")

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = logits.argmax(dim=-1).item()

class_labels = model.config.id2label
predicted_class_label = class_labels[predicted_class_id]
print(f"The predicted class label is: {predicted_class_label}")
```

</hfoption>
</hfoptions>

## HGNetV2Config

[[autodoc]] HGNetV2Config


## HGNetV2Backbone

[[autodoc]] HGNetV2Backbone
    - forward


## HGNetV2ForImageClassification

[[autodoc]] HGNetV2ForImageClassification
    - forward