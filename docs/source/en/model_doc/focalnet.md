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
*This model was released on 2022-03-22 and added to Hugging Face Transformers on 2023-04-23 and contributed by [nielsr](https://huggingface.co/nielsr).*

# FocalNet

[FocalNets](https://huggingface.co/papers/2203.11926) replaces self-attention with a focal modulation mechanism to model token interactions in vision. This mechanism includes hierarchical contextualization via depth-wise convolutions, gated aggregation, and element-wise modulation. Experiments demonstrate that FocalNets outperform self-attention models on image classification, object detection, and segmentation with similar computational costs. Specifically, FocalNets achieve 82.3% and 83.9% top-1 accuracy on ImageNet-1K for tiny and base sizes, respectively. After pretraining on ImageNet-22K, they reach 86.5% and 87.3% top-1 accuracy. In object detection with Mask R-CNN, FocalNet base outperforms Swin by 2.1 points. For semantic segmentation with UPerNet, FocalNet base surpasses Swin by 2.4 points at single-scale and 2.8 points at multi-scale. Using large FocalNet and Mask2former, 58.5 mIoU is achieved for ADE20K semantic segmentation, and 57.9 PQ for COCO Panoptic Segmentation. With huge FocalNet and DINO, 64.3 and 64.4 mAP are achieved on COCO minival and test-dev, respectively, setting new state-of-the-art results.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="microsoft/focalnet-tiny", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-tiny")
model = AutoModelForImageClassification.from_pretrained("microsoft/focalnet-tiny", dtype="auto")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])
```

</hfoption>
</hfoptions>

## FocalNetConfig

[[autodoc]] FocalNetConfig

## FocalNetModel

[[autodoc]] FocalNetModel
    - forward

## FocalNetForMaskedImageModeling

[[autodoc]] FocalNetForMaskedImageModeling
    - forward

## FocalNetForImageClassification

[[autodoc]] FocalNetForImageClassification
    - forward

