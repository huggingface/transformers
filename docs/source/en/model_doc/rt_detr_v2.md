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

# RT-DETRv2

## Overview

The RT-DETRv2 model was proposed in [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140) by Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu.

RT-DETRv2 refines RT-DETR by introducing selective multi-scale feature extraction, a discrete sampling operator for broader deployment compatibility, and improved training strategies like dynamic data augmentation and scale-adaptive hyperparameters. These changes enhance flexibility and practicality while maintaining real-time performance.

The abstract from the paper is the following:

*In this report, we present RT-DETRv2, an improved Real-Time DEtection TRansformer (RT-DETR). RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance. To improve the flexibility, we suggest setting a distinct number of sampling points for features at different scales in the deformable attention to achieve selective multi-scale feature extraction by the decoder. To enhance practicality, we propose an optional discrete sampling operator to replace the grid_sample operator that is specific to RT-DETR compared to YOLOs. This removes the deployment constraints typically associated with DETRs. For the training strategy, we propose dynamic data augmentation and scale-adaptive hyperparameters customization to improve performance without loss of speed.*

This model was contributed by [jadechoghari](https://huggingface.co/jadechoghari).
The original code can be found [here](https://github.com/lyuwenyu/RT-DETR).

## Usage tips 

This second version of RT-DETR improves how the decoder finds objects in an image. 

- **better sampling** – adjusts offsets so the model looks at the right areas
- **flexible attention** – can use smooth (bilinear) or fixed (discrete) sampling
- **optimized processing** – improves how attention weights mix information

```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = RTDetrImageProcessor.from_pretrained("jadechoghari/rtdetr_v2_r18vd")
>>> model = RTDetrV2ForObjectDetection.from_pretrained("jadechoghari/rtdetr_v2_r18vd")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

>>> for result in results:
...     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
...         score, label = score.item(), label_id.item()
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
cat: 0.97 [341.14, 25.11, 639.98, 372.89]
cat: 0.96 [12.78, 56.35, 317.67, 471.34]
remote: 0.95 [39.96, 73.12, 175.65, 117.44]
sofa: 0.86 [-0.11, 2.97, 639.89, 473.62]
sofa: 0.82 [-0.12, 1.78, 639.87, 473.52]
remote: 0.79 [333.65, 76.38, 370.69, 187.48]
```

## RTDetrV2Config

[[autodoc]] RTDetrV2Config


## RTDetrV2Model

[[autodoc]] RTDetrV2Model
    - forward
 
## RTDetrV2ForObjectDetection

[[autodoc]] RTDetrV2ForObjectDetection
    - forward
