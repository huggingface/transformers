<!--Copyright 2024 The HuggingFace Team. All rights reserved.

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

The RT-DETR model was proposed in [RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer
](https://arxiv.org/abs/2407.17140) by Wenyu Lv, Yian Zhao, Qinyao Chang, Kui Huang, Guanzhong Wang, Yi Liu.

The abstract from the paper is the following:

In this report, we present RT-DETRv2, an improved Real-Time DEtection TRansformer (RT-DETR). RT-DETRv2 builds upon the previous state-of-the-art real-time detector, RT-DETR, and opens up a set of bag-of-freebies for flexibility and practicality, as well as optimizing the training strategy to achieve enhanced performance. To improve the flexibility, we suggest setting a distinct number of sampling points for features at different scales in the deformable attention to achieve selective multi-scale feature extraction by the decoder. To enhance practicality, we propose an optional discrete sampling operator to replace the grid_sample operator that is specific to RT-DETR compared to YOLOs. This removes the deployment constraints typically associated with DETRs. For the training strategy, we propose dynamic data augmentation and scale-adaptive hyperparameters customization to improve performance without loss of speed. Source code and pre-trained models will be available at this [https URL](https://github.com/lyuwenyu/RT-DETR).

Tips:

The framework of RT-DETRv2 remains the same as RT-DETR, with only modifications to the deformable attention module of the decoder.

```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
>>> model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

>>> for result in results:
...     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
...         score, label = score.item(), label_id.item()
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
sofa: 0.97 [0.14, 0.38, 640.13, 476.21]
cat: 0.96 [343.38, 24.28, 640.14, 371.5]
cat: 0.96 [13.23, 54.18, 318.98, 472.22]
remote: 0.95 [40.11, 73.44, 175.96, 118.48]
remote: 0.92 [333.73, 76.58, 369.97, 186.99]
```

This model was contributed by [sangbumchoi](https://huggingface.co/danelcsb).
The original code can be found [here](https://github.com/lyuwenyu/RT-DETR).


## RTDetrV2Config

[[autodoc]] RTDetrV2Config

## RTDetrResNetConfig

[[autodoc]] RTDetrResNetConfig

## RTDetrModel

[[autodoc]] RTDetrModel
    - forward

## RTDetrForObjectDetection

[[autodoc]] RTDetrForObjectDetection
    - forward

## RTDetrResNetBackbone

[[autodoc]] RTDetrResNetBackbone
    - forward
