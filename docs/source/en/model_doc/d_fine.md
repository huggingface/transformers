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

# D-FINE

## Overview

The D-FINE model was proposed in [D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842) by
Yansong Peng, Hebei Li, Peixi Wu, Yueyi Zhang, Xiaoyan Sun, Feng Wu

The abstract from the paper is the following:

*We introduce D-FINE, a powerful real-time object detector that achieves outstanding localization precision by redefining the bounding box regression task in DETR models. D-FINE comprises two key components: Fine-grained Distribution Refinement (FDR) and Global Optimal Localization Self-Distillation (GO-LSD). 
FDR transforms the regression process from predicting fixed coordinates to iteratively refining probability distributions, providing a fine-grained intermediate representation that significantly enhances localization accuracy. GO-LSD is a bidirectional optimization strategy that transfers localization knowledge from refined distributions to shallower layers through self-distillation, while also simplifying the residual prediction tasks for deeper layers. Additionally, D-FINE incorporates lightweight optimizations in computationally intensive modules and operations, achieving a better balance between speed and accuracy. Specifically, D-FINE-L / X achieves 54.0% / 55.8% AP on the COCO dataset at 124 / 78 FPS on an NVIDIA T4 GPU. When pretrained on Objects365, D-FINE-L / X attains 57.1% / 59.3% AP, surpassing all existing real-time detectors. Furthermore, our method significantly enhances the performance of a wide range of DETR models by up to 5.3% AP with negligible extra parameters and training costs. Our code and pretrained models: this https URL.*

This model was contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber). 
The original code can be found [here](https://github.com/Peterande/D-FINE).

## Usage tips 

This D-FINE version of RT-DETR improves how the decoder finds objects in an image. 

```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import DFineForObjectDetection, RTDetrImageProcessor

>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = RTDetrImageProcessor.from_pretrained("vladislavbro/dfine_x_coco")
>>> model = DFineForObjectDetection.from_pretrained("vladislavbro/dfine_x_coco")

>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

>>> for result in results:
...     for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
...         score, label = score.item(), label_id.item()
...         box = [round(i, 2) for i in box.tolist()]
...         print(f"{model.config.id2label[label]}: {score:.2f} {box}")
cat: 0.96 [344.4865,  23.4047, 639.8372, 374.2650]
cat: 0.96 [11.7123,  53.5185, 316.6395, 472.3320]
remote: 0.95 [40.4605,  73.6995, 175.6157, 117.5686]
sofa: 0.92 [0.58968, 1.88410, 640.25000, 474.74000]
remote: 0.89 [333.4805,  77.0410, 370.7715, 187.2985]

## DFineConfig

[[autodoc]] DFineConfig

## DFineResNetConfig

[[autodoc]] DFineResNetConfig

## DFineModel

[[autodoc]] DFineModel
    - forward

## DFineModelForObjectDetection

[[autodoc]] DFineModelForObjectDetection
    - forward

## DFineResNetBackbone

[[autodoc]] DFineResNetBackbone
    - forward