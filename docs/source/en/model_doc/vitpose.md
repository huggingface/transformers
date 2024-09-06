<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# VitPose

## Overview

The VitPose model was proposed in [ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation](https://arxiv.org/abs/2204.12484) by Yufei Xu, Jing Zhang, Qiming Zhang, Dacheng Tao. VitPose employs a standard, non-hierarchical [Vision Transformer](https://arxiv.org/pdf/2010.11929v2) as backbone for the task of keypoint estimation. A simple decoder head is added on top to predict the heatmaps from a given image. Despite its simplicity, the model gets state-of-the-art results on the challenging MS COCO Keypoint Detection benchmark.

The abstract from the paper is the following:

*Although no specific domain knowledge is considered in the design, plain vision transformers have shown excellent performance in visual recognition tasks. However, little effort has been made to reveal the potential of such simple structures for pose estimation tasks. In this paper, we show the surprisingly good capabilities of plain vision transformers for pose estimation from various aspects, namely simplicity in model structure, scalability in model size, flexibility in training paradigm, and transferability of knowledge between models, through a simple baseline model called ViTPose. Specifically, ViTPose employs plain and non-hierarchical vision transformers as backbones to extract features for a given person instance and a lightweight decoder for pose estimation. It can be scaled up from 100M to 1B parameters by taking the advantages of the scalable model capacity and high parallelism of transformers, setting a new Pareto front between throughput and performance. Besides, ViTPose is very flexible regarding the attention type, input resolution, pre-training and finetuning strategy, as well as dealing with multiple pose tasks. We also empirically demonstrate that the knowledge of large ViTPose models can be easily transferred to small ones via a simple knowledge token. Experimental results show that our basic ViTPose model outperforms representative methods on the challenging MS COCO Keypoint Detection benchmark, while the largest model sets a new state-of-the-art.*


This model was contributed by [nielsr](https://huggingface.co/nielsr) and [sangbumchoi](https://github.com/SangbumChoi).
The original code can be found [here](https://github.com/ViTAE-Transformer/ViTPose).

## Usage Tips

- To enable MoE (Mixture of Experts) function in the backbone, the user has to give appropriate input indices to the backbone model. 
  However, it is not used in default parameters.
- The current model utilizes a 2-step inference pipeline. The first step involves placing a bounding box around the region corresponding to the person.
  After that, the second step uses VitPose to predict the keypoints.

```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import VitPoseImageProcessor, VitPoseForPoseEstimation

>>> url = 'http://images.cocodataset.org/val2017/000000000139.jpg' 
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = VitPoseImageProcessor.from_pretrained("nielsr/vitpose-base-simple")
>>> model = VitPoseForPoseEstimation.from_pretrained("nielsr/vitpose-base-simple")

>>> boxes = [[[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]]]

>>> pixel_values = image_processor(image, boxes=boxes, return_tensors="pt").pixel_values

>>> with torch.no_grad():
...     outputs = model(pixel_values)

>>> pose_results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)[0]

>>> for pose_result in pose_results:
...     for keypoint in pose_result['keypoints']:
...         x, y, score = keypoint
...         print(f"coordinate : [{x}, {y}], score : {score}")
```
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vitpose-coco.jpg" alt="drawing" width="600"/>


## VitPoseImageProcessor

[[autodoc]] VitPoseImageProcessor
    - preprocess

## VitPoseConfig

[[autodoc]] VitPoseConfig

## VitPoseForPoseEstimation

[[autodoc]] VitPoseForPoseEstimation
    - forward