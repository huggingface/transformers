<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ViTPose

## Overview

The ViTPose model was proposed in [ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation](https://arxiv.org/abs/2204.12484) by Yufei Xu, Jing Zhang, Qiming Zhang, Dacheng Tao. ViTPose employs a standard, non-hierarchical [Vision Transformer](vit) as backbone for the task of keypoint estimation. A simple decoder head is added on top to predict the heatmaps from a given image. Despite its simplicity, the model gets state-of-the-art results on the challenging MS COCO Keypoint Detection benchmark.

The abstract from the paper is the following:

*Although no specific domain knowledge is considered in the design, plain vision transformers have shown excellent performance in visual recognition tasks. However, little effort has been made to reveal the potential of such simple structures for pose estimation tasks. In this paper, we show the surprisingly good capabilities of plain vision transformers for pose estimation from various aspects, namely simplicity in model structure, scalability in model size, flexibility in training paradigm, and transferability of knowledge between models, through a simple baseline model called ViTPose. Specifically, ViTPose employs plain and non-hierarchical vision transformers as backbones to extract features for a given person instance and a lightweight decoder for pose estimation. It can be scaled up from 100M to 1B parameters by taking the advantages of the scalable model capacity and high parallelism of transformers, setting a new Pareto front between throughput and performance. Besides, ViTPose is very flexible regarding the attention type, input resolution, pre-training and finetuning strategy, as well as dealing with multiple pose tasks. We also empirically demonstrate that the knowledge of large ViTPose models can be easily transferred to small ones via a simple knowledge token. Experimental results show that our basic ViTPose model outperforms representative methods on the challenging MS COCO Keypoint Detection benchmark, while the largest model sets a new state-of-the-art.*


This model was contributed by [nielsr](https://huggingface.co/nielsr) and [sangbumchoi](https://github.com/SangbumChoi).
The original code can be found [here](https://github.com/ViTAE-Transformer/ViTPose).

## Usage Tips

The current model utilizes a 2-step inference pipeline. The first step involves placing a bounding box around the region corresponding to the person. After that, the second step uses ViTPose to predict the keypoints.

```py
>>> import torch
>>> import requests

>>> from PIL import Image
>>> from transformers import ViTPoseImageProcessor, ViTPoseForPoseEstimation

>>> url = 'http://images.cocodataset.org/val2017/000000000139.jpg' 
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = ViTPoseImageProcessor.from_pretrained("nielsr/vitpose-base-simple")
>>> model = ViTPoseForPoseEstimation.from_pretrained("nielsr/vitpose-base-simple")

>>> boxes = [[[412.8, 157.61, 53.05, 138.01], [384.43, 172.21, 15.12, 35.74]]]

>>> pixel_values = image_processor(image, boxes=boxes, return_tensors="pt").pixel_values

>>> with torch.no_grad():
...     outputs = model(pixel_values)

>>> pose_results = image_processor.post_process_pose_estimation(outputs, boxes=boxes[0])

>>> for pose_result in pose_results:
...     for keypoint in pose_result['keypoints']:
...         x, y, score = keypoint
...         print(f"coordinate : [{x}, {y}], score : {score}")
coordinate : [428.25335693359375, 170.24496459960938], score : 0.8717536330223083
coordinate : [429.13037109375, 167.39605712890625], score : 0.8820509910583496
coordinate : [428.23681640625, 167.72825622558594], score : 0.7663289308547974
coordinate : [433.1866455078125, 167.2566680908203], score : 0.933370053768158
coordinate : [440.34075927734375, 166.58522033691406], score : 0.8911094069480896
coordinate : [439.90283203125, 177.54049682617188], score : 0.9118685722351074
coordinate : [445.50372314453125, 178.04055786132812], score : 0.751734733581543
coordinate : [436.45819091796875, 199.42474365234375], score : 0.8745120167732239
coordinate : [433.68255615234375, 200.17333984375], score : 0.5155676603317261
coordinate : [430.5008544921875, 218.7760009765625], score : 0.8757728338241577
coordinate : [420.5921630859375, 213.15621948242188], score : 0.9036439657211304
coordinate : [445.17218017578125, 222.87921142578125], score : 0.8029380440711975
coordinate : [452.07672119140625, 222.17730712890625], score : 0.8517846465110779
coordinate : [441.92657470703125, 255.0374755859375], score : 0.8607744574546814
coordinate : [451.2308349609375, 254.36398315429688], score : 0.8495950698852539
coordinate : [443.9051513671875, 287.5822448730469], score : 0.703719437122345
coordinate : [455.88482666015625, 285.6434631347656], score : 0.8391701579093933
```


## ViTPoseImageProcessor

[[autodoc]] ViTPoseImageProcessor
    - preprocess

## ViTPoseConfig

[[autodoc]] ViTPoseConfig

## ViTPoseForPoseEstimation

[[autodoc]] ViTPoseForPoseEstimation
    - forward