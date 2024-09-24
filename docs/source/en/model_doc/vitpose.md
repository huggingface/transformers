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

- To enable MoE (Mixture of Experts) function in the backbone, user has to give appropriate configuration such as `num_experts` and input value `dataset_index` to the backbone model. 
  However, it is not used in default parameters. Below is the code snippet for usage of MoE function.
```py
>>> from transformers import VitPoseBackboneConfig, VitPoseBackbone
>>> import torch

>>> config = VitPoseBackboneConfig(num_experts=3, out_indices=[-1])
>>> model = VitPoseBackbone(config)

>>> pixel_values = torch.randn(3, 3, 256, 192)
>>> dataset_index = torch.tensor([1, 2, 3])
>>> outputs = model(pixel_values, dataset_index)
```

- ViTPose is a so-called top-down keypoint detection model. This means that one first uses an object detector, like [RT-DETR](rt-detr), to detect people (or other instances) in an image. Next, ViTPose takes the cropped images as input and predicts the keypoints.

```py
import math

import cv2
import numpy as np
import requests
import torch
from PIL import Image

from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    VitPoseForPoseEstimation,
    VitPoseImageProcessor,
)


url = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Stage 1. Run Object Detector
# User can replace this object_detector part
person_image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
inputs = person_image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3
)

def pascal_voc_to_coco(bboxes: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from the Pascal VOC format to the COCO format.

    In other words, converts from (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format
    to (top_left_x, top_left_y, width, height).

    Args:
        bboxes (`np.ndarray` of shape `(batch_size, 4)):
            Bounding boxes in Pascal VOC format.

    Returns:
        `np.ndarray` of shape `(batch_size, 4) in COCO format.
    """
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

    return bboxes

# 0 index indicates human label in COCO
boxes = results[0]["boxes"][results[0]["labels"] == 0]
boxes = [pascal_voc_to_coco(boxes.cpu().numpy())]

image_processor = VitPoseImageProcessor.from_pretrained("nielsr/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained("nielsr/vitpose-base-simple")

# Stage 2. Run ViTPose
pixel_values = image_processor(image, boxes=boxes, return_tensors="pt").pixel_values

with torch.no_grad():
    outputs = model(pixel_values)

pose_results = image_processor.post_process_pose_estimation(outputs, boxes=boxes)[0]

for pose_result in pose_results:
    for keypoint in pose_result["keypoints"]:
        x, y, score = keypoint
        print(f"coordinate : [{x}, {y}], score : {score}")

def draw_points(pose_kpt_color, kpts, img, kpt_score_thr, radius, show_keypoint_weight):
    if pose_kpt_color is not None:
        assert len(pose_kpt_color) == len(kpts)
    for kid, kpt in enumerate(kpts):
        x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
        if kpt_score > kpt_score_thr:
            color = tuple(int(c) for c in pose_kpt_color[kid])
            if show_keypoint_weight:
                cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(img, transparency, img, 1 - transparency, 0, dst=img)
            else:
                cv2.circle(img, (int(x_coord), int(y_coord)), radius, color, -1)

def draw_links(skeleton, pose_link_color, kpts, img, kpt_score_thr, thickness, show_keypoint_weight):
    img_h, img_w, _ = img.shape
    if skeleton is not None and pose_link_color is not None:
        assert len(pose_link_color) == len(skeleton)
        for sk_id, sk in enumerate(skeleton):
            pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
            pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
            if (
                pos1[0] > 0
                and pos1[0] < img_w
                and pos1[1] > 0
                and pos1[1] < img_h
                and pos2[0] > 0
                and pos2[0] < img_w
                and pos2[1] > 0
                and pos2[1] < img_h
                and kpts[sk[0], 2] > kpt_score_thr
                and kpts[sk[1], 2] > kpt_score_thr
            ):
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 1
                    )
                    cv2.fillConvexPoly(img, polygon, color)
                    transparency = max(0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(img, transparency, img, 1 - transparency, 0, dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

def visualize_keypoints(
    img,
    pose_result,
    skeleton=None,
    kpt_score_thr=0.3,
    pose_kpt_color=None,
    pose_link_color=None,
    radius=4,
    thickness=1,
    show_keypoint_weight=False,
):
    """Draw keypoints and links on an image.

    Args:
        img (`numpy.ndarray`): 
            The image to draw poses on. It will be modified in-place.
        pose_result (`List[numpy.ndarray]`): 
            The poses to draw. Each element is a set of K keypoints as a Kx3 numpy.ndarray, where each keypoint
            is represented as x, y, score.
        skeleton (`List[tuple]`, *optional*): 
            Skeleton definition.
        kpt_score_thr (`float`, *optional*, defaults to 0.3): 
            Minimum score of keypoints to be shown.
        pose_kpt_color (`numpy.ndarray`, *optional*): 
            Color of N keypoints. If None, the keypoints will not be drawn.
        pose_link_color (`numpy.ndarray`, *optional*): 
            Color of M links. If None, the links will not be drawn.
        radius (`int`, *optional*, defaults to 4):
            Radius of keypoint circles.
        thickness (`int`, *optional*, defaults to 1): 
            Thickness of lines.
        show_keypoint_weight (`bool`, *optional*, defaults to False): 
            Whether to adjust keypoint and link visibility based on the keypoint scores.
    
    Returns:
        `numpy.ndarray`: Image with drawn keypoints and links.
    """
    for kpts in pose_result:
        kpts = np.array(kpts, copy=False)

        # draw each point on image
        draw_points(pose_kpt_color, kpts, img, kpt_score_thr, radius, show_keypoint_weight)

        # draw links
        draw_links(skeleton, pose_link_color, kpts, img, kpt_score_thr, thickness, show_keypoint_weight)

    return img

# Note: skeleton and color palette are dataset-specific
skeleton = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]

palette = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ]
)

pose_link_color = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

pose_results = [result["keypoints"] for result in pose_results]

result = visualize_keypoints(
    np.array(image),
    pose_result,
    skeleton=skeleton,
    kpt_score_thr=0.3,
    pose_kpt_color=pose_kpt_color,
    pose_link_color=pose_link_color,
    radius=4,
    thickness=1,
    show_keypoint_weight=False,
)

pose_image = Image.fromarray(result)
pose_image
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