<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2022-04-26 and added to Hugging Face Transformers on 2025-01-08 and contributed by [nielsr](https://huggingface.co/nielsr) and [SangbumChoi](https://github.com/SangbumChoi).*



```py
from transformers import AutoModelForPoseEstimation, AutoTokenizer

model = AutoModelForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")
tokenizer = AutoTokenizer.from_pretrained("usyd-community/vitpose-base-simple")

inputs = tokenizer("Input data", return_tensors="pt")
outputs = model(**inputs)
```

# ViTPose

[ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation](https://huggingface.co/papers/2204.12484) employs a standard, non-hierarchical Vision Transformer as a backbone for keypoint estimation, adding a simple decoder to predict heatmaps. Despite its simplicity, ViTPose achieves state-of-the-art results on the MS COCO Keypoint Detection benchmark. The model can be scaled from 100M to 1B parameters, demonstrating scalability and high parallelism. It is flexible in terms of attention type, input resolution, pre-training, and fine-tuning strategies, and can handle multiple pose tasks. Knowledge transfer between large and small models is facilitated via a knowledge token.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vitpose-architecture.png"
alt="drawing" width="600"/>

<small> ViTPose architecture. Taken from the <a href="https://huggingface.co/papers/2204.12484">original paper.</a> </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr) and [sangbumchoi](https://github.com/SangbumChoi).
The original code can be found [here](https://github.com/ViTAE-Transformer/ViTPose).

## Usage Tips

ViTPose is a so-called top-down keypoint detection model. This means that one first uses an object detector, like [RT-DETR](rt_detr.md), to detect people (or other instances) in an image. Next, ViTPose takes the cropped images as input and predicts the keypoints for each of them.

```py
import torch
import requests
import numpy as np

from PIL import Image

from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

device = "cuda" if torch.cuda.is_available() else "cpu"

url = "http://images.cocodataset.org/val2017/000000000139.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# ------------------------------------------------------------------------
# Stage 1. Detect humans on the image
# ------------------------------------------------------------------------

# You can choose any detector of your choice
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

inputs = person_image_processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = person_model(**inputs)

results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
)
result = results[0]  # take first image results

# Human label refers 0 index in COCO dataset
person_boxes = result["boxes"][result["labels"] == 0]
person_boxes = person_boxes.cpu().numpy()

# Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

# ------------------------------------------------------------------------
# Stage 2. Detect keypoints for each person found
# ------------------------------------------------------------------------

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
image_pose_result = pose_results[0]  # results for first image
```

### ViTPose++ models

The best [checkpoints](https://huggingface.co/collections/usyd-community/vitpose-677fcfd0a0b2b5c8f79c4335) are those of the [ViTPose++ paper](https://huggingface.co/papers/2212.04246). ViTPose++ models employ a so-called [Mixture-of-Experts (MoE)](https://huggingface.co/blog/moe) architecture for the ViT backbone, resulting in better performance.

The ViTPose+ checkpoints use 6 experts, hence 6 different dataset indices can be passed. 
An overview of the various dataset indices is provided below:

- 0: [COCO validation 2017](https://cocodataset.org/#overview) dataset, using an object detector that gets 56 AP on the "person" class
- 1: [AiC](https://github.com/fabbrimatteo/AiC-Dataset) dataset
- 2: [MPII](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset) dataset
- 3: [AP-10K](https://github.com/AlexTheBad/AP-10K) dataset
- 4: [APT-36K](https://github.com/pandorgan/APT-36K) dataset
- 5: [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) dataset

Pass the `dataset_index` argument in the forward of the model to indicate which experts to use for each example in the batch. Example usage is shown below:

```python
image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-base")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-base", device=device)

inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

dataset_index = torch.tensor([0], device=device) # must be a tensor of shape (batch_size,)

with torch.no_grad():
    outputs = model(**inputs, dataset_index=dataset_index)
```

The ViTPose+ checkpoints use 6 experts, hence 6 different dataset indices can be passed. 
An overview of the various dataset indices is provided below:

- 0: [COCO validation 2017](https://cocodataset.org/#overview) dataset, using an object detector that gets 56 AP on the "person" class
- 1: [AiC](https://github.com/fabbrimatteo/AiC-Dataset) dataset
- 2: [MPII](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset) dataset
- 3: [AP-10K](https://github.com/AlexTheBad/AP-10K) dataset
- 4: [APT-36K](https://github.com/pandorgan/APT-36K) dataset
- 5: [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody) dataset

### Visualization

To visualize the various keypoints, one can either leverage the `supervision` [library](https://github.com/roboflow/supervision (requires `pip install supervision`):

```python
import supervision as sv

xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

key_points = sv.KeyPoints(
    xy=xy, confidence=scores
)

edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=1
)
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.RED,
    radius=2
)
annotated_frame = edge_annotator.annotate(
    scene=image.copy(),
    key_points=key_points
)
annotated_frame = vertex_annotator.annotate(
    scene=annotated_frame,
    key_points=key_points
)
```

Alternatively, one can also visualize the keypoints using [OpenCV](https://opencv.org/) (requires `pip install opencv-python`):

```python
import math
import cv2

def draw_points(image, keypoints, scores, pose_keypoint_color, keypoint_score_threshold, radius, show_keypoint_weight):
    if pose_keypoint_color is not None:
        assert len(pose_keypoint_color) == len(keypoints)
    for kid, (kpt, kpt_score) in enumerate(zip(keypoints, scores)):
        x_coord, y_coord = int(kpt[0]), int(kpt[1])
        if kpt_score > keypoint_score_threshold:
            color = tuple(int(c) for c in pose_keypoint_color[kid])
            if show_keypoint_weight:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)
                transparency = max(0, min(1, kpt_score))
                cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
            else:
                cv2.circle(image, (int(x_coord), int(y_coord)), radius, color, -1)

def draw_links(image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold, thickness, show_keypoint_weight, stick_width = 2):
    height, width, _ = image.shape
    if keypoint_edges is not None and link_colors is not None:
        assert len(link_colors) == len(keypoint_edges)
        for sk_id, sk in enumerate(keypoint_edges):
            x1, y1, score1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]), scores[sk[0]])
            x2, y2, score2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]), scores[sk[1]])
            if (
                x1 > 0
                and x1 < width
                and y1 > 0
                and y1 < height
                and x2 > 0
                and x2 < width
                and y2 > 0
                and y2 < height
                and score1 > keypoint_score_threshold
                and score2 > keypoint_score_threshold
            ):
                color = tuple(int(c) for c in link_colors[sk_id])
                if show_keypoint_weight:
                    X = (x1, x2)
                    Y = (y1, y2)
                    mean_x = np.mean(X)
                    mean_y = np.mean(Y)
                    length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    polygon = cv2.ellipse2Poly(
                        (int(mean_x), int(mean_y)), (int(length / 2), int(stick_width)), int(angle), 0, 360, 1
                    )
                    cv2.fillConvexPoly(image, polygon, color)
                    transparency = max(0, min(1, 0.5 * (keypoints[sk[0], 2] + keypoints[sk[1], 2])))
                    cv2.addWeighted(image, transparency, image, 1 - transparency, 0, dst=image)
                else:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=thickness)

# Note: keypoint_edges and color palette are dataset-specific
keypoint_edges = model.config.edges

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

link_colors = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
keypoint_colors = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]

numpy_image = np.array(image)

for pose_result in image_pose_result:
    scores = np.array(pose_result["scores"])
    keypoints = np.array(pose_result["keypoints"])

    # draw each point on image
    draw_points(numpy_image, keypoints, scores, keypoint_colors, keypoint_score_threshold=0.3, radius=4, show_keypoint_weight=False)

    # draw links
    draw_links(numpy_image, keypoints, scores, keypoint_edges, link_colors, keypoint_score_threshold=0.3, thickness=1, show_keypoint_weight=False)

pose_image = Image.fromarray(numpy_image)
pose_image
```
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/vitpose-coco.jpg" alt="drawing" width="600"/>

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ViTPose. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

- A demo of ViTPose on images and video can be found [here](https://huggingface.co/spaces/hysts/ViTPose-transformers).
- A notebook illustrating inference and visualization can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/ViTPose/Inference_with_ViTPose_for_human_pose_estimation.ipynb).

## VitPoseImageProcessor

[[autodoc]] VitPoseImageProcessor
    - preprocess
    - post_process_pose_estimation

## VitPoseConfig

[[autodoc]] VitPoseConfig

## VitPoseForPoseEstimation

[[autodoc]] VitPoseForPoseEstimation
    - forward

