<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.


-->

# SuperGlue

## Overview

The SuperGlue model was proposed
in [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763) by Paul-Edouard Sarlin, Daniel
DeTone, Tomasz Malisiewicz and Andrew Rabinovich.

 TODO: Add a description of the model

[//]: # (This model is the result of a self-supervised training of a fully-convolutional network for interest point detection and)

[//]: # (description. The model is able to detect interest points that are repeatable under homographic transformations and)

[//]: # (provide a descriptor for each point. The use of the model in its own is limited, but it can be used as a feature)

[//]: # (extractor for other tasks such as homography estimation, image matching, etc.)

The abstract from the paper is the following:

*This paper introduces SuperGlue, a neural network that matches two sets of local features by jointly finding correspondences and rejecting non-matchable points. Assignments are estimated by solving a differentiable optimal transport problem, whose costs are predicted by a graph neural network. We introduce a flexible context aggregation mechanism based on attention, enabling SuperGlue to reason about the underlying 3D scene and feature assignments jointly. Compared to traditional, hand-designed heuristics, our technique learns priors over geometric transformations and regularities of the 3D world through end-to-end training from image pairs. SuperGlue outperforms other learned approaches and achieves state-of-the-art results on the task of pose estimation in challenging real-world indoor and outdoor environments. The proposed method performs matching in real-time on a modern GPU and can be readily integrated into modern SfM or SLAM systems. The code and trained weights are publicly available at this [URL](https://github.com/magicleap/SuperGluePretrainedNetwork).*

## How to use

Here is a quick example of using the model to detect interest points in an image:

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = AutoModel.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
```

The outputs contain the list of keypoint coordinates with their respective score and description (a 256-long vector).

You can also feed multiple images to the model. Due to the nature of SuperPoint, to output a dynamic number of keypoints,
you will need to use the mask attribute to retrieve the respective information :

```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url_image_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
url_image_2 = "http://images.cocodataset.org/test-stuff2017/000000000568.jpg"
image_2 = Image.open(requests.get(url_image_2, stream=True).raw)

images = [image_1, image_2]

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = AutoModel.from_pretrained("magic-leap-community/superpoint")

inputs = processor(images, return_tensors="pt")
outputs = model(**inputs)

for i in range(len(images)):
    image_mask = outputs.mask[i]
    image_indices = torch.nonzero(image_mask).squeeze()
    image_keypoints = outputs.keypoints[i][image_indices]
    image_scores = outputs.scores[i][image_indices]
    image_descriptors = outputs.descriptors[i][image_indices]
```

You can then print the keypoints on the image to visualize the result :
```python
import cv2
for keypoint, score in zip(image_keypoints, image_scores):
    keypoint_x, keypoint_y = int(keypoint[0].item()), int(keypoint[1].item())
    color = tuple([score.item() * 255] * 3)
    image = cv2.circle(image, (keypoint_x, keypoint_y), 2, color)
cv2.imwrite("output_image.png", image)
```

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/magicleap/SuperPointPretrainedNetwork).

## SuperGlueConfig

[[autodoc]] SuperGlueConfig

## SuperGlueImageProcessor

[[autodoc]] SuperGlueImageProcessor

- preprocess

## SuperGlueForImageMatching

[[autodoc]] SuperGlueForImageMatching

- forward
