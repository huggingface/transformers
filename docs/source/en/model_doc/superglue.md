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

The SuperGlue model was proposed in [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763) by Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz and Andrew Rabinovich.

This model consists of matching two sets of interest points detected in an image. Paired with the 
[SuperPoint model](https://huggingface.co/magic-leap-community/superpoint), it can be used to match two images and 
estimate the pose between them. This model is useful for tasks such as image matching, homography estimation, etc.

The abstract from the paper is the following:

*This paper introduces SuperGlue, a neural network that matches two sets of local features by jointly finding correspondences 
and rejecting non-matchable points. Assignments are estimated by solving a differentiable optimal transport problem, whose costs 
are predicted by a graph neural network. We introduce a flexible context aggregation mechanism based on attention, enabling 
SuperGlue to reason about the underlying 3D scene and feature assignments jointly. Compared to traditional, hand-designed heuristics, 
our technique learns priors over geometric transformations and regularities of the 3D world through end-to-end training from image 
pairs. SuperGlue outperforms other learned approaches and achieves state-of-the-art results on the task of pose estimation in 
challenging real-world indoor and outdoor environments. The proposed method performs matching in real-time on a modern GPU and 
can be readily integrated into modern SfM or SLAM systems. The code and trained weights are publicly available at this [URL](https://github.com/magicleap/SuperGluePretrainedNetwork).*

## How to use

Here is a quick example of using the model. Since this model is an image matching model, it requires pairs of images to be matched. 
The raw outputs contain the list of keypoints detected by the keypoint detector as well as the list of matches with their corresponding 
matching scores.
```python
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import requests

url_image1 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg"
image1 = Image.open(requests.get(url_image1, stream=True).raw)
url_image2 = "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg"
image_2 = Image.open(requests.get(url_image2, stream=True).raw)

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superglue_outdoor")
model = AutoModel.from_pretrained("magic-leap-community/superglue_outdoor")

inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

You can use the `post_process_keypoint_matching` method from the `SuperGlueImageProcessor` to get the keypoints and matches in a more readable format:

```python
image_sizes = [[(image.height, image.width) for image in images]]
outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
for i, output in enumerate(outputs):
    print("For the image pair", i)
    for keypoint0, keypoint1, matching_score in zip(
            output["keypoints0"], output["keypoints1"], output["matching_scores"]
    ):
        print(
            f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}."
        )

```

From the outputs, you can visualize the matches between the two images using the following code:
```python
import matplotlib.pyplot as plt
import numpy as np

# Create side by side image
merged_image = np.zeros((max(image1.height, image2.height), image1.width + image2.width, 3))
merged_image[: image1.height, : image1.width] = np.array(image1) / 255.0
merged_image[: image2.height, image1.width :] = np.array(image2) / 255.0
plt.imshow(merged_image)
plt.axis("off")

# Retrieve the keypoints and matches
output = outputs[0]
keypoints0 = output["keypoints0"]
keypoints1 = output["keypoints1"]
matching_scores = output["matching_scores"]
keypoints0_x, keypoints0_y = keypoints0[:, 0].numpy(), keypoints0[:, 1].numpy()
keypoints1_x, keypoints1_y = keypoints1[:, 0].numpy(), keypoints1[:, 1].numpy()

# Plot the matches
for keypoint0_x, keypoint0_y, keypoint1_x, keypoint1_y, matching_score in zip(
        keypoints0_x, keypoints0_y, keypoints1_x, keypoints1_y, matching_scores
):
    plt.plot(
        [keypoint0_x, keypoint1_x + image1.width],
        [keypoint0_y, keypoint1_y],
        color=plt.get_cmap("RdYlGn")(matching_score.item()),
        alpha=0.9,
        linewidth=0.5,
    )
    plt.scatter(keypoint0_x, keypoint0_y, c="black", s=2)
    plt.scatter(keypoint1_x + image1.width, keypoint1_y, c="black", s=2)

# Save the plot
plt.savefig("matched_image.png", dpi=300, bbox_inches='tight')
plt.close()
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/01ZYaLB1NL5XdA8u7yCo4.png)

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/magicleap/SuperGluePretrainedNetwork).

## SuperGlueConfig

[[autodoc]] SuperGlueConfig

## SuperGlueImageProcessor

[[autodoc]] SuperGlueImageProcessor

- preprocess

## SuperGlueForKeypointMatching

[[autodoc]] SuperGlueForKeypointMatching

- forward
- post_process_keypoint_matching