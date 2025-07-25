<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.


-->

# EfficientLoFTR

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The EfficientLoFTR model was proposed in [Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed](https://arxiv.org/abs/2403.04765) by Yifan Wang, Xingyi He, Sida Peng, Dongli Tan and Xiaowei Zhou.

This model consists of matching two images together by finding pixel correspondences. It can be used to estimate the pose between them. 
This model is useful for tasks such as image matching, homography estimation, etc.

The abstract from the paper is the following:

*We present a novel method for efficiently producing semidense matches across images. Previous detector-free matcher 
LoFTR has shown remarkable matching capability in handling large-viewpoint change and texture-poor scenarios but suffers
from low efficiency. We revisit its design choices and derive multiple improvements for both efficiency and accuracy. 
One key observation is that performing the transformer over the entire feature map is redundant due to shared local 
information, therefore we propose an aggregated attention mechanism with adaptive token selection for efficiency. 
Furthermore, we find spatial variance exists in LoFTR’s fine correlation module, which is adverse to matching accuracy. 
A novel two-stage correlation layer is proposed to achieve accurate subpixel correspondences for accuracy improvement. 
Our efficiency optimized model is ∼ 2.5× faster than LoFTR which can even surpass state-of-the-art efficient sparse 
matching pipeline SuperPoint + LightGlue. Moreover, extensive experiments show that our method can achieve higher 
accuracy compared with competitive semi-dense matchers, with considerable efficiency benefits. This opens up exciting 
prospects for large-scale or latency-sensitive applications such as image retrieval and 3D reconstruction. 
Project page: [https://zju3dv.github.io/efficientloftr/](https://zju3dv.github.io/efficientloftr/).*

## How to use

Here is a quick example of using the model. 
```python
import torch

from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from transformers.image_utils import load_image


image1 = load_image("https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg")
image2 = load_image("https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg")

images = [image1, image2]

processor = AutoImageProcessor.from_pretrained("stevenbucaille/efficientloftr")
model = AutoModelForKeypointMatching.from_pretrained("stevenbucaille/efficientloftr")

inputs = processor(images, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
```

You can use the `post_process_keypoint_matching` method from the `ImageProcessor` to get the keypoints and matches in a more readable format:

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

From the post processed outputs, you can visualize the matches between the two images using the following code:
```python
images_with_matching = processor.visualize_keypoint_matching(images, outputs)
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/632885ba1558dac67c440aa8/2nJZQlFToCYp_iLurvcZ4.png)

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/zju3dv/EfficientLoFTR).

## EfficientLoFTRConfig

[[autodoc]] EfficientLoFTRConfig

## EfficientLoFTRImageProcessor

[[autodoc]] EfficientLoFTRImageProcessor

- preprocess
- post_process_keypoint_matching
- visualize_keypoint_matching

## EfficientLoFTRModel

[[autodoc]] EfficientLoFTRModel

- forward

## EfficientLoFTRForKeypointMatching

[[autodoc]] EfficientLoFTRForKeypointMatching

- forward