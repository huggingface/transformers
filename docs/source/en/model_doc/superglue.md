<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-11-26 and added to Hugging Face Transformers on 2025-01-20 and contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).*

# SuperGlue

[SuperGlue](https://huggingface.co/papers/1911.11763) matches two sets of local features by jointly finding correspondences and rejecting non-matchable points using a differentiable optimal transport problem. It employs a graph neural network to predict costs and a flexible context aggregation mechanism based on attention to reason about the 3D scene and feature assignments. SuperGlue learns priors over geometric transformations and 3D world regularities through end-to-end training from image pairs, outperforming traditional heuristics and achieving state-of-the-art results in pose estimation. The model operates in real-time on modern GPUs and can be integrated into SfM or SLAM systems.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="keypoint-matching", model="magic-leap-community/superglue_outdoor", dtype="auto")
pipeline(["path/to/image1.png", "path/to/image2.png"])
```

</hfoption>
<hfoption id="AutoModel">

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

</hfoption>
</hfoptions>

## SuperGlueConfig

[[autodoc]] SuperGlueConfig

## SuperGlueImageProcessor

[[autodoc]] SuperGlueImageProcessor

- preprocess

## SuperGlueImageProcessorFast

[[autodoc]] SuperGlueImageProcessorFast
    - preprocess
    - post_process_keypoint_matching
    - visualize_keypoint_matching

## SuperGlueForKeypointMatching

[[autodoc]] SuperGlueForKeypointMatching

- forward
- post_process_keypoint_matching

