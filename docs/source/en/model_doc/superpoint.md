<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the MIT License; you may not use this file except in compliance with
the License.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2017-12-20 and added to Hugging Face Transformers on 2024-03-19 and contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).*

# SuperPoint

[SuperPoint: Self-Supervised Interest Point Detection and Description](https://huggingface.co/papers/1712.07629) is a fully-convolutional network trained self-supervised for detecting and describing interest points in images. It computes interest point locations and descriptors in a single forward pass. The model uses Homographic Adaptation, a multi-scale, multi-homography technique, to enhance detection repeatability and enable cross-domain adaptation. Trained on MS-COCO, SuperPoint outperforms traditional detectors and achieves state-of-the-art results in homography estimation on the HPatches dataset.

<hfoptions id="usage">
<hfoption id="AutoModel">

```py
import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint", dtype="auto")

inputs = processor(image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

image_size = (image.height, image.width)
processed_outputs = processor.post_process_keypoint_detection(outputs, [image_size])

plt.axis("off")
plt.imshow(image)
plt.scatter(
    processed_outputs[0]["keypoints"][:, 0],
    processed_outputs[0]["keypoints"][:, 1],
    c=processed_outputs[0]["scores"] * 100,
    s=processed_outputs[0]["scores"] * 50,
    alpha=0.8
)
```

</hfoption>
</hfoptions>

## SuperPointConfig

[[autodoc]] SuperPointConfig

## SuperPointImageProcessor

[[autodoc]] SuperPointImageProcessor

- preprocess
- post_process_keypoint_detection

## SuperPointImageProcessorFast

[[autodoc]] SuperPointImageProcessorFast
    - preprocess
    - post_process_keypoint_detection

## SuperPointForKeypointDetection

[[autodoc]] SuperPointForKeypointDetection

- forward

