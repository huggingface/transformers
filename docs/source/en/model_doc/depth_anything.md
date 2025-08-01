<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# Depth Anything

[Depth Anything](https://huggingface.co/papers/2401.10891) is designed to be a foundation model for monocular depth estimation (MDE). It is jointly trained on labeled and ~62M unlabeled images to enhance the dataset. It uses a pretrained [DINOv2](./dinov2) model as an image encoder to inherit its existing rich semantic priors, and [DPT](./dpt) as the decoder. A teacher model is trained on unlabeled images to create pseudo-labels. The student model is trained on a combination of the pseudo-labels and labeled images. To improve the student model's performance, strong perturbations are added to the unlabeled images to challenge the student model to learn more visual knowledge from the image.

You can find all the original Depth Anything checkpoints under the [Depth Anything](https://huggingface.co/collections/LiheYoung/depth-anything-release-65b317de04eec72abf6b55aa) collection.

> [!TIP]
> Click on the Depth Anything models in the right sidebar for more examples of how to apply Depth Anything to different vision tasks.

The example below demonstrates how to obtain a depth map with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", dtype=torch.bfloat16, device=0)
pipe("http://images.cocodataset.org/val2017/000000039769.jpg")["depth"]
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-base-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-base-hf", dtype=torch.bfloat16)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)
predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.detach().cpu().numpy() * 255
Image.fromarray(depth.astype("uint8"))
```

</hfoption>
</hfoptions>

## Notes

- [DepthAnythingV2](./depth_anything_v2), released in June 2024, uses the same architecture as Depth Anything and is compatible with all code examples and existing workflows. It uses synthetic data and a larger capacity teacher model to achieve much finer and robust depth predictions.

## DepthAnythingConfig

[[autodoc]] DepthAnythingConfig

## DepthAnythingForDepthEstimation

[[autodoc]] DepthAnythingForDepthEstimation
    - forward