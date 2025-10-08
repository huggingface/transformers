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
*This model was released on 2024-01-19 and added to Hugging Face Transformers on 2024-01-25 and contributed by [nielsr](https://huggingface.co/nielsr).*
# Depth Anything

[Depth Anything](https://huggingface.co/papers/2401.10891) is a robust monocular depth estimation model based on the DPT architecture. Trained on approximately 62 million images, it achieves state-of-the-art results in both relative and absolute depth estimation. The model leverages large-scale unlabeled data, enhanced by data augmentation and auxiliary supervision from pre-trained encoders, to improve generalization and robustness. Extensive zero-shot evaluations on six public datasets and random photos demonstrate its impressive capabilities, and fine-tuning with metric depth information from NYUv2 and KITTI sets new benchmarks. Additionally, the improved depth model enhances depth-conditioned ControlNet performance.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", dtype="auto")
pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
```

</hfoption>
<hfoption id="AutoModel">

```python
import torch
import requests
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-base-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-base-hf", dtype="auto")
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
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

## DepthAnythingConfig

[[autodoc]] DepthAnythingConfig

## DepthAnythingForDepthEstimation

[[autodoc]] DepthAnythingForDepthEstimation
    - forward

