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
*This model was released on 2024-12-18 and added to Hugging Face Transformers on 2025-03-21.*

# Prompt Depth Anything

[Prompt Depth Anything](https://huggingface.co/papers/2412.14015) introduces prompting into depth foundation models, enabling accurate metric depth estimation up to 4K resolution. This model uses low-cost LiDAR as a prompt integrated at multiple scales within the depth decoder. To overcome training challenges with limited datasets, it employs a scalable data pipeline featuring synthetic LiDAR simulation and real data pseudo GT depth generation. The approach achieves state-of-the-art results on ARKitScenes and ScanNet++ datasets and enhances applications like 3D reconstruction and robotic grasping.

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="depth-estimation", model="depth-anything/prompt-depth-anything-vits-hf", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vits-hf", dtype="auto")
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

## PromptDepthAnythingConfig

[[autodoc]] PromptDepthAnythingConfig

## PromptDepthAnythingForDepthEstimation

[[autodoc]] PromptDepthAnythingForDepthEstimation
    - forward

## PromptDepthAnythingImageProcessor

[[autodoc]] PromptDepthAnythingImageProcessor
    - preprocess
    - post_process_depth_estimation

## PromptDepthAnythingImageProcessorFast

[[autodoc]] PromptDepthAnythingImageProcessorFast
    - preprocess
    - post_process_depth_estimation