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
*This model was released on 2024-06-13 and added to Hugging Face Transformers on 2024-07-05.*

# Depth Anything V2

[Depth Anything V2](https://huggingface.co/papers/2406.09414) enhances monocular depth estimation by replacing real images with synthetic data, increasing the teacher model's capacity, and using large-scale pseudo-labeled real images to train student models. This results in finer and more robust depth predictions, offering efficiency and accuracy improvements over models based on Stable Diffusion. Available in various sizes (25M to 1.3B parameters), these models can be fine-tuned for metric depth tasks. The paper also introduces a new evaluation benchmark with precise annotations and diverse scenes to support future research.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", dtype="auto")
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

## Usage tips

- DepthAnythingV2 was released in June 2024. It uses the same architecture as Depth Anything and is compatible with all code examples and existing workflows.
- The model uses synthetic data and a larger capacity teacher model to achieve much finer and more robust depth predictions.

## DepthAnythingConfig

[[autodoc]] DepthAnythingConfig

## DepthAnythingForDepthEstimation

[[autodoc]] DepthAnythingForDepthEstimation
    - forward

