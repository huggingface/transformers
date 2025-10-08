<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2022-01-19 and added to Hugging Face Transformers on 2022-03-22 and contributed by [nielsr](https://huggingface.co/nielsr).*

# GLPN

[GLPN](https://huggingface.co/papers/2201.07436) combines SegFormer's hierarchical mix-Transformer with a lightweight decoder for monocular depth estimation. The decoder integrates global context and local connectivity through a selective feature fusion module, enhancing fine detail recovery. This approach achieves state-of-the-art performance on the NYU Depth V2 dataset with improved generalization and robustness, while maintaining lower computational complexity compared to previous decoders.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="depth-estimation", model="vinvino02/glpn-kitti", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("vinvino02/glpn-kitti")
model = AutoModelForDepthEstimation.from_pretrained("vinvino02/glpn-kitti", dtype="auto")
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

## GLPNConfig

[[autodoc]] GLPNConfig

## GLPNFeatureExtractor

[[autodoc]] GLPNFeatureExtractor
    - __call__

## GLPNImageProcessor

[[autodoc]] GLPNImageProcessor
    - preprocess

## GLPNImageProcessorFast

[[autodoc]] GLPNImageProcessorFast
    - preprocess

## GLPNModel

[[autodoc]] GLPNModel
    - forward

## GLPNForDepthEstimation

[[autodoc]] GLPNForDepthEstimation
    - forward

