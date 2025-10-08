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
*This model was released on 2021-03-24 and added to Hugging Face Transformers on 2022-03-28 and contributed by [nielsr](https://huggingface.co/nielsr).*

# DPT

[DPT](https://huggingface.co/papers/2103.13413) leverages Vision Transformers (ViT) as a backbone for dense prediction tasks such as semantic segmentation and depth estimation. It constructs tokens from various stages of the ViT into image-like representations at different resolutions, which are then combined into full-resolution predictions using a convolutional decoder. The transformer backbone maintains a constant and high-resolution processing with a global receptive field at every stage, enabling finer-grained and more coherent predictions compared to fully-convolutional networks. Experiments demonstrate significant improvements on dense prediction tasks, particularly with large training datasets. For monocular depth estimation, DPT achieves up to a 28% relative performance boost over state-of-the-art fully-convolutional networks. In semantic segmentation, it sets a new state of the art on ADE20K with 49.02% mIoU. Additionally, DPT outperforms on smaller datasets like NYUv2, KITTI, and Pascal Context, establishing new benchmarks in these areas.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="depth-estimation", model="Intel/dpt-large", dtype="auto")
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

image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model = AutoModelForDepthEstimation.from_pretrained("Intel/dpt-large", dtype="auto")
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

## DPTConfig

[[autodoc]] DPTConfig

## DPTFeatureExtractor

[[autodoc]] DPTFeatureExtractor
    - __call__
    - post_process_semantic_segmentation

## DPTImageProcessor

[[autodoc]] DPTImageProcessor
    - preprocess
    - post_process_semantic_segmentation

## DPTImageProcessorFast

[[autodoc]] DPTImageProcessorFast
    - preprocess
    - post_process_semantic_segmentation

## DPTModel

[[autodoc]] DPTModel
    - forward

## DPTForDepthEstimation

[[autodoc]] DPTForDepthEstimation
    - forward

## DPTForSemanticSegmentation

[[autodoc]] DPTForSemanticSegmentation
    - forward

