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

# Distill Any Depth

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Distill Any Depth model was proposed in [Distill Any Depth: Distillation Creates a Stronger Monocular Depth Estimator](https://arxiv.org/abs/2502.19204) by Xiankang He, Dongyan Guo, Hongji Li, Ruibo Li, Ying Cui, Chi Zhang. 


The original code can be found [here](https://github.com/Westlake-AGI-Lab/Distill-Any-Depth).

## Usage example

There are 2 main ways to use Distill Any Depth: either using the pipeline API, which abstracts away all the complexity for you, or by using the `DistillAnyDepthForDepthEstimation` class yourself.

### Pipeline API

The pipeline allows to use the model in a few lines of code:

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> # load pipe
>>> pipe = pipeline(task="depth-estimation", model="xingyang1/Distill-Any-Depth-small-hf")

>>> # load image
>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # inference
>>> depth = pipe(image)["depth"]
```

### Using the model yourself

If you want to do the pre- and postprocessing yourself, here's how to do that:

```python
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("xingyang1/Distill-Any-Depth-small-hf")
>>> model = AutoModelForDepthEstimation.from_pretrained("xingyang1/Distill-Any-Depth-small-hf")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size and visualize the prediction
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     target_sizes=[(image.height, image.width)],
... )

>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
>>> depth = depth.detach().cpu().numpy() * 255
>>> depth = Image.fromarray(depth.astype("uint8"))
```

## DistillAnyDepthConfig

[[autodoc]] DistillAnyDepthConfig

## DistillAnyDepthForDepthEstimation

[[autodoc]] DistillAnyDepthForDepthEstimation
    - forward