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

# Depth Anything V2

## Overview

Depth Anything V2 was introduced in [the paper of the same name](https://arxiv.org/abs/2406.09414) by Lihe Yang et al. It uses the same architecture as the original [Depth Anything model](depth_anything), but uses synthetic data and a larger capacity teacher model to achieve much finer and robust depth predictions.

The abstract from the paper is the following:

*This work presents Depth Anything V2. Without pursuing fancy techniques, we aim to reveal crucial findings to pave the way towards building a powerful monocular depth estimation model. Notably, compared with V1, this version produces much finer and more robust depth predictions through three key practices: 1) replacing all labeled real images with synthetic images, 2) scaling up the capacity of our teacher model, and 3) teaching student models via the bridge of large-scale pseudo-labeled real images. Compared with the latest models built on Stable Diffusion, our models are significantly more efficient (more than 10x faster) and more accurate. We offer models of different scales (ranging from 25M to 1.3B params) to support extensive scenarios. Benefiting from their strong generalization capability, we fine-tune them with metric depth labels to obtain our metric depth models. In addition to our models, considering the limited diversity and frequent noise in current test sets, we construct a versatile evaluation benchmark with precise annotations and diverse scenes to facilitate future research.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/depth_anything_overview.jpg"
alt="drawing" width="600"/>

<small> Depth Anything overview. Taken from the <a href="https://arxiv.org/abs/2401.10891">original paper</a>.</small>

The Depth Anything models were contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/DepthAnything/Depth-Anything-V2).

## Usage example

There are 2 main ways to use Depth Anything V2: either using the pipeline API, which abstracts away all the complexity for you, or by using the `DepthAnythingForDepthEstimation` class yourself.

### Pipeline API

The pipeline allows to use the model in a few lines of code:

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> # load pipe
>>> pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

>>> # load image
>>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> # inference
>>> depth = pipe(image)["depth"]
```

### Using the model yourself

If you want to do the pre- and post-processing yourself, here's how to do that:

```python
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
>>> model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # interpolate to original size and visualize the prediction
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     target_size=[image.size[::-1]],
...     normalize=True,
... )

>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = post_processed_output[0]["depth"]
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with Depth Anything.

- [Monocular depth estimation task guide](../tasks/depth_estimation)
- [Depth Anything V2 demo](https://huggingface.co/spaces/depth-anything/Depth-Anything-V2).
- A notebook showcasing inference with [`DepthAnythingForDepthEstimation`] can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Depth%20Anything/Predicting_depth_in_an_image_with_Depth_Anything.ipynb). 🌎
- [Core ML conversion of the `small` variant for use on Apple Silicon](https://huggingface.co/apple/coreml-depth-anything-v2-small).

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DepthAnythingConfig

[[autodoc]] DepthAnythingConfig

## DepthAnythingForDepthEstimation

[[autodoc]] DepthAnythingForDepthEstimation
    - forward