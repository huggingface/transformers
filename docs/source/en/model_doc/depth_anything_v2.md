<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Depth-Anything-V2-Small model is under the Apache-2.0 license.
Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Depth Anything V2

## Overview

The Depth Anything V2 model was proposed in [Depth Anything V2: Unleashing the Upgraded Power of V2 Large-Scale Unlabeled Data](https://arxiv.org/abs/2406.09414) by Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao. Depth Anything V2 is based on the [DPT](https://arxiv.org/abs/2103.13413) architecture, trained on ~62 million images, obtaining state-of-the-art results for both relative and absolute depth estimation.

The abstract from the paper is the following:

*This work presents Depth Anything V2. Without pursuing fancy techniques, we aim to reveal crucial findings to pave the way towards building a powerful monocular depth estimation model. Notably, compared with V1, this version produces much finer and more robust depth predictions through three key practices: 

    1. Replacing all labeled real images with synthetic images, 
    2. Scaling up the capacity of our teacher model, and 
    3. Teaching student models via the bridge of large-scale pseudo-labeled real images. 

Compared with the latest models built on Stable Diffusion, our models are significantly more efficient (more than 10x faster) and more accurate. We offer models of different scales (ranging from 25M to 1.3B params) to support extensive scenarios. Benefiting from their strong generalization capability, we fine-tune them with metric depth labels to obtain our metric depth models. In addition to our models, considering the limited diversity and frequent noise in current test sets, we construct a versatile evaluation benchmark with precise annotations and diverse scenes to facilitate future research. Our better v2 depth model also results in a better depth-conditioned ControlNet.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/depth_anything_overview.jpg"
alt="drawing" width="600"/>

<small> Depth Anything overview. Taken from the <a href="https://arxiv.org/abs/2406.09414">original paper</a>.</small>

This model was contributed by [MackinationsAi](https://huggingface.co/MackinationsAi).
The code can be found here [Upgraded-Depth-Anything-V2](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2) which is an upgraded version of the original DepthAnything/Depth-Anything-V2 github repo.

## Usage example

There are 2 main ways to use Depth Anything V2: either using the pipeline API, which abstracts away all the complexity for you, or by using the `DepthAnythingV2ForDepthEstimation` class yourself.

### Pipeline API

The pipeline allows to use the model in a few lines of code:

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests
>>> from safetensors.torch import load_file

>>> # load pipe
>>> pipe = pipeline(task="depth-estimation", model="MackinationsAi/Depth-Anything-V2_Safetensors/depth_anything_v2_vits.safetensors")

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
>>> from safetensors.torch import load_file

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("MackinationsAi/Depth-Anything-V2_Safetensors/depth_anything_v2_vits.safetensors")

>>> model_config = "MackinationsAi/Depth-Anything-V2_Safetensors/depth_anything_v2_vits.safetensors"

>>> state_dict = load_file(f"{model_config}")

>>> model = AutoModelForDepthEstimation.from_pretrained(model_config, state_dict=state_dict)

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> # perform inference
>>> with torch.no_grad():
...     outputs = model(**inputs)
...     predicted_depth = outputs.predicted_depth

>>> # interpolate the prediction to the original image size
>>> prediction = torch.nn.functional.interpolate(
...     predicted_depth.unsqueeze(1),
...     size=image.size[::-1],
...     mode="bicubic",
...     align_corners=False,
...)

>>> # visualize the prediction
>>> output = prediction.squeeze().cpu().numpy()
>>> formatted = (output * 255 / np.max(output)).astype("uint8")
>>> depth = Image.fromarray(formatted)
>>> depth.show()
```

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with the Upgraded Depth Anything V2 models.

- [Monocular depth estimation task guide](../tasks/depth_estimation)
- A notebook showcasing inference with [`DepthAnythingV2ForDepthEstimation`] can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Depth%20Anything/Predicting_depth_in_an_image_with_Depth_Anything.ipynb). 🌎

If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource.

## DepthAnythingV2Config

[[autodoc]] DepthAnythingV2Config

## DepthAnythingV2ForDepthEstimation

[[autodoc]] DepthAnythingV2ForDepthEstimation
    - forward