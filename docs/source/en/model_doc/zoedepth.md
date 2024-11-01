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

# ZoeDepth

## Overview

The ZoeDepth model was proposed in [ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth](https://arxiv.org/abs/2302.12288) by Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, Matthias Müller. ZoeDepth extends the [DPT](dpt) framework for metric (also called absolute) depth estimation. ZoeDepth is pre-trained on 12 datasets using relative depth and fine-tuned on two domains (NYU and KITTI) using metric depth. A lightweight head is used with a novel bin adjustment design called metric bins module for each domain. During inference, each input image is automatically routed to the appropriate head using a latent classifier.

The abstract from the paper is the following:

*This paper tackles the problem of depth estimation from a single image. Existing work either focuses on generalization performance disregarding metric scale, i.e. relative depth estimation, or state-of-the-art results on specific datasets, i.e. metric depth estimation. We propose the first approach that combines both worlds, leading to a model with excellent generalization performance while maintaining metric scale. Our flagship model, ZoeD-M12-NK, is pre-trained on 12 datasets using relative depth and fine-tuned on two datasets using metric depth. We use a lightweight head with a novel bin adjustment design called metric bins module for each domain. During inference, each input image is automatically routed to the appropriate head using a latent classifier. Our framework admits multiple configurations depending on the datasets used for relative depth pre-training and metric fine-tuning. Without pre-training, we can already significantly improve the state of the art (SOTA) on the NYU Depth v2 indoor dataset. Pre-training on twelve datasets and fine-tuning on the NYU Depth v2 indoor dataset, we can further improve SOTA for a total of 21% in terms of relative absolute error (REL). Finally, ZoeD-M12-NK is the first model that can jointly train on multiple datasets (NYU Depth v2 and KITTI) without a significant drop in performance and achieve unprecedented zero-shot generalization performance to eight unseen datasets from both indoor and outdoor domains.*

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/zoedepth_architecture_bis.png"
alt="drawing" width="600"/>

<small> ZoeDepth architecture. Taken from the <a href="https://arxiv.org/abs/2302.12288">original paper.</a> </small>

This model was contributed by [nielsr](https://huggingface.co/nielsr).
The original code can be found [here](https://github.com/isl-org/ZoeDepth).

## Usage tips

- ZoeDepth is an absolute (also called metric) depth estimation model, unlike DPT which is a relative depth estimation model. This means that ZoeDepth is able to estimate depth in metric units like meters.

The easiest to perform inference with ZoeDepth is by leveraging the [pipeline API](../main_classes/pipelines.md):

```python
>>> from transformers import pipeline
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> pipe = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")
>>> result = pipe(image)
>>> depth = result["depth"]
```

Alternatively, one can also perform inference using the classes:

```python
>>> from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
>>> import torch
>>> import numpy as np
>>> from PIL import Image
>>> import requests

>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
>>> model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti")

>>> # prepare image for the model
>>> inputs = image_processor(images=image, return_tensors="pt")

>>> with torch.no_grad():   
...     outputs = model(pixel_values)

>>> # interpolate to original size and visualize the prediction
>>> ## ZoeDepth dynamically pads the input image. Thus we pass the original image size as argument
>>> ## to `post_process_depth_estimation` to remove the padding and resize to original dimensions.
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_sizes=[(image.height, image.width)],
... )

>>> predicted_depth = post_processed_output[0]["predicted_depth"]
>>> depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
>>> depth = depth.detach().cpu().numpy() * 255
>>> depth = Image.fromarray(depth.astype("uint8"))
```

<Tip>
<p>In the <a href="https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L131">original implementation</a> ZoeDepth model performs inference on both the original and flipped images and averages out the results. The <code>post_process_depth_estimation</code> function can handle this for us by passing the flipped outputs to the optional <code>outputs_flipped</code> argument:</p>
<pre><code class="language-Python">&gt;&gt;&gt; with torch.no_grad():   
...     outputs = model(pixel_values)
...     outputs_flipped = model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))
&gt;&gt;&gt; post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_sizes=[(image.height, image.width)],
...     outputs_flipped=outputs_flipped,
... )
</code></pre>
</Tip>

## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with ZoeDepth.

- A demo notebook regarding inference with ZoeDepth models can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ZoeDepth). 🌎

## ZoeDepthConfig

[[autodoc]] ZoeDepthConfig

## ZoeDepthImageProcessor

[[autodoc]] ZoeDepthImageProcessor
    - preprocess

## ZoeDepthForDepthEstimation

[[autodoc]] ZoeDepthForDepthEstimation
    - forward