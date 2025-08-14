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

# ZoeDepth

[ZoeDepth](https://huggingface.co/papers/2302.12288) is a depth estimation model that combines the generalization performance of relative depth estimation (how far objects are from each other) and metric depth estimation (precise depth measurement on metric scale) from a single image. It is pre-trained on 12 datasets using relative depth and 2 datasets (NYU Depth v2 and KITTI) for metric accuracy. A lightweight head with a metric bin module for each domain is used, and during inference, it automatically selects the appropriate head for each input image with a latent classifier.

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/zoedepth_architecture_bis.png"
alt="drawing" width="600"/>

You can find all the original ZoeDepth checkpoints under the [Intel](https://huggingface.co/Intel?search=zoedepth) organization.

The example below demonstrates how to estimate depth with [`Pipeline`] or the [`AutoModel`] class.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import requests
import torch
from transformers import pipeline
from PIL import Image

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
pipeline = pipeline(
    task="depth-estimation",
    model="Intel/zoedepth-nyu-kitti",
    dtype=torch.float16,
    device=0
)
results = pipeline(image)
results["depth"]
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
import requests
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(
    "Intel/zoedepth-nyu-kitti"
)
model = AutoModelForDepthEstimation.from_pretrained(
    "Intel/zoedepth-nyu-kitti",
    device_map="auto"
)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = image_processor(image, return_tensors="pt").to("cuda")

with torch.no_grad():
  outputs = model(inputs)

# interpolate to original size and visualize the prediction
## ZoeDepth dynamically pads the input image, so pass the original image size as argument
## to `post_process_depth_estimation` to remove the padding and resize to original dimensions.
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    source_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.detach().cpu().numpy() * 255
Image.fromarray(depth.astype("uint8"))
```

</hfoption>
</hfoptions>

## Notes

- In the [original implementation](https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L131) ZoeDepth performs inference on both the original and flipped images and averages the results. The `post_process_depth_estimation` function handles this by passing the flipped outputs to the optional `outputs_flipped` argument as shown below.
   ```py
    with torch.no_grad():
        outputs = model(pixel_values)
        outputs_flipped = model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))
        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            source_sizes=[(image.height, image.width)],
            outputs_flipped=outputs_flipped,
        )
   ```
   
## Resources
- Refer to this [notebook](https://github.com/NielsRogge/Transformers-Tutorials/tree/master/ZoeDepth) for an inference example.

## ZoeDepthConfig

[[autodoc]] ZoeDepthConfig

## ZoeDepthImageProcessor

[[autodoc]] ZoeDepthImageProcessor
    - preprocess

## ZoeDepthImageProcessorFast

[[autodoc]] ZoeDepthImageProcessorFast
    - preprocess

## ZoeDepthForDepthEstimation

[[autodoc]] ZoeDepthForDepthEstimation
    - forward