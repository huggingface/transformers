<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Monocular depth estimation

Monocular depth estimation is a computer vision task that involves predicting the depth information of a scene from a
single image. In other words, it is the process of estimating the distance of objects in a scene from
a single camera viewpoint.

Monocular depth estimation has various applications, including 3D reconstruction, augmented reality, autonomous driving,
and robotics. It is a challenging task as it requires the model to understand the complex relationships between objects
in the scene and the corresponding depth information, which can be affected by factors such as lighting conditions,
occlusion, and texture. 

There are two main depth estimation categories:

- **Absolute depth estimation**: This task variant aims to provide exact depth measurements from the camera. The term is used interchangeably with metric depth estimation, where depth is provided in precise measurements in meters or feet. Absolute depth estimation models output depth maps with numerical values that represent real-world distances.

- **Relative depth estimation**: Relative depth estimation aims to predict the depth order of objects or points in a scene without providing the precise measurements. These models output a depth map that indicates which parts of the scene are closer or farther relative to each other without the actual distances to A and B.

In this guide, we will see how to infer with [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large), a state-of-the-art zero-shot relative depth estimation model, and [ZoeDepth](https://huggingface.co/docs/transformers/main/en/model_doc/zoedepth), an absolute depth estimation model.

<Tip>

Check the [Depth Estimation](https://huggingface.co/tasks/depth-estimation) task page to view all compatible architectures and checkpoints.

</Tip>

Before we begin, we need to install the latest version of Transformers:

```bash
pip install -q -U transformers
```

## Depth estimation pipeline

The simplest way to try out inference with a model supporting depth estimation is to use the corresponding [`pipeline`].
Instantiate a pipeline from a [checkpoint on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads):

```py
>>> from transformers import pipeline
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
>>> pipe = pipeline("depth-estimation", model=checkpoint, device=device)
```

Next, choose an image to analyze:

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="Photo of a bee"/>
</div>

Pass the image to the pipeline.

```py
>>> predictions = pipe(image)
```

The pipeline returns a dictionary with two entries. The first one, called `predicted_depth`, is a tensor with the values
being the depth expressed in meters for each pixel.
The second one, `depth`, is a PIL image that visualizes the depth estimation result.

Let's take a look at the visualized result:

```py
>>> predictions["depth"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization.png" alt="Depth estimation visualization"/>
</div>

## Depth estimation inference by hand

Now that you've seen how to use the depth estimation pipeline, let's see how we can replicate the same result by hand.

Start by loading the model and associated processor from a [checkpoint on the Hugging Face Hub](https://huggingface.co/models?pipeline_tag=depth-estimation&sort=downloads).
Here we'll use the same checkpoint as before:

```py
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation

>>> checkpoint = "Intel/zoedepth-nyu-kitti"

>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)
```

Prepare the image input for the model using the `image_processor` that will take care of the necessary image transformations
such as resizing and normalization:

```py
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
```

Pass the prepared inputs through the model:

```py
>>> import torch

>>> with torch.no_grad():   
...     outputs = model(pixel_values)
...     outputs_flip = model(pixel_values=torch.flip(inputs.pixel_values, dims=[3]))
```

<Tip>
According to the [original implementation](https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L131) ZoeDepth model performs inference on both the original and flipped images by default and averages out the results. The `post_process_depth_estimation` function will handle this for us by passing the flipped outputs to the optional `outputs_flip` argument.
</Tip>

Let's post-process the results to remove any padding and resize the depth map to match the original image size. The `post_process_depth_estimation` outputs a list of dicts containing the `"predicted_depth"` and the `"depth"` image visualization.

```py
>>> post_processed_output = image_processor.post_process_depth_estimation(
...     outputs,
...     source_size=[image.size[::-1]],
...     outputs_flip=outputs_flip,
...     normalize=True,
... )
>>> depth = post_processed_output[0]["depth"]
```

<Tip>
Due to ZoeDepth's dynamic padding of the input image during inference we need to pass the original image size (`image.size[::-1]`) as input to the `post_process_depth_estimation` function to get the final depth map with the same dimensions as the original image.
</Tip>

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization-zoe.png" alt="Depth estimation visualization"/>
</div>
