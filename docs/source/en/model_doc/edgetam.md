<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-01-13 and added to Hugging Face Transformers on 2025-09-29.*
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# EdgeTAM

## Overview

The EdgeTAM model was proposed in [EdgeTAM: On-Device Track Anything Model](https://huggingface.co/papers/2501.07256) Chong Zhou, Chenchen Zhu, Yunyang Xiong, Saksham Suri, Fanyi Xiao, Lemeng Wu, Raghuraman Krishnamoorthi, Bo Dai, Chen Change Loy, Vikas Chandra, Bilge Soran.

EdgeTAM is an efficient adaptation of SAM 2 that introduces a 2D Spatial Perceiver architecture to optimize memory attention mechanisms for real-time video segmentation on mobile devices.

The abstract from the paper is the following:

*On top of Segment Anything Model (SAM), SAM 2 further extends its capability from image to video inputs through a memory bank mechanism and obtains a remarkable performance compared with previous methods, making it a foundation model for video segmentation task. In this paper, we aim at making SAM 2 much more efficient so that it even runs on mobile devices while maintaining a comparable performance. Despite several works optimizing SAM for better efficiency, we find they are not sufficient for SAM 2 because they all focus on compressing the image encoder, while our benchmark shows that the newly introduced memory attention blocks are also the latency bottleneck. Given this observation, we propose EdgeTAM, which leverages a novel 2D Spatial Perceiver to reduce the computational cost. In particular, the proposed 2D Spatial Perceiver encodes the densely stored frame-level memories with a lightweight Transformer that contains a fixed set of learnable queries. Given that video segmentation is a dense prediction task, we find preserving the spatial structure of the memories is essential so that the queries are split into global-level and patch-level groups. We also propose a distillation pipeline that further improves the performance without inference overhead. As a result, EdgeTAM achieves 87.7, 70.0, 72.3, and 71.7 J&F on DAVIS 2017, MOSE, SA-V val, and SA-V test, while running at 16 FPS on iPhone 15 Pro Max.*

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/facebookresearch/EdgeTAM).

## Usage example

### Automatic Mask Generation with Pipeline

EdgeTAM can be used for automatic mask generation to segment all objects in an image using the `mask-generation` pipeline:

```python
>>> from transformers import pipeline

>>> generator = pipeline("mask-generation", model="yonigozlan/edgetam-1", device=0)
>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> outputs = generator(image_url, points_per_batch=64)

>>> len(outputs["masks"])  # Number of masks generated
39
```

### Basic Image Segmentation

#### Single Point Click

You can segment objects by providing a single point click on the object you want to segment:

```python
>>> from transformers import Sam2Processor, EdgeTamModel
from accelerate import Accelerator
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = Accelerator().device

>>> model = EdgeTamModel.from_pretrained("yonigozlan/edgetam-1").to(device)
>>> processor = Sam2Processor.from_pretrained("yonigozlan/edgetam-1")

>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
>>> input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]

>>> # The model outputs multiple mask predictions ranked by quality score
>>> print(f"Generated {masks.shape[1]} masks with shape {masks.shape}")
Generated 3 masks with shape torch.Size([1, 3, 1200, 1800])
>>> print(f"IoU scores: {outputs.iou_scores.squeeze()}")
IoU scores: tensor([0.0463, 0.4859, 0.7616], device='cuda:0')
```

#### Multiple Points for Refinement

You can provide multiple points to refine the segmentation:

```python
>>> # Add both positive and negative points to refine the mask
>>> input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
>>> input_labels = [[[1, 1]]]  # Both positive clicks

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
>>> print(f"IoU scores: {outputs.iou_scores.squeeze()}")
IoU scores: tensor([0.8362, 0.6900, 0.2120], device='cuda:0')
```

#### Bounding Box Input

EdgeTAM also supports bounding box inputs for segmentation:

```python
>>> # Define bounding box as [x_min, y_min, x_max, y_max]
>>> input_boxes = [[[75, 275, 1725, 850]]]

>>> inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
>>> print(f"IoU scores: {outputs.iou_scores.squeeze()}")
IoU scores: tensor([0.9301, 0.9348, 0.6605], device='cuda:0')
```

#### Multiple Objects Segmentation

You can segment multiple objects simultaneously:

```python
>>> # Define points for two different objects
>>> input_points = [[[[500, 375]], [[650, 750]]]]  # Points for two objects in same image
>>> input_labels = [[[1], [1]]]  # Positive clicks for both objects

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Each object gets its own mask
>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
>>> print(f"Generated masks for {masks.shape[0]} objects")
Generated masks for 2 objects
>>> print(f"IoU scores: {outputs.iou_scores.squeeze()}")
IoU scores: tensor([0.7616, 0.9465], device='cuda:0')
```

### Batch Inference

#### Batched Images

Process multiple images simultaneously for improved efficiency:

```python
>>> from transformers import Sam2Processor, EdgeTamModel
from accelerate import Accelerator
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = Accelerator().device

>>> model = EdgeTamModel.from_pretrained("yonigozlan/edgetam-1").to(device)
>>> processor = Sam2Processor.from_pretrained("yonigozlan/edgetam-1")

>>> # Load multiple images
>>> image_urls = [
...     "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/dog-sam.png"
... ]
>>> raw_images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_urls]

>>> # Single point per image
>>> input_points = [[[[500, 375]]], [[[770, 200]]]]  # One point for each image
>>> input_labels = [[[1]], [[1]]]  # Positive clicks for both images

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(model.device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Post-process masks for each image
>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
>>> print(f"Processed {len(all_masks)} images, each with {all_masks[0].shape[0]} objects")
Processed 2 images, each with 1 objects
>>> print(f"IoU scores: {outputs.iou_scores.squeeze()}")
IoU scores: tensor([0.7618, 0.7999], device='cuda:0')
```

#### Batched Objects per Image

Segment multiple objects within each image using batch inference:

```python
>>> # Multiple objects per image - different numbers of objects per image
>>> input_points = [
...     [[[500, 375]], [[650, 750]]],  # Truck image: 2 objects
...     [[[770, 200]]]  # Dog image: 1 object
... ]
>>> input_labels = [
...     [[1], [1]],  # Truck image: positive clicks for both objects
...     [[1]]  # Dog image: positive click for the object
... ]

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
```

#### Batched Images with Batched Objects and Multiple Points

Handle complex batch scenarios with multiple points per object:

```python
>>> # Add groceries image for more complex example
>>> groceries_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/groceries.jpg"
>>> groceries_image = Image.open(requests.get(groceries_url, stream=True).raw).convert("RGB")
>>> raw_images = [raw_images[0], groceries_image]  # Use truck and groceries images

>>> # Complex batching: multiple images, multiple objects, multiple points per object
>>> input_points = [
...     [[[500, 375]], [[650, 750]]],  # Truck image: 2 objects with 1 point each
...     [[[400, 300]], [[630, 300], [550, 300]]]  # Groceries image: obj1 has 1 point, obj2 has 2 points
... ]
>>> input_labels = [
...     [[1], [1]],  # Truck image: positive clicks
...     [[1], [1, 1]]  # Groceries image: positive clicks for refinement
... ]

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
```

#### Batched Bounding Boxes

Process multiple images with bounding box inputs:

```python
>>> # Multiple bounding boxes per image (using truck and groceries images)
>>> input_boxes = [
...     [[75, 275, 1725, 850], [425, 600, 700, 875], [1375, 550, 1650, 800], [1240, 675, 1400, 750]],  # Truck image: 4 boxes
...     [[450, 170, 520, 350], [350, 190, 450, 350], [500, 170, 580, 350], [580, 170, 640, 350]]  # Groceries image: 4 boxes
... ]

>>> # Update images for this example
>>> raw_images = [raw_images[0], groceries_image]  # truck and groceries

>>> inputs = processor(images=raw_images, input_boxes=input_boxes, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
>>> print(f"Processed {len(input_boxes)} images with {len(input_boxes[0])} and {len(input_boxes[1])} boxes respectively")
Processed 2 images with 4 and 4 boxes respectively
>>> print(f"IoU scores: {outputs.iou_scores.squeeze()}")
IoU scores: tensor([0.9301, 0.9348, 0.6605, 0.9465], device='cuda:0')
```

### Using Previous Masks as Input

EdgeTAM can use masks from previous predictions as input to refine segmentation:

```python
>>> # Get initial segmentation
>>> input_points = [[[[500, 375]]]]
>>> input_labels = [[[1]]]
>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Use the best mask as input for refinement
>>> mask_input = outputs.pred_masks[:, :, torch.argmax(outputs.iou_scores.squeeze())]

>>> # Add additional points with the mask input
>>> new_input_points = [[[[500, 375], [450, 300]]]]
>>> new_input_labels = [[[1, 1]]]
>>> inputs = processor(
...     input_points=new_input_points,
...     input_labels=new_input_labels,
...     original_sizes=inputs["original_sizes"],
...     return_tensors="pt",
... ).to(device)

>>> with torch.no_grad():
...     refined_outputs = model(
...         **inputs,
...         input_masks=mask_input,
...         image_embeddings=outputs.image_embeddings,
...         multimask_output=False,
...     )
```

## EdgeTamConfig

[[autodoc]] EdgeTamConfig

## EdgeTamVisionConfig

[[autodoc]] EdgeTamVisionConfig

## EdgeTamMaskDecoderConfig

[[autodoc]] EdgeTamMaskDecoderConfig

## EdgeTamPromptEncoderConfig

[[autodoc]] EdgeTamPromptEncoderConfig

## EdgeTamVisionModel

[[autodoc]] EdgeTamVisionModel
    - forward

## EdgeTamModel

[[autodoc]] EdgeTamModel
    - forward
    - get_image_features
