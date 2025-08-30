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
<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# SAM2

## Overview

SAM2 (Segment Anything Model 2) was proposed in [Segment Anything in Images and Videos](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/) by Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer.

The model can be used to predict segmentation masks of any object of interest given an input image or video, and input points or bounding boxes.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam2_header.gif)

The abstract from the paper is the following:

*We present Segment Anything Model 2 (SAM 2), a foundation model towards solving promptable visual segmentation in images and videos. We build a data engine, which improves model and data via user interaction, to collect the largest video segmentation dataset to date. Our model is a simple transformer architecture with streaming memory for real-time video processing. SAM 2 trained on our data provides strong performance across a wide range of tasks. In video segmentation, we observe better accuracy, using 3x fewer interactions than prior approaches. In image segmentation, our model is more accurate and 6x faster than the Segment Anything Model (SAM). We believe that our data, model, and insights will serve as a significant milestone for video segmentation and related perception tasks. We are releasing a version of our model, the dataset and an interactive demo.*

Tips:

- Batch & Video Support: SAM2 natively supports batch processing and seamless video segmentation, while original SAM is designed for static images and simpler one-image-at-a-time workflows.
- Accuracy & Generalization: SAM2 shows improved segmentation quality, robustness, and zero-shot generalization to new domains compared to the original SAM, especially with mixed prompts.

This model was contributed by [sangbumchoi](https://github.com/SangbumChoi) and [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/facebookresearch/sam2/tree/main).

## Usage example

### Automatic Mask Generation with Pipeline

SAM2 can be used for automatic mask generation to segment all objects in an image using the `mask-generation` pipeline:

```python
>>> from transformers import pipeline

>>> generator = pipeline("mask-generation", model="facebook/sam2.1-hiera-large", device=0)
>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> outputs = generator(image_url, points_per_batch=64)

>>> len(outputs["masks"])  # Number of masks generated
39
```

### Basic Image Segmentation

#### Single Point Click

You can segment objects by providing a single point click on the object you want to segment:

```python
>>> from transformers import Sam2Processor, Sam2Model, infer_device
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = infer_device()

>>> model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
>>> processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

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
Generated 3 masks with shape torch.Size(1, 3, 1500, 2250)
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
```

#### Bounding Box Input

SAM2 also supports bounding box inputs for segmentation:

```python
>>> # Define bounding box as [x_min, y_min, x_max, y_max]
>>> input_boxes = [[[75, 275, 1725, 850]]]

>>> inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt").to(device)

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
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
```

### Batch Inference

#### Batched Images

Process multiple images simultaneously for improved efficiency:

```python
>>> from transformers import Sam2Processor, Sam2Model, infer_device
>>> import torch
>>> from PIL import Image
>>> import requests

>>> device = infer_device()

>>> model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(device)
>>> processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

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
```

### Using Previous Masks as Input

SAM2 can use masks from previous predictions as input to refine segmentation:

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

<!-- TODO replace with sam2 resources -->
<!-- ## Resources -->
<!-- A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM.

- [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything_2.ipynb) for using the model. -->

## Sam2Config

[[autodoc]] Sam2Config

## Sam2HieraDetConfig

[[autodoc]] Sam2HieraDetConfig

## Sam2VisionConfig

[[autodoc]] Sam2VisionConfig

## Sam2MaskDecoderConfig

[[autodoc]] Sam2MaskDecoderConfig

## Sam2PromptEncoderConfig

[[autodoc]] Sam2PromptEncoderConfig

## Sam2Processor

[[autodoc]] Sam2Processor
    - __call__
    - post_process_masks

## Sam2ImageProcessorFast

[[autodoc]] Sam2ImageProcessorFast

## Sam2HieraDetModel

[[autodoc]] Sam2HieraDetModel
    - forward

## Sam2VisionModel

[[autodoc]] Sam2VisionModel
    - forward

## Sam2Model

[[autodoc]] Sam2Model
    - forward
