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

SAM2 (Segment Anything Model 2) was proposed in [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) by Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer.

The model can be used to predict segmentation masks of any object of interest given an input image or video, and input points or bounding boxes.

![example image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam2_header.gif)

The abstract from the paper is the following:

*We present Segment Anything Model 2 (SAM 2), a foundation model towards solving promptable visual segmentation in images and videos. We build a data engine, which improves model and data via user interaction, to collect the largest video segmentation dataset to date. Our model is a simple transformer architecture with streaming memory for real-time video processing. SAM 2 trained on our data provides strong performance across a wide range of tasks. In video segmentation, we observe better accuracy, using 3x fewer interactions than prior approaches. In image segmentation, our model is more accurate and 6x faster than the Segment Anything Model (SAM). We believe that our data, model, and insights will serve as a significant milestone for video segmentation and related perception tasks. We are releasing our main model, dataset, as well as code for model training and our demo.*

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
>>> from transformers import Sam2Processor, Sam2Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")
>>> processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

>>> image_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
>>> raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

>>> input_points = [[[[500, 375]]]]  # Single point click, 4 dimensions (image_dim, object_dim, point_per_object_dim, coordinates)
>>> input_labels = [[[1]]]  # 1 for positive click, 0 for negative click, 3 dimensions (image_dim, object_dim, point_label)

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )[0]

>>> # The model outputs multiple mask predictions ranked by quality score
>>> print(f"Generated {masks.shape[0]} masks with shape {masks.shape}")
Generated 3 masks with shape torch.Size([3, 1500, 2250])
```

#### Multiple Points for Refinement

You can provide multiple points to refine the segmentation:

```python
>>> # Add both positive and negative points to refine the mask
>>> input_points = [[[[500, 375], [1125, 625]]]]  # Multiple points for refinement
>>> input_labels = [[[1, 1]]]  # Both positive clicks

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )[0]
```

#### Bounding Box Input

SAM2 also supports bounding box inputs for segmentation:

```python
>>> # Define bounding box as [x_min, y_min, x_max, y_max]
>>> input_boxes = [[[75, 275, 1725, 850]]]

>>> inputs = processor(images=raw_image, input_boxes=input_boxes, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )[0]
```

#### Multiple Objects Segmentation

You can segment multiple objects simultaneously:

```python
>>> # Define points for two different objects
>>> input_points = [[[[500, 375]], [[650, 750]]]]  # Points for two objects in same image
>>> input_labels = [[[1], [1]]]  # Positive clicks for both objects

>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Each object gets its own mask
>>> masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )[0]
>>> print(f"Generated masks for {masks.shape[0]} objects")
Generated masks for 2 objects
```

### Batch Inference

#### Batched Images

Process multiple images simultaneously for improved efficiency:

```python
>>> from transformers import Sam2Processor, Sam2Model
>>> import torch
>>> from PIL import Image
>>> import requests

>>> model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large")
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

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> # Post-process masks for each image
>>> all_masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )
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

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )
>>> print(f"Truck image: {all_masks[0].shape[0]} objects, Dog image: {all_masks[1].shape[0]} objects")
Truck image: 2 objects, Dog image: 1 objects
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

>>> inputs = processor(images=raw_images, input_points=input_points, input_labels=input_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )
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

>>> inputs = processor(images=raw_images, input_boxes=input_boxes, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs, multimask_output=False)

>>> all_masks = processor.post_process_masks(
...     outputs.pred_masks.cpu(), inputs["original_sizes"], inputs["reshaped_input_sizes"]
... )
>>> print(f"Processed {len(input_boxes)} images with {len(input_boxes[0])} and {len(input_boxes[1])} boxes respectively")
Processed 2 images with 4 and 4 boxes respectively
```

### Video Segmentation and Tracking

SAM2's key strength is its ability to track objects across video frames. Here's how to use it for video segmentation:

#### Basic Video Tracking

```python
>>> from transformers import Sam2VideoModel, Sam2Processor
>>> import torch

>>> model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-large")
>>> processor = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")

>>> # Load video frames (example assumes you have a list of PIL Images)
>>> # video_frames = [Image.open(f"frame_{i:05d}.jpg") for i in range(num_frames)]

>>> # For this example, we'll use the video loading utility
>>> from transformers.video_utils import load_video
>>> video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
>>> video_frames, _ = load_video(video_url)

>>> # Initialize video inference session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device="cuda" if torch.cuda.is_available() else "cpu"
... )

>>> # Add click on first frame to select object
>>> ann_frame_idx = 0
>>> ann_obj_id = 1
>>> points = [[[[210, 350]]]]
>>> labels = [[[1]]]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=ann_obj_id,
...     input_points=points,
...     input_labels=labels,
... )

>>> # Segment the object on the first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )
>>> print(f"Segmentation shape: {outputs.video_res_masks.shape}")
Segmentation shape: torch.Size([1, 1, 480, 854])

>>> # Propagate through the entire video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_segments[sam2_video_output.frame_idx] = sam2_video_output.video_res_masks

>>> print(f"Tracked object through {len(video_segments)} frames")
Tracked object through 180 frames
```

#### Multi-Object Video Tracking

Track multiple objects simultaneously across video frames:

```python
>>> # Reset for new tracking session
>>> inference_session.reset_inference_session()

>>> # Add multiple objects on the first frame
>>> ann_frame_idx = 0
>>> obj_ids = [2, 3]
>>> input_points = [[[[200, 300]]], [[[400, 150]]]]  # Points for two objects
>>> input_labels = [[[1]], [[1]]]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=obj_ids,
...     input_points=input_points,
...     input_labels=input_labels,
... )

>>> # Get masks for both objects on first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )

>>> # Propagate both objects through video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_segments[sam2_video_output.frame_idx] = {
...         obj_id: sam2_video_output.video_res_masks[i]
...         for i, obj_id in enumerate(inference_session.obj_ids)
...     }

>>> print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames
```

#### Refining Video Segmentation

You can add additional clicks on any frame to refine the tracking:

```python
>>> # Add refinement click on a later frame
>>> refine_frame_idx = 50
>>> ann_obj_id = 2  # Refining first object
>>> points = [[[[220, 280]]]]  # Additional point
>>> labels = [[[1]]]  # Positive click

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=refine_frame_idx,
...     obj_ids=ann_obj_id,
...     input_points=points,
...     input_labels=labels,
... )

>>> # Re-propagate with the additional information
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_segments[sam2_video_output.frame_idx] = sam2_video_output.video_res_masks
```

### Streaming Video Inference

For real-time applications, SAM2 supports processing video frames as they arrive:

```python
>>> # Initialize session for streaming
>>> inference_session = processor.init_video_session(
...     inference_device="cuda" if torch.cuda.is_available() else "cpu"
... )

>>> # Process frames one by one
>>> for frame_idx, frame in enumerate(video_frames[:10]):  # Process first 10 frames
...     inputs = processor(images=frame, device="cuda" if torch.cuda.is_available() else "cpu", return_tensors="pt")
...
...     if frame_idx == 0:
...         # Add point input on first frame
...         processor.add_inputs_to_inference_session(
...             inference_session=inference_session,
...             frame_idx=0,
...             obj_ids=1,
...             input_points=[[[[210, 350], [250, 220]]]],
...             input_labels=[[[1, 1]]],
...             original_size=inputs.original_sizes[0], # need to be provided when using streaming video inference
...         )
...
...     # Process current frame
...     sam2_video_output = model(
...         inference_session=inference_session,
...         frame=inputs.pixel_values[0],
...     )
...
...     print(f"Frame {frame_idx}: mask shape {sam2_video_output.video_res_masks.shape}")
```

#### Video Batch Processing for Multiple Objects

Track multiple objects simultaneously in video by adding them all at once:

```python
>>> # Initialize video session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device="cuda" if torch.cuda.is_available() else "cpu"
... )

>>> # Add multiple objects on the first frame using batch processing
>>> ann_frame_idx = 0
>>> obj_ids = [2, 3]  # Track two different objects
>>> input_points = [
...     [[[200, 300], [230, 250], [275, 175]]],  # Object 2: 3 points (2 positive, 1 negative)
...     [[[400, 150]]]                           # Object 3: 1 point
... ]
>>> input_labels = [
...     [[1, 1, 0]],  # Object 2: positive, positive, negative for refinement
...     [[1]]         # Object 3: positive
... ]

>>> processor.add_inputs_to_inference_session(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
...     obj_ids=obj_ids,
...     input_points=input_points,
...     input_labels=input_labels,
... )

>>> # Get masks for all objects on the first frame
>>> outputs = model(
...     inference_session=inference_session,
...     frame_idx=ann_frame_idx,
... )
>>> print(f"Generated masks for {outputs.video_res_masks.shape[0]} objects")
Generated masks for 2 objects

>>> # Propagate all objects through the video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_segments[sam2_video_output.frame_idx] = {
...         obj_id: sam2_video_output.video_res_masks[i]
...         for i, obj_id in enumerate(inference_session.obj_ids)
...     }

>>> print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames
```

### Using Previous Masks as Input

SAM2 can use masks from previous predictions as input to refine segmentation:

```python
>>> # Get initial segmentation
>>> input_points = [[[[500, 375]]]]
>>> input_labels = [[[1]]]
>>> inputs = processor(images=raw_image, input_points=input_points, input_labels=input_labels, return_tensors="pt")

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> # Use the best mask as input for refinement
>>> mask_input = outputs.pred_masks[:, torch.argmax(outputs.iou_scores)]

>>> # Add additional points with the mask input
>>> new_input_points = [[[[500, 375], [450, 300]]]]
>>> new_input_labels = [[[1, 1]]]
>>> inputs = processor(
...     input_points=new_input_points,
...     input_labels=new_input_labels,
...     original_sizes=inputs["original_sizes"],
...     return_tensors="pt",
... )

>>> with torch.no_grad():
...     refined_outputs = model(
...         **inputs,
...         input_masks=mask_input,
...         multimask_output=False,
...     )
```

## Resources
<!-- TODO replace with sam2 resources -->
A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM.

- [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/segment_anything.ipynb) for using the model.
- [Demo notebook](https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb) for using the automatic mask generation pipeline.
- [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Run_inference_with_MedSAM_using_HuggingFace_Transformers.ipynb) for inference with MedSAM, a fine-tuned version of SAM on the medical domain. 🌎
- [Demo notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb) for fine-tuning the model on custom data. 🌎

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
    - init_video_session
    - add_inputs_to_inference_session

## Sam2ImageProcessorFast

[[autodoc]] Sam2ImageProcessorFast

## Sam2VideoProcessor

[[autodoc]] Sam2VideoProcessor

## Sam2VideoInferenceSession

[[autodoc]] Sam2VideoInferenceSession

## Sam2HieraDetModel

[[autodoc]] Sam2HieraDetModel
    - forward

## Sam2VisionModel

[[autodoc]] Sam2VisionModel
    - forward

## Sam2Model

[[autodoc]] Sam2Model
    - forward

## Sam2VideoModel

[[autodoc]] Sam2VideoModel
    - forward
    - propagate_in_video_iterator
