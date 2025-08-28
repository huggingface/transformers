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

# SAM2 Video

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

### Video Segmentation and Tracking

SAM2's key strength is its ability to track objects across video frames. Here's how to use it for video segmentation:

#### Basic Video Tracking

```python
>>> from transformers import Sam2VideoModel, Sam2VideoProcessor, infer_device
>>> import torch

>>> device = infer_device()
>>> model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to(device, dtype=torch.bfloat16)
>>> processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")

>>> # Load video frames (example assumes you have a list of PIL Images)
>>> # video_frames = [Image.open(f"frame_{i:05d}.jpg") for i in range(num_frames)]

>>> # For this example, we'll use the video loading utility
>>> from transformers.video_utils import load_video
>>> video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
>>> video_frames, _ = load_video(video_url)

>>> # Initialize video inference session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     dtype=torch.bfloat16,
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
>>> video_res_masks = processor.post_process_masks(
...     [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
... )[0]
>>> print(f"Segmentation shape: {video_res_masks.shape}")
Segmentation shape: torch.Size([1, 1, 480, 854])

>>> # Propagate through the entire video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = video_res_masks

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
>>> input_points = [[[[200, 300]], [[400, 150]]]]  # Points for two objects (batched)
>>> input_labels = [[[1], [1]]]

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
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = {
...         obj_id: video_res_masks[i]
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
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = video_res_masks
```

### Streaming Video Inference

For real-time applications, SAM2 supports processing video frames as they arrive:

```python
>>> # Initialize session for streaming
>>> inference_session = processor.init_video_session(
...     inference_device=device,
...     dtype=torch.bfloat16,
... )

>>> # Process frames one by one
>>> for frame_idx, frame in enumerate(video_frames[:10]):  # Process first 10 frames
...     inputs = processor(images=frame, device=device, return_tensors="pt")
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
...     sam2_video_output = model(inference_session=inference_session, frame=inputs.pixel_values[0])
...
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=inputs.original_sizes, binarize=False
...     )[0]
...     print(f"Frame {frame_idx}: mask shape {video_res_masks.shape}")
```

#### Video Batch Processing for Multiple Objects

Track multiple objects simultaneously in video by adding them all at once:

```python
>>> # Initialize video session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     dtype=torch.bfloat16,
... )

>>> # Add multiple objects on the first frame using batch processing
>>> ann_frame_idx = 0
>>> obj_ids = [2, 3]  # Track two different objects
>>> input_points = [
...     [[[200, 300], [230, 250], [275, 175]], [[400, 150]]]
... ]  # Object 2: 3 points (2 positive, 1 negative); Object 3: 1 point
>>> input_labels = [
...     [[1, 1, 0], [1]]
... ]  # Object 2: positive, positive, negative; Object 3: positive

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
>>> video_res_masks = processor.post_process_masks(
...     [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
... )[0]
>>> print(f"Generated masks for {video_res_masks.shape[0]} objects")
Generated masks for 2 objects

>>> # Propagate all objects through the video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = {
...         obj_id: video_res_masks[i]
...         for i, obj_id in enumerate(inference_session.obj_ids)
...     }

>>> print(f"Tracked {len(inference_session.obj_ids)} objects through {len(video_segments)} frames")
Tracked 2 objects through 180 frames
```

<!-- TODO replace with sam2 resources -->
## Resources

A list of official Hugging Face and community (indicated by 🌎) resources to help you get started with SAM.

- [Demo notebook 🌎](https://colab.research.google.com/drive/1Z0NGLE7p8qnc9UpuI8KBETHd2xBbOEhv?usp=sharing) for using the model, contributed by [Sangbum Choi](https://github.com/SangbumChoi).

## Sam2VideoConfig

[[autodoc]] Sam2VideoConfig

## Sam2VideoMaskDecoderConfig

[[autodoc]] Sam2VideoMaskDecoderConfig

## Sam2VideoPromptEncoderConfig

[[autodoc]] Sam2VideoPromptEncoderConfig

## Sam2VideoProcessor

[[autodoc]] Sam2VideoProcessor
    - __call__
    - post_process_masks
    - init_video_session
    - add_inputs_to_inference_session

## Sam2VideoVideoProcessor

[[autodoc]] Sam2VideoVideoProcessor

## Sam2VideoInferenceSession

[[autodoc]] Sam2VideoInferenceSession

## Sam2VideoModel

[[autodoc]] Sam2VideoModel
    - forward
    - propagate_in_video_iterator
