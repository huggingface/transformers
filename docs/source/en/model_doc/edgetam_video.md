<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2025-01-13 and added to Hugging Face Transformers on 2025-09-29.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# EdgeTAMVideo

## Overview

The EdgeTAM model was proposed in [EdgeTAM: On-Device Track Anything Model](https://huggingface.co/papers/2501.07256) Chong Zhou, Chenchen Zhu, Yunyang Xiong, Saksham Suri, Fanyi Xiao, Lemeng Wu, Raghuraman Krishnamoorthi, Bo Dai, Chen Change Loy, Vikas Chandra, Bilge Soran.

EdgeTAM is an efficient adaptation of SAM 2 that introduces a 2D Spatial Perceiver architecture to optimize memory attention mechanisms for real-time video segmentation on mobile devices.

The abstract from the paper is the following:

*On top of Segment Anything Model (SAM), SAM 2 further extends its capability from image to video inputs through a memory bank mechanism and obtains a remarkable performance compared with previous methods, making it a foundation model for video segmentation task. In this paper, we aim at making SAM 2 much more efficient so that it even runs on mobile devices while maintaining a comparable performance. Despite several works optimizing SAM for better efficiency, we find they are not sufficient for SAM 2 because they all focus on compressing the image encoder, while our benchmark shows that the newly introduced memory attention blocks are also the latency bottleneck. Given this observation, we propose EdgeTAM, which leverages a novel 2D Spatial Perceiver to reduce the computational cost. In particular, the proposed 2D Spatial Perceiver encodes the densely stored frame-level memories with a lightweight Transformer that contains a fixed set of learnable queries. Given that video segmentation is a dense prediction task, we find preserving the spatial structure of the memories is essential so that the queries are split into global-level and patch-level groups. We also propose a distillation pipeline that further improves the performance without inference overhead. As a result, EdgeTAM achieves 87.7, 70.0, 72.3, and 71.7 J&F on DAVIS 2017, MOSE, SA-V val, and SA-V test, while running at 16 FPS on iPhone 15 Pro Max.*

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).
The original code can be found [here](https://github.com/facebookresearch/EdgeTAM).

## Usage example

### Video Segmentation and Tracking

EdgeTAM Video's key strength is its ability to track objects across video frames efficiently on mobile devices. Here's how to use it for video segmentation:

#### Basic Video Tracking

```python
>>> from transformers import EdgeTamVideoModel, Sam2VideoProcessor
from accelerate import Accelerator
>>> import torch

>>> device = Accelerator().device
>>> model = EdgeTamVideoModel.from_pretrained("yonigozlan/edgetam-video-1").to(device, dtype=torch.bfloat16)
>>> processor = Sam2VideoProcessor.from_pretrained("yonigozlan/edgetam-video-1")

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
Segmentation shape: torch.Size([1, 1, 540, 960])

>>> # Propagate through the entire video
>>> video_segments = {}
>>> for sam2_video_output in model.propagate_in_video_iterator(inference_session):
...     video_res_masks = processor.post_process_masks(
...         [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
...     )[0]
...     video_segments[sam2_video_output.frame_idx] = video_res_masks

>>> print(f"Tracked object through {len(video_segments)} frames")
Tracked object through 200 frames
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
Tracked 2 objects through 200 frames
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

For real-time applications, EdgeTAM Video supports processing video frames as they arrive:

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

Frame 0: mask shape torch.Size([1, 1, 540, 960])
...
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
Tracked 2 objects through 200 frames
```

## EdgeTamVideoMaskDecoderConfig

[[autodoc]] EdgeTamVideoMaskDecoderConfig

## EdgeTamVideoPromptEncoderConfig

[[autodoc]] EdgeTamVideoPromptEncoderConfig

## EdgeTamVideoConfig

[[autodoc]] EdgeTamVideoConfig

## EdgeTamVideoInferenceSession

[[autodoc]] EdgeTamVideoInferenceSession

## EdgeTamVideoModel

[[autodoc]] EdgeTamVideoModel
    - forward
