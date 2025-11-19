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
*This model was released on 2025-11-19 and added to Hugging Face Transformers on 2025-11-19.*

# SAM3 Video

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

## Overview

SAM3 (Segment Anything Model 3) was introduced in [SAM 3: Segment Anything with Concepts](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).

SAM3 Video performs **Promptable Concept Segmentation (PCS)** on videos. PCS takes text as input (e.g., "yellow school bus"), and predicts instance and semantic masks for **every single object** matching the concept, while preserving object identities across video frames.

The model combines a detection module (SAM3) with a tracking module (SAM2-style tracker) to enable robust object tracking across video frames using text prompts.

The abstract from the paper is the following:

*We present Segment Anything Model (SAM) 3, a unified model that detects, segments, and tracks objects in images and videos based on concept prompts, which we define as either short noun phrases (e.g., "yellow school bus"), image exemplars, or a combination of both. Promptable Concept Segmentation (PCS) takes such prompts and returns segmentation masks and unique identities for all matching object instances. To advance PCS, we build a scalable data engine that produces a high-quality dataset with 4M unique concept labels, including hard negatives, across images and videos. Our model consists of an image-level detector and a memory-based video tracker that share a single backbone. Recognition and localization are decoupled with a presence head, which boosts detection accuracy. SAM 3 doubles the accuracy of existing systems in both image and video PCS, and improves previous SAM capabilities on visual segmentation tasks. We open source SAM 3 along with our new Segment Anything with Concepts (SA-Co) benchmark for promptable concept segmentation.*

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan) and [ronghanghu](https://huggingface.co/ronghanghu).

## Usage example

### Video Segmentation and Tracking

#### Pre-loaded Video Inference

Process a video with all frames already available using text prompts:

```python
>>> from transformers import Sam3VideoModel, Sam3VideoProcessor
>>> from accelerate import Accelerator
>>> import torch

>>> device = Accelerator().device
>>> model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
>>> processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

>>> # Load video frames
>>> from transformers.video_utils import load_video
>>> video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
>>> video_frames, _ = load_video(video_url)

>>> # Initialize video inference session
>>> inference_session = processor.init_video_session(
...     video=video_frames,
...     inference_device=device,
...     processing_device="cpu",
...     video_storage_device="cpu",
...     dtype=torch.bfloat16,
... )

>>> # Add text prompt to detect and track objects
>>> text = "person"
>>> inference_session = processor.add_text_prompt(
...     inference_session=inference_session,
...     text=text,
... )

>>> # Process all frames in the video
>>> outputs_per_frame = {}
>>> for model_outputs in model.propagate_in_video_iterator(
...     inference_session=inference_session, max_frame_num_to_track=50
... ):
...     processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
...     outputs_per_frame[model_outputs.frame_idx] = processed_outputs

>>> print(f"Processed {len(outputs_per_frame)} frames")
Processed 51 frames

>>> # Access results for a specific frame
>>> frame_0_outputs = outputs_per_frame[0]
>>> print(f"Detected {len(frame_0_outputs['object_ids'])} objects")
>>> print(f"Object IDs: {frame_0_outputs['object_ids'].tolist()}")
>>> print(f"Scores: {frame_0_outputs['scores'].tolist()}")
>>> print(f"Boxes shape (XYXY format, absolute coordinates): {frame_0_outputs['boxes'].shape}")
>>> print(f"Masks shape: {frame_0_outputs['masks'].shape}")
```

#### Streaming Video Inference

<div class="warning">
⚠️ **Note on Streaming Inference Quality**: Streaming inference disables hotstart heuristics that remove unmatched and duplicate objects, as these require access to future frames to make informed decisions. This may result in more false positive detections and duplicate object tracks compared to pre-loaded video inference. For best results, use pre-loaded video inference when all frames are available.
</div>

For real-time applications, SAM3 Video supports processing video frames as they arrive:

```python
>>> # Initialize session for streaming
>>> streaming_inference_session = processor.init_video_session(
...     inference_device=device,
...     processing_device="cpu",
...     video_storage_device="cpu",
...     dtype=torch.bfloat16,
... )

>>> # Add text prompt
>>> text = "person"
>>> streaming_inference_session = processor.add_text_prompt(
...     inference_session=streaming_inference_session,
...     text=text,
... )

>>> # Process frames one by one (streaming mode)
>>> streaming_outputs_per_frame = {}
>>> for frame_idx, frame in enumerate(video_frames[:50]):  # Process first 50 frames
...     # First, process the frame using the processor
...     inputs = processor(images=frame, device=device, return_tensors="pt")
...
...     # Process frame using streaming inference - pass the processed pixel_values
...     model_outputs = model(
...         inference_session=streaming_inference_session,
...         frame=inputs.pixel_values[0],  # Provide processed frame - this enables streaming mode
...         reverse=False,
...     )
...
...     # Post-process outputs with original_sizes for proper resolution handling
...     processed_outputs = processor.postprocess_outputs(
...         streaming_inference_session,
...         model_outputs,
...         original_sizes=inputs.original_sizes,  # Required for streaming inference
...     )
...     streaming_outputs_per_frame[frame_idx] = processed_outputs
...
...     if (frame_idx + 1) % 10 == 0:
...         print(f"Processed {frame_idx + 1} frames...")

>>> print(f"✓ Streaming inference complete! Processed {len(streaming_outputs_per_frame)} frames")
✓ Streaming inference complete! Processed 50 frames

>>> # Access results
>>> frame_0_outputs = streaming_outputs_per_frame[0]
>>> print(f"Detected {len(frame_0_outputs['object_ids'])} objects in first frame")
>>> print(f"Boxes are in XYXY format (absolute pixel coordinates): {frame_0_outputs['boxes'].shape}")
>>> print(f"Masks are at original video resolution: {frame_0_outputs['masks'].shape}")
```

## Sam3VideoConfig

[[autodoc]] Sam3VideoConfig

## Sam3VideoProcessor

[[autodoc]] Sam3VideoProcessor
    - __call__
    - postprocess_outputs
    - init_video_session
    - add_text_prompt

## Sam3VideoInferenceSession

[[autodoc]] Sam3VideoInferenceSession

## Sam3VideoSegmentationOutput

[[autodoc]] Sam3VideoSegmentationOutput

## Sam3VideoModel

[[autodoc]] Sam3VideoModel
    - forward
    - propagate_in_video_iterator

