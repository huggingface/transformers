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

*This model was released on 2024-07-29 and added to Hugging Face Transformers on 2025-08-14 and contributed by [SangbumChoi](https://github.com/SangbumChoi) and [yonigozlan](https://huggingface.co/yonigozlan).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# SAM2 Video

[Segment Anything Model 2](https://huggingface.co/papers/2304.02643) presents a foundation model for promptable visual segmentation in images and videos. It incorporates a data engine that enhances both the model and data through user interaction, resulting in the largest video segmentation dataset available. SAM 2 features a simple transformer architecture with streaming memory for real-time video processing. The model demonstrates superior performance across various tasks, achieving better accuracy in video segmentation with 3x fewer interactions and improved speed and accuracy in image segmentation compared to the original SAM. SAM 2 supports batch and video processing natively, offers enhanced segmentation quality and robustness, and exhibits superior zero-shot generalization with mixed prompts.

<hfoptions id="usage">
<hfoption id="Sam2VideoModel">

```py
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor
from transformers.video_utils import load_video

model = Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny", dtype="auto"
processor = Sam2VideoProcessor.from_pretrained("facebook/sam2.1-hiera-tiny")

video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    dtype=torch.bfloat16,
)

ann_frame_idx = 0
ann_obj_id = 1
points = [[[[210, 350]]]]
labels = [[[1]]]

processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=ann_obj_id,
    input_points=points,
    input_labels=labels,
)

outputs = model(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
)
video_res_masks = processor.post_process_masks(
    [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
)[0]
print(f"Segmentation shape: {video_res_masks.shape}")

video_segments = {}
for sam2_video_output in model.propagate_in_video_iterator(inference_session):
    video_res_masks = processor.post_process_masks(
        [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]
    video_segments[sam2_video_output.frame_idx] = video_res_masks

print(f"Tracked object through {len(video_segments)} frames")
```

</hfoption>
</hfoptions>

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

