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

*This model was released on 2025-01-13 and added to Hugging Face Transformers on 2025-09-29 and contributed by [yonigozlan](https://huggingface.co/yonigozlan).*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
    </div>
</div>

# EdgeTAMVideo

[EdgeTAM: On-Device Track Anything Model](https://huggingface.co/papers/2501.07256) extends SAM 2 for efficient real-time video segmentation on mobile devices by introducing a 2D Spatial Perceiver. This architecture optimizes memory attention mechanisms, addressing latency issues caused by memory attention blocks. The 2D Spatial Perceiver uses a lightweight Transformer with fixed learnable queries, split into global and patch-level groups to preserve spatial structure. Additionally, a distillation pipeline enhances performance without increasing inference time. EdgeTAM achieves high J&F scores on DAVIS 2017, MOSE, SA-V val, and SA-V test, while operating at 16 FPS on iPhone 15 Pro Max.

<hfoptions id="usage">
<hfoption id="EdgeTamModel">

```py
import torch
from transformers.video_utils import load_video
from transformers import EdgeTamVideoModel, AutoProcessor

model = EdgeTamVideoModel.from_pretrained("yonigozlan/edgetam-video-1", dtype="auto")
processor = Sam2VideoProcessor.from_pretrained("yonigozlan/edgetam-video-1")

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

