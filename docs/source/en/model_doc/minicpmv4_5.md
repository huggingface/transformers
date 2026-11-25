<!--Copyright 2026 OpenBMB and the HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2025-09-16 and contributed to Hugging Face Transformers on 2026-11-24.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# MiniCPM-V 4.5

[MiniCPM-V 4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5) is a multimodal model from [OpenBMB](https://github.com/OpenBMB) ([paper](https://huggingface.co/papers/2509.18154)). For images it is [MiniCPM-V 2.6](minicpmv2_6) with a [Qwen3](qwen3) 8B language model. For videos, groups of up to 3 consecutive frames go through the resampler together, with each frame's timestamp added to the keys, so a group becomes 64 tokens, as many as one image.

The original code can be found [here](https://github.com/OpenBMB/MiniCPM-V).

## Usage example

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openbmb/MiniCPM-V-4_5"
model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto")
processor = AutoProcessor.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "http://images.cocodataset.org/val2017/000000039769.jpg"},
            {"type": "text", "text": "What is in this image?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=64)
print(processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0])
```

Videos go through the same chat template:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/sample_demo_1.mp4"},
            {"type": "text", "text": "Describe the video."},
        ],
    }
]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)
generated = model.generate(**inputs, max_new_tokens=64)
print(processor.post_process_image_text_to_text(generated[:, inputs["input_ids"].shape[1] :])[0])
```

## Notes

- Frames are sampled at `fps` (5 by default), as in the original `encode_video`. Videos with `fps * int(duration) > max_num_frames` (37 s and longer by default) are packed `ceil(duration * fps / max_num_frames)` frames per group, at most `max_num_packing` (3), so at most `max_num_frames * max_num_packing` frames (540) are sampled. Each group is placed as `<image>` + 64 × `<|video_pad|>` + `</image>`, one line per group.
- With `do_sample_frames=False` and no video metadata, the frames are taken as sampled at `fps`.
- At the defaults, videos of 108 s and longer are sampled to 540 frames (180 groups); a 120 s clip peaked at 45 GiB in bf16 on a GB10. Lower `max_num_frames` to use less memory.

## MiniCPMV4_5Config

[[autodoc]] MiniCPMV4_5Config

## MiniCPMV4_5VisionConfig

[[autodoc]] MiniCPMV4_5VisionConfig

## MiniCPMV4_5VisionTransformer

[[autodoc]] MiniCPMV4_5VisionTransformer
    - forward

## MiniCPMV4_5Model

[[autodoc]] MiniCPMV4_5Model
    - forward
    - get_image_features
    - get_video_features

## MiniCPMV4_5ForConditionalGeneration

[[autodoc]] MiniCPMV4_5ForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features

## MiniCPMV4_5Processor

[[autodoc]] MiniCPMV4_5Processor
    - __call__

## MiniCPMV4_5VideoProcessor

[[autodoc]] MiniCPMV4_5VideoProcessor
    - preprocess
