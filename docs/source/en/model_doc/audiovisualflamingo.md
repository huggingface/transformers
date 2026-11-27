<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-07-17 and contributed to Hugging Face Transformers on 2026-11-19.*

# Audio-Visual Flamingo

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Audio-Visual Flamingo (AVF) was proposed in [Audio-Visual Flamingo: Open Audio-Visual Intelligence for Long and Complex Videos](https://huggingface.co/papers/2607.16107).
It is an open audio-visual language model for joint understanding and reasoning over images, videos and audio.

AVF pairs a SigLIP vision tower and a Whisper-style audio encoder with a Qwen2.5-7B language model, through one projector
per modality. Images are encoded at multiple scales (S2) from tiles of the image resolution, and video frames are
pooled over time. The audio track of a video is aligned with its frames: the video is split into 30 second segments, in
which the visual tokens are followed by the audio tokens of the same segment. Video and audio tokens also receive a
constrained rotary time embedding (CRTE) of their timestamps.

This model was contributed by [Lasha Koroshinadze](https://huggingface.co/lashahub).
The original code can be found [here](https://github.com/NVIDIA/audio-flamingo).

## Usage examples

### Video with audio

The audio track of a video is loaded and interleaved with its frames when `load_audio_from_video=True`.

```python
from transformers import AudioVisualFlamingoForConditionalGeneration, AutoProcessor

model_id = "nvidia/nemotron-labs-audio-visual-flamingo-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AudioVisualFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"},
            {"type": "text", "text": "Describe the visual scene and the audio of this video."},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation, add_generation_prompt=True, tokenize=True, return_dict=True, load_audio_from_video=True
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0])
```

### Images, audio and batches

Images, audio and text can be combined in a conversation, and conversations can be batched.

```python
conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "audio", "url": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dogs_barking_in_sync_with_the_music.wav"},
                {"type": "text", "text": "Describe the audio in detail."},
            ],
        }
    ],
]
inputs = processor.apply_chat_template(
    conversations, add_generation_prompt=True, tokenize=True, return_dict=True
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Usage tips

- Audio longer than `max_audio_len` seconds (600 by default) is truncated. It can be changed with
  `AutoProcessor.from_pretrained(model_id, max_audio_len=...)`, keeping in mind that 30 seconds of audio take 750
  tokens of the 32k token context.
- Videos are sampled to `num_frames` frames (128 by default) evenly spaced from the first to the last frame, which can
  be changed with `apply_chat_template(..., processor_kwargs={"num_frames": ...})` (sampling by `fps` isn't
  supported). Every 8 frames are pooled into 256 tokens.
- A single image is encoded at three scales: as 1 and 2x2 tiles of 448 pixels of the resized image, and as the grid of 9
  to 12 tiles whose aspect ratio is the closest to the image one. Each image of a multi-image prompt is encoded as a
  single tile, as in training.
- The chat template marks the assistant turns, so `apply_chat_template(..., return_assistant_tokens_mask=True)`
  returns the mask of the tokens to train on.

## AudioVisualFlamingoConfig

[[autodoc]] AudioVisualFlamingoConfig

## AudioVisualFlamingoProcessor

[[autodoc]] AudioVisualFlamingoProcessor
    - __call__

## AudioVisualFlamingoImageProcessor

[[autodoc]] AudioVisualFlamingoImageProcessor
    - preprocess

## AudioVisualFlamingoVideoProcessor

[[autodoc]] AudioVisualFlamingoVideoProcessor

## AudioVisualFlamingoModel

[[autodoc]] AudioVisualFlamingoModel
    - forward
    - get_image_features
    - get_video_features
    - get_audio_features

## AudioVisualFlamingoForConditionalGeneration

[[autodoc]] AudioVisualFlamingoForConditionalGeneration
    - forward
    - get_image_features
    - get_video_features
    - get_audio_features
