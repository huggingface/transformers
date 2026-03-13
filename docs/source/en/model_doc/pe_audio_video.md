<!--Copyright 2025 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-16.*

# PE Audio Video (Perception Encoder Audio-Video)

## Overview

TODO

## Usage

### Basic usage

```py

model = PeAudioVideoModel.from_pretrained("facebook/pe-av-large", device_map="cuda", dtype=torch.bfloat16)
processor = PeAudioVideoProcessor.from_pretrained("facebook/pe-av-large")

from huggingface_hub import hf_hub_download

video_path = hf_hub_download(
    repo_id="eustlb/dummy-video-dataset", filename="audiobox.mp4", repo_type="dataset"
)

video_path2 = hf_hub_download(
    repo_id="eustlb/dummy-video-dataset", filename="glass_breaking.mp4", repo_type="dataset"
)

audio_path = hf_hub_download(
    repo_id="eustlb/dummy-video-dataset", filename="audiobox.mp4", repo_type="dataset"
)

audio_path2 = hf_hub_download(
    repo_id="eustlb/dummy-video-dataset", filename="glass_breaking.mp4", repo_type="dataset"
)

video_files = [video_path, video_path2]
descriptions = ["A woman and a man speaking", "A glass breaking"]
audio_files = [audio_path, audio_path2]

inputs = processor(
    videos=video_files, text=descriptions, audio=audio_files, return_tensors="pt", padding=True
)

with torch.inference_mode(), torch.autocast(model.device.type, dtype=torch.bfloat16):
    outputs = model(**inputs.to(model.device, dtype=model.dtype))

audio_embeds = outputs.audio_embeds  # Audio-only embeddings
video_embeds = outputs.video_embeds  # Video-only embeddings
audio_video_embeds = outputs.audio_video_embeds  # Joint audio-video embeddings
text_audio_embeds = outputs.text_audio_embeds  # Text embeddings aligned to audio
text_video_embeds = outputs.text_video_embeds  # Text embeddings aligned to video
text_audio_video_embeds = outputs.text_audio_video_embeds  # Text embeddings aligned to audio-video
audio_plus_text_embeds = outputs.audio_plus_text_embeds  # Joint audio and text embedding
video_plus_text_embeds = outputs.video_plus_text_embeds  # Joint video and text embedding
```

## PeAudioVideoProcessor

[[autodoc]] PeAudioVideoProcessor
    - __call__

## PeAudioVideoConfig

[[autodoc]] PeAudioVideoConfig

## PeAudioVideoEncoderConfig

[[autodoc]] PeAudioVideoEncoderConfig

## PeAudioVideoModel

[[autodoc]] PeAudioVideoModel
    - forward

## PeAudioVideoEncoder

[[autodoc]] PeAudioVideoEncoder
    - forward
