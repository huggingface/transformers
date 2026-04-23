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
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-16.*

# PE Audio Video

[PE Audio Video](https://huggingface.co/papers/2504.13181) is the joint audio–video branch of Meta's Perception Encoder family. It encodes audio and video streams together with a shared text tower, producing contrastive embeddings for every pairwise combination, audio-text, video-text, audio-video, and audio+text-video, from a single forward pass.

Internally the model aligns the video feature sequence to the audio's temporal resolution via nearest-neighbor interpolation, so clips with different frame rates from sample rates stay in lockstep. The text encoder weights are tied across the audio and video branches.

You can find all the official PE Audio Video checkpoints under the [perception-encoder-audio-visual](https://huggingface.co/collections/facebook/perception-encoder-audio-visual) collection.

## Quickstart

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, PeAudioVideoModel
from transformers.video_utils import load_video

processor = AutoProcessor.from_pretrained("facebook/pe-av-large")
model = PeAudioVideoModel.from_pretrained(
    "facebook/pe-av-large",
    device_map="auto",
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio = ds[0]["audio"]["array"]
video, _ = load_video("https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4")
labels = ["a person playing tennis with background crowd", "a dog barking in a park"]

audio_inputs = processor.feature_extractor(audio, sampling_rate=48_000, return_tensors="pt").to(model.device)
video_inputs = processor.video_processor(video, num_frames=16, return_tensors="pt").to(model.device)
text_inputs = processor.tokenizer(labels, padding=True, return_tensors="pt").to(model.device)
inputs = {**audio_inputs, **video_inputs, **text_inputs}

with torch.no_grad():
    outputs = model(**inputs)

print("audio-text:", outputs.logits_audio_text.sigmoid().tolist())
print("video-text:", outputs.logits_video_text.sigmoid().tolist())
print("audio-video:", outputs.logits_audio_video.sigmoid().tolist())
```

## Usage tips and notes

- [`PeAudioVideoModel`] requires at least two of `input_ids`, `input_values`, `pixel_values_videos` — if only two are provided it dispatches to the audio-only or video-only sub-model. Passing all three triggers the joint audio-video-text path and the full set of logit matrices in [`PeAudioVideoOutput`].
- Audio uses `padding_mask` and video uses `padding_mask_videos` simultaneously. They are independent masks; do not conflate them with `attention_mask`, which is reserved for the text tower.
- Audio–video alignment runs per-batch-element inside `_align_video_hidden_state`, so batches with very different audio/video lengths iterate rather than vectorizing. Keep batch items roughly balanced for throughput.
- The text tower's weights are tied across branches via `_tied_weights_keys` — do not try to load separate text encoders for the audio and video halves.

## PeAudioVideoConfig

[[autodoc]] PeAudioVideoConfig

## PeAudioVideoEncoderConfig

[[autodoc]] PeAudioVideoEncoderConfig

## PeAudioVideoProcessor

[[autodoc]] PeAudioVideoProcessor

## PeAudioVideoEncoder

[[autodoc]] PeAudioVideoEncoder
    - forward

## PeAudioVideoModel

[[autodoc]] PeAudioVideoModel
    - forward
