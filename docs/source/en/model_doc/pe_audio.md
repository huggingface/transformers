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

# PE Audio

[PE Audio](https://huggingface.co/papers/2504.13181) is the audio branch of Meta's Perception Encoder family. It contrastively aligns raw waveforms with text into a shared embedding space, trained on paired audio–caption data for cross-modal retrieval and zero-shot audio classification.

Two heads are exposed on top of the same encoder. [`PeAudioModel`] returns one pooled embedding per clip for clip-level retrieval, while [`PeAudioFrameLevelModel`] returns one embedding every 40 ms for event localization and fine-grained temporal analysis.

You can find all the official PE Audio checkpoints under the [perception-encoder-audio-visual](https://huggingface.co/collections/facebook/perception-encoder-audio-visual) collection.

## Quickstart

```py
import torch
from datasets import load_dataset
from transformers import AutoProcessor, PeAudioModel

processor = AutoProcessor.from_pretrained("facebook/pe-av-large")
model = PeAudioModel.from_pretrained(
    "facebook/pe-av-large",
    device_map="auto",
)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio = ds[0]["audio"]["array"]
labels = ["a dog barking", "a person speaking", "music playing"]

audio_inputs = processor.feature_extractor(audio, sampling_rate=48_000, return_tensors="pt").to(model.device)
text_inputs = processor.tokenizer(labels, padding=True, return_tensors="pt").to(model.device)
inputs = {**audio_inputs, **text_inputs}

with torch.no_grad():
    outputs = model(**inputs)

probs = outputs.logits_audio_text.sigmoid()
print({label: p.item() for label, p in zip(labels, probs[0])})
```

## Usage tips and notes

- Audio must be mono (`feature_size=1`) and resampled to 48 kHz — the feature extractor warns but does not resample for you. Stereo input is not supported.
- Variable-length audio is handled with `padding_mask` (not the usual `attention_mask`). The mask is downsampled internally by `dac_config.hop_length` before it reaches the encoder, so pass the raw waveform-resolution mask that the feature extractor returns.
- [`PeAudioModel`] returns logits of shape `(n_audio, n_text)`. [`PeAudioFrameLevelModel`] returns `(n_audio, n_text, n_frames)` with one frame every 40 ms. Pick the class that matches the task — they share weights so swapping is cheap.
- The text tower is a shared encoder loaded via `AutoModel` from `config.text_config`. The tokenizer is attached to the processor via `AutoTokenizer`, not a dedicated class.

## PeAudioConfig

[[autodoc]] PeAudioConfig

## PeAudioEncoderConfig

[[autodoc]] PeAudioEncoderConfig

## PeAudioFeatureExtractor

[[autodoc]] PeAudioFeatureExtractor
    - __call__

## PeAudioProcessor

[[autodoc]] PeAudioProcessor

## PeAudioEncoder

[[autodoc]] PeAudioEncoder
    - forward

## PeAudioModel

[[autodoc]] PeAudioModel
    - forward

## PeAudioFrameLevelModel

[[autodoc]] PeAudioFrameLevelModel
    - forward
