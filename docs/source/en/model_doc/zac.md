<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was contributed to Hugging Face Transformers on 2026-11-15.*

# ZAC-1

<div class="flex flex-wrap space-x-1">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

ZAC-1 (Zyphra Audio Codec 1) is the 44.1 kHz speech codec of [ZONOS3-Mini](./zonos3_mini), released by Zyphra with
[ZONOS3-Mini](https://www.zyphra.com/post/zonos3-mini). It encodes audio into 15 frames per second of 10 codebooks of
4096 codes:

- **Codebook 0 is semantic**: the features of a frozen [Wav2Vec2-BERT 2.0](./wav2vec2-bert) teacher (its first 16 layers),
  averaged onto the code frames, go through a ConvNeXt encoder and one codebook.
- **Codebooks 1-9 are acoustic**: a causal convolutional front end and patched causal transformer stages encode the
  audio at 44160 Hz into a latent; nine residual codebooks quantize the latent minus the decoded semantic prediction.
- **The decoder is causal**: transformer stages with sliding-window attention bring the codes from 15 Hz to 240 Hz
  features, and a multi-resolution iSTFT head (two branches with 736- and 2208-sample windows, mixed by a linear FIR
  filter) synthesizes the audio.

Audio is encoded in lanes of 20 seconds; every lane after the first re-reads the previous 10 seconds as left context.

Two decoders are released: `keep2_plain` (2 layers per transformer stage, the default) and `full_mr` (32/12/12/12/12
layers). The checkpoint [Zyphra/ZAC-1-hf](https://huggingface.co/Zyphra/ZAC-1-hf) holds the `keep2_plain` decoder on
`main` and the `full_mr` decoder on the `full_mr` branch (`AutoModel.from_pretrained("Zyphra/ZAC-1-hf",
revision="full_mr")`).

The decoder outputs audio at the raw decoder level: multiply it by `10 ** (-3 / 20)` (ZAC-1's `output_gain()`, a 3 dB
headroom) for playback level. The [`Zonos3MiniProcessor`] applies this gain when decoding speech.

The original code can be found at [Zyphra/ZAC-1](https://github.com/Zyphra/ZAC-1).

## Usage

### Encode and decode

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

model_id = "Zyphra/ZAC-1-hf"
model = AutoModel.from_pretrained(model_id, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio = dataset[0]["audio"]["array"]

inputs = feature_extractor(audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(model.device)
audio_codes = model.encode(**inputs).audio_codes  # (batch_size, 10, num_frames)
audio_values = model.decode(audio_codes).audio_values  # (batch_size, 1, num_frames * 2940)

# or both at once
outputs = model(**inputs)
audio_codes, audio_values = outputs.audio_codes, outputs.audio_values

# playback level
audio_values = audio_values * 10 ** (-3 / 20)
```

### Batched inference

The feature extractor pads the clips; `audio_codes_mask` marks each clip's frames.

```python
audios = [dataset[i]["audio"]["array"] for i in range(3)]
inputs = feature_extractor(audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(model.device)
encoded = model.encode(**inputs)
for codes, mask in zip(encoded.audio_codes, encoded.audio_codes_mask):
    codes = codes[:, mask.bool()]  # (10, num_frames)
    audio_values = model.decode(codes[None]).audio_values
```

### `torch.compile`

The decoder runs on fixed-size tensors and compiles to a full graph; the encoder's lane schedule depends on the audio
length.

```python
model.decode = torch.compile(model.decode, fullgraph=True)
audio_values = model.decode(audio_codes).audio_values
```

The encoder runs in the model's dtype. In bfloat16, the Wav2Vec2-BERT teacher's features differ slightly from the
original encoder's, so the codes may differ slightly too; float32 reproduces them exactly.

## ZacConfig

[[autodoc]] ZacConfig

## ZacFeatureExtractor

[[autodoc]] ZacFeatureExtractor
    - __call__

## ZacModel

[[autodoc]] ZacModel
    - encode
    - decode
    - forward
