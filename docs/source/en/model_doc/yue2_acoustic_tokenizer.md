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
*This model was contributed to Hugging Face Transformers on 2026-11-13.*

# YuE2 acoustic tokenizer

## Overview

The acoustic tokenizer of [YuE2](https://huggingface.co/m-a-p/YuE2-3B) is a variational autoencoder (VAE), following the Oobleck architecture of [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) (described in Section 2.1 of [this paper](https://huggingface.co/papers/2404.10301)).

It turns 48 kHz stereo audio into continuous 64-dimensional latents at 25 Hz (one latent frame per 1920 samples) and back. YuE2 generates these latents with flow matching, and this model decodes them into the song.

The weights were released as [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) under the [CC-BY-NC-4.0](https://huggingface.co/m-a-p/YuE2-Vae/blob/main/LICENSE) license.

Transformers modeling code was contributed by [Eric Bezzam](https://huggingface.co/bezzam). The original code can be found on [GitHub](https://github.com/multimodal-art-projection/YuE/blob/main/src/yue2/modeling_vae.py).

## Usage

The encoder returns the mean of the latent posterior, or a sample of it with `sample=True`.

```python
import soundfile as sf
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, AutoModel

model_id, revision = "m-a-p/YuE2-Vae", "9a63959a7e34304bd31ef7bf9940ba03ee978c51"
model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
sampling_rate = feature_extractor.sampling_rate  # 48 kHz

# 2 seconds of a stereo song, from 1 minute in
path = hf_hub_download("m-a-p/YuE2-3B", "assets/audio/tonight-awake.mp3")
audio, _ = sf.read(path, start=60 * sampling_rate, frames=2 * sampling_rate, dtype="float32")  # (num_samples, audio_channels)
inputs = feature_extractor(audio.T, sampling_rate=sampling_rate, return_tensors="pt")
audio = inputs["input_values"].to(model.device)  # (batch_size, audio_channels, num_samples): (1, 2, 96000)

latents = model.encode(audio).latents  # (batch_size, num_frames, latent_dim): (1, 50, 64)
audio_values = model.decode(latents).audio_values  # (batch_size, audio_channels, num_samples): (1, 2, 95936)
```

### Speed-up with `torch.compile`

You can speed up inference with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html). The first few calls will be slower due to compilation overhead, but subsequent calls will be faster.

On an NVIDIA A100-SXM4-80GB, we observed a speed-up of ~2.1 for a batch size of 4 with 10 seconds of audio each (228 ms to 109 ms for encoding and decoding, [script](https://gist.github.com/ebezzam/d03a9cc1d2bfc31c39c8e3704f73e5bc#file-benchmark_torch_compile-py)).

```python
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, AutoModel

model_id, revision = "m-a-p/YuE2-Vae", "9a63959a7e34304bd31ef7bf9940ba03ee978c51"
model = AutoModel.from_pretrained(model_id, revision=revision, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, revision=revision)
sampling_rate = feature_extractor.sampling_rate  # 48 kHz

# 4 consecutive excerpts of 10 seconds of a stereo song, from 1 minute in
path = hf_hub_download("m-a-p/YuE2-3B", "assets/audio/tonight-awake.mp3")
excerpt_length = 10 * sampling_rate
audio, _ = sf.read(path, start=60 * sampling_rate, frames=4 * excerpt_length, dtype="float32")  # (num_samples, audio_channels)
excerpts = [audio[i : i + excerpt_length].T for i in range(0, len(audio), excerpt_length)]
inputs = feature_extractor(excerpts, sampling_rate=sampling_rate, return_tensors="pt")
audio = inputs["input_values"].to(model.device)  # (batch_size, audio_channels, num_samples)

model.forward = torch.compile(model.forward, fullgraph=True)

# Warmup (includes compilation on first call)
for _ in range(10):
    with torch.inference_mode():
        _ = model(audio)

with torch.inference_mode():
    output = model(audio)
print("Audio values shape:", output.audio_values.shape)
```

## YuE2AcousticTokenizerConfig

[[autodoc]] YuE2AcousticTokenizerConfig

## YuE2AcousticTokenizerEncoderConfig

[[autodoc]] YuE2AcousticTokenizerEncoderConfig

## YuE2AcousticTokenizerDecoderConfig

[[autodoc]] YuE2AcousticTokenizerDecoderConfig

## YuE2AcousticTokenizerModel

[[autodoc]] YuE2AcousticTokenizerModel
    - encode
    - decode
    - forward

## YuE2AcousticTokenizerEncoderModel

[[autodoc]] YuE2AcousticTokenizerEncoderModel
    - forward

## YuE2AcousticTokenizerDecoderModel

[[autodoc]] YuE2AcousticTokenizerDecoderModel
    - forward
