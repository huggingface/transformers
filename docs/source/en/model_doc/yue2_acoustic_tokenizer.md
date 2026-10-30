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
*This model was contributed to Hugging Face Transformers on 2026-10-30.*

# YuE2 acoustic tokenizer

## Overview

The acoustic tokenizer of [YuE2](https://huggingface.co/m-a-p/YuE2-3B): the Oobleck variational autoencoder of [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools), with SnakeBeta activations, released as [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae). It turns 48 kHz stereo audio into continuous 64-dimensional latents at 25 Hz (one latent frame per 1920 samples) and back. YuE2 generates these latents with flow matching, and this model decodes them into the song.

The weights are released under the [CC-BY-NC-4.0](https://huggingface.co/m-a-p/YuE2-Vae/blob/main/LICENSE) license.

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam). The original code can be found on [GitHub](https://github.com/multimodal-art-projection/YuE/blob/main/src/yue2/modeling_vae.py).

## Usage

The encoder returns the mean of the latent posterior, or a sample of it with `sample=True`.

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("bezzam/yue2-acoustic-tokenizer", device_map="auto")

# 2 seconds of 48 kHz stereo audio: (batch_size, audio_channels, num_samples)
audio = 0.1 * torch.randn(1, 2, 2 * 48000, device=model.device)

latents = model.encode(audio).latents  # (batch_size, num_frames, latent_dim): (1, 50, 64)
audio_values = model.decode(latents).audio_values  # (batch_size, audio_channels, num_samples)
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
