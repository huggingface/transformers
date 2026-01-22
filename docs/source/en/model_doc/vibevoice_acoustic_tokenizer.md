<!--Copyright 2026 Microsoft and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-08-26 and added to Hugging Face Transformers on 2026-01-22.*

# VibeVoice Acoustic Tokenizer

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>


## Overview

[VibeVoice](https://huggingface.co/papers/2508.19205) is a novel framework for synthesizing high-fidelity, long-form speech with multiple speakers by employing a next-token diffusion approach within a Large Language Model (LLM) structure. It's designed to capture the authentic conversational "vibe" and is particularly suited for generating audio content like podcasts and multi-participant audiobooks.

One key feature of VibeVoice is the use of two continuous speech tokenizers, one for extracting acoustic features and another for semantic features.

A model checkpoint is available at [bezzam/VibeVoice-AcousticTokenizer](https://huggingface.co/bezzam/VibeVoice-AcousticTokenizer)

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

## Architecture

The architecture is a mirror-symmetric encoder-decoder structure. The encoder employs a hierarchical design with 7 stages of ConvNeXt-like blocks, which use 1D depth-wise causal convolutionsfor efficient streaming processing. Six downsampling layers achieve a cumulative 3200X downsampling rate from a 24kHz input, yielding 7.5 tokens/frames per second. Each encoder/decoder component has approximately 340M parameters. The training objective follows that of [DAC](./dac), including its discriminator and loss designs.

Acoustic Tokenizer adopts the principles of a Variational Autoencoder (VAE). The encoder which maps the input audio to the parameters of a latent distribution, namely the mean. Along with a fixed standard deviation, a latent vector is then sampling using the reparameterization trick. Please refer to the [technical report](https://huggingface.co/papers/2508.19205) for further details.


## Usage

Below is example usage to encode and decode audio:

```python
import torch
from scipy.io import wavfile
from transformers import AutoFeatureExtractor, VibeVoiceAcousticTokenizerModel
from transformers.audio_utils import load_audio_librosa


model_id = "bezzam/VibeVoice-AcousticTokenizer"
sampling_rate = 24000

# load audio
audio = load_audio_librosa(
    "https://huggingface.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav",
    sampling_rate=sampling_rate,
)

# load model
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
model = VibeVoiceAcousticTokenizerModel.from_pretrained(model_id, device_map="auto")
print("Model loaded on device:", model.device)
print("Model dtype:", model.dtype)

# preprocess audio
inputs = feature_extractor(
    audio,
    sampling_rate=sampling_rate,
    padding=True,
    pad_to_multiple_of=3200,
    return_tensors="pt",
).to(model.device, model.dtype)
print("Input audio shape:", inputs.input_values.shape)
# Input audio shape: torch.Size([1, 1, 224000])

# encode
with torch.no_grad():
    encoded_outputs = model.encode(inputs.input_values)
print("Latent shape:", encoded_outputs.latents.shape)
# Latent shape: torch.Size([1, 70, 64])

# VAE sampling
with torch.no_grad():
    encoded_outputs = model.sample(encoded_outputs.latents)
print("Noisy latents shape:", encoded_outputs.latents.shape)
# Noisy latents shape: torch.Size([1, 70, 64])

# decode
with torch.no_grad():
    decoded_outputs = model.decode(**encoded_outputs)
print("Reconstructed audio shape:", decoded_outputs.audio.shape)
# Reconstructed audio shape: torch.Size([1, 1, 224000])

# Save audio
output_fp = "vibevoice_acoustic_tokenizer_reconstructed.wav"
wavfile.write(output_fp, sampling_rate, decoded_outputs.audio.squeeze().float().cpu().numpy())
print(f"Reconstructed audio saved to : {output_fp}")
```


## VibeVoiceAcousticTokenizerConfig

[[autodoc]] VibeVoiceAcousticTokenizerConfig


## VibeVoiceAcousticTokenizerFeatureExtractor

[[autodoc]] VibeVoiceAcousticTokenizerFeatureExtractor
    - __call__


## VibeVoiceAcousticTokenizerModel

[[autodoc]] VibeVoiceAcousticTokenizerModel
    - encode
    - sample
    - decode
    - forward