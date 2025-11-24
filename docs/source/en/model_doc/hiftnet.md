<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# HiFTNet

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The HiFTNet model is a neural vocoder that converts mel spectrograms to waveforms. It combines a Neural Source Filter with an Inverse STFT Network (ISTFTNet) to achieve high-quality speech synthesis with efficient computation.

HiFTNet was introduced in the paper "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis" and further improved with the HiFT architecture described in [HiFTNet](https://arxiv.org/abs/2309.09493).

This model was integrated from the [Chatterbox](https://github.com/resemble-ai/chatterbox) implementation.

## Usage example

Here is a quick example of how to convert mel spectrograms to audio using this model:

```python
>>> from transformers import HiFTNetModel
>>> import torch

>>> # Load the model
>>> model = HiFTNetModel.from_pretrained("path/to/model")

>>> # Prepare mel spectrogram input
>>> # Shape: (batch_size, mel_time_steps, mel_bins)
>>> mel_spectrogram = torch.randn(1, 100, 80)

>>> # Generate waveform
>>> waveform = model(mel_spectrogram)
>>> # waveform shape: (batch_size, audio_samples)
>>> print(waveform.shape)  # torch.Size([1, 48000]) for a 100-frame mel
```

For streaming or real-time synthesis with source caching:

```python
>>> from transformers import HiFTNetModel
>>> import torch

>>> model = HiFTNetModel.from_pretrained("path/to/model")

>>> # First chunk
>>> mel_chunk1 = torch.randn(1, 80, 50)  # (batch, mel_bins, time)
>>> waveform1, source_cache = model.generate(mel_chunk1)

>>> # Second chunk with caching for seamless continuation
>>> mel_chunk2 = torch.randn(1, 80, 50)
>>> waveform2, source_cache = model.generate(mel_chunk2, cache_source=source_cache)
```

## HiFTNetConfig

[[autodoc]] HiFTNetConfig

## HiFTNetModel

[[autodoc]] HiFTNetModel
    - forward
    - generate

