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

# Vocos

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Vocos model was proposed in [**Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis**](https://huggingface.co/papers/2306.00814) by Hubert Siuzdak.

Vocos is a GAN-based neural vocoder designed for high quality audio synthesis in text to speech (TTS) pipelines and related tasks. Traditional time-domain vocoders rely on transposed convolutions for upsampling, which degrades temporal resolution across all layers and introduces aliasing artifacts into synthesized speech.
Instead, Vocos represents audio signals in the time-frequency domain, it's trained to predict the complex Short Time Fourier Transform (STFT) coefficients, magnitude and phase, and uses the computationally inverse STFT (ISTFT) for upsampling, which maintains the same temporal resolution throughout the network and converts directly to speech waveforms.

Vocos delivers the same high audio quality while achieving  30× faster inference speed on CPU and outperforming HiFi-GAN in both VISQOL and PESQ scores.

Vocos supports both a mel-spectrogram variant and an EnCodec-based variant (24 kHz), accessible via the same [VocosModel](#vocosmodel) and [VocosProcessor](#vocosprocessor) API. See usage below.

The abstract of the paper states the following:

*Recent advancements in neural vocoding are predominantly driven by Generative Adversarial Networks (GANs) operating in the time-domain. While effective, this approach neglects the inductive bias offered by time-frequency representations, resulting in reduntant and computionally-intensive upsampling operations. Fourier-based time-frequency representation is an appealing alternative, aligning more accurately with human auditory perception, and benefitting from well-established fast algorithms for its computation. Nevertheless, direct reconstruction of complex-valued spectrograms has been historically problematic, primarily due to phase recovery issues. This study seeks to close this gap by presenting Vocos, a new model that directly generates Fourier spectral coefficients. Vocos not only matches the state-of-the-art in audio quality, as demonstrated in our evaluations, but it also substantially improves computational efficiency, achieving an order of magnitude increase in speed compared to prevailing time-domain neural vocoding approaches.*

Demos can be found in this [post](https://gemelo-ai.github.io/vocos/).

This model was contributed by [Manal El Aidouni](https://huggingface.co/Manel). The original code can be found [here](https://github.com/gemelo-ai/vocos) and original checkpoints [here](https://huggingface.co/charactr/vocos-mel-24khz) and [here](https://huggingface.co/charactr/vocos-encodec-24khz).

## Usage

There are two Vocos variants that can be used with the `VocosModel`.

### Mel-spectrogram variant 

You can use `VocosProcessor`  to turn a raw waveform into mel-spectrogram features and feed them into `VocosModel` to generate high quality audio. You can also plug `VocosModel` in as a standalone vocoder component within a larger audio generation pipeline (for example the [YuE](https://github.com/multimodal-art-projection/YuE) model). To use the mel-spectrogram pathway, pass audio without specifying the bandwidth:

```python 
from datasets import load_dataset, Audio
from transformers import VocosModel, VocosProcessor
    
# load model and processor
model_id = "Manel/vocos-mel-24khz"
processor = VocosProcessor.from_pretrained(model_id)
model = VocosModel.from_pretrained(model_id)
    
# load audio sample
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
audio_sample= ds[0]["audio"]["array"]

inputs = processor(audio=audio)
# -- `inputs.features` shape (batch, mel, frame): [1, 100, 550]
audio = model(**inputs)
# -- `audio` shape (batch, time): [1, 140544]
```

### Neural audio codec variant (with EnCodec)

Instead of reconstructing audio from mel spectrograms, recent research has started to utilize learned neural audio codec features. Vocos supports [EnCodec](./encodec)-based reconstruction for high-quality audio generation.

The EnCodec neural audio codec encodes the input audio into discrete tokens with Encodec’s Residual Vector Quantization (RVQ). These codes are then converted into embedding that serve as input to `VocosModel`.

A desired target `bandwidth` value is required for EnCodec. The supported bandwidths are  [1.5, 3, 6, 12] kbps.  The selected bandwidth determines the number of quantizers/codebooks used by the RVQ of Encodec, namely [2, 4, 6, 8] quantizers respectively.

```python 
from datasets import load_dataset, Audio
from transformers import VocosModel, VocosProcessor

# load model and processor
model_id = "Manel/vocos-encodec-24khz"
processor = VocosProcessor.from_pretrained(model_id)
model = VocosModel.from_pretrained(model_id)

# load audio sample
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))  
audio_sample = ds[0]["audio"]["array"]

inputs = processor(audio=audio_sample, bandwidth=6.0)
# -- `inputs.features` shape (batch, channels, frame): [1, 128, 440]
audio = model(**inputs)
# -- `audio` shape (batch, time): [1, 140800]
```

The EnCodec variant can also process precomputed RVQ codes directly. You provide quantized audio codes as input to the processor, which converts them into embeddings for the VocosModel.

```python
from transformers import VocosModel, VocosProcessor
import torch

# load model and processor
model_id = "Manel/vocos-encodec-24khz"
model = VocosModel.from_pretrained(model_id)
processor = VocosProcessor.from_pretrained(model_id)

# Generate random codes for 6 kbps (8 codeboooks, 200 frames)
audio_codes = torch.randint(low=0, high=1024, size=(8, 200))  
inputs = processor(codes=audio_codes, bandwidth=6.0)
# -- `inputs.features` shape (batch, channels, frame): [1, 128, 200]
audio = model(**inputs)
# -- `audio` shape (batch, time): [1, 64000]
```


## VocosConfig

[[autodoc]] VocosConfig

## VocosFeatureExtractor

[[autodoc]] VocosFeatureExtractor

## VocosProcessor

[[autodoc]] VocosProcessor
    - __call__

## VocosModel

[[autodoc]] VocosModel
    - forward
