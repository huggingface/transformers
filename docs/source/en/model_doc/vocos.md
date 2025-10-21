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

This model was contributed by [Manal El Aidouni](https://huggingface.co/Manel) and [Eric Bezzam](https://huggingface.co/bezzam). The Transformers-compatible checkpoints can be found below:
- Mel-spectrogram variant: https://huggingface.co/hf-audio/vocos-mel-24khz
- EnCodec variant: https://huggingface.co/hf-audio/vocos-encodec-24khz


The original code can be found [here](https://github.com/gemelo-ai/vocos) and original checkpoints [here](https://huggingface.co/charactr/vocos-mel-24khz) and [here](https://huggingface.co/charactr/vocos-encodec-24khz).

## Usage

There are two Vocos variants that can be used with the `VocosModel`.

### Mel-spectrogram variant 

You can use `VocosProcessor`  to turn a raw waveform into mel-spectrogram features and feed them into `VocosModel` to generate high quality audio. You can also plug `VocosModel` in as a standalone vocoder component within a larger audio generation pipeline (for example the [YuE](https://github.com/multimodal-art-projection/YuE) model).

To use the mel-spectrogram pathway, pass audio to the processor without specifying the bandwidth:

```python 
from datasets import load_dataset, Audio
from transformers import VocosModel, VocosProcessor
from scipy.io.wavfile import write as write_wav
    
# load model and processor
model_id = "hf-audio/vocos-mel-24khz"
processor = VocosProcessor.from_pretrained(model_id)
model = VocosModel.from_pretrained(model_id, device_map="auto")
sampling_rate = processor.feature_extractor.sampling_rate
    
# load audio sample
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
audio = ds[0]["audio"]["array"]

inputs = processor(audio=audio, sampling_rate=sampling_rate).to(model.device)
print(inputs.audio_spectrogram.shape)
# -- (batch, mel, frame): [1, 100, 549]
outputs = model(**inputs)
audio = outputs.audio
print(audio.shape)
# --  (batch, time): [1, 140288]

# save audio to file
write_wav("vocos.wav", sampling_rate, audio[0].detach().cpu().numpy())
```

To listen from an IPython notebook:
```python
from IPython.display import Audio

Audio(audio.detach().numpy(), rate=sampling_rate)
```

### Neural audio codec variant (with EnCodec)

Instead of reconstructing audio from mel spectrograms, recent research has started to utilize learned neural audio codec features. Vocos supports [EnCodec](./encodec)-based reconstruction for high-quality audio generation.

The EnCodec neural audio codec encodes the input audio into discrete tokens with Encodec’s Residual Vector Quantization (RVQ). These codes are then converted into embedding that serve as input to `VocosModel`.

A desired target `bandwidth` value is required for EnCodec. The supported bandwidths are [1.5, 3, 6, 12] kbps.  The selected bandwidth determines the number of quantizers/codebooks used by the RVQ of Encodec, namely [2, 4, 6, 8] quantizers respectively.

```python 
from datasets import load_dataset, Audio
from transformers import VocosModel, VocosProcessor
from scipy.io.wavfile import write as write_wav
    
# can be chosen from [1.5, 3, 6, 12]
bandwidth = 6.0

# load model and processor
model_id = "hf-audio/vocos-encodec-24khz"
processor = VocosProcessor.from_pretrained(model_id)
model = VocosModel.from_pretrained(model_id, device_map="auto")
sampling_rate = processor.feature_extractor.sampling_rate
    
# load audio sample
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
audio = ds[0]["audio"]["array"]

inputs = processor(audio=audio, bandwidth=bandwidth, sampling_rate=sampling_rate).to(model.device)
print(inputs.input_features.shape)
# -- (batch, codes, frame): [1, 128, 440]
outputs = model(**inputs)
audio = outputs.audio
print(audio.shape)
# --  (batch, time): [1, 140800]

# save audio to file
write_wav("vocos_encodec.wav", sampling_rate, audio[0].detach().cpu().numpy())
```

### Reconstructing audio from Bark tokens

The EnCodec variant can also process precomputed RVQ codes directly. You provide quantized audio codes as input to the processor, which converts them into embeddings for the Vocos model.

Bark is a text-to-speech model that encodes input text into discrete EnCodec RVQ codes, then uses EnCodec to convert those codes into an audio waveform. The Vocos vocoder is often integrated with Bark instead of relying only on the EnCodec's decoder for better audio quality.

Below is an example using the Transformers implementation of  [Bark](./bark) to generate quantized codes from text, then decoding them with `VocosProcessor` and `VocosModel`:

```python 
from transformers import VocosModel, VocosProcessor, BarkProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig, BarkCoarseGenerationConfig, BarkFineGenerationConfig
from scipy.io.wavfile import write as write_wav

# load the Bark model and processor
bark_id = "suno/bark-small"
bark_processor = BarkProcessor.from_pretrained(bark_id)
bark = BarkModel.from_pretrained(bark_id, device_map="auto")

text_prompt = "We've been messing around with this new model called Vocos."
bark_inputs = bark_processor(text_prompt, return_tensors="pt").to(bark.device)

# building generation configs for each stage
semantic_generation_config = BarkSemanticGenerationConfig(**bark.generation_config.semantic_config)
coarse_generation_config = BarkCoarseGenerationConfig(**bark.generation_config.coarse_acoustics_config)
fine_generation_config = BarkFineGenerationConfig(**bark.generation_config.fine_acoustics_config)

# generating the RVQ codes
semantic_tokens = bark.semantic.generate(
    **bark_inputs,
    semantic_generation_config=semantic_generation_config,
)

coarse_tokens = bark.coarse_acoustics.generate(
    semantic_tokens,
    semantic_generation_config=semantic_generation_config,
    coarse_generation_config=coarse_generation_config,
    codebook_size=bark.generation_config.codebook_size,
)

fine_tokens = bark.fine_acoustics.generate(
    coarse_tokens,
    semantic_generation_config=semantic_generation_config,
    coarse_generation_config=coarse_generation_config,
    fine_generation_config=fine_generation_config,
    codebook_size=bark.generation_config.codebook_size,
)

codes = fine_tokens.squeeze(0)
# -- `codes` shape (8 codebooks, * frames)

# Reconstruct audio with Vocos from codes
vocos_id = "hf-audio/vocos-encodec-24khz"
processor = VocosProcessor.from_pretrained(vocos_id)
vocos_model = VocosModel.from_pretrained(vocos_id, device_map="auto")
sampling_rate = processor.feature_extractor.sampling_rate

# generate audio
inputs = processor(codes=codes.to("cpu"), bandwidth=6.0).to(vocos_model.device)   
audio = vocos_model(**inputs).audio

# save audio to file
write_wav("vocos_bark.wav", sampling_rate, audio[0].detach().cpu().numpy())
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
