<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# VOCOS

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Vocos model was proposed in  [**Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis**](https://arxiv.org/abs/2306.00814) by Hubert Siuzdak.

Vocos is a GAN-based neural vocoder designed for high quality audio synthesis in text to speech (TTS) pipelines and related tasks. Traditional time-domain vocoders rely on transposed convolutions for upsampling, which degrades temporal resolution across all layers and introduces aliasing artifacts into synthesized speech.
Instead, Vocos represents audio signals in the time-frequency domain, it's trained to predict the complex Short Time Fourier Transform (STFT) coefficients, magnitude and phase, and uses the computationally inverse STFT (ISTFT) for upsampling, which maintains the same temporal resolution throughout the network and converts directly to speech waveforms.

Vocos delivers the same high audio quality while achieving  30× faster inference speed on CPU and outperforming HiFi-GAN in both VISQOL and PESQ scores.


The abstract of the paper states the following:

*Recent advancements in neural vocoding are predominantly driven by Generative Adversarial Networks (GANs) operating in the time-domain. While effective, this approach neglects the inductive bias offered by time-frequency representations, resulting in reduntant and computionally-intensive upsampling operations. Fourier-based time-frequency representation is an appealing alternative, aligning more accurately with human auditory perception, and benefitting from well-established fast algorithms for its computation. Nevertheless, direct reconstruction of complex-valued spectrograms has been historically problematic, primarily due to phase recovery issues. This study seeks to close this gap by presenting Vocos, a new model that directly generates Fourier spectral coefficients. Vocos not only matches the state-of-the-art in audio quality, as demonstrated in our evaluations, but it also substantially improves computational efficiency, achieving an order of magnitude increase in speed compared to prevailing time-domain neural vocoding approaches.*



Demos can be found in this [post](https://gemelo-ai.github.io/vocos/).



This model was contributed by [Manal El Aidouni](https://huggingface.co/Manel). The original code can be found [here](https://github.com/gemelo-ai/vocos) and original checkpoints  [here](https://huggingface.co/charactr/vocos-mel-24khz) and [here](https://huggingface.co/charactr/vocos-encodec-24khz).




## Usage example 

There are two Vocos variants introduced `VocosModel` and `VocosWithEncodecModel`.


## Mel-spectrogram variant 

> “*Reconstructing audio waveforms from mel-spectrograms has become a fundamental task for vocoders in contemporary speech synthesis pipelines.*” - paper.

You can use `VocosFeatureExtractor` to turn a raw waveform into mel-spectrogram features and feed them into VocosModel to generate high quality speech like the example below. You can also plug `VocosModel` in as a standalone vocoder component within a larger audio generation pipeline (for example the YuE model).

```python 
from datasets import load_dataset, Audio
from transformers import VocosModel, VocosFeatureExtractor

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# load model and feature extractor
model = VocosModel.from_pretrained("Manel/Vocos")
feature_extractor = VocosFeatureExtractor.from_pretrained("Manel/Vocos")

# load audio sample
ds = ds.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio_sample= ds[0]["audio"]["array"]
            
inputs = feature_extractor(audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

audio = model(inputs.input_features)

```


## Neural audio EnCodec variant

> “*While traditionally, neural vocoders reconstruct the audio waveform from a mel-scaled spectrogram an approach widely adopted in many speech synthesis pipelines, recent research has started to utilize learnt features often in a quantized form.*” - paper. 

`VocosWithEncodecModel` integrates a pretrained EnCodec neural audio codec into Vocos for end-to-end audio compression and reconstruction. It accepts either raw audio or precomputed EnCodec RVQ codes (quantized audio features) and reconstructs a higher quality audio.


You need to provide a `bandwidth_id`, which is an index in [0, 1, 2, 3] range that selects the desired bandwidth for the EnCodec quantization [1.5, 3, 6, 12] kbps respectively. 

#### Reconstructing audio from raw audio:
```python 
from datasets import load_dataset, Audio
from transformers import VocosWithEncodecModel

model = VocosWithEncodecModel.from_pretrained("Manel/Vocos-Encodec")

# load audio sample
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=24000))  
audio_sample = ds[0]["audio"]["array"]

# reconstructing audio from raw audio
audio_sample = torch.tensor(audio_sample, dtype=torch.float32).unsqueeze(0)

bandwidth_id = torch.tensor([0], dtype=torch.long)
audio = model(audio=audio_sample, bandwidth_id=bandwidth_id)
```

#### Reconstructing audio from EnCodec RVQ codes:

```python

from transformers import VocosWithEncodecModel

model = VocosWithEncodecModel.from_pretrained("Manel/Vocos-Encodec")

# 8 codeboooks, 200 frames
audio_codes = torch.randint(low=0, high=1024, size=(8, 200))  
audio = model(codes=audio_codes, bandwidth_id=bandwidth_id)

```

#### Reconstructing audio from Bark tokens:


Bark is a text to audio model that encodes the input text into discrete tokens which are integer indices into codebooks, similar to Encodec’s RVQ codes, it then uses the Encodec’s decoder to convert those codes into an audio waveform. The Vocos vocoder is usually integrated with Bark instead of relying only on the Encodec model. 

Here is a quick example of how to encode and decode an audio using `Bark` with `VocosWithEncodecModel` model:


```python 
from bark import text_to_semantic
from bark.generation import generate_coarse, generate_fine, codec_decode
from transformers import VocosWithEncodecModel
from IPython.display import Audio
import torchaudio

preload_models()
model = VocosWithEncodecModel.from_pretrained("Manel/Vocos-Encodec")
bandwidth_id = torch.tensor([0]) 

# convert your text prompt into Bark semantic tokens
text_prompt = "So, you've heard about neural vocoding? [laughs] We've been messing around with this new model called Vocos."
semantic_tokens = text_to_semantic(text_prompt, history_prompt=None, temp=0.7, silent=False)

# generates EnCodec audio tokens from semantic tokens
coarse_codes = generate_coarse(semantic_tokens, history_prompt=None, temp=0.7, silent=False,use_kv_caching=True)
fine_codes = generate_fine(coarse_codes, history_prompt=None, temp=0.5)
bark_codes = torch.from_numpy(fine_codes)

# reconstruct audio from Bark produces codes
audio = model(codes=bark_codes, bandwidth_id=bandwidth_id)

# listen to the audio
audio = torchaudio.functional.resample(audio.cpu(), orig_freq=24000, new_freq=44100).numpy()
Audio(audio, rate=44100)

```


## VocosConfig

[[autodoc]] VocosConfig

## VocosWithEncodecConfig

[[autodoc]] VocosWithEncodecConfig

## VocosModel

[[autodoc]] VocosModel
    - forward

## VocosWithEncodecModel

[[autodoc]] VocosWithEncodecModel
    - forward