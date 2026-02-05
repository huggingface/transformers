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
*This model was released on 2023-06-01 and added to Hugging Face Transformers on 2026-01-23.*

# Vocos

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The Vocos model was proposed in [**Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis**](https://huggingface.co/papers/2306.00814) by Hubert Siuzdak.

Vocos is a GAN-based neural vocoder designed for high quality audio synthesis in text to speech (TTS) pipelines and related tasks. Traditional time-domain vocoders rely on transposed convolutions for upsampling, which degrades temporal resolution across all layers and introduces aliasing artifacts into synthesized speech.
Instead, Vocos represents audio signals in the time-frequency domain, it's trained to predict the complex Short Time Fourier Transform (STFT) coefficients, magnitude and phase, and uses the computationally inverse STFT (ISTFT) for upsampling, which maintains the same temporal resolution throughout the network and converts directly to speech waveforms.

Vocos delivers the same high audio quality while achieving  30× faster inference speed on CPU and outperforming HiFi-GAN in both VISQOL and PESQ scores.



The abstract of the paper states the following:

*Recent advancements in neural vocoding are predominantly driven by Generative Adversarial Networks (GANs) operating in the time-domain. While effective, this approach neglects the inductive bias offered by time-frequency representations, resulting in reduntant and computionally-intensive upsampling operations. Fourier-based time-frequency representation is an appealing alternative, aligning more accurately with human auditory perception, and benefitting from well-established fast algorithms for its computation. Nevertheless, direct reconstruction of complex-valued spectrograms has been historically problematic, primarily due to phase recovery issues. This study seeks to close this gap by presenting Vocos, a new model that directly generates Fourier spectral coefficients. Vocos not only matches the state-of-the-art in audio quality, as demonstrated in our evaluations, but it also substantially improves computational efficiency, achieving an order of magnitude increase in speed compared to prevailing time-domain neural vocoding approaches.*


Vocos is available in two variants:

- `VocosModel` : Mel-spectrogram based vocoder documented in this card. 

- `VocosEncodecModel` : EnCodec based vocoder can be found [here](https://huggingface.co/docs/transformers/model_doc/vocos_encodec).



You can find demos in this [post](https://gemelo-ai.github.io/vocos/). The original implementation can be found [here](https://github.com/gemelo-ai/vocos) and the original checkpoint is available [here](https://huggingface.co/charactr/vocos-mel-24khz).

This model was contributed by [Manal El Aidouni](https://huggingface.co/Manel) and [Eric Bezzam](https://huggingface.co/bezzam). 

## Usage


You can extract mel-spectrogram features from an audio using `VocosFeatureExtractor` and feed them into `VocosModel` to generate high quality audio. You can also plug `VocosModel` in as a standalone vocoder component within a larger audio generation pipeline (for example the [YuE](https://github.com/multimodal-art-projection/YuE) model).


```python 
from datasets import load_dataset, Audio
from transformers import VocosFeatureExtractor, VocosModel
from scipy.io.wavfile import write as write_wav
    
# load model and processor
model_id = "hf-audio/vocos-mel-24khz"
feature_extractor = VocosFeatureExtractor.from_pretrained(model_id)
model = VocosModel.from_pretrained(model_id, device_map="auto")
sampling_rate = feature_extractor.sampling_rate
    
# load audio sample
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
audio = ds[0]["audio"]["array"]

inputs = feature_extractor(audio=audio, sampling_rate=sampling_rate).to(model.device)

print(inputs.input_features.shape) # (batch_size, num_mel_bins, frame) [1, 100, 550]

outputs = model(**inputs)

audio = outputs.audio

print(audio.shape) # (batch_size, time) [1, 140544]

# save audio to file
write_wav("vocos.wav", sampling_rate, audio[0].detach().cpu().numpy())
```

In case of processing multiple audio files in batch,  you can remove padding from reconstructed audios using `attention_mask` returned in output as such:


```python 
inputs = feature_extractor(audio=[audio1, audio2], return_tensors="pt")

outputs = model(**inputs)

reconstructed_audio, attention_mask = outputs.audio, outputs.attention_mask

unpadded_audios = [reconstructed_audio[i][attention_mask[i].bool()].detach().cpu().numpy() for i in range(reconstructed_audio.shape[0])]

```

## VocosConfig

[[autodoc]] VocosConfig

## VocosFeatureExtractor

[[autodoc]] VocosFeatureExtractor

## VocosModel

[[autodoc]] VocosModel
    - forward
