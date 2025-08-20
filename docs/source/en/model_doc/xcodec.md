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
*This model was released on 2024-08-30 and added to Hugging Face Transformers on 2025-08-15.*

# X-Codec

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The X-Codec model was proposed in [Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model](https://huggingface.co/papers/2408.17175) by Zhen Ye, Peiwen Sun, Jiahe Lei, Hongzhan Lin, Xu Tan, Zheqi Dai, Qiuqiang Kong, Jianyi Chen, Jiahao Pan, Qifeng Liu, Yike Guo, Wei Xue.

The X-Codec model is a neural audio codec that integrates semantic information from self-supervised models (e.g., HuBERT) alongside traditional acoustic information. This enables:

- **Music continuation**: Better modeling of musical semantics yields more coherent continuations.
- **Text-to-Sound Synthesis**: X-Codec captures semantic alignment between text prompts and generated audio.
- **Semantic aware audio tokenization**: X-Codec is used as an audio tokenizer in the YuE lyrics to song generation model.

The abstract of the paper states the following:

*Recent advancements in audio generation have been significantly propelled by the capabilities of Large Language Models (LLMs). The existing research on audio LLM has primarily focused on enhancing the architecture and scale of audio language models, as well as leveraging larger datasets, and generally, acoustic codecs, such as EnCodec, are used for audio tokenization. However, these codecs were originally designed for audio compression, which may lead to suboptimal performance in the context of audio LLM. Our research aims to address the shortcomings of current audio LLM codecs, particularly their challenges in maintaining semantic integrity in generated audio. For instance, existing methods like VALL-E, which condition acoustic token generation on text transcriptions, often suffer from content inaccuracies and elevated word error rates (WER) due to semantic misinterpretations of acoustic tokens, resulting in word skipping and errors. To overcome these issues, we propose a straightforward yet effective approach called X-Codec. X-Codec incorporates semantic features from a pre-trained semantic encoder before the Residual Vector Quantization (RVQ) stage and introduces a semantic reconstruction loss after RVQ. By enhancing the semantic ability of the codec, X-Codec significantly reduces WER in speech synthesis tasks and extends these benefits to non-speech applications, including music and sound generation. Our experiments in text-to-speech, music continuation, and text-to-sound tasks demonstrate that integrating semantic information substantially improves the overall performance of language models in audio generation.* 

Model cards:
- [xcodec-hubert-librispeech](https://huggingface.co/hf-audio/xcodec-hubert-librispeech) (for speech)
- [xcodec-wavlm-mls](https://huggingface.co/hf-audio/xcodec-wavlm-mls) (for speech)
- [xcodec-wavlm-more-data](https://huggingface.co/hf-audio/xcodec-wavlm-more-data) (for speech)
- [xcodec-hubert-general](https://huggingface.co/hf-audio/xcodec-hubert-general) (for general audio)
- [xcodec-hubert-general-balanced](https://huggingface.co/hf-audio/xcodec-hubert-general-balanced) (for general audio)

This model was contributed by [Manal El Aidouni](https://huggingface.co/Manel). The original code can be found [here](https://github.com/zhenye234/xcodec) and original checkpoints for the five different models [here](https://github.com/zhenye234/xcodec?tab=readme-ov-file#available-models).

Demos can be found on this [page](https://x-codec-audio.github.io/).


## Usage example 

Here is a quick example of how to encode and decode an audio using this model:

```python 
from datasets import load_dataset, Audio
from transformers import XcodecModel, AutoFeatureExtractor
dummy_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# load model and feature extractor
model_id = "hf-audio/xcodec-hubert-librispeech"
model = XcodecModel.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

# load audio sample
dummy_dataset = dummy_dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio_sample = dummy_dataset[-1]["audio"]["array"]
inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

# encode and decode
encoder_outputs = model.encode(inputs["input_values"])
decoder_outputs = model.decode(encoder_outputs.audio_codes)
audio_values = decoder_outputs.audio_values

# or the equivalent with a forward pass
audio_values = model(inputs["input_values"]).audio_values

```
To listen to the original and reconstructed audio, run the snippet below and then open the generated `original.wav` and `reconstruction.wav` files in your music player to compare.

```python
import soundfile as sf

original = audio_sample
reconstruction = audio_values[0].cpu().detach().numpy()
sampling_rate = feature_extractor.sampling_rate

sf.write("original.wav", original, sampling_rate)
sf.write("reconstruction.wav", reconstruction.T, sampling_rate)
```


## XcodecConfig

[[autodoc]] XcodecConfig


## XcodecModel

[[autodoc]] XcodecModel
    - decode
    - encode
    - forward