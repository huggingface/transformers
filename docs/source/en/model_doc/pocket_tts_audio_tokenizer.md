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
*This model was contributed to Hugging Face Transformers on 2026-10-20.*

# Pocket TTS audio tokenizer

## Overview

The audio tokenizer of [Pocket TTS](https://huggingface.co/kyutai/pocket-tts): a codec with [Mimi](./mimi)'s architecture (SEANet convolutions and a small causal transformer on each side) whose residual vector quantizer is replaced by a continuous 32-dimensional bottleneck, the VAE-GAN of [Continuous Audio Language Models](https://huggingface.co/papers/2509.06926). It turns 24 kHz audio into latents at 12.5 Hz and back. Its encoder and decoder can be loaded on their own ([`PocketTTSAudioTokenizerEncoderModel`], [`PocketTTSAudioTokenizerDecoderModel`]). It is the `audio_tokenizer` of [`PocketTTSProcessor`], shared by every Pocket TTS checkpoint: the processor encodes voice prompts with it and decodes the generated latents.

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb). The original code is [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts).

## Usage

```python
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

model = AutoModel.from_pretrained("eustlb/pocket-tts-audio-tokenizer", device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained("eustlb/pocket-tts-audio-tokenizer")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=24000, return_tensors="pt").to(model.device)

latents = model.encode(**inputs).latents
audio_values = model.decode(latents).audio_values
```

## PocketTTSAudioTokenizerConfig

[[autodoc]] PocketTTSAudioTokenizerConfig

## PocketTTSAudioTokenizerEncoderConfig

[[autodoc]] PocketTTSAudioTokenizerEncoderConfig

## PocketTTSAudioTokenizerDecoderConfig

[[autodoc]] PocketTTSAudioTokenizerDecoderConfig

## PocketTTSAudioTokenizerModel

[[autodoc]] PocketTTSAudioTokenizerModel
    - encode
    - decode
    - forward

## PocketTTSAudioTokenizerEncoderModel

[[autodoc]] PocketTTSAudioTokenizerEncoderModel
    - forward

## PocketTTSAudioTokenizerDecoderModel

[[autodoc]] PocketTTSAudioTokenizerDecoderModel
    - forward
