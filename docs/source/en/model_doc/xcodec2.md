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
*This model was released on 2025-02-06 and added to Hugging Face Transformers on 2025-04-29.*

# X-Codec2

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The X-Codec2 model was proposed in [Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis](https://huggingface.co/papers/2502.04128).

X-Codec2 is a neural audio codec designed to improve speech synthesis and general audio generation for large language model (LLM) pipelines. It extends the original X-Codec by refining how semantic and acoustic information is integrated and tokenized, enabling efficient and high-fidelity audio representation.

Its architecture is based on [X-Codec](./xcodec) with several major differences:

- **Unified Semantic-Acoustic Tokenization**: X-Codec2 fuses outputs from a semantic encoder (e.g., Wav2Vec2-BERT) and an acoustic encoder into a single embedding, capturing both high-level meaning (e.g., text content, emotion) and low-level audio details (e.g., timbre).
- **Single-Stage Vector Quantization (VQ)**: Unlike the multi-layer residual VQ in most approaches (e.g., [X-Codec](./xcodec), [DAC](./dac), [EnCodec](./encodec)), X-Codec2 uses a single-layer Feature-Space Quantization (FSQ) for stability and compatibility with causal, autoregressive LLMs.
- **Semantic Supervision During Training**: It adds a semantic reconstruction loss, ensuring that the discrete tokens preserve meaningful linguistic and emotional information — crucial for TTS tasks.
- **Transformer-Friendly Design**: The 1D token structure of X-Codec2 naturally aligns with the autoregressive modeling in LLMs like LLaMA, improving training efficiency and downstream compatibility.

## Usage example 

Here is a quick example of how to encode and decode an audio using this model:

```python 
>>> from datasets import load_dataset, Audio
>>> from transformers import XCodec2Model, AutoFeatureExtractor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> # load model and feature extractor
>>> model = XCodec2Model.from_pretrained("Steveeeeeeen/XCodec2")
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("Steveeeeeeen/XCodec2")

>>> # load audio sample
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"])
>>> audio_values = model.decode(encoder_outputs.audio_codes)[0]
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"]).audio_values
```

This model was contributed by [Steven Zheng](https://huggingface.co/Steveeeeeeen) and [Eric Bezzam](https://huggingface.co/bezzam).
The original code can be found [here](https://github.com/zhenye234/X-Codec-2.0).


## Xcodec2Config

[[autodoc]] Xcodec2Config

## Xcodec2Model

[[autodoc]] Xcodec2Model
    - decode
    - encode
    - forward
