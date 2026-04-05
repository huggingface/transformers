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
*This model was released on 2025-02-06 and added to Hugging Face Transformers on 2026-03-19.*

# X-Codec2

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The X-Codec2 model was proposed in [Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis](https://huggingface.co/papers/2502.04128).

X-Codec2 is a neural audio codec designed to improve speech synthesis and general audio generation for large language model (LLM) pipelines. It extends the original X-Codec by refining how semantic and acoustic information is integrated and tokenized, enabling efficient and high-fidelity audio representation.

About its architecture:
- **Unified Semantic-Acoustic Tokenization**: X-Codec2 fuses outputs from a semantic encoder (e.g., Wav2Vec2-BERT) and an acoustic encoder into a single embedding, capturing both high-level meaning (e.g., text content, emotion) and low-level audio details (e.g., timbre).
- **Single-Stage Feature Scalar Quantization (FSQ)**: Unlike the multi-layer residual VQ in most approaches (e.g., [DAC](./dac), [EnCodec](./encodec), [X-Codec](./xcodec), [Mimi](./mimi.md)), X-Codec2 uses a single-layer of Feature Scalar Quantization (FSQ) for stability and compatibility with causal, autoregressive LLMs.
- **Transformer-Friendly Design**: The 1D token structure of X-Codec2 naturally aligns with the autoregressive modeling in LLMs like LLaMA, improving training efficiency and downstream compatibility.

A model checkpoint is available at [bezzam/xcodec2](https://huggingface.co/bezzam/xcodec2).

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam) and [Steven Zheng](https://huggingface.co/Steveeeeeeen).
The original modeling code can be found [here](https://huggingface.co/HKUSTAudio/xcodec2/blob/main/modeling_xcodec2.py), while their training code is [here](https://github.com/zhenye234/X-Codec-2.0).

## Usage example 

Here is a quick example of how to encode and decode an audio using this model:

```python 
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, Xcodec2Model

model_id = "bezzam/xcodec2"
model = Xcodec2Model.from_pretrained(model_id, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio = dataset[0]["audio"]["array"]
inputs = feature_extractor(audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(
    model.device, model.dtype
)
print("Input waveform shape:", inputs["audio"].shape)
# Input waveform shape: torch.Size([1, 1, 94080])

# encoder and decoder
audio_codes = model.encode(**inputs).audio_codes
print("Audio codes shape:", audio_codes.shape)
# Audio codes shape: torch.Size([1, 1, 294])
audio_values = model.decode(audio_codes).audio_values
print("Audio values shape:", audio_values.shape)
# Audio values shape: torch.Size([1, 1, 94080])

# Equivalently, you can do encoding and decoding in one step
model_output = model(**inputs)
audio_codes = model_output.audio_codes
audio_values = model_output.audio_values
```

### Batch processing

The original [checkpoint](https://huggingface.co/HKUSTAudio/xcodec2) and code via PyPI does not support batch processing, but it is possible with this version!

```python
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, Xcodec2Model

batch_size = 2
model_id = "bezzam/xcodec2"
model = Xcodec2Model.from_pretrained(model_id, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audios = [dataset[i]["audio"]["array"] for i in range(batch_size)]
inputs = feature_extractor(audio=audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(
    model.device, model.dtype
)
print("Input waveform shape:", inputs["audio"].shape)
# Input waveform shape: torch.Size([2, 1, 94080])

# encoder and decoder
encoder_output = model.encode(**inputs)
audio_codes = encoder_output.audio_codes
print("Audio codes shape:", audio_codes.shape)
# Audio codes shape: torch.Size([2, 1, 294])
audio_values = model.decode(audio_codes).audio_values
print("Audio values shape:", audio_values.shape)
# Audio values shape: torch.Size([2, 1, 94080])

# Equivalently, you can do encoding and decoding in one step
model_output = model(**inputs)
audio_codes = model_output.audio_codes
audio_values = model_output.audio_values
```

## Xcodec2Config

[[autodoc]] Xcodec2Config

## Xcodec2FeatureExtractor

[[autodoc]] Xcodec2FeatureExtractor
    - __call__

## Xcodec2Model

[[autodoc]] Xcodec2Model
    - decode
    - encode
    - forward
