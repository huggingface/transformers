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

# NeuCodec

<div class="flex flex-wrap space-x-1">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The NeuCodec model was proposed in [Finite Scalar Quantization Enables Redundant and Transmission-Robust Neural Audio Compression at Low Bit-rates](https://arxiv.org/abs/2509.09550).

NeuCodec is a neural audio codec based on XCodec2.

## Usage example 

Here is a quick example of how to encode and decode an audio using this model:

```python 
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

model_id = "neuphonic/neucodec"
model = AutoModel.from_pretrained(model_id, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audio = dataset[0]["audio"]["array"]
inputs = feature_extractor(audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(
    model.device, model.dtype
)
print("Input waveform shape:", inputs["input_values"].shape)
# Input waveform shape: torch.Size([1, 1, 93760])

# encoder and decoder
audio_codes = model.encode(**inputs).audio_codes
print("Audio codes shape:", audio_codes.shape)
# Audio codes shape: torch.Size([1, 1, 292])
audio_values = model.decode(audio_codes).audio_values
print("Audio values shape:", audio_values.shape)

# Equivalently, you can do encoding and decoding in one step
model_output = model(**inputs)
audio_codes = model_output.audio_codes
audio_values = model_output.audio_values
```

### Batch processing

This implementation also supports batched input!

```python
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

batch_size = 2
model_id = "neuphonic/neucodec"
model = AutoModel.from_pretrained(model_id, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audios = [dataset[i]["audio"]["array"] for i in range(batch_size)]
inputs = feature_extractor(audio=audios, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(
    model.device, model.dtype
)
print("Input waveform shape:", inputs["input_values"].shape)
# Input waveform shape: torch.Size([2, 1, 93760])

# encoder and decoder
encoder_output = model.encode(**inputs)
audio_codes = encoder_output.audio_codes
print("Audio codes shape:", audio_codes.shape)
# Audio codes shape: torch.Size([2, 1, 292])
audio_values = model.decode(audio_codes).audio_values
print("Audio values shape:", audio_values.shape)

# Equivalently, you can do encoding and decoding in one step
model_output = model(**inputs)
audio_codes = model_output.audio_codes
audio_values = model_output.audio_values
```

### Speed-up with `torch.compile`

You can speed up inference with [`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html). The first few calls will be slower due to compilation overhead, but subsequent calls will be faster.

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

batch_size = 4
model_id = "neuphonic/neucodec"
model = AutoModel.from_pretrained(model_id, device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
audios = [dataset[i]["audio"]["array"] for i in range(batch_size)]
inputs = feature_extractor(
    audio=audios, sampling_rate=feature_extractor.sampling_rate, padding=True, return_tensors="pt"
).to(model.device, model.dtype)

compiled_model = torch.compile(model, fullgraph=True)

# Warmup (includes compilation on first call)
for _ in range(10):
    with torch.inference_mode():
        _ = compiled_model(**inputs)

with torch.inference_mode():
    output = compiled_model(**inputs)
print("Audio values shape:", output.audio_values.shape)
```

## NeuCodecConfig

[[autodoc]] NeuCodecConfig

## NeuCodecFeatureExtractor

[[autodoc]] NeuCodecFeatureExtractor
    - __call__

## NeuCodecModel

[[autodoc]] NeuCodecModel
    - decode
    - encode
    - forward
