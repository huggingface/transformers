<!--Copyright 2025 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-12-05.*

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# LASR

## Overview

TODO

## Usage

### Basic usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="path/to/lasr-model")
out = pipe("path/to/audio.mp3")
print(out)
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("path/to/lasr-model")
model = AutoModelForCTC.from_pretrained("path/to/lasr-model", dtype="auto", device_map=device)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))
```

</hfoption>
</hfoptions>

### Making The Model Go Brrr

TODO

### Training

TODO

## LasrTokenizer

[[autodoc]] LasrTokenizer

## LasrFeatureExtractor

[[autodoc]] LasrFeatureExtractor
    - __call__

## LasrProcessor

[[autodoc]] LasrProcessor
    - __call__
    - batch_decode
    - decode

## LasrEncoderConfig

[[autodoc]] LasrEncoderConfig

## LasrCTCConfig

[[autodoc]] LasrCTCConfig

## LasrEncoder

[[autodoc]] LasrEncoder

## LasrForCTC

[[autodoc]] LasrForCTC
