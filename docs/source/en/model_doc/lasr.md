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

LASR is the architecture used by MedASR, a speech-to-text model released by Google Health AI based on the [Conformer architecture](https://huggingface.co/papers/2005.08100) pre-trained for medical dictation. MedASR is intended as a starting point for developers, and is well-suited for dictation tasks involving medical terminologies, such as radiology dictation. While MedASR has been extensively pre-trained on a corpus of medical audio data, it may occasionally exhibit performance variability when encountering terms outside of its pre-training data, such as non-standard medication names or consistent handling of temporal data (dates, times, or durations).

## Usage

### Basic usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="google/medasr")
out = pipe("path/to/audio.mp3")
print(out)
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio

processor = AutoProcessor.from_pretrained("google/medasr")
model = AutoModelForCTC.from_pretrained("google/medasr", dtype="auto", device_map="auto")

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

### Training

Here is a minimal example for preparing data and running a single training step with LASR/MedASR using PyTorch and 🤗 Transformers. This snippet demonstrates how to prepare batches with audio and text, feed them to the corresponding model, and compute the loss for training.

```python
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio

# Load processor and model
processor = AutoProcessor.from_pretrained("google/medasr")
model = AutoModelForCTC.from_pretrained("google/medasr", dtype="auto", device_map="auto")

# Load a small example dataset and prepare batch
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:5]]
text_samples = [el for el in ds["text"][:5]]

# Passing `text` to the processor will prepare the `labels`
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(device, dtype=model.dtype)

outputs = model(**inputs)
outputs.loss.backward()
```

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
