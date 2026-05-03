<!--Copyright 2026 The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-05-16 and added to Hugging Face Transformers on 2026-05-03.*

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# Conformer

## Overview

Conformer, introduced in [Conformer: Convolution-augmented Transformer for Speech Recognition](https://huggingface.co/papers/2005.08100), is an encoder architecture for automatic speech recognition. It combines self-attention for long-range dependencies with convolution modules for local acoustic patterns.

**Model Architecture**

- **Conformer Encoder**: A stack of Conformer blocks that process mel-spectrogram features after convolutional subsampling. Each block uses two feed-forward modules around multi-head self-attention and a depthwise convolution module.
- [**ConformerForCTC**](#conformerforctc): a Conformer encoder with a Connectionist Temporal Classification (CTC) head for automatic speech recognition.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo). Model checkpoints are to be found under [the NVIDIA organization](https://huggingface.co/nvidia/models?search=conformer_ctc).

## Usage

### Basic usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```python
from transformers import pipeline


pipe = pipeline("automatic-speech-recognition", model="nvidia/stt_en_conformer_ctc_small")
out = pipe("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
print(out)
```

</hfoption>
<hfoption id="AutoModel">

```python
from datasets import Audio, load_dataset

from transformers import AutoModelForCTC, AutoProcessor


processor = AutoProcessor.from_pretrained("nvidia/stt_en_conformer_ctc_small")
model = AutoModelForCTC.from_pretrained("nvidia/stt_en_conformer_ctc_small", device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:5]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.batch_decode(outputs, skip_special_tokens=True))
```

</hfoption>
</hfoptions>

### Training

The example below prepares a batch of audio and text, passes it through the Conformer CTC model, and computes the training loss.

```python
from datasets import Audio, load_dataset

from transformers import AutoModelForCTC, AutoProcessor


processor = AutoProcessor.from_pretrained("nvidia/stt_en_conformer_ctc_small")
model = AutoModelForCTC.from_pretrained("nvidia/stt_en_conformer_ctc_small", device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:5]]
text_samples = [el for el in ds["text"][:5]]

# Passing `text` to the processor will prepare the `labels`
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)

outputs = model(**inputs)
outputs.loss.backward()
```

## ConformerTokenizer

[[autodoc]] ConformerTokenizer

## ConformerFeatureExtractor

[[autodoc]] ConformerFeatureExtractor
    - __call__

## ConformerProcessor

[[autodoc]] ConformerProcessor
    - __call__
    - batch_decode
    - decode

## ConformerEncoderConfig

[[autodoc]] ConformerEncoderConfig

## ConformerCTCConfig

[[autodoc]] ConformerCTCConfig

## ConformerEncoder

[[autodoc]] ConformerEncoder

## ConformerForCTC

[[autodoc]] ConformerForCTC
