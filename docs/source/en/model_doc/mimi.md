<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Mimi

## Overview

The Mimi model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Mimi is a high-fidelity audio codec model developed by the Kyutai team. It can be used to project audio waveforms into quantized latent spaces, and vice versa. In other words, it can be used to map audio waveforms into “audio tokens”, known as “codebooks”.


Its architecture is based on [Encodec](model_doc/encodec) with several major differences:
* it uses a much lower frame-rate.
* it uses additional transformers for encoding and decoding for better latent contextualization
* it uses a different quantization scheme: one codebook is dedicated to semantic projection.



## Usage example 

Here is a quick example of how to encode and decode an audio using this model:

```python 
>>> from datasets import load_dataset, Audio
>>> from transformers import MimiModel, AutoFeatureExtractor
>>> librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

>>> # load model and feature extractor
>>> model = MimiModel.from_pretrained("kmhf/mimi")  # TODO(YL): modify once official
>>> feature_extractor = AutoFeatureExtractor.from_pretrained("kmhf/mimi")

>>> # load audio sample
>>> librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
>>> audio_sample = librispeech_dummy[-1]["audio"]["array"]
>>> inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

>>> encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
>>> audio_values = model.decode(encoder_outputs.audio_codes, inputs["padding_mask"])[0]
>>> # or the equivalent with a forward pass
>>> audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

This model was contributed by [Yoach Lacombe (ylacombe)](https://huggingface.co/ylacombe).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## MimiConfig

[[autodoc]] MimiConfig

## MimiModel

[[autodoc]] MimiModel
    - decode
    - encode
    - forward
