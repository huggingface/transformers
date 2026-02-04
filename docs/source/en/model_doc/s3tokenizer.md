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

# S3Tokenizer

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The S3Tokenizer model is a speech tokenizer that converts raw audio into discrete tokens at 25 tokens/second. 
It uses a mel-spectrogram encoder with Finite Scalar Quantization (FSQ) to produce high-quality speech representations suitable for speech language models.

This model was contributed by [xingchensong](https://github.com/xingchensong).
The original code can be found [here](https://github.com/xingchensong/S3Tokenizer).

## Usage example

Here is a quick example of how to tokenize audio using this model:

```python
>>> from transformers import S3TokenizerModel
>>> import torch

>>> # Load the model
>>> model = S3TokenizerModel.from_pretrained("path/to/model")

>>> # Prepare audio (16kHz sample rate expected)
>>> audio = torch.randn(1, 16000)  # 1 second of audio at 16kHz

>>> # Tokenize the audio
>>> outputs = model(audio)
>>> speech_tokens = outputs.speech_tokens  # Discrete tokens
>>> speech_token_lens = outputs.speech_token_lens  # Length of token sequence
```

## S3TokenizerConfig

[[autodoc]] S3TokenizerConfig

## S3TokenizerFeatureExtractor

[[autodoc]] S3TokenizerFeatureExtractor
    - __call__

## S3TokenizerModel

[[autodoc]] S3TokenizerModel
    - forward

## S3TokenizerOutput

[[autodoc]] transformers.models.s3tokenizer.modeling_s3tokenizer.S3TokenizerOutput

## Utilities

[[autodoc]] transformers.models.s3tokenizer.modeling_s3tokenizer.drop_invalid_tokens

