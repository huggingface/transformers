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

# PhoneticXeus

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

PhoneticXeus is a multilingual IPA phone recognition model that uses a CNN feature encoder followed by an
E-Branchformer encoder with a CTC head. The E-Branchformer architecture runs self-attention and a Convolutional
Gating MLP (cgMLP) as parallel branches merged via depthwise convolution, rather than the sequential
attention-convolution pattern used in Conformer models. The model employs intermediate CTC (interCTC)
self-conditioning at specified encoder layers.

The model was released by the [Changeling Lab](https://huggingface.co/changelinglab) and can be found at
[changelinglab/PhoneticXeus](https://huggingface.co/changelinglab/PhoneticXeus).

## Usage tips

- PhoneticXeus outputs IPA (International Phonetic Alphabet) phone sequences rather than text transcriptions.
- The vocabulary consists of 428 IPA phoneme tokens.
- It uses `Wav2Vec2FeatureExtractor` for audio preprocessing and `PhoneticXeusTokenizer` for CTC decoding.
- The E-Branchformer encoder uses parallel self-attention and cgMLP branches instead of the sequential
  attention and convolution used in Conformer models.
- InterCTC self-conditioning feeds intermediate CTC predictions back into the encoder at layers 4, 8, and 12.

## Resources

- [Automatic speech recognition task guide](../tasks/asr)

## PhoneticXeusConfig

[[autodoc]] PhoneticXeusConfig

## PhoneticXeusModel

[[autodoc]] PhoneticXeusModel
    - forward

## PhoneticXeusForCTC

[[autodoc]] PhoneticXeusForCTC
    - forward

## PhoneticXeusTokenizer

[[autodoc]] PhoneticXeusTokenizer

## PhoneticXeusProcessor

[[autodoc]] PhoneticXeusProcessor
