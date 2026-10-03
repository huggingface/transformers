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
*This model was contributed to Hugging Face Transformers on 2026-10-02.*

# Nemotron Speech Encoder

## Overview

The Nemotron speech encoder is the Transformer audio encoder shared by NVIDIA Nemotron speech models. It stacks
`subsampling_factor` consecutive log-mel frames, projects them to the hidden size, and encodes them with pre-norm
Transformer layers using rotary position embeddings. The rotary embedding can rotate only a fraction of each head
(`partial_rotary_factor`), and queries and keys can be normalized per head before it (`use_qk_norm`).

It is the audio tower of [Nemotron 3 Diarization](./nemotron3_diarization), and is meant to be used as a sub-model
through [`AutoModel.from_config`] with a [`NemotronSpeechEncoderConfig`].

## NemotronSpeechEncoderConfig

[[autodoc]] NemotronSpeechEncoderConfig

## NemotronSpeechEncoder

[[autodoc]] NemotronSpeechEncoder
    - forward
