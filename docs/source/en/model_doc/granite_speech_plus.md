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
*This model was released on 2026-04-23 and added to Hugging Face Transformers on 2026-04-23.*

# Granite Speech Plus

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

Granite Speech Plus is a variant of [Granite Speech](./granite_speech) whose projector consumes the concatenation of
the encoder's final hidden states with an arbitrary subset of its intermediate hidden states (along the feature
dimension). The selected intermediate layers are controlled by the `encoder_hidden_layers` config field on
[`GraniteSpeechPlusConfig`]; when it is `None`, the model behaves identically to Granite Speech. When it is set, the
projector's `encoder_hidden_size` must equal `encoder_config.hidden_dim * (len(encoder_hidden_layers) + 1)`.

The rest of the architecture — speech encoder, query transformer projector, language model, and optional LoRA adapter
— is inherited unchanged from Granite Speech. See the [Granite Speech documentation](./granite_speech) for usage
examples; the same [`GraniteSpeechProcessor`] and [`GraniteSpeechFeatureExtractor`] are used here.

## GraniteSpeechPlusConfig

[[autodoc]] GraniteSpeechPlusConfig

## GraniteSpeechPlusEncoderConfig

[[autodoc]] GraniteSpeechPlusEncoderConfig

## GraniteSpeechPlusForConditionalGeneration

[[autodoc]] GraniteSpeechPlusForConditionalGeneration
    - forward
    - get_audio_features
