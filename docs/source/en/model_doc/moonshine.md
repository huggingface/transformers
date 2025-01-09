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

# moonshine

## Overview

The moonshine model was proposed in [Moonshine: Speech Recognition for Live Transcription and Voice Commands
](https://arxiv.org/abs/2410.15608) by Nat Jeffries, Evan King, Manjunath Kudlur, Guy Nicholson, James Wang, Pete Warden.

The abstract from the paper is the following:

*This paper introduces Moonshine, a family of speech recognition models optimized for live transcription and voice command processing. Moonshine is based on an encoder-decoder transformer architecture and employs Rotary Position Embedding (RoPE) instead of traditional absolute position embeddings. The model is trained on speech segments of various lengths, but without using zero-padding, leading to greater efficiency for the encoder during inference time. When benchmarked against OpenAI's Whisper tiny-en, Moonshine Tiny demonstrates a 5x reduction in compute requirements for transcribing a 10-second speech segment while incurring no increase in word error rates across standard evaluation datasets. These results highlight Moonshine's potential for real-time and resource-constrained applications.*

Tips:

- Moonshine improves upon Whisper's architecture:
  1. It uses SwiGLU activation instead of GELU in the decoder layers
  2. Most importantly, it replaces absolute position embeddings with Rotary Position Embeddings (RoPE). This allows Moonshine to handle audio inputs of any length, unlike Whisper which is restricted to fixed 30-second windows.

This model was contributed by [Eustache Le Bihan (eustlb)](https://huggingface.co/eustlb).
The original code can be found [here](https://github.com/usefulsensors/moonshine).

## Resources

- [Automatic speech recognition task guide](../tasks/asr)

## MoonshineConfig

[[autodoc]] MoonshineConfig

## MoonshineModel

[[autodoc]] MoonshineModel
    - forward
    - _mask_input_features

## MoonshineForConditionalGeneration

[[autodoc]] MoonshineForConditionalGeneration
    - forward
    - generate

