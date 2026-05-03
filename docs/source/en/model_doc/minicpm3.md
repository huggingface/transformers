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

# MiniCPM3

## Overview

The MiniCPM3 model was proposed in [MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies](https://huggingface.co/papers/2404.06395) by OpenBMB.

MiniCPM3-4B is a dense language model that uses Multi-head Latent Attention (MLA) for efficient KV cache compression, combined with embedding scaling, depth-dependent residual scaling, and logit scaling for stable training. Despite its compact 4B parameter size, it achieves performance comparable to larger 7B-9B models.

This model was contributed by [aliyevaladddin](https://github.com/aliyevaladddin).
The original code can be found [here](https://huggingface.co/openbmb/MiniCPM3-4B).

## MiniCPM3Config

[[autodoc]] MiniCPM3Config

## MiniCPM3Model

[[autodoc]] MiniCPM3Model
    - forward

## MiniCPM3ForCausalLM

[[autodoc]] MiniCPM3ForCausalLM
    - forward

## MiniCPM3ForSequenceClassification

[[autodoc]] MiniCPM3ForSequenceClassification
    - forward
