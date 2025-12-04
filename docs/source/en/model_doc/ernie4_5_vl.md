<!--Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2025-06-30 and added to Hugging Face Transformers on TBD.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# Ernie 4.5 VL

## Overview

The Ernie 4.5 VL model was released in the [Ernie 4.5 Model Family](https://ernie.baidu.com/blog/posts/ernie4.5/) release by baidu.
This family of models contains multiple different architectures and model sizes. The Vision-Language series in specific is
composed of a novel multimodal heterogeneous structure, sharing paremeters across modalities and dedicating parameters
to specific modalities. This becomes especially apparent in the Mixture of Expert (MoE) which is composed of
- Dedicated Text Experts
- Dedicated Vision Experts
- Shared Experts
This architecture has the advantage to enhance multimodal understanding without compromising, and even improving, performance on text-related tasks. TODO image of the moe?

Other models from the family can be found at [Ernie 4.5](./ernie4_5) and at [Ernie 4.5 MoE](./ernie4_5_moe.md).


TODO: tips, usage, etc.


## Ernie4_5_VLConfig

[[autodoc]] Ernie4_5_VLConfig

## Ernie4_5_VLTextConfig

[[autodoc]] Ernie4_5_VLTextConfig

## Ernie4_5_VLVisionConfig

[[autodoc]] Ernie4_5_VLVisionConfig

## Ernie4_5_VLImageProcessor

[[autodoc]] Ernie4_5_VLImageProcessor
    - preprocess

## Ernie4_5_VLImageProcessorFast

[[autodoc]] Ernie4_5_VLImageProcessorFast
    - preprocess

## Ernie4_5_VLVideoProcessor

[[autodoc]] Ernie4_5_VLVideoProcessor
    - preprocess

## Ernie4_5_VLProcessor

[[autodoc]] Ernie4_5_VLProcessor

## Ernie4_5_VLTextModel

[[autodoc]] Ernie4_5_VLTextModel
    - forward

## Ernie4_5_VLVisionTransformerPretrainedModel

[[autodoc]] Ernie4_5_VLVisionTransformerPretrainedModel
    - forward

## Ernie4_5_VLVariableResolutionResamplerModel

[[autodoc]] Ernie4_5_VLVariableResolutionResamplerModel
    - forward

## Ernie4_5_VLModel

[[autodoc]] Ernie4_5_VLModel
    - forward

## Ernie4_5_VLForConditionalGeneration

[[autodoc]] Ernie4_5_VLForConditionalGeneration
    - forward
