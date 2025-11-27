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

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# ernie4_5_vl

## Overview

The ernie4_5_vl model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

In this report, we propose PaddleOCR-VL, a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios.

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


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
