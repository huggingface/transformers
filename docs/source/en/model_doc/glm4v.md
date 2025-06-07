<!--Copyright 2025 The ZhipuAI Inc. and The HuggingFace Inc. team. All rights reserved.

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

# GLM-4.1V

It will be completed after the model is released.

## Glm4vConfig

[[autodoc]] Glm4vConfig

## Glm4vTextConfig

[[autodoc]] Glm4vTextConfig

## Glm4vImageProcessor

[[autodoc]] Glm4vImageProcessor
    - preprocess

## Glm4vVideoProcessor

[[autodoc]] Glm4vVideoProcessor
    - preprocess

## Glm4vImageProcessorFast

[[autodoc]] Glm4vImageProcessorFast
    - preprocess

## Glm4vProcessor

[[autodoc]] Glm4vProcessor

## Glm4vTextModel

[[autodoc]] Glm4vTextModel
    - forward

## Glm4vModel

[[autodoc]] Glm4vModel
    - forward

## Glm4vForConditionalGeneration

[[autodoc]] Glm4vForConditionalGeneration
    - forward
