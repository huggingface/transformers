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
*This model was released on 2025-07-28 and added to Hugging Face Transformers on 2025-08-08.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">    </div>
</div>

# Glm4vMoe

## Overview

Vision-language models (VLMs) have become a key cornerstone of intelligent systems. As real-world AI tasks grow increasingly complex, VLMs urgently need to enhance reasoning capabilities beyond basic multimodal perception — improving accuracy, comprehensiveness, and intelligence — to enable complex problem solving, long-context understanding, and multimodal agents.

Through our open-source work, we aim to explore the technological frontier together with the community while empowering more developers to create exciting and innovative applications.

[GLM-4.5V](https://huggingface.co/papers/2508.06471) ([Github repo](https://github.com/zai-org/GLM-V)) is based on ZhipuAI’s next-generation flagship text foundation model GLM-4.5-Air (106B parameters, 12B active).  It continues the technical approach of [GLM-4.1V-Thinking](https://huggingface.co/papers/2507.01006), achieving SOTA performance among models of the same scale on 42 public vision-language benchmarks.  It covers common tasks such as image, video, and document understanding, as well as GUI agent operations.

![bench_45](https://raw.githubusercontent.com/zai-org/GLM-V/refs/heads/main/resources/bench_45v.jpeg)

Beyond benchmark performance, GLM-4.5V focuses on real-world usability. Through efficient hybrid training, it can handle diverse types of visual content, enabling full-spectrum vision reasoning, including:
- **Image reasoning** (scene understanding, complex multi-image analysis, spatial recognition)
- **Video understanding** (long video segmentation and event recognition)
- **GUI tasks** (screen reading, icon recognition, desktop operation assistance)
- **Complex chart & long document parsing** (research report analysis, information extraction)
- **Grounding** (precise visual element localization)

The model also introduces a **Thinking Mode** switch, allowing users to balance between quick responses and deep reasoning. This switch works the same as in the `GLM-4.5` language model.

## Glm4vMoeConfig

[[autodoc]] Glm4vMoeConfig

## Glm4vMoeTextConfig

[[autodoc]] Glm4vMoeTextConfig

## Glm4vMoeTextModel

[[autodoc]] Glm4vMoeTextModel
    - forward

## Glm4vMoeModel

[[autodoc]] Glm4vMoeModel
    - forward

## Glm4vMoeForConditionalGeneration

[[autodoc]] Glm4vMoeForConditionalGeneration
    - forward
