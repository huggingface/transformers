<!--Copyright 2025 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was released on 2025-08-12 and added to Hugging Face Transformers on 2025-08-08.*

# Glm4vMoe

## Overview

The GLM-V model was proposed in [GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](https://huggingface.co/papers/2507.01006v6).

The abstract from the paper is the following:

> *We present GLM-4.1V-Thinking, GLM-4.5V, and GLM-4.6V, a family of vision-language models (VLMs) designed to advance
general-purpose multimodal understanding and reasoning. In this report, we share our key findings in the development of
the reasoning-centric training framework. We first develop a capable vision foundation model with significant potential
through large-scale pre-training, which arguably sets the upper bound for the final performance. We then propose
Reinforcement Learning with Curriculum Sampling (RLCS) to unlock the full potential of the model, leading to
comprehensive capability enhancement across a diverse range of tasks, including STEM problem solving, video
understanding, content recognition, coding, grounding, GUI-based agents, and long document interpretation. In a
comprehensive evaluation across 42 public benchmarks, GLM-4.5V achieves state-of-the-art performance on nearly all tasks
among open-source models of similar size, and demonstrates competitive or even superior results compared to
closed-source models such as Gemini-2.5-Flash on challenging tasks including Coding and GUI Agents. Meanwhile, the
smaller GLM-4.1V-9B-Thinking remains highly competitive-achieving superior results to the much larger Qwen2.5-VL-72B on
29 benchmarks. We open-source both GLM-4.1V-9B-Thinking and GLM-4.5V. We further introduce the GLM-4.6V series,
open-source multimodal models with native tool use and a 128K context window. A brief overview is available at this
https URL. Code, models and more information are released at https://github.com/zai-org/GLM-V*

## Support Model

This Model type support these model of zai-org:

+ [GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V)
+ [GLM-4.6V](https://huggingface.co/zai-org/GLM-4.6V)

This model was contributed by [Raushan Turganbay](https://huggingface.co/RaushanTurganbay) and [Yuxuan Zhang](https://huggingface.co/ZHANGYUXUAN-zR).

## Glm4vMoeConfig

[[autodoc]] Glm4vMoeConfig

## Glm4vMoeVisionConfig

[[autodoc]] Glm4vMoeVisionConfig

## Glm4vMoeTextConfig

[[autodoc]] Glm4vMoeTextConfig

## Glm4vMoeVisionModel

[[autodoc]] Glm4vMoeVisionModel
    - forward

## Glm4vMoeTextModel

[[autodoc]] Glm4vMoeTextModel
    - forward

## Glm4vMoeModel

[[autodoc]] Glm4vMoeModel
    - forward

## Glm4vMoeForConditionalGeneration

[[autodoc]] Glm4vMoeForConditionalGeneration
    - forward
