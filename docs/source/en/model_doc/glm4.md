<!--Copyright 2025 The GLM & ZhipuAI team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2024-06-18 and added to Hugging Face Transformers on 2025-04-09.*

# Glm4

## Overview

The GLM family welcomes new members [GLM-4-0414](https://huggingface.co/papers/2406.12793) series models.

The **GLM-4-32B-0414** series models, featuring 32 billion parameters. Its performance is comparable to OpenAI’s GPT
series and DeepSeek’s V3/R1 series. It also supports very user-friendly local deployment features. GLM-4-32B-Base-0414
was pre-trained on 15T of high-quality data, including substantial reasoning-type synthetic data. This lays the
foundation for subsequent reinforcement learning extensions. In the post-training stage, we employed human preference
alignment for dialogue scenarios. Additionally, using techniques like rejection sampling and reinforcement learning, we
enhanced the model’s performance in instruction following, engineering code, and function calling, thus strengthening
the atomic capabilities required for agent tasks. GLM-4-32B-0414 achieves good results in engineering code, Artifact
generation, function calling, search-based Q&A, and report generation. In particular, on several benchmarks, such as
code generation or specific Q&A tasks, GLM-4-32B-Base-0414 achieves comparable performance with those larger models like
GPT-4o and DeepSeek-V3-0324 (671B).

**GLM-Z1-32B-0414** is a reasoning model with deep thinking capabilities. This was developed based on GLM-4-32B-0414
through cold start, extended reinforcement learning, and further training on tasks including mathematics, code, and
logic. Compared to the base model, GLM-Z1-32B-0414 significantly improves mathematical abilities and the capability to
solve complex tasks. During training, we also introduced general reinforcement learning based on pairwise ranking
feedback, which enhances the model's general capabilities.

**GLM-Z1-Rumination-32B-0414** is a deep reasoning model with rumination capabilities (against OpenAI's Deep Research).
Unlike typical deep thinking models, the rumination model is capable of deeper and longer thinking to solve more
open-ended and complex problems (e.g., writing a comparative analysis of AI development in two cities and their future
development plans). Z1-Rumination is trained through scaling end-to-end reinforcement learning with responses graded by
the ground truth answers or rubrics and can make use of search tools during its deep thinking process to handle complex
tasks. The model shows significant improvements in research-style writing and complex tasks.

Finally, **GLM-Z1-9B-0414** is a surprise. We employed all the aforementioned techniques to train a small model (9B).
GLM-Z1-9B-0414 exhibits excellent capabilities in mathematical reasoning and general tasks. Its overall performance is
top-ranked among all open-source models of the same size. Especially in resource-constrained scenarios, this model
achieves an excellent balance between efficiency and effectiveness, providing a powerful option for users seeking
lightweight deployment.

## Glm4Config

[[autodoc]] Glm4Config

## Glm4Model

[[autodoc]] Glm4Model
    - forward

## Glm4ForCausalLM

[[autodoc]] Glm4ForCausalLM
    - forward

## Glm4ForSequenceClassification

[[autodoc]] Glm4ForSequenceClassification
    - forward

## Glm4ForTokenClassification

[[autodoc]] Glm4ForTokenClassification
    - forward
