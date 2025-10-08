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

*This model was released on 2025-06-06 and added to Hugging Face Transformers on 2025-06-25.*

# dots.llm1

[dots.llm1](https://huggingface.co/papers/2506.05767) is a large-scale Mixture of Experts (MoE) model that activates 14B parameters out of 142B, offering performance on par with leading models while lowering training and inference costs. Using a high-quality corpus and an efficient data processing pipeline, dots.llm1 matches Qwen2.5-72B after pretraining and post-training. The model is open-sourced with intermediate training checkpoints to aid research into large language model learning dynamics.

## Dots1Config

[[autodoc]] Dots1Config

## Dots1Model

[[autodoc]] Dots1Model
    - forward

## Dots1ForCausalLM

[[autodoc]] Dots1ForCausalLM
    - forward

