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
*This model was released on 2025-12-09 and added to Hugging Face Transformers on 2025-12-16.*

# Jais2

## Overview

Jais2 a next-generation Arabic open-weight LLM trained on the richest Arabic-first dataset to date. Built from the ground up with 8B and 70B parameters, Jais 2 understands Arabic the way it's truly spoken across dialects, cuulutre, and modern expression. It is developed by MBZUAI, Inception and Cerebras Systems and based on the transformer architecture with modifications including:

- LayerNorm instead of RMSNorm
- ReLU² activation function
- Rotary Position Embeddings (RoPE)

## Jais2Config

[[autodoc]] Jais2Config

## Jais2Model

[[autodoc]] Jais2Model
    - forward

## Jais2ForCausalLM

[[autodoc]] Jais2ForCausalLM
    - forward
