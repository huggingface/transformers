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


⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->

# MiniCPM5-15B-A2.5B

MiniCPM5-15B-A2.5B is a decoder-only language model with 15 billion parameters and 2.5 billion active parameters per token. It uses gated grouped-query attention. The first decoder layer has a dense feed-forward network. Each subsequent layer uses a shared expert and selects 16 of 160 routed experts per token.

## MiniCPM5Config

[[autodoc]] MiniCPM5Config

## MiniCPM5Model

[[autodoc]] MiniCPM5Model
    - forward

## MiniCPM5MoEForCausalLM

[[autodoc]] MiniCPM5MoEForCausalLM
    - forward
