<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2026-04-30.*

# DeepSeek-V4

[DeepSeek-V4](https://huggingface.co/deepseek-ai) is a family of MoE language models released by DeepSeek. Relative
to DeepSeek-V3, V4 replaces MLA with sliding-window attention plus a per-layer KV Compressor, swaps residual
connections for Hyper-Connections, routes the first few layers via a static token-id hash, and drops expert groups.

This implementation covers the `DeepSeek-V4-Flash`, `DeepSeek-V4-Pro`, and their `-Base` pretrained siblings. All
four share the same architecture; they differ only in width / depth / expert count and weights.

## DeepseekV4Config

[[autodoc]] DeepseekV4Config

## DeepseekV4Model

[[autodoc]] DeepseekV4Model
    - forward

## DeepseekV4ForCausalLM

[[autodoc]] DeepseekV4ForCausalLM
    - forward
