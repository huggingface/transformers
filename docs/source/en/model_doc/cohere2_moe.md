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
*This model was contributed to Hugging Face Transformers on 2026-05-20.*

# Cohere2 MoE

[Command A+] is a Mixture-of-Experts (MoE) language model from Cohere. It features a hybrid attention pattern combining sliding window and full attention layers, shared and routed experts, and supports a very large context window.

## Cohere2MoeConfig

[[autodoc]] Cohere2MoeConfig

## Cohere2MoeModel

[[autodoc]] Cohere2MoeModel
    - forward

## Cohere2MoeForCausalLM

[[autodoc]] Cohere2MoeForCausalLM
    - forward
