<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Open-Llama

## Overview

The Open-Llama model was proposed in [Open-Llama project](https://github.com/s-JoL/Open-Llama) by community developer s-JoL.

The model is mainly based on LLaMA with some modifications, incorporating memory-efficient attention from Xformers, stable embedding from Bloom, and shared input-output embedding from PaLM.
And the model is pre-trained on both Chinese and English, which gives it better performance on Chinese language tasks.

This model was contributed by [s-JoL](https://huggingface.co/s-JoL).
The original code can be found [Open-Llama](https://github.com/s-JoL/Open-Llama).
Checkpoint and usage can be found at [s-JoL/Open-Llama-V1](https://huggingface.co/s-JoL/Open-Llama-V1).


## OpenLlamaConfig

[[autodoc]] OpenLlamaConfig

## OpenLlamaModel

[[autodoc]] OpenLlamaModel
    - forward

## OpenLlamaForCausalLM

[[autodoc]] OpenLlamaForCausalLM
    - forward

## OpenLlamaForSequenceClassification

[[autodoc]] OpenLlamaForSequenceClassification
    - forward
