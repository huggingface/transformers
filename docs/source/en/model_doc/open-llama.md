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
*This model was released on 2023-04-16 and added to Hugging Face Transformers on 2023-06-20.*

# Open-Llama

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.

If you run into any issues running this model, please reinstall the last version that supported this model: v4.31.0.
You can do so by running the following command: `pip install -U transformers==4.31.0`.

</Tip>

<Tip warning={true}>

This model differs from the [OpenLLaMA models](https://huggingface.co/models?search=openllama) on the Hugging Face Hub, which primarily use the [LLaMA](llama) architecture.

</Tip>

## Overview

The Open-Llama model was proposed in the open source Open-Llama project by community developer s-JoL.

The model is mainly based on LLaMA with some modifications, incorporating memory-efficient attention from Xformers, stable embedding from Bloom, and shared input-output embedding from PaLM.
And the model is pre-trained on both Chinese and English, which gives it better performance on Chinese language tasks.

This model was contributed by [s-JoL](https://huggingface.co/s-JoL).
The original code was released on GitHub by [s-JoL](https://github.com/s-JoL), but is now removed.

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
