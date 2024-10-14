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

# OLMo

## Overview

Molmo is a family of open vision-language models developed by the Allen Institute for AI. Molmo models are trained on PixMo, a dataset of 1 million, highly-curated image-text pairs. 
The Molmo family is open source and are not trained on data derived from other vision-language models. 
Molmo models support instruction following and visual grounding through generating pointing output.

More details are in the [paper](https://huggingface.co/papers/2409.17146) and the [blog post](https://molmo.allenai.org/blog)
## OlmoConfig

[[autodoc]] OlmoConfig

## OlmoModel

[[autodoc]] OlmoModel
    - forward

## OlmoForCausalLM

[[autodoc]] OlmoForCausalLM
    - forward
