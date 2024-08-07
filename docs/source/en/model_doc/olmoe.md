<!--

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# OLMoE

## Overview

The OLMoE model was proposed in [TODO](TODO) by TODO.

OLMoE is a series of **O**pen **L**anguage **Mo**dels which are **M**ixture-**o**f-**E**xperts designed to enable the science of language models. We release all code, checkpoints, logs, and details involved in training these models.

The abstract from the paper is the following:

*TODO*

This model was contributed by [Muennighoff](https://hf.co/Muennighoff).
The original code can be found [here](https://github.com/allenai/OLMoE).


## OlmoeConfig

[[autodoc]] OlmoeConfig

## OlmoeModel

[[autodoc]] OlmoeModel
    - forward

## OlmoeForCausalLM

[[autodoc]] OlmoeForCausalLM
    - forward
