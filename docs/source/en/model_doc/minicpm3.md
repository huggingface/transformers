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
# MiniCPM3

## Overview

[MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B) is the 3rd generation of the MiniCPM series. The overall performance of MiniCPM3-4B surpasses Phi-3.5-mini-Instruct and GPT-3.5-Turbo-0125, being comparable with many recent 7B~9B models.

Compared to MiniCPM1.0/MiniCPM2.0, MiniCPM3-4B has a more powerful and versatile skill set to enable more general usage. MiniCPM3-4B supports function calling, along with code interpreter capabilities.

## MiniCPM3Config

[[autodoc]] MiniCPM3Config

## MiniCPM3Model

[[autodoc]] MiniCPM3Model
    - forward

## MiniCPM3ForCausalLM

[[autodoc]] MiniCPM3ForCausalLM
    - forward

## MiniCPM3ForSequenceClassification

[[autodoc]] MiniCPM3ForSequenceClassification
    - forward

## MiniCPM3ForTokenClassification

[[autodoc]] MiniCPM3ForTokenClassification
    - forward
