<!--Copyright 2024 JetMoe team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# JetMoe

## Overview

**JetMoe-8B** is an 8B Mixture-of-Experts (MoE) language model developed by [Yikang Shen](https://scholar.google.com.hk/citations?user=qff5rRYAAAAJ) and [MyShell](https://myshell.ai/).
JetMoe project aims to provide a LLaMA2-level performance and efficient language model with a limited budget.
To achieve this goal, JetMoe uses a sparsely activated architecture inspired by the [ModuleFormer](https://arxiv.org/abs/2306.04640). 
Each JetMoe block consists of two MoE layers: Mixture of Attention Heads and Mixture of MLP Experts.
Given the input tokens, it activates a subset of its experts to process them.
This sparse activation schema enables JetMoe to achieve much better training throughput than similar size dense models. 
The training throughput of JetMoe-8B is around 100B tokens per day on a cluster of 96 H100 GPUs with a straightforward 3-way pipeline parallelism strategy.

This model was contributed by [Yikang Shen](https://huggingface.co/YikangS).


## JetMoeConfig

[[autodoc]] JetMoeConfig

## JetMoeModel

[[autodoc]] JetMoeModel
    - forward

## JetMoeForCausalLM

[[autodoc]] JetMoeForCausalLM
    - forward

## JetMoeForSequenceClassification

[[autodoc]] JetMoeForSequenceClassification
    - forward
