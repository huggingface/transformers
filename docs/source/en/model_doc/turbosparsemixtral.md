<!--Copyright 2023 Mistral AI and The HuggingFace Team. All rights reserved.
<!--Copyright 2023 SJTU IPADS. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# TurboSparse-Mixtral

## Overview

TurboSparse-Mixtral was introduced in the [paper](https://arxiv.org/abs/2406.05955) by Yixin Song, Haotong Xie, Zhengyan Zheng, Bo wen, Jianxiang Gao, Li Ma, Zeyu Mi, Haibo Chen.

The introduction of the paper says:

Mixtral-8x7B is the second large language model (LLM) released by [mistral.ai](https://mistral.ai/), after [Mistral-7B](mistral).
*Exploiting activation sparsity is a promising approach to significantly accelerating the inference process of large language models (LLMs) without compromising performance. However, activation sparsity is determined by activation functions, and commonly used ones like SwiGLU and GeGLU exhibit limited sparsity. Simply replacing these functions with ReLU fails to achieve sufficient sparsity. Moreover, inadequate training data can further increase the risk of performance degradation.To address these challenges, we propose a novel dReLU function, designed to improve LLM activation sparsity, along with a high-quality training data mixture ratio to facilitate effective sparsification. Additionally, we leverage sparse activation patterns within the Feed-Forward Network (FFN) experts of Mixture-of-Experts (MoE) models to further boost efficiency. By applying our neuron sparsification method to the Mistral and Mixtral models, only 2.5 billion and 4.3 billion parameters are activated per inference iteration, respectively, while achieving even more powerful model performance. Evaluation results demonstrate that this sparsity achieves a 2-5× decoding speedup.*

### Architectural details

TurboSparse-Mixtral 47B is a decoder-only Transformer with the following architectural choices:

- TurboSparse-Mixtral is a Mixture of Experts (MoE) model with 8 experts per MLP, with a total of 45 billion parameters. To learn more about mixture-of-experts, refer to the [blog post](https://huggingface.co/blog/moe).
- dReLU activation structure. To push more sparsity, TurboSparse-Mixtral replace the original SwiGLU with dReLU.

### License

`TurboSparse-Mixtral` is released under the Apache 2.0 license.

## TurboSparseMixtralConfig

[[autodoc]] TurboSparseMixtralConfig

## TurboSparseMixtralModel

[[autodoc]] TurboSparseMixtralModel
    - forward

## TurboSparseMixtralForCausalLM

[[autodoc]] TurboSparseMixtralForCausalLM
    - forward

## TurboSparseMixtralForSequenceClassification

[[autodoc]] TurboSparseMixtralForSequenceClassification
    - forward

## TurboSparseMixtralForTokenClassification

[[autodoc]] TurboSparseMixtralForTokenClassification
    - forward
