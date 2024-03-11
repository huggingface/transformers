<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# LLaMA-MoE

## Overview

The LLaMA-MoE model was proposed in [LLaMA-MoE: Building Mixture-of-Experts from LLaMA with Continual Pre-training](https://github.com/pjlab-sys4nlp/llama-moe/blob/main/docs/LLaMA_MoE.pdf) by LLaMA-MoE Team. LLaMA-MoE is a series of open-sourced Mixture-of-Expert (MoE) models based on LLaMA and SlimPajama. The authors build LLaMA-MoE by first partitioning LLaMA's FFNs into sparse experts and insert top-K gate for each layer of experts, and then continually pre-train the initialized MoE model with an optimized data sampling weights from Sheared LLaMA and filtered datasets from SlimPajama.

The abstract from the paper is the following:

*Despite the significant advancements of decoder-style dense large language models (LLMs), e.g., LLaMA and ChatGPT, there remains limited exploration of sparse language models. Sparsely activated models, decoupling model size from computation costs, provide a practicable way to extrapolate the scaling law and attract increasing attention. Although sparse models are more efficient and flexible in terms of quality and computation cost, they still suffer from data-hungry and instability problems to training from scratch in a large-scale setting. Motivated by these limits, we investigate building a sparsely activated Mixture-of-Experts (MoE) model from existing decoder-style large language models. Specifically, based on the most well-known open-source LLaMA-2, we obtain an MoE model by: (1) Expert Construction, which partitions the parameters of original Feed- Forward Networks (FFNs) in the LLaMA models into multiple functional modules as experts; and (2) Continual pre-training, which further trains the transformed MoE model and additional gate networks for expert routing. After these stages, the model could maintain its language abilities and routes the input tokens to specific experts. Meanwhile, only part of the total parameters are activated. In this report, we present the LLaMA-MoE-v1 series, converting a LLaMA-2-7B model into MoE models and training them continually. In particular, we introduce two different sizes of MoE models that activate 3.0B and 3.5B parameters, respectively. Empirically, by training 200B tokens, LLaMA-MoE-v1-3.5B models significantly outperform dense models that contain similar activation parameters, while LLaMA-MoE-v1-3.0B performs comparably with them. LLaMA-MoE-v1 series also provide a feasible framework to train MoE models from the existing LLMs in a more cost-effective approach. It is worth noting that our framework can be easily applied to more decoder-style LLMs. The source code and models can be obtained at https://github.com/pjlab-sys4nlp/llama-moe.*

This model was contributed by [LLaMA-MoE Team](https://huggingface.co/llama-moe). The original code can be found [here](https://github.com/pjlab-sys4nlp/llama-moe/tree/main/smoe/models/llama_moe).

## LlamaMoEConfig

[[autodoc]] LlamaMoEConfig


## LlamaMoEModel

[[autodoc]] LlamaMoEModel
    - forward


## LlamaMoEForCausalLM

[[autodoc]] LlamaMoEForCausalLM
    - forward


## LlamaMoEForSequenceClassification

[[autodoc]] transformers.LlamaMoEForSequenceClassification
    - forward


## LlamaMoEForQuestionAnswering

[[autodoc]] LlamaMoEForQuestionAnswering
    - forward
