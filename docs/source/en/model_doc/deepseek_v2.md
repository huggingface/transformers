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
*This model was released on 2024-05-07 and added to Hugging Face Transformers on 2025-07-09.*

# DeepSeek-V2

## Overview

The DeepSeek-V2 model was proposed in [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://huggingface.co/papers/2405.04434) by DeepSeek-AI Team.

The abstract from the paper is the following:
We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models.

This model was contributed by [VladOS95-cyber](https://github.com/VladOS95-cyber).
The original code can be found [here](https://huggingface.co/deepseek-ai/DeepSeek-V2).

### Usage tips
The model uses Multi-head Latent Attention (MLA) and DeepSeekMoE architectures for efficient inference and cost-effective training. It employs an auxiliary-loss-free strategy for load balancing and multi-token prediction training objective. The model can be used for various language tasks after being pre-trained on 14.8 trillion tokens and going through Supervised Fine-Tuning and Reinforcement Learning stages.

## DeepseekV2Config

[[autodoc]] DeepseekV2Config

## DeepseekV2Model

[[autodoc]] DeepseekV2Model
    - forward

## DeepseekV2ForCausalLM

[[autodoc]] DeepseekV2ForCausalLM
    - forward

## DeepseekV2ForSequenceClassification

[[autodoc]] DeepseekV2ForSequenceClassification
    - forward