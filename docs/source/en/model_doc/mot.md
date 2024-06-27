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

# MixtureOfTokens

## Overview

The MixtureOfTokens model was proposed in [MIXTURE OF TOKENS: EFFICIENT LLMS THROUGH
CROSS-EXAMPLE AGGREGATION](https://arxiv.org/abs/2310.15961) by Szymon Antoniak, Sebastian Jaszczur, Michał Krutul, Maciej Pióro, Jakub Krajewski, Jan Ludziejewski, Tomasz Odrzygóźdź, Marek Cygan.

The abstract from the paper is the following:

Despite the promise of Mixture of Experts (MoE) models in increasing parameter counts of Transformer models while maintaining training and inference costs,
their application carries notable drawbacks. The key strategy of these models is to,
for each processed token, activate at most a few experts - subsets of an extensive
feed-forward layer. But this approach is not without its challenges. The operation
of matching experts and tokens is discrete, which makes MoE models prone to issues like training instability and uneven expert utilization. Existing techniques designed to address these concerns, such as auxiliary losses or balance-aware matching, result either in lower model performance or are more difficult to train. In response to these issues, we propose Mixture of Tokens, a fully-differentiable model
that retains the benefits of MoE architectures while avoiding the aforementioned
difficulties. Rather than routing tokens to experts, this approach mixes tokens from
different examples prior to feeding them to experts, enabling the model to learn
from all token-expert combinations. Importantly, this mixing can be disabled to
avoid mixing of different sequences during inference. Crucially, this method is
fully compatible with both masked and causal Large Language Model training
and inference.

Tips:

During inference, the model's computational performance is derived from combining tokens across batches into groups of a specified size, denoted as *group_size* in the model configuration. If the batch size is not evenly divisible by *group_size*, the model will internally pad the batch to ensure divisibility. To achieve optimal performance, it is advisable to conduct batched inference using a batch size that is a multiple of *group_size*.


This model was contributed by [jaszczur](https://huggingface.co/jaszczur).
The original code can be found [here](https://github.com/llm-random/llm-random/blob/main/research/conditional/moe_layers/continuous_moe.py).


## MoTConfig

[[autodoc]] MoTConfig

<frameworkcontent>
<pt>

## MoTModel

[[autodoc]] MoTModel
    - forward

## MoTLMHeadModel

[[autodoc]] MoTLMHeadModel
    - forward

## MoTForQuestionAnswering

[[autodoc]] MoTForQuestionAnswering
    - forward

## MoTForSequenceClassification

[[autodoc]] MoTForSequenceClassification
    - forward

## MoTForTokenClassification

[[autodoc]] MoTForTokenClassification
    - forward

</pt>
<tf>
