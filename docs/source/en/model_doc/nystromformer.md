<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Nyströmformer

## Overview

The Nyströmformer model was proposed in [*Nyströmformer: A Nyström-Based Algorithm for Approximating Self-Attention*](https://arxiv.org/abs/2102.03902) by Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn
Fung, Yin Li, and Vikas Singh.

The abstract from the paper is the following:

*Transformers have emerged as a powerful tool for a broad range of natural language processing tasks. A key component
that drives the impressive performance of Transformers is the self-attention mechanism that encodes the influence or
dependence of other tokens on each specific token. While beneficial, the quadratic complexity of self-attention on the
input sequence length has limited its application to longer sequences -- a topic being actively studied in the
community. To address this limitation, we propose Nyströmformer -- a model that exhibits favorable scalability as a
function of sequence length. Our idea is based on adapting the Nyström method to approximate standard self-attention
with O(n) complexity. The scalability of Nyströmformer enables application to longer sequences with thousands of
tokens. We perform evaluations on multiple downstream tasks on the GLUE benchmark and IMDB reviews with standard
sequence length, and find that our Nyströmformer performs comparably, or in a few cases, even slightly better, than
standard self-attention. On longer sequence tasks in the Long Range Arena (LRA) benchmark, Nyströmformer performs
favorably relative to other efficient self-attention methods. Our code is available at this https URL.*

This model was contributed by [novice03](https://huggingface.co/novice03). The original code can be found [here](https://github.com/mlpen/Nystromformer).

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## NystromformerConfig

[[autodoc]] NystromformerConfig

## NystromformerModel

[[autodoc]] NystromformerModel
    - forward

## NystromformerForMaskedLM

[[autodoc]] NystromformerForMaskedLM
    - forward

## NystromformerForSequenceClassification

[[autodoc]] NystromformerForSequenceClassification
    - forward

## NystromformerForMultipleChoice

[[autodoc]] NystromformerForMultipleChoice
    - forward

## NystromformerForTokenClassification

[[autodoc]] NystromformerForTokenClassification
    - forward

## NystromformerForQuestionAnswering

[[autodoc]] NystromformerForQuestionAnswering
    - forward
