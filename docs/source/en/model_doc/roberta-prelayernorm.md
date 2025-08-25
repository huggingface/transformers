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
*This model was released on 2019-04-01 and added to Hugging Face Transformers on 2022-12-19.*

# RoBERTa-PreLayerNorm

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## Overview

The RoBERTa-PreLayerNorm model was proposed in [fairseq: A Fast, Extensible Toolkit for Sequence Modeling](https://huggingface.co/papers/1904.01038) by Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, Michael Auli.
It is identical to using the `--encoder-normalize-before` flag in [fairseq](https://fairseq.readthedocs.io/).

The abstract from the paper is the following:

*fairseq is an open-source sequence modeling toolkit that allows researchers and developers to train custom models for translation, summarization, language modeling, and other text generation tasks. The toolkit is based on PyTorch and supports distributed training across multiple GPUs and machines. We also support fast mixed-precision training and inference on modern GPUs.*

This model was contributed by [andreasmaden](https://huggingface.co/andreasmadsen).
The original code can be found [here](https://github.com/princeton-nlp/DinkyTrain).

## Usage tips

- The implementation is the same as [Roberta](roberta) except instead of using _Add and Norm_ it does _Norm and Add_. _Add_ and _Norm_ refers to the Addition and LayerNormalization as described in [Attention Is All You Need](https://huggingface.co/papers/1706.03762).
- This is identical to using the `--encoder-normalize-before` flag in [fairseq](https://fairseq.readthedocs.io/).

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## RobertaPreLayerNormConfig

[[autodoc]] RobertaPreLayerNormConfig

## RobertaPreLayerNormModel

[[autodoc]] RobertaPreLayerNormModel
    - forward

## RobertaPreLayerNormForCausalLM

[[autodoc]] RobertaPreLayerNormForCausalLM
    - forward

## RobertaPreLayerNormForMaskedLM

[[autodoc]] RobertaPreLayerNormForMaskedLM
    - forward

## RobertaPreLayerNormForSequenceClassification

[[autodoc]] RobertaPreLayerNormForSequenceClassification
    - forward

## RobertaPreLayerNormForMultipleChoice

[[autodoc]] RobertaPreLayerNormForMultipleChoice
    - forward

## RobertaPreLayerNormForTokenClassification

[[autodoc]] RobertaPreLayerNormForTokenClassification
    - forward

## RobertaPreLayerNormForQuestionAnswering

[[autodoc]] RobertaPreLayerNormForQuestionAnswering
    - forward
