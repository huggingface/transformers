<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ConvBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=convbert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-convbert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/conv-bert-base">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The ConvBERT model was proposed in [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496) by Zihang Jiang, Weihao Yu, Daquan Zhou, Yunpeng Chen, Jiashi Feng, Shuicheng
Yan.

The abstract from the paper is the following:

*Pre-trained language models like BERT and its variants have recently achieved impressive performance in various
natural language understanding tasks. However, BERT heavily relies on the global self-attention block and thus suffers
large memory footprint and computation cost. Although all its attention heads query on the whole input sequence for
generating the attention map from a global perspective, we observe some heads only need to learn local dependencies,
which means the existence of computation redundancy. We therefore propose a novel span-based dynamic convolution to
replace these self-attention heads to directly model local dependencies. The novel convolution heads, together with the
rest self-attention heads, form a new mixed attention block that is more efficient at both global and local context
learning. We equip BERT with this mixed attention design and build a ConvBERT model. Experiments have shown that
ConvBERT significantly outperforms BERT and its variants in various downstream tasks, with lower training cost and
fewer model parameters. Remarkably, ConvBERTbase model achieves 86.4 GLUE score, 0.7 higher than ELECTRAbase, while
using less than 1/4 training cost. Code and pre-trained models will be released.*

ConvBERT training tips are similar to those of BERT.

This model was contributed by [abhishek](https://huggingface.co/abhishek). The original implementation can be found
here: https://github.com/yitu-opensource/ConvBert

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## ConvBertConfig

[[autodoc]] ConvBertConfig

## ConvBertTokenizer

[[autodoc]] ConvBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## ConvBertTokenizerFast

[[autodoc]] ConvBertTokenizerFast

## ConvBertModel

[[autodoc]] ConvBertModel
    - forward

## ConvBertForMaskedLM

[[autodoc]] ConvBertForMaskedLM
    - forward

## ConvBertForSequenceClassification

[[autodoc]] ConvBertForSequenceClassification
    - forward

## ConvBertForMultipleChoice

[[autodoc]] ConvBertForMultipleChoice
    - forward

## ConvBertForTokenClassification

[[autodoc]] ConvBertForTokenClassification
    - forward

## ConvBertForQuestionAnswering

[[autodoc]] ConvBertForQuestionAnswering
    - forward

## TFConvBertModel

[[autodoc]] TFConvBertModel
    - call

## TFConvBertForMaskedLM

[[autodoc]] TFConvBertForMaskedLM
    - call

## TFConvBertForSequenceClassification

[[autodoc]] TFConvBertForSequenceClassification
    - call

## TFConvBertForMultipleChoice

[[autodoc]] TFConvBertForMultipleChoice
    - call

## TFConvBertForTokenClassification

[[autodoc]] TFConvBertForTokenClassification
    - call

## TFConvBertForQuestionAnswering

[[autodoc]] TFConvBertForQuestionAnswering
    - call
