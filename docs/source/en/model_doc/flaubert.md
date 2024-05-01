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

# FlauBERT

<div class="flex flex-wrap space-x-1">
<a href="https://huggingface.co/models?filter=flaubert">
<img alt="Models" src="https://img.shields.io/badge/All_model_pages-flaubert-blueviolet">
</a>
<a href="https://huggingface.co/spaces/docs-demos/flaubert_small_cased">
<img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">
</a>
</div>

## Overview

The FlauBERT model was proposed in the paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) by Hang Le et al. It's a transformer model pretrained using a masked language
modeling (MLM) objective (like BERT).

The abstract from the paper is the following:

*Language models have become a key step to achieve state-of-the art results in many different Natural Language
Processing (NLP) tasks. Leveraging the huge amount of unlabeled texts nowadays available, they provide an efficient way
to pre-train continuous word representations that can be fine-tuned for a downstream task, along with their
contextualization at the sentence level. This has been widely demonstrated for English using contextualized
representations (Dai and Le, 2015; Peters et al., 2018; Howard and Ruder, 2018; Radford et al., 2018; Devlin et al.,
2019; Yang et al., 2019b). In this paper, we introduce and share FlauBERT, a model learned on a very large and
heterogeneous French corpus. Models of different sizes are trained using the new CNRS (French National Centre for
Scientific Research) Jean Zay supercomputer. We apply our French language models to diverse NLP tasks (text
classification, paraphrasing, natural language inference, parsing, word sense disambiguation) and show that most of the
time they outperform other pretraining approaches. Different versions of FlauBERT as well as a unified evaluation
protocol for the downstream tasks, called FLUE (French Language Understanding Evaluation), are shared to the research
community for further reproducible experiments in French NLP.*

This model was contributed by [formiel](https://huggingface.co/formiel). The original code can be found [here](https://github.com/getalp/Flaubert).

Tips:
- Like RoBERTa, without the sentence ordering prediction (so just trained on the MLM objective).

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## FlaubertConfig

[[autodoc]] FlaubertConfig

## FlaubertTokenizer

[[autodoc]] FlaubertTokenizer

<frameworkcontent>
<pt>

## FlaubertModel

[[autodoc]] FlaubertModel
    - forward

## FlaubertWithLMHeadModel

[[autodoc]] FlaubertWithLMHeadModel
    - forward

## FlaubertForSequenceClassification

[[autodoc]] FlaubertForSequenceClassification
    - forward

## FlaubertForMultipleChoice

[[autodoc]] FlaubertForMultipleChoice
    - forward

## FlaubertForTokenClassification

[[autodoc]] FlaubertForTokenClassification
    - forward

## FlaubertForQuestionAnsweringSimple

[[autodoc]] FlaubertForQuestionAnsweringSimple
    - forward

## FlaubertForQuestionAnswering

[[autodoc]] FlaubertForQuestionAnswering
    - forward

</pt>
<tf>

## TFFlaubertModel

[[autodoc]] TFFlaubertModel
    - call

## TFFlaubertWithLMHeadModel

[[autodoc]] TFFlaubertWithLMHeadModel
    - call

## TFFlaubertForSequenceClassification

[[autodoc]] TFFlaubertForSequenceClassification
    - call

## TFFlaubertForMultipleChoice

[[autodoc]] TFFlaubertForMultipleChoice
    - call

## TFFlaubertForTokenClassification

[[autodoc]] TFFlaubertForTokenClassification
    - call

## TFFlaubertForQuestionAnsweringSimple

[[autodoc]] TFFlaubertForQuestionAnsweringSimple
    - call

</tf>
</frameworkcontent>



