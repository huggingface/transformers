<!--Copyright 2023 The HuggingFace and Baidu Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ErnieM

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

<Tip warning={true}>

This model is in maintenance mode only, we don't accept any new PRs changing its code.
If you run into any issues running this model, please reinstall the last version that supported this model: v4.40.2.
You can do so by running the following command: `pip install -U transformers==4.40.2`.

</Tip>

## Overview

The ErnieM model was proposed in [ERNIE-M: Enhanced Multilingual Representation by Aligning
Cross-lingual Semantics with Monolingual Corpora](https://huggingface.co/papers/2012.15674)  by Xuan Ouyang, Shuohuan Wang, Chao Pang, Yu Sun,
Hao Tian, Hua Wu, Haifeng Wang.

The abstract from the paper is the following:

*Recent studies have demonstrated that pre-trained cross-lingual models achieve impressive performance in downstream cross-lingual tasks. This improvement benefits from learning a large amount of monolingual and parallel corpora. Although it is generally acknowledged that parallel corpora are critical for improving the model performance, existing methods are often constrained by the size of parallel corpora, especially for lowresource languages. In this paper, we propose ERNIE-M, a new training method that encourages the model to align the representation of multiple languages with monolingual corpora, to overcome the constraint that the parallel corpus size places on the model performance. Our key insight is to integrate back-translation into the pre-training process. We generate pseudo-parallel sentence pairs on a monolingual corpus to enable the learning of semantic alignments between different languages, thereby enhancing the semantic modeling of cross-lingual models. Experimental results show that ERNIE-M outperforms existing cross-lingual models and delivers new state-of-the-art results in various cross-lingual downstream tasks.*
This model was contributed by [Susnato Dhar](https://huggingface.co/susnato). The original code can be found [here](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ernie_m).


## Usage tips

- Ernie-M is a BERT-like model so it is a stacked Transformer Encoder.
- Instead of using MaskedLM for pretraining (like BERT) the authors used two novel techniques: `Cross-attention Masked Language Modeling` and `Back-translation Masked Language Modeling`. For now these two LMHead objectives are not implemented here.
- It is a multilingual language model.
- Next Sentence Prediction was not used in pretraining process.

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Multiple choice task guide](../tasks/multiple_choice)

## ErnieMConfig

[[autodoc]] ErnieMConfig


## ErnieMTokenizer

[[autodoc]] ErnieMTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary


## ErnieMModel

[[autodoc]] ErnieMModel
    - forward

## ErnieMForSequenceClassification

[[autodoc]] ErnieMForSequenceClassification
    - forward


## ErnieMForMultipleChoice

[[autodoc]] ErnieMForMultipleChoice
    - forward


## ErnieMForTokenClassification

[[autodoc]] ErnieMForTokenClassification
    - forward


## ErnieMForQuestionAnswering

[[autodoc]] ErnieMForQuestionAnswering
    - forward

## ErnieMForInformationExtraction

[[autodoc]] ErnieMForInformationExtraction
    - forward
