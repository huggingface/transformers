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

# RoCBert

## Overview

The RoCBert model was proposed in [RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining](https://aclanthology.org/2022.acl-long.65.pdf)  by HuiSu, WeiweiShi, XiaoyuShen, XiaoZhou, TuoJi, JiaruiFang, JieZhou.
It's a pretrained Chinese language model that is robust under various forms of adversarial attacks.

The abstract from the paper is the following:

*Large-scale pretrained language models have achieved SOTA results on NLP tasks. However, they have been shown
vulnerable to adversarial attacks especially for logographic languages like Chinese. In this work, we propose
ROCBERT: a pretrained Chinese Bert that is robust to various forms of adversarial attacks like word perturbation,
synonyms, typos, etc. It is pretrained with the contrastive learning objective which maximizes the label consistency
under different synthesized adversarial examples. The model takes as input multimodal information including the
semantic, phonetic and visual features. We show all these features are important to the model robustness since the
attack can be performed in all the three forms. Across 5 Chinese NLU tasks, ROCBERT outperforms strong baselines under
three blackbox adversarial algorithms without sacrificing the performance on clean testset. It also performs the best
in the toxic content detection task under human-made attacks.*

This model was contributed by [weiweishi](https://huggingface.co/weiweishi).

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## RoCBertConfig

[[autodoc]] RoCBertConfig
    - all

## RoCBertTokenizer

[[autodoc]] RoCBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## RoCBertModel

[[autodoc]] RoCBertModel
    - forward

## RoCBertForPreTraining

[[autodoc]] RoCBertForPreTraining
    - forward

## RoCBertForCausalLM

[[autodoc]] RoCBertForCausalLM
    - forward

## RoCBertForMaskedLM

[[autodoc]] RoCBertForMaskedLM
    - forward

## RoCBertForSequenceClassification

[[autodoc]] transformers.RoCBertForSequenceClassification
    - forward

## RoCBertForMultipleChoice

[[autodoc]] transformers.RoCBertForMultipleChoice
    - forward

## RoCBertForTokenClassification

[[autodoc]] transformers.RoCBertForTokenClassification
    - forward

## RoCBertForQuestionAnswering

[[autodoc]] RoCBertForQuestionAnswering
    - forward
