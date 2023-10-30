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

# CamemBERT

## Overview

The CamemBERT model was proposed in [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) by
Louis Martin, Benjamin Muller, Pedro Javier Ortiz Suárez, Yoann Dupont, Laurent Romary, Éric Villemonte de la
Clergerie, Djamé Seddah, and Benoît Sagot. It is based on Facebook's RoBERTa model released in 2019. It is a model
trained on 138GB of French text.

The abstract from the paper is the following:

*Pretrained language models are now ubiquitous in Natural Language Processing. Despite their success, most available
models have either been trained on English data or on the concatenation of data in multiple languages. This makes
practical use of such models --in all languages except English-- very limited. Aiming to address this issue for French,
we release CamemBERT, a French version of the Bi-directional Encoders for Transformers (BERT). We measure the
performance of CamemBERT compared to multilingual models in multiple downstream tasks, namely part-of-speech tagging,
dependency parsing, named-entity recognition, and natural language inference. CamemBERT improves the state of the art
for most of the tasks considered. We release the pretrained model for CamemBERT hoping to foster research and
downstream applications for French NLP.*

Tips:

- This implementation is the same as RoBERTa. Refer to the [documentation of RoBERTa](roberta) for usage examples
  as well as the information relative to the inputs and outputs.

This model was contributed by [camembert](https://huggingface.co/camembert). The original code can be found [here](https://camembert-model.fr/).

## Documentation resources

- [Text classification task guide](../tasks/sequence_classification)
- [Token classification task guide](../tasks/token_classification)
- [Question answering task guide](../tasks/question_answering)
- [Causal language modeling task guide](../tasks/language_modeling)
- [Masked language modeling task guide](../tasks/masked_language_modeling)
- [Multiple choice task guide](../tasks/multiple_choice)

## CamembertConfig

[[autodoc]] CamembertConfig

## CamembertTokenizer

[[autodoc]] CamembertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CamembertTokenizerFast

[[autodoc]] CamembertTokenizerFast

## CamembertModel

[[autodoc]] CamembertModel

## CamembertForCausalLM

[[autodoc]] CamembertForCausalLM

## CamembertForMaskedLM

[[autodoc]] CamembertForMaskedLM

## CamembertForSequenceClassification

[[autodoc]] CamembertForSequenceClassification

## CamembertForMultipleChoice

[[autodoc]] CamembertForMultipleChoice

## CamembertForTokenClassification

[[autodoc]] CamembertForTokenClassification

## CamembertForQuestionAnswering

[[autodoc]] CamembertForQuestionAnswering

## TFCamembertModel

[[autodoc]] TFCamembertModel

## TFCamembertForCasualLM

[[autodoc]] TFCamembertForCausalLM

## TFCamembertForMaskedLM

[[autodoc]] TFCamembertForMaskedLM

## TFCamembertForSequenceClassification

[[autodoc]] TFCamembertForSequenceClassification

## TFCamembertForMultipleChoice

[[autodoc]] TFCamembertForMultipleChoice

## TFCamembertForTokenClassification

[[autodoc]] TFCamembertForTokenClassification

## TFCamembertForQuestionAnswering

[[autodoc]] TFCamembertForQuestionAnswering
