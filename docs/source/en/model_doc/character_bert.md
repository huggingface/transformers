<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CharacterBERT

## Overview

The CharacterBERT model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## CharacterBertConfig

[[autodoc]] CharacterBertConfig
    - all

## CharacterBertTokenizer

[[autodoc]] CharacterBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CharacterBertTokenizerFast

[[autodoc]] CharacterBertTokenizerFast

## TFCharacterBertTokenizer

[[autodoc]] TFCharacterBertTokenizer

## CharacterBert specific outputs

[[autodoc]] models.character_bert.modeling_character_bert.CharacterBertForPreTrainingOutput

[[autodoc]] models.character_bert.modeling_tf_character_bert.TFCharacterBertForPreTrainingOutput

[[autodoc]] models.character_bert.modeling_flax_character_bert.FlaxCharacterBertForPreTrainingOutput

## CharacterBertModel

[[autodoc]] CharacterBertModel
    - forward

## CharacterBertForPreTraining

[[autodoc]] CharacterBertForPreTraining
    - forward

## CharacterBertLMHeadModel

[[autodoc]] CharacterBertLMHeadModel
    - forward

## CharacterBertForMaskedLM

[[autodoc]] CharacterBertForMaskedLM
    - forward

## CharacterBertForNextSentencePrediction

[[autodoc]] CharacterBertForNextSentencePrediction
    - forward

## CharacterBertForSequenceClassification

[[autodoc]] CharacterBertForSequenceClassification
    - forward

## CharacterBertForMultipleChoice

[[autodoc]] CharacterBertForMultipleChoice
    - forward

## CharacterBertForTokenClassification

[[autodoc]] CharacterBertForTokenClassification
    - forward

## CharacterBertForQuestionAnswering

[[autodoc]] CharacterBertForQuestionAnswering
    - forward
