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

# CharacterBERT

## Overview

The CharacterBERT model was proposed in [CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary Representations From Characters](https://aclanthology.org/2020.coling-main.609/) by [Hicham El Boukkouri](https://scholar.google.com/citations?user=rK_ER-YAAAAJ&hl=fr), [Olivier Ferret](https://scholar.google.com/citations?user=-mCQhtIAAAAJ&hl=fr), [Thomas Lavergne](https://scholar.google.com/citations?user=l7XLFhEAAAAJ&hl=fr), [Hiroshi Noji](https://scholar.google.com/citations?user=OODRveoAAAAJ&hl=fr), [Pierre Zweigenbaum](https://scholar.google.com/citations?user=0LjUNAsAAAAJ&hl=fr) and [Junichi Tsujii](https://scholar.google.com/citations?user=h3aNnAIAAAAJ&hl=fr).

This model is a version of BERT that allows for token-level inputs and outputs. It does not use a subword vocabulary and instead processes each input token at the character-level, making it more flexible and easy to use as well robust to misspellings.

The abstract from the paper is the following:

*Due to the compelling improvements brought by **BERT**, many recent representation models adopted the **Transformer architecture** as their main building block, consequently inheriting the **wordpiece tokenization system** despite it not being intrinsically linked to the notion of Transformers. While this system is thought to achieve a **good balance between the flexibility of characters and the efficiency of full words**, using predefined wordpiece vocabularies from the general domain is not always suitable, especially when building models for **specialized domains** (e.g., the medical domain). Moreover, adopting a wordpiece tokenization **shifts the focus from the word level to the subword level**, making the models **conceptually more complex** and arguably **less convenient in practice**. For these reasons, we propose CharacterBERT, a new variant of BERT that **drops the wordpiece system altogether and uses a Character-CNN module instead** to represent entire words by consulting their characters. We show that this new model **improves the performance of BERT** on a variety of medical domain tasks while at the same time producing **robust**, **word-level** and **open-vocabulary** representations.*

Tips:

- The model is **token-level** but relies on **characters** internally. This means the tokenization can be a simple **split on whitespace**.
- The WordPiece vocabulary is replaced with a much smaller character (byte) vocabulary. By encoding all texts in **UTF-8**, any characters from any langage can fit in a byte vocabulary of size **28=256 + special tokens**.
- Instead of **token ids**, the model requires **tokens** to be represented as **lists of character ids**. Therefore, the input shape changes from `(batch_size, sequence_length)` to `(batch_size, sequence_length, token_length)`.

This model was contributed by [helboukkouri](https://huggingface.co/helboukkouri).
The original code can be found [here](https://github.com/helboukkouri/character-bert).


## CharacterBertConfig

[[autodoc]] CharacterBertConfig
    - all

## CharacterBertTokenizer

[[autodoc]] CharacterBertTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## CharacterBert specific outputs

[[autodoc]] models.character_bert.modeling_character_bert.CharacterBertForPreTrainingOutput

## CharacterBertModel

[[autodoc]] CharacterBertModel
    - forward

## CharacterBertForPreTraining

[[autodoc]] CharacterBertForPreTraining
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
