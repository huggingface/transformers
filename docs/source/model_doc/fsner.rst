.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

FSNER
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FSNER model was proposed in `Example-Based Named Entity Recognition <https://arxiv.org/abs/2008.10570>`__ by
Morteza Ziyadi, Yuting Sun, Abhishek Goswami, Jade Huang, Weizhu Chen. To identify entity spans in a new domain, it
uses a train-free few-shot learning approach inspired by question-answering.

The abstract from the paper is the following:

*We present a novel approach to named entity recognition (NER) in the presence of scarce data that we call
example-based NER. Our train-free few-shot learning approach takes inspiration from question-answering to identify
entity spans in a new and unseen domain. In comparison with the current state-of-the-art, the proposed method performs
significantly better, especially when using a low number of support examples.*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by `sayef <https://huggingface.co/sayef>`__.

Implementation Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The base model `fsner-bert-base-uncased` is trained for 10 epochs on the datasets ontonotes5, conll2003, wnut2017,
  and fin (Alvarado et al.).
- The span selection or sorting is based on the product of span-start probability and span-end probability after
  applying softmax, where in the original paper it was summation without softmax.
- Use `get_start_end_token_scores` method of :class:`~transformers.FSNERModel` for extracting start and end
  probabilities for each corresponding entity supports.
- Use `extract_entity_from_scores` method of :class:`~transformers.FSNERTokenizer` or
  :class:`~transformers.FSNERTokenizerFast` for extracting entity span from the query.

FSNERConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FSNERConfig
    :members:


FSNERTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FSNERTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary, extract_entity_from_scores


FSNERTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FSNERTokenizerFast
    :members: extract_entity_from_scores


FSNERModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FSNERModel
    :members: forward, get_start_end_token_scores

