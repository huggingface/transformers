.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

RoBERTaJapanese
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RoBERTa model trained on Japanese text.

There are models with two different tokenization methods:

- Tokenize with MeCab and BPE. This requires some extra dependencies, `fugashi
  <https://github.com/polm/fugashi>`__ which is a wrapper around `MeCab <https://taku910.github.io/mecab/>`__.

You should ``pip install transformers["ja"]`` (or ``pip install -e .["ja"]`` if you install
from source) to install dependencies.

Example of using a model with MeCab and BPE tokenization:

.. code-block::

    >>> import torch
    >>> from transformers import AutoModel, AutoTokenizer 

    >>> robertajapanese = AutoModel.from_pretrained("cl-tohoku/roberta-base-japanese")
    >>> tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/roberta-base-japanese", use_fast=False)

    >>> ## Input Japanese Text
    >>> line = "吾輩は猫である。"

    >>> inputs = tokenizer(line, return_tensors="pt")

    >>> print(tokenizer.decode(inputs['input_ids'][0]))
    <s> 吾輩 は 猫 で ある 。</s>

    >>> outputs = robertajapanese(**inputs)

Tips:

- This implementation is the same as XLMRoBERTa, except for tokenization method. Refer to the :doc:`documentation of XLMRoBERTa
  <xlmroberta>` for more usage examples.


RobertaJapaneseTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.RobertaJapaneseTokenizer
    :members: 
