.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

ByT5
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ByT5 model was presented in `ByT5: Towards a token-free future with pre-trained byte-to-byte models
<https://arxiv.org/abs/2105.13626>`_ by Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir
Kale, Adam Roberts, Colin Raffel.

The abstract from the paper is the following:

*Most widely-used pre-trained language models operate on sequences of tokens corresponding to word or subword units.
Encoding text as a sequence of tokens requires a tokenizer, which is typically created as an independent artifact from
the model. Token-free models that instead operate directly on raw text (bytes or characters) have many benefits: they
can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by
removing complex and error-prone text preprocessing pipelines. Since byte or character sequences are longer than token
sequences, past work on token-free models has often introduced new model architectures designed to amortize the cost of
operating directly on raw text. In this paper, we show that a standard Transformer architecture can be used with
minimal modifications to process byte sequences. We carefully characterize the trade-offs in terms of parameter count,
training FLOPs, and inference speed, and show that byte-level models are competitive with their token-level
counterparts. We also demonstrate that byte-level models are significantly more robust to noise and perform better on
tasks that are sensitive to spelling and pronunciation. As part of our contribution, we release a new set of
pre-trained byte-level Transformer models based on the T5 architecture, as well as all code and data used in our
experiments.*

This model was contributed by `patrickvonplaten <https://huggingface.co/patrickvonplaten>`__. The original code can be
found `here <https://github.com/google-research/byt5>`__.


ByT5's architecture is based on the T5 model, so one can refer to :doc:`T5's documentation page <t5>`.


Example
_______________________________________________________________________________________________________________________

ByT5 works on raw UTF-8 bytes, so it can be used without a tokenizer:

.. code-block::

    from transformers import T5ForConditionalGeneration
    import torch

    model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')

    input_ids = torch.tensor([list("Life is like a box of chocolates.".encode("utf-8"))]) + 3  # add 3 for special tokens
    labels = torch.tensor([list("La vie est comme une boîte de chocolat.".encode("utf-8"))]) + 3  # add 3 for special tokens

    loss = model(input_ids, labels=labels).loss # forward pass


For batched inference and training it is however recommended to make use of the tokenizer:

.. code-block::

    from transformers import T5ForConditionalGeneration, AutoTokenizer

    model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    model_inputs = tokenizer(["Life is like a box of chocolates.", "Today is Monday."], padding="longest", return_tensors="pt")
    labels = tokenizer(["La vie est comme une boîte de chocolat.", "Aujourd'hui c'est lundi."], padding="longest", return_tensors="pt").input_ids

    loss = model(**model_inputs, labels=labels).loss # forward pass

ByT5Tokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.ByT5Tokenizer

See :class:`~transformers.ByT5Tokenizer` for all details.
