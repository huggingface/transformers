.. 
    Copyright 2021 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

CANINE
-----------------------------------------------------------------------------------------------------------------------

Overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CANINE model was proposed in `CANINE: Pre-training an Efficient Tokenization-Free Encoder for Language Representation <https://arxiv.org/abs/2103.06874>`__ 
by Jonathan H. Clark, Dan Garrette, Iulia Turc, John Wieting. It's the first paper that trains a Transformer without using an explicit
tokenizer (such as Byte Pair Encoding (BPE), WordPiece or SentencePiece). Instead, the model is trained directly at a Unicode character-level. 
Training at a character-level inevitably comes with a longer sequence length, which CANINE solves with an efficient downsampling strategy, 
before applying a deep Transformer encoder.

The abstract from the paper is the following:

*Pipelined NLP systems have largely been superseded by end-to-end neural modeling, yet nearly all commonly-used models still require an 
explicit tokenization step. While recent tokenization approaches based on data-derived subword lexicons are less brittle than manually 
engineered tokenizers, these techniques are not equally suited to all languages, and the use of any fixed vocabulary may limit a model's
ability to adapt. In this paper, we present CANINE, a neural encoder that operates directly on character sequences, without explicit 
tokenization or vocabulary, and a pre-training strategy that operates either directly on characters or optionally uses subwords as a 
soft inductive bias. To use its finer-grained input effectively and efficiently, CANINE combines downsampling, which reduces the input 
sequence length, with a deep transformer stack, which encodes context. CANINE outperforms a comparable mBERT model by 2.8 F1 on TyDi QA,
a challenging multilingual benchmark, despite having 28% fewer model parameters.*

Tips:

- CANINE is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left. One can use :class:`~transformers.CanineTokenizer` to prepare text for the model. 
- CANINE uses no less than 3 Transformer encoders internally: 2 "shallow" encoders (which only consist of a single layer) and 1 "deep"
  encoder (which is a regular BERT encoder). First, a "shallow" encoder is used to contextualize the character embeddings, using local 
  attention. Next, a "deep" encoder is applied. Finally, a "shallow" encoder is used to create the final character embeddings. 
- CANINE uses a max sequence length of 2048 characters by default. The model internally downsamples the sequence length to 512
  using strided convolutions before applying the regular BERT encoder. 
- Classification can be done by placing a linear layer on top of the final hidden state of the special [CLS] token (which as a 
  predefined Unicode code point). For token classification tasks however, the downsampled sequence of tokens needs to be upsampled
  again to match the length of the original character sequence (which is 2048). The details for this can be found in the paper. 

This model was contributed by `nielsr <https://huggingface.co/nielsr`__. The original code can be found `here <https://github.com/google-research/language/tree/master/language/canine>`__.

CanineConfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CanineConfig
    :members:


CanineTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CanineTokenizer
    :members: build_inputs_with_special_tokens, get_special_tokens_mask,
        create_token_type_ids_from_sequences, save_vocabulary


CanineModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CanineModel
    :members: forward


CanineForSequenceClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CanineForSequenceClassification
    :members: forward


CanineForMultipleChoice
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CanineForMultipleChoice
    :members: forward


CanineForTokenClassification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CanineForTokenClassification
    :members: forward


CanineForQuestionAnswering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.CanineForQuestionAnswering
    :members: forward
