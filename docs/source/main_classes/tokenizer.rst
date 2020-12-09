.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Tokenizer
-----------------------------------------------------------------------------------------------------------------------

A tokenizer is in charge of preparing the inputs for a model. The library contains tokenizers for all the models. Most
of the tokenizers are available in two flavors: a full python implementation and a "Fast" implementation based on the
Rust library `tokenizers <https://github.com/huggingface/tokenizers>`__. The "Fast" implementations allows:

1. a significant speed-up in particular when doing batched tokenization and
2. additional methods to map between the original string (character and words) and the token space (e.g. getting the
   index of the token comprising a given character or the span of characters corresponding to a given token). Currently
   no "Fast" implementation is available for the SentencePiece-based tokenizers (for T5, ALBERT, CamemBERT, XLMRoBERTa
   and XLNet models).

The base classes :class:`~transformers.PreTrainedTokenizer` and :class:`~transformers.PreTrainedTokenizerFast`
implement the common methods for encoding string inputs in model inputs (see below) and instantiating/saving python and
"Fast" tokenizers either from a local file or directory or from a pretrained tokenizer provided by the library
(downloaded from HuggingFace's AWS S3 repository). They both rely on
:class:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase` that contains the common methods, and
:class:`~transformers.tokenization_utils_base.SpecialTokensMixin`.

:class:`~transformers.PreTrainedTokenizer` and :class:`~transformers.PreTrainedTokenizerFast` thus implement the main
methods for using all the tokenizers:

- Tokenizing (splitting strings in sub-word token strings), converting tokens strings to ids and back, and
  encoding/decoding (i.e., tokenizing and converting to integers).
- Adding new tokens to the vocabulary in a way that is independent of the underlying structure (BPE, SentencePiece...).
- Managing special tokens (like mask, beginning-of-sentence, etc.): adding them, assigning them to attributes in the
  tokenizer for easy access and making sure they are not split during tokenization.

:class:`~transformers.BatchEncoding` holds the output of the tokenizer's encoding methods (``__call__``,
``encode_plus`` and ``batch_encode_plus``) and is derived from a Python dictionary. When the tokenizer is a pure python
tokenizer, this class behaves just like a standard python dictionary and holds the various model inputs computed by
these methods (``input_ids``, ``attention_mask``...). When the tokenizer is a "Fast" tokenizer (i.e., backed by
HuggingFace `tokenizers library <https://github.com/huggingface/tokenizers>`__), this class provides in addition
several advanced alignment methods which can be used to map between the original string (character and words) and the
token space (e.g., getting the index of the token comprising a given character or the span of characters corresponding
to a given token).


PreTrainedTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PreTrainedTokenizer
    :special-members: __call__
    :members:


PreTrainedTokenizerFast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PreTrainedTokenizerFast
    :special-members: __call__
    :members:


BatchEncoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BatchEncoding
    :members:
