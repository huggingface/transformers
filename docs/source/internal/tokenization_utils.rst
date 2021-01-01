.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Utilities for Tokenizers
-----------------------------------------------------------------------------------------------------------------------

This page lists all the utility functions used by the tokenizers, mainly the class
:class:`~transformers.tokenization_utils_base.PreTrainedTokenizerBase` that implements the common methods between
:class:`~transformers.PreTrainedTokenizer` and :class:`~transformers.PreTrainedTokenizerFast` and the mixin
:class:`~transformers.tokenization_utils_base.SpecialTokensMixin`.

Most of those are only useful if you are studying the code of the tokenizers in the library.

PreTrainedTokenizerBase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.tokenization_utils_base.PreTrainedTokenizerBase
    :special-members: __call__
    :members:


SpecialTokensMixin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.tokenization_utils_base.SpecialTokensMixin
    :members:


Enums and namedtuples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.tokenization_utils_base.ExplicitEnum

.. autoclass:: transformers.tokenization_utils_base.PaddingStrategy

.. autoclass:: transformers.tokenization_utils_base.TensorType

.. autoclass:: transformers.tokenization_utils_base.TruncationStrategy

.. autoclass:: transformers.tokenization_utils_base.CharSpan

.. autoclass:: transformers.tokenization_utils_base.TokenSpan
