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
