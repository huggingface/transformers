Utilities for Generation
-----------------------------------------------------------------------------------------------------------------------

This page lists all the utility functions used by :meth:`~transformers.PretrainedModel.generate`,
:meth:`~transformers.PretrainedModel.greedy_search`, :meth:`~transformers.PretrainedModel.sample`,
:meth:`~transformers.PretrainedModel.beam_search`, and :meth:`~transformers.PretrainedModel.beam_sample`.

Most of those are only useful if you are studying the code of the generate methods in the library.

LogitsProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~transformers.LogitsProcessor` can be used to modify the prediction scores of a language model head for
generation.

.. autoclass:: transformers.LogitsProcessor
    :members: __call__

.. autoclass:: transformers.LogitsProcessorList
    :members: __call__

.. autoclass:: transformers.MinLengthLogitsProcessor
    :members: __call__

.. autoclass:: transformers.TemperatureLogitsWarper
    :members: __call__

.. autoclass:: transformers.RepetitionPenaltyLogitsProcessor
    :members: __call__

.. autoclass:: transformers.TopPLogitsWarper
    :members: __call__

.. autoclass:: transformers.TopKLogitsWarper
    :members: __call__

.. autoclass:: transformers.NoRepeatNGramLogitsProcessor
    :members: __call__

.. autoclass:: transformers.NoBadWordsLogitsProcessor
    :members: __call__

BeamSearch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BeamScorer
    :members: process, finalize

.. autoclass:: transformers.BeamSearchScorer
    :members: process, finalize
