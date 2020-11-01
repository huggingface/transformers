Utilities for Generation
-----------------------------------------------------------------------------------------------------------------------

This page lists all the utility functions used by :meth:`~transformers.PretrainedModel.generate`,
:meth:`~transformers.PretrainedModel.greedy_search`, :meth:`~transformers.PretrainedModel.sample`,
:meth:`~transformers.PretrainedModel.beam_search`, and :meth:`~transformers.PretrainedModel.beam_sample`.

Most of those are only useful if you are studying the code of the generate methods in the library.

LogitsProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~transformers.generation_logits_process.LogitsProcessor` can be used to modify the prediction scores of a
language model head for generation.

.. autoclass:: transformers.generation_logits_process.LogitsProcessor
    :members: __call__

.. autoclass:: transformers.generation_logits_process.LogitsProcessorList
    :members: __call__

.. autoclass:: transformers.generation_logits_process.MinLengthLogitsProcessor
    :members: __call__

.. autoclass:: transformers.generation_logits_process.TemperatureLogitsWarper
    :members: __call__

.. autoclass:: transformers.generation_logits_process.RepetitionPenaltyLogitsProcessor
    :members: __call__

.. autoclass:: transformers.generation_logits_process.TopPLogitsWarper
    :members: __call__

.. autoclass:: transformers.generation_logits_process.TopKLogitsWarper
    :members: __call__

.. autoclass:: transformers.generation_logits_process.NoRepeatNGramLogitsProcessor
    :members: __call__

.. autoclass:: transformers.generation_logits_process.NoBadWordsLogitsProcessor
    :members: __call__

BeamSearch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
