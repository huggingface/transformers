.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Utilities for Generation
-----------------------------------------------------------------------------------------------------------------------

This page lists all the utility functions used by :meth:`~transformers.PretrainedModel.generate`,
:meth:`~transformers.PretrainedModel.greedy_search`, :meth:`~transformers.PretrainedModel.sample`,
:meth:`~transformers.PretrainedModel.beam_search`, :meth:`~transformers.PretrainedModel.beam_sample`, and
:meth:`~transformers.PretrainedModel.group_beam_search`.

Most of those are only useful if you are studying the code of the generate methods in the library.

LogitsProcessor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~transformers.LogitsProcessor` can be used to modify the prediction scores of a language model head for
generation.

.. autoclass:: transformers.LogitsProcessor
    :members: __call__

.. autoclass:: transformers.LogitsProcessorList
    :members: __call__

.. autoclass:: transformers.LogitsWarper
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

.. autoclass:: transformers.PrefixConstrainedLogitsProcessor
    :members: __call__

.. autoclass:: transformers.HammingDiversityLogitsProcessor
    :members: __call__

BeamSearch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.BeamScorer
    :members: process, finalize

.. autoclass:: transformers.BeamSearchScorer
    :members: process, finalize

Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.top_k_top_p_filtering

.. autofunction:: transformers.tf_top_k_top_p_filtering
