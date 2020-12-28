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

Generate Outputs
-----------------------------------------------------------------------------------------------------------------------

The output of :meth:`~transformers.PretrainedModel.generate` is an instance of a subclass of
:class:`~transformers.file_utils.ModelOutput`. This output is a data structures containing all the information returned
by :meth:`~transformers.PretrainedModel.generate`, but that can also be used as tuples or dictionaries.

Let's see of this looks on an example:

.. code-block::

    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
    sequences = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)

The ``sequences`` object is a :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`, as we can see in the
documentation of that class below, it means it has a ``sequences`` an optional ``scores``, an optional
``hidden_states`` and an optional ``attentions`` attribute. Here we have the ``scores`` since we passed along
``output_scores=True``, but we don't have ``hidden_states`` and ``attentions`` because we didn't pass
``output_hidden_states=True`` or ``output_attentions=True``.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you
will get ``None``. Here for instance ``sequences.scores`` are all the generated prediction scores of the language
modeling head, and ``sequences.attentions`` is ``None``.

When considering our ``sequences`` object as tuple, it only considers the attributes that don't have ``None`` values.
Here for instance, it has two elements, ``loss`` then ``logits``, so

.. code-block::

    sequences[:2]

will return the tuple ``(sequences.sequences, sequences.scores)`` for instance.

When considering our ``sequences`` object as dictionary, it only considers the attributes that don't have ``None``
values. Here for instance, it has two keys that are ``sequences`` and ``scores``.

We document here all generation sequences.


GreedySearchOutput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.generation_utils.GreedySearchDecoderOnlyOutput
    :members:

.. autoclass:: transformers.generation_utils.GreedySearchEncoderDecoderOutput
    :members:


SampleOutput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.generation_utils.SampleDecoderOnlyOutput
    :members:

.. autoclass:: transformers.generation_utils.SampleEncoderDecoderOutput
    :members:


BeamSearchOutput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.generation_utils.BeamSearchDecoderOnlyOutput
    :members:

.. autoclass:: transformers.generation_utils.BeamSearchEncoderDecoderOutput
    :members:


BeamSampleOutput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.generation_utils.BeamSampleDecoderOnlyOutput
    :members:

.. autoclass:: transformers.generation_utils.BeamSampleEncoderDecoderOutput
    :members:


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
