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

This page lists all the utility functions used by :meth:`~transformers.generation_utils.GenerationMixin.generate`,
:meth:`~transformers.generation_utils.GenerationMixin.greedy_search`,
:meth:`~transformers.generation_utils.GenerationMixin.sample`,
:meth:`~transformers.generation_utils.GenerationMixin.beam_search`,
:meth:`~transformers.generation_utils.GenerationMixin.beam_sample`, and
:meth:`~transformers.generation_utils.GenerationMixin.group_beam_search`.

Most of those are only useful if you are studying the code of the generate methods in the library.

Generate Outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output of :meth:`~transformers.generation_utils.GenerationMixin.generate` is an instance of a subclass of
:class:`~transformers.file_utils.ModelOutput`. This output is a data structure containing all the information returned
by :meth:`~transformers.generation_utils.GenerationMixin.generate`, but that can also be used as tuple or dictionary.

Here's an example:

.. code-block::

    from transformers import GPT2Tokenizer, GPT2LMHeadModel

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
    generation_output = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)

The ``generation_output`` object is a :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`, as we can
see in the documentation of that class below, it means it has the following attributes:

- ``sequences``: the generated sequences of tokens
- ``scores`` (optional): the prediction scores of the language modelling head, for each generation step
- ``hidden_states`` (optional): the hidden states of the model, for each generation step
- ``attentions`` (optional): the attention weights of the model, for each generation step

Here we have the ``scores`` since we passed along ``output_scores=True``, but we don't have ``hidden_states`` and
``attentions`` because we didn't pass ``output_hidden_states=True`` or ``output_attentions=True``.

You can access each attribute as you would usually do, and if that attribute has not been returned by the model, you
will get ``None``. Here for instance ``generation_output.scores`` are all the generated prediction scores of the
language modeling head, and ``generation_output.attentions`` is ``None``.

When using our ``generation_output`` object as a tuple, it only keeps the attributes that don't have ``None`` values.
Here, for instance, it has two elements, ``loss`` then ``logits``, so

.. code-block::

    generation_output[:2]

will return the tuple ``(generation_output.sequences, generation_output.scores)`` for instance.

When using our ``generation_output`` object as a dictionary, it only keeps the attributes that don't have ``None``
values. Here, for instance, it has two keys that are ``sequences`` and ``scores``.

We document here all output types.


GreedySearchOutput
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: transformers.generation_utils.GreedySearchDecoderOnlyOutput
    :members:

.. autoclass:: transformers.generation_utils.GreedySearchEncoderDecoderOutput
    :members:

.. autoclass:: transformers.generation_flax_utils.FlaxGreedySearchOutput
    :members:


SampleOutput
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: transformers.generation_utils.SampleDecoderOnlyOutput
    :members:

.. autoclass:: transformers.generation_utils.SampleEncoderDecoderOutput
    :members:

.. autoclass:: transformers.generation_flax_utils.FlaxSampleOutput
    :members:


BeamSearchOutput
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: transformers.generation_utils.BeamSearchDecoderOnlyOutput
    :members:

.. autoclass:: transformers.generation_utils.BeamSearchEncoderDecoderOutput
    :members:


BeamSampleOutput
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. autoclass:: transformers.ForcedBOSTokenLogitsProcessor
    :members: __call__

.. autoclass:: transformers.ForcedEOSTokenLogitsProcessor
    :members: __call__

.. autoclass:: transformers.InfNanRemoveLogitsProcessor
    :members: __call__

.. autoclass:: transformers.FlaxLogitsProcessor
    :members: __call__

.. autoclass:: transformers.FlaxLogitsProcessorList
    :members: __call__

.. autoclass:: transformers.FlaxLogitsWarper
    :members: __call__

.. autoclass:: transformers.FlaxTemperatureLogitsWarper
    :members: __call__

.. autoclass:: transformers.FlaxTopPLogitsWarper
    :members: __call__

.. autoclass:: transformers.FlaxTopKLogitsWarper
    :members: __call__

.. autoclass:: transformers.FlaxForcedBOSTokenLogitsProcessor
    :members: __call__

.. autoclass:: transformers.FlaxForcedEOSTokenLogitsProcessor
    :members: __call__

.. autoclass:: transformers.FlaxMinLengthLogitsProcessor
    :members: __call__


StoppingCriteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~transformers.StoppingCriteria` can be used to change when to stop generation (other than EOS token).

.. autoclass:: transformers.StoppingCriteria
    :members: __call__

.. autoclass:: transformers.StoppingCriteriaList
    :members: __call__

.. autoclass:: transformers.MaxLengthCriteria
    :members: __call__

.. autoclass:: transformers.MaxTimeCriteria
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
