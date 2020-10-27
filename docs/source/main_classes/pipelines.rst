Pipelines
-----------------------------------------------------------------------------------------------------------------------

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of
the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the
:doc:`task summary <../task_summary>` for examples of use.

There are two categories of pipeline abstractions to be aware about:

- The :func:`~transformers.pipeline` which is the most powerful object encapsulating all other pipelines.
- The other task-specific pipelines:

    - :class:`~transformers.ConversationalPipeline`
    - :class:`~transformers.FeatureExtractionPipeline`
    - :class:`~transformers.FillMaskPipeline`
    - :class:`~transformers.QuestionAnsweringPipeline`
    - :class:`~transformers.SummarizationPipeline`
    - :class:`~transformers.TextClassificationPipeline`
    - :class:`~transformers.TextGenerationPipeline`
    - :class:`~transformers.TokenClassificationPipeline`
    - :class:`~transformers.TranslationPipeline`
    - :class:`~transformers.ZeroShotClassificationPipeline`
    - :class:`~transformers.Text2TextGenerationPipeline`

The pipeline abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `pipeline` abstraction is a wrapper around all the other available pipelines. It is instantiated as any other
pipeline but requires an additional argument which is the `task`.

.. autofunction:: transformers.pipeline


The task specific pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ConversationalPipeline
=======================================================================================================================

.. autoclass:: transformers.Conversation

.. autoclass:: transformers.ConversationalPipeline
    :special-members: __call__
    :members:

FeatureExtractionPipeline
=======================================================================================================================

.. autoclass:: transformers.FeatureExtractionPipeline
    :special-members: __call__
    :members:

FillMaskPipeline
=======================================================================================================================

.. autoclass:: transformers.FillMaskPipeline
    :special-members: __call__
    :members:

NerPipeline
=======================================================================================================================

This class is an alias of the :class:`~transformers.TokenClassificationPipeline` defined below. Please refer to that
pipeline for documentation and usage examples.

QuestionAnsweringPipeline
=======================================================================================================================

.. autoclass:: transformers.QuestionAnsweringPipeline
    :special-members: __call__
    :members:

SummarizationPipeline
=======================================================================================================================

.. autoclass:: transformers.SummarizationPipeline
    :special-members: __call__
    :members:

TextClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.TextClassificationPipeline
    :special-members: __call__
    :members:

TextGenerationPipeline
=======================================================================================================================

.. autoclass:: transformers.TextGenerationPipeline
    :special-members: __call__
    :members:

Text2TextGenerationPipeline
=======================================================================================================================

.. autoclass:: transformers.Text2TextGenerationPipeline
    :special-members: __call__
    :members:

TokenClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.TokenClassificationPipeline
    :special-members: __call__
    :members:

ZeroShotClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.ZeroShotClassificationPipeline
    :special-members: __call__
    :members:

Parent class: :obj:`Pipeline`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Pipeline
    :members:
