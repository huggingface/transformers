Pipelines
----------------------------------------------------

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most
of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering.

There are two categories of pipeline abstractions to be aware about:

- The :func:`~transformers.pipeline` which is the most powerful object encapsulating all other pipelines
- The other task-specific pipelines, such as :class:`~transformers.TokenClassificationPipeline`
  or :class:`~transformers.QuestionAnsweringPipeline`

The pipeline abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `pipeline` abstraction is a wrapper around all the other available pipelines. It is instantiated as any
other pipeline but requires an additional argument which is the `task`.

.. autofunction:: transformers.pipeline


The task specific pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parent class: Pipeline
=========================================

.. autoclass:: transformers.Pipeline
    :members: predict, transform, save_pretrained

TokenClassificationPipeline
==========================================

.. autoclass:: transformers.TokenClassificationPipeline

NerPipeline
==========================================

This class is an alias of the :class:`~transformers.TokenClassificationPipeline` defined above. Please refer to that pipeline for
documentation and usage examples.

FillMaskPipeline
==========================================

.. autoclass:: transformers.FillMaskPipeline

FeatureExtractionPipeline
==========================================

.. autoclass:: transformers.FeatureExtractionPipeline

TextClassificationPipeline
==========================================

.. autoclass:: transformers.TextClassificationPipeline

QuestionAnsweringPipeline
==========================================

.. autoclass:: transformers.QuestionAnsweringPipeline


SummarizationPipeline
==========================================

.. autoclass:: transformers.SummarizationPipeline


TextGenerationPipeline
==========================================

.. autoclass:: transformers.TextGenerationPipeline
