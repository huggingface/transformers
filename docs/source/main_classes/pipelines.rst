Pipelines
----------------------------------------------------

The pipelines are a great an easy way to use models for inference. These pipelines are objects that abstract most
of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering.

There are two categories of pipeline abstractions to be aware about:

- The :class:`~transformers.pipeline` which is the most powerful object encapsulating all other pipelines
- The other task-specific pipelines, such as :class:`~transformers.NerPipeline`
  or :class:`~transformers.QuestionAnsweringPipeline`

The pipeline abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.pipeline
    :members:


The task specific pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NerPipeline
==========================================

.. autoclass:: transformers.NerPipeline
    :members:

TokenClassificationPipeline
==========================================

This class is an alias of the :class:`~transformers.NerPipeline` defined above. Please refer to that pipeline for
documentation and usage examples.

FillMaskPipeline
==========================================

.. autoclass:: transformers.FillMaskPipeline
    :members:

FeatureExtractionPipeline
==========================================

.. autoclass:: transformers.FeatureExtractionPipeline
    :members:

TextClassificationPipeline
==========================================

.. autoclass:: transformers.TextClassificationPipeline
    :members:

QuestionAnsweringPipeline
==========================================

.. autoclass:: transformers.QuestionAnsweringPipeline
    :members:

