Utilities for pipelines
-----------------------------------------------------------------------------------------------------------------------

This page lists all the utility functions the library provides for pipelines.

Most of those are only useful if you are studying the code of the models in the library.


Argument handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.pipelines.ArgumentHandler

.. autoclass:: transformers.pipelines.ZeroShotClassificationArgumentHandler

.. autoclass:: transformers.pipelines.QuestionAnsweringArgumentHandler


Data format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.pipelines.PipelineDataFormat
    :members:

.. autoclass:: transformers.pipelines.CsvPipelineDataFormat
    :members:

.. autoclass:: transformers.pipelines.JsonPipelineDataFormat
    :members:

.. autoclass:: transformers.pipelines.PipedPipelineDataFormat
    :members:


Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.pipelines.get_framework

.. autoclass:: transformers.pipelines.PipelineException
