.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

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

.. autoclass:: transformers.pipelines.PipelineException
