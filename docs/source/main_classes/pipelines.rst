.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Pipelines
-----------------------------------------------------------------------------------------------------------------------

The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of
the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity
Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering. See the
:doc:`task summary <../task_summary>` for examples of use.

There are two categories of pipeline abstractions to be aware about:

- The :func:`~transformers.pipeline` which is the most powerful object encapsulating all other pipelines.
- The other task-specific pipelines:

    - :class:`~transformers.AutomaticSpeechRecognitionPipeline`
    - :class:`~transformers.ConversationalPipeline`
    - :class:`~transformers.FeatureExtractionPipeline`
    - :class:`~transformers.FillMaskPipeline`
    - :class:`~transformers.ImageClassificationPipeline`
    - :class:`~transformers.QuestionAnsweringPipeline`
    - :class:`~transformers.SummarizationPipeline`
    - :class:`~transformers.TextClassificationPipeline`
    - :class:`~transformers.TextGenerationPipeline`
    - :class:`~transformers.TokenClassificationPipeline`
    - :class:`~transformers.TranslationPipeline`
    - :class:`~transformers.ZeroShotClassificationPipeline`
    - :class:`~transformers.Text2TextGenerationPipeline`
    - :class:`~transformers.TableQuestionAnsweringPipeline`

The pipeline abstraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `pipeline` abstraction is a wrapper around all the other available pipelines. It is instantiated as any other
pipeline but requires an additional argument which is the `task`.

.. autofunction:: transformers.pipeline


The task specific pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AutomaticSpeechRecognitionPipeline
=======================================================================================================================

.. autoclass:: transformers.AutomaticSpeechRecognitionPipeline
    :special-members: __call__
    :members:

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

ImageClassificationPipeline
=======================================================================================================================

.. autoclass:: transformers.ImageClassificationPipeline
    :special-members: __call__
    :members:

NerPipeline
=======================================================================================================================

.. autoclass:: transformers.NerPipeline

See :class:`~transformers.TokenClassificationPipeline` for all details.

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

TableQuestionAnsweringPipeline
=======================================================================================================================

.. autoclass:: transformers.TableQuestionAnsweringPipeline
    :special-members: __call__


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

TranslationPipeline
=======================================================================================================================

.. autoclass:: transformers.TranslationPipeline
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
