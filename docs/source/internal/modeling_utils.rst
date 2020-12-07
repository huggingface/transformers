.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Custom Layers and Utilities
-----------------------------------------------------------------------------------------------------------------------

This page lists all the custom layers used by the library, as well as the utility functions it provides for modeling.

Most of those are only useful if you are studying the code of the models in the library.


Pytorch custom modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_utils.Conv1D

.. autoclass:: transformers.modeling_utils.PoolerStartLogits
    :members: forward

.. autoclass:: transformers.modeling_utils.PoolerEndLogits
    :members: forward

.. autoclass:: transformers.modeling_utils.PoolerAnswerClass
    :members: forward

.. autoclass:: transformers.modeling_utils.SquadHeadOutput

.. autoclass:: transformers.modeling_utils.SQuADHead
    :members: forward

.. autoclass:: transformers.modeling_utils.SequenceSummary
    :members: forward


PyTorch Helper Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.apply_chunking_to_forward

.. autofunction:: transformers.modeling_utils.find_pruneable_heads_and_indices

.. autofunction:: transformers.modeling_utils.prune_layer

.. autofunction:: transformers.modeling_utils.prune_conv1d_layer

.. autofunction:: transformers.modeling_utils.prune_linear_layer

TensorFlow custom layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_tf_utils.TFConv1D

.. autoclass:: transformers.modeling_tf_utils.TFSharedEmbeddings
    :members: call

.. autoclass:: transformers.modeling_tf_utils.TFSequenceSummary
    :members: call


TensorFlow loss functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_tf_utils.TFCausalLanguageModelingLoss
    :members:

.. autoclass:: transformers.modeling_tf_utils.TFMaskedLanguageModelingLoss
    :members:

.. autoclass:: transformers.modeling_tf_utils.TFMultipleChoiceLoss
    :members:

.. autoclass:: transformers.modeling_tf_utils.TFQuestionAnsweringLoss
    :members:

.. autoclass:: transformers.modeling_tf_utils.TFSequenceClassificationLoss
    :members:

.. autoclass:: transformers.modeling_tf_utils.TFTokenClassificationLoss
    :members:


TensorFlow Helper Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.modeling_tf_utils.cast_bool_to_primitive

.. autofunction:: transformers.modeling_tf_utils.get_initializer

.. autofunction:: transformers.modeling_tf_utils.keras_serializable

.. autofunction:: transformers.modeling_tf_utils.shape_list
