..
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Models
-----------------------------------------------------------------------------------------------------------------------

The base classes :class:`~transformers.PreTrainedModel`, :class:`~transformers.TFPreTrainedModel`, and
:class:`~transformers.FlaxPreTrainedModel` implement the common methods for loading/saving a model either from a local
file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace's AWS
S3 repository).

:class:`~transformers.PreTrainedModel` and :class:`~transformers.TFPreTrainedModel` also implement a few methods which
are common among all the models to:

- resize the input token embeddings when new tokens are added to the vocabulary
- prune the attention heads of the model.

The other methods that are common to each model are defined in :class:`~transformers.modeling_utils.ModuleUtilsMixin`
(for the PyTorch models) and :class:`~transformers.modeling_tf_utils.TFModuleUtilsMixin` (for the TensorFlow models) or
for text generation, :class:`~transformers.generation_utils.GenerationMixin` (for the PyTorch models),
:class:`~transformers.generation_tf_utils.TFGenerationMixin` (for the TensorFlow models) and
:class:`~transformers.generation_flax_utils.FlaxGenerationMixin` (for the Flax/JAX models).


PreTrainedModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PreTrainedModel
    :special-members: push_to_hub
    :members:


.. _from_pretrained-torch-dtype:

Model Instantiation dtype
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Under Pytorch a model normally gets instantiated with ``torch.float32`` format. This can be an issue if one tries to
load a model whose weights are in fp16, since it'd require twice as much memory. To overcome this limitation, you can
either explicitly pass the desired ``dtype`` using ``torch_dtype`` argument:

.. code-block:: python

    model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype=torch.float16)

or, if you want the model to always load in the most optimal memory pattern, you can use the special value ``"auto"``,
and then ``dtype`` will be automatically derived from the model's weights:

.. code-block:: python

    model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype="auto")

Models instantiated from scratch can also be told which ``dtype`` to use with:

.. code-block:: python

    config = T5Config.from_pretrained("t5")
    model = AutoModel.from_config(config)

Due to Pytorch design, this functionality is only available for floating dtypes.



ModuleUtilsMixin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_utils.ModuleUtilsMixin
    :members:


TFPreTrainedModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPreTrainedModel
    :special-members: push_to_hub
    :members:


TFModelUtilsMixin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_tf_utils.TFModelUtilsMixin
    :members:


FlaxPreTrainedModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.FlaxPreTrainedModel
    :special-members: push_to_hub
    :members:


Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.generation_utils.GenerationMixin
    :members:

.. autoclass:: transformers.generation_tf_utils.TFGenerationMixin
    :members:

.. autoclass:: transformers.generation_flax_utils.FlaxGenerationMixin
    :members:


Pushing to the Hub
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.file_utils.PushToHubMixin
    :members:
