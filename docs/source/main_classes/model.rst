Models
-----------------------------------------------------------------------------------------------------------------------

The base classes :class:`~transformers.PreTrainedModel` and :class:`~transformers.TFPreTrainedModel` implement the
common methods for loading/saving a model either from a local file or directory, or from a pretrained model
configuration provided by the library (downloaded from HuggingFace's AWS S3 repository).

:class:`~transformers.PreTrainedModel` and :class:`~transformers.TFPreTrainedModel` also implement a few methods which
are common among all the models to:

- resize the input token embeddings when new tokens are added to the vocabulary
- prune the attention heads of the model.

The other methods that are common to each model are defined in :class:`~transformers.modeling_utils.ModuleUtilsMixin`
(for the PyTorch models) and :class:`~transformers.modeling_tf_utils.TFModuleUtilsMixin` (for the TensorFlow models) or
for text generation, :class:`~transformers.generation_utils.GenerationMixin` (for the PyTorch models) and
:class:`~transformers.generation_tf_utils.TFGenerationMixin` (for the TensorFlow models)


PreTrainedModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PreTrainedModel
    :members:


ModuleUtilsMixin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_utils.ModuleUtilsMixin
    :members:


TFPreTrainedModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPreTrainedModel
    :members:


TFModelUtilsMixin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_tf_utils.TFModelUtilsMixin
    :members:


Generative models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.generation_utils.GenerationMixin
    :members:

.. autoclass:: transformers.generation_tf_utils.TFGenerationMixin
    :members: