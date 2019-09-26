Models
----------------------------------------------------

The base class ``PreTrainedModel`` implements the common methods for loading/saving a model either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace's AWS S3 repository).

``PreTrainedModel`` also implements a few methods which are common among all the models to:

- resize the input token embeddings when new tokens are added to the vocabulary
- prune the attention heads of the model.

``PreTrainedModel``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.PreTrainedModel
    :members:

``TFPreTrainedModel``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFPreTrainedModel
    :members:
