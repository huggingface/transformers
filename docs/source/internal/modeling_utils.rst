Custom Layers and Utilities
---------------------------

This page lists all the custom layers used by the library, as well as the utility functions it provides for modeling.
Most of those are only useful if you are studying the code of the models in the library.


``PyTorch Helper Functions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.apply_chunking_to_forward


``TensorFlow custom layers``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.modeling_tf_utils.TFConv1D

.. autoclass:: transformers.modeling_tf_utils.TFSharedEmbeddings
    :members: call

.. autoclass:: transformers.modeling_tf_utils.TFSequenceSummary


``TensorFlow loss functions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


``TensorFlow Helper Functions``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: transformers.modeling_tf_utils.cast_bool_to_primitive

.. autofunction:: transformers.modeling_tf_utils.get_initializer

.. autofunction:: transformers.modeling_tf_utils.keras_serializable

.. autofunction:: transformers.modeling_tf_utils.shape_list