Trainer
----------

The :class:`~transformers.Trainer` and :class:`~transformers.TFTrainer` classes provide an API for feature-complete
training in most standard use cases. It's used in most of the :doc:`example scripts <../examples>`.

Before instantiating your :class:`~transformers.Trainer`/:class:`~transformers.TFTrainer`, create a 
:class:`~transformers.TrainingArguments`/:class:`~transformers.TFTrainingArguments` to access all the points of
customization during training.

The API supports distributed training on multiple GPUs/TPUs, mixed precision through `NVIDIA Apex
<https://github.com/NVIDIA/apex>`__ for PyTorch and :obj:`tf.keras.mixed_precision` for TensorFlow.

``Trainer`` 
~~~~~~~~~~~

.. autoclass:: transformers.Trainer
    :members:

``TFTrainer`` 
~~~~~~~~~~~~~

.. autoclass:: transformers.TFTrainer
    :members:

``TrainingArguments``
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TrainingArguments
    :members:

``TFTrainingArguments``
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.TFTrainingArguments
    :members:

Utilities
~~~~~~~~~

.. autoclass:: transformers.EvalPrediction

.. autofunction:: transformers.set_seed

.. autofunction:: transformers.torch_distributed_zero_first
