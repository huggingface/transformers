Utilities for Trainer
-----------------------------------------------------------------------------------------------------------------------

This page lists all the utility functions used by :class:`~transformers.Trainer`.

Most of those are only useful if you are studying the code of the Trainer in the library.

Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.EvalPrediction

.. autofunction:: transformers.set_seed

.. autofunction:: transformers.torch_distributed_zero_first


Callbacks internals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.trainer_callback.CallbackHandler

Distributed Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.trainer_pt_utils.DistributedTensorGatherer
    :members:
