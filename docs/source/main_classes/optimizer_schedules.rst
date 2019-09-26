Optimizer
----------------------------------------------------

The ``.optimization`` module provides:

- an optimizer with weight decay fixed that can be used to fine-tuned models, and
- several schedules in the form of schedule objects that inherit from ``_LRSchedule``:

``AdamW``
~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AdamW
    :members:

Schedules
----------------------------------------------------

Learning Rate Schedules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: transformers.ConstantLRSchedule
    :members:


.. autoclass:: transformers.WarmupConstantSchedule
    :members:

.. image:: /imgs/warmup_constant_schedule.png
    :target: /imgs/warmup_constant_schedule.png
    :alt:


.. autoclass:: transformers.WarmupCosineSchedule
    :members:

.. image:: /imgs/warmup_cosine_schedule.png
    :target: /imgs/warmup_cosine_schedule.png
    :alt:


.. autoclass:: transformers.WarmupCosineWithHardRestartsSchedule
    :members:

.. image:: /imgs/warmup_cosine_hard_restarts_schedule.png
    :target: /imgs/warmup_cosine_hard_restarts_schedule.png
    :alt:



.. autoclass:: transformers.WarmupLinearSchedule
    :members:

.. image:: /imgs/warmup_linear_schedule.png
    :target: /imgs/warmup_linear_schedule.png
    :alt:
