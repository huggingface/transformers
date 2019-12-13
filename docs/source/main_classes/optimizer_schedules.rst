Optimizer
----------------------------------------------------

The ``.optimization`` module provides:

- an optimizer with weight decay fixed that can be used to fine-tuned models, and
- several schedules in the form of schedule objects that inherit from ``_LRSchedule``:
- a gradient accumulation class to accumulate the gradients of multiple batches

``AdamW``
~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AdamW
    :members:

``AdamWeightDecay``
~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AdamWeightDecay
    :members:

.. autofunction:: transformers.create_optimizer
    :members:

Schedules
----------------------------------------------------

Learning Rate Schedules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: transformers.get_constant_schedule


.. autofunction:: transformers.get_constant_schedule_with_warmup

.. image:: /imgs/warmup_constant_schedule.png
    :target: /imgs/warmup_constant_schedule.png
    :alt:


.. autofunction:: transformers.get_cosine_schedule_with_warmup
    :members:

.. image:: /imgs/warmup_cosine_schedule.png
    :target: /imgs/warmup_cosine_schedule.png
    :alt:


.. autofunction:: transformers.get_cosine_with_hard_restarts_schedule_with_warmup

.. image:: /imgs/warmup_cosine_hard_restarts_schedule.png
    :target: /imgs/warmup_cosine_hard_restarts_schedule.png
    :alt:



.. autofunction:: transformers.get_linear_schedule_with_warmup

.. image:: /imgs/warmup_linear_schedule.png
    :target: /imgs/warmup_linear_schedule.png
    :alt:

``Warmup``
~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Warmup
    :members:

Gradient Strategies
----------------------------------------------------

``GradientAccumulator``
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.GradientAccumulator
