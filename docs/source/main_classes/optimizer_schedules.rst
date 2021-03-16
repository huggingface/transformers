.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Optimization
-----------------------------------------------------------------------------------------------------------------------

The ``.optimization`` module provides:

- an optimizer with weight decay fixed that can be used to fine-tuned models, and
- several schedules in the form of schedule objects that inherit from ``_LRSchedule``:
- a gradient accumulation class to accumulate the gradients of multiple batches

AdamW (PyTorch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AdamW
    :members:

AdaFactor (PyTorch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.Adafactor

AdamWeightDecay (TensorFlow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.AdamWeightDecay

.. autofunction:: transformers.create_optimizer

Schedules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learning Rate Schedules (Pytorch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: transformers.SchedulerType

.. autofunction:: transformers.get_scheduler

.. autofunction:: transformers.get_constant_schedule


.. autofunction:: transformers.get_constant_schedule_with_warmup

.. image:: /imgs/warmup_constant_schedule.png
    :target: /imgs/warmup_constant_schedule.png
    :alt:


.. autofunction:: transformers.get_cosine_schedule_with_warmup

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


.. autofunction:: transformers.get_polynomial_decay_schedule_with_warmup


Warmup (TensorFlow)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: transformers.WarmUp
    :members:

Gradient Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GradientAccumulator (TensorFlow)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: transformers.GradientAccumulator
