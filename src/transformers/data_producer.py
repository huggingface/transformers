# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DataProducer protocol for online/async training.

Enables reinforcement-learning methods (PPO, GRPO, REINFORCE, online DPO) and
curriculum learning by letting the model generate its own training data. Instead
of iterating over a fixed dataset, the Trainer calls
``data_producer.produce(model, step)`` to get a fresh ``Dataset`` each rollout.

Quick start::

    from datasets import Dataset
    from transformers import Trainer, TrainingArguments
    from transformers.data_producer import BaseDataProducer, ProducerConfig

    class MyProducer(BaseDataProducer):
        def produce(self, model, global_step, **kwargs):
            completions = model.generate(self.prompts, max_new_tokens=128)
            rewards = self.reward_fn(completions)
            return Dataset.from_dict({"completion": completions, "reward": rewards})

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./out", max_steps=5000),
        data_producer=MyProducer(ProducerConfig(mini_epochs=2, max_rollouts=100)),
    )
    trainer.train()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from .trainer_callback import TrainerCallback


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProducerConfig:
    """Configuration for a :class:`DataProducer`.

    Args:
        mini_epochs: Number of training passes over each produced dataset.
            Higher values amortise expensive generation across more gradient
            updates.
        max_rollouts: Maximum number of produce-then-train rounds.  ``None``
            means training is bounded only by ``TrainingArguments.max_steps``.
        steps_per_generation: Number of optimisation steps to take on each
            produced dataset before calling ``produce()`` again.  Maps to the
            GRPO ``steps_per_generation`` parameter.  ``None`` means the entire
            produced dataset is consumed (one full epoch) before regenerating.
        num_iterations: Number of times to reuse each generation across
            optimisation steps.  Maps to the GRPO *μ* parameter.
        async_prefetch: If ``True``, the next dataset is produced in a
            background thread while the current one is being trained on.
        eval_during_produce: Switch the model to ``eval()`` mode during
            ``produce()``.  Recommended for generation quality.
        empty_cache_before_produce: Call ``torch.cuda.empty_cache()`` before
            each ``produce()`` call.
        empty_cache_after_produce: Call ``torch.cuda.empty_cache()`` after
            each ``produce()`` call.
    """

    mini_epochs: int = 1
    max_rollouts: int | None = None
    steps_per_generation: int | None = None
    num_iterations: int = 1
    async_prefetch: bool = False
    eval_during_produce: bool = True
    empty_cache_before_produce: bool = False
    empty_cache_after_produce: bool = False

    def __post_init__(self):
        if self.mini_epochs < 1:
            raise ValueError(f"mini_epochs must be >= 1, got {self.mini_epochs}")
        if self.max_rollouts is not None and self.max_rollouts < 1:
            raise ValueError(f"max_rollouts must be >= 1 or None, got {self.max_rollouts}")
        if self.num_iterations < 1:
            raise ValueError(f"num_iterations must be >= 1, got {self.num_iterations}")
        if self.steps_per_generation is not None and self.steps_per_generation < 1:
            raise ValueError(f"steps_per_generation must be >= 1 or None, got {self.steps_per_generation}")


# ---------------------------------------------------------------------------
# DataProducer protocol
# ---------------------------------------------------------------------------


class DataProducer(ABC):
    """Abstract base class for online data producers.

    Subclass this and implement :meth:`produce` to supply fresh training data
    each rollout round.  The Trainer calls ``produce(model, step)`` and wraps
    the returned ``Dataset`` in a ``DataLoader`` automatically.
    """

    config: ProducerConfig

    @abstractmethod
    def produce(
        self,
        model: Any,
        global_step: int,
        *,
        processing_class: Any = None,
        accelerator: Any = None,
        args: Any = None,
        **kwargs,
    ) -> Dataset:
        """Generate a fresh training dataset.

        Args:
            model: The current model (may be wrapped by DDP/FSDP/DeepSpeed).
            global_step: The current global training step.
            processing_class: The tokeniser / processor attached to the Trainer.
            accelerator: The ``Accelerator`` instance from the Trainer.
            args: The ``TrainingArguments`` from the Trainer.

        Returns:
            A ``torch.utils.data.Dataset`` to train on for this rollout.
        """
        ...


class BaseDataProducer(DataProducer):
    """Convenience base class with a default :class:`ProducerConfig` and
    lifecycle hooks.

    Subclass this and override :meth:`produce`.  Optionally override
    :meth:`on_rollout_begin` / :meth:`on_rollout_end` for custom logging or
    bookkeeping.
    """

    def __init__(self, config: ProducerConfig | None = None):
        self.config = config or ProducerConfig()

    def on_rollout_begin(self, global_step: int) -> None:
        """Called before each ``produce()`` invocation."""

    def on_rollout_end(self, dataset: Dataset, global_step: int) -> None:
        """Called after each ``produce()`` invocation with the produced dataset."""


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


class AsyncDataProducer:
    """Wraps a synchronous :class:`DataProducer` for background-thread data
    generation.

    While the Trainer trains on the current rollout, this wrapper produces the
    next dataset in a background thread.  The first call to :meth:`produce` is
    synchronous; subsequent calls return the prefetched result and start the
    next prefetch.
    """

    def __init__(self, inner: DataProducer):
        self._inner = inner
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-producer")
        self._pending: Future | None = None

    @property
    def config(self) -> ProducerConfig:
        return self._inner.config

    def produce(self, model: Any, global_step: int, **kwargs) -> Dataset:
        """Return the prefetched dataset (blocking) and start prefetching the
        next one.  On the very first call, produces synchronously."""
        if self._pending is not None:
            dataset = self._pending.result()
        else:
            dataset = self._inner.produce(model, global_step, **kwargs)

        # Start prefetching the next dataset
        self._pending = self._executor.submit(self._inner.produce, model, global_step + 1, **kwargs)
        return dataset

    def on_rollout_begin(self, global_step: int) -> None:
        if hasattr(self._inner, "on_rollout_begin"):
            self._inner.on_rollout_begin(global_step)

    def on_rollout_end(self, dataset: Dataset, global_step: int) -> None:
        if hasattr(self._inner, "on_rollout_end"):
            self._inner.on_rollout_end(dataset, global_step)

    def shutdown(self) -> None:
        """Shut down the background thread pool."""
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Callback integration
# ---------------------------------------------------------------------------


class DataProducerCallback(TrainerCallback):
    """Marker class: if a :class:`DataProducer` also inherits from this, the
    Trainer will automatically register it as a callback, giving the producer
    access to all :class:`TrainerCallback` lifecycle events (``on_train_begin``,
    ``on_step_end``, etc.)."""
