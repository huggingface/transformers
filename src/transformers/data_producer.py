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
from collections import deque
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
        prefetch_depth: How many rollouts to produce ahead of training when
            ``async_prefetch`` is enabled.  With depth *N*, the producer
            keeps *N* rollouts queued.  Higher values keep the GPU more
            saturated but increase off-policy staleness — each additional
            rollout in the queue was generated with a model that is
            ``~steps_per_generation × num_iterations`` more optimizer
            steps behind.  Default is 1 (one rollout ahead).
        sync_warmup_rollouts: Number of initial rollouts to produce
            synchronously before switching to async prefetch.  During
            warmup, each rollout is generated on-policy (using the
            latest model weights) so the model can bootstrap learning
            from sparse reward signals.  After the warmup period, async
            prefetch resumes for maximum throughput.  ``0`` (default)
            disables warmup and uses async prefetch from the start.
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
    prefetch_depth: int = 1
    sync_warmup_rollouts: int = 0
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
        if self.prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {self.prefetch_depth}")
        if self.sync_warmup_rollouts < 0:
            raise ValueError(f"sync_warmup_rollouts must be >= 0, got {self.sync_warmup_rollouts}")


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

    While the Trainer trains on the current rollout, this wrapper produces
    upcoming datasets in a background thread.  The ``prefetch_depth``
    (from :class:`ProducerConfig`) controls how many rollouts are queued
    ahead of training:

    * ``prefetch_depth=1`` (default): one rollout is produced in the
      background while the current one is trained on.  This is the
      sweet spot for most setups — it hides generation latency without
      introducing off-policy staleness.
    * ``prefetch_depth=N``: *N* rollouts are queued.  Useful when
      generation is much faster than training (e.g. vLLM server mode)
      and you want to keep the GPU fully saturated, at the cost of
      increased off-policy staleness.

    The first call to :meth:`produce` is synchronous; it returns the
    first dataset and seeds the prefetch queue.
    """

    def __init__(self, inner: DataProducer):
        self._inner = inner
        self._depth = inner.config.prefetch_depth
        self._warmup_remaining = inner.config.sync_warmup_rollouts
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-producer")
        self._queue: deque[Future] = deque()
        self._initialized = False

    @property
    def config(self) -> ProducerConfig:
        return self._inner.config

    def produce(self, model: Any, global_step: int, **kwargs) -> Dataset:
        """Return the next dataset, blocking if the prefetch hasn't finished.

        On the very first call, the current dataset is produced synchronously
        and the prefetch queue is seeded with ``prefetch_depth`` futures.
        Subsequent calls pop the oldest future from the queue and submit a
        new one to maintain the queue at ``prefetch_depth``.

        When ``sync_warmup_rollouts > 0``, the first *N* rollouts are
        produced synchronously (on-policy) so the model can bootstrap
        learning from sparse reward signals before async prefetch begins.
        """
        # During warmup, produce synchronously (on-policy) without prefetching
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            logger.info(
                f"AsyncDataProducer: sync warmup rollout (remaining={self._warmup_remaining})"
            )
            return self._inner.produce(model, global_step, **kwargs)

        if not self._initialized:
            # First async call: produce synchronously, then seed the queue
            dataset = self._inner.produce(model, global_step, **kwargs)
            for i in range(1, self._depth + 1):
                self._queue.append(
                    self._executor.submit(self._inner.produce, model, global_step + i, **kwargs)
                )
            self._initialized = True
            return dataset

        # Subsequent calls: consume oldest prefetched result
        dataset = self._queue.popleft().result()

        # Submit a new future to keep the queue full
        next_step = global_step + self._depth
        self._queue.append(
            self._executor.submit(self._inner.produce, model, next_step, **kwargs)
        )
        return dataset

    def on_rollout_begin(self, global_step: int) -> None:
        if hasattr(self._inner, "on_rollout_begin"):
            self._inner.on_rollout_begin(global_step)

    def on_rollout_end(self, dataset: Dataset, global_step: int) -> None:
        if hasattr(self._inner, "on_rollout_end"):
            self._inner.on_rollout_end(dataset, global_step)

    def shutdown(self) -> None:
        """Shut down the background thread pool and cancel pending futures."""
        for future in self._queue:
            future.cancel()
        self._queue.clear()
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Callback integration
# ---------------------------------------------------------------------------


class DataProducerCallback(TrainerCallback):
    """Marker class: if a :class:`DataProducer` also inherits from this, the
    Trainer will automatically register it as a callback, giving the producer
    access to all :class:`TrainerCallback` lifecycle events (``on_train_begin``,
    ``on_step_end``, etc.)."""
