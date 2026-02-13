# Copyright 2020-present the HuggingFace Inc. team.
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
DataProducer protocol for online/async training with the HuggingFace Trainer.

A ``DataProducer`` generates fresh training data each rollout round,
enabling online RL methods (PPO, GRPO, REINFORCE, online DPO) and
curriculum learning without any changes to the core training loop.

Quick start::

    from transformers import Trainer, TrainingArguments
    from transformers.data_producer import BaseDataProducer, ProducerConfig, RolloutDataset

    class MyProducer(BaseDataProducer):
        def __init__(self, prompts, reward_fn):
            super().__init__(ProducerConfig(mini_epochs=2, max_rollouts=100))
            self.prompts = prompts
            self.reward_fn = reward_fn

        def produce(self, model, global_step, **kwargs):
            completions = model.generate(self.prompts, max_new_tokens=256)
            rewards = self.reward_fn(completions)
            return RolloutDataset(prompts=self.prompts, completions=completions, rewards=rewards)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./out", max_steps=5000),
        data_producer=MyProducer(prompts, reward_fn),
    )
    trainer.train()
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from .trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from torch import nn


logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class ProducerConfig:
    """Configuration for a :class:`DataProducer`.

    Args:
        mini_epochs: Number of training passes over each produced dataset.
            Use ``>1`` to amortise expensive generation across multiple
            gradient updates (common in PPO/GRPO).
        max_rollouts: Maximum number of produce-then-train rounds.  Set to
            ``None`` for unlimited (training stops at ``args.max_steps``).
        async_prefetch: If ``True``, the next dataset is produced in a
            background thread while the current one is being trained on.
        eval_during_produce: If ``True``, switch the model to eval mode
            during ``produce()`` (common for generation).
        empty_cache_before_produce: Call ``torch.cuda.empty_cache()`` before
            ``produce()`` to free memory for generation.
        empty_cache_after_produce: Call ``torch.cuda.empty_cache()`` after
            ``produce()`` to free memory for training.
    """

    mini_epochs: int = 1
    max_rollouts: int | None = None
    async_prefetch: bool = False
    eval_during_produce: bool = True
    empty_cache_before_produce: bool = False
    empty_cache_after_produce: bool = False


# ======================================================================
# Protocol / base classes
# ======================================================================


class DataProducer(ABC):
    """Abstract protocol for online data production.

    Subclasses must implement :meth:`produce` and provide a :attr:`config`.

    The Trainer calls ``produce(model, global_step, ...)`` each rollout
    round to obtain a fresh ``Dataset`` for training.
    """

    config: ProducerConfig

    @abstractmethod
    def produce(
        self,
        model: nn.Module,
        global_step: int,
        processing_class: Any = None,
        accelerator: Any = None,
        args: Any = None,
        **kwargs,
    ) -> Dataset:
        """Generate a fresh training dataset.

        Args:
            model: The current model (may be in eval mode if
                ``config.eval_during_produce`` is set).
            global_step: Current training step.
            processing_class: Tokenizer / processor from the Trainer.
            accelerator: The Accelerate accelerator.
            args: TrainingArguments.
            **kwargs: Reserved for future use.

        Returns:
            A ``torch.utils.data.Dataset``.  Prefer map-style datasets
            (with ``__len__``) when ``mini_epochs > 1``.
        """
        ...


class BaseDataProducer(DataProducer):
    """Convenience base class with lifecycle hooks.

    Subclasses only need to implement :meth:`produce`.  Optional hooks:

    - ``on_rollout_begin(global_step)`` — called before each ``produce()``.
    - ``on_rollout_end(dataset, global_step)`` — called after each ``produce()``.
    """

    def __init__(self, config: ProducerConfig | None = None):
        self.config = config or ProducerConfig()

    def on_rollout_begin(self, global_step: int) -> None:
        """Hook called before ``produce()``.  Override for logging/setup."""
        pass

    def on_rollout_end(self, dataset: Dataset, global_step: int) -> None:
        """Hook called after ``produce()``.  Override for logging/cleanup."""
        pass


# ======================================================================
# Async wrapper
# ======================================================================


class AsyncDataProducer(DataProducer):
    """Wraps a :class:`DataProducer` for background-thread data generation.

    While the Trainer trains on the current dataset, the next dataset is
    produced in a background thread.  ``produce()`` blocks until the
    prefetched result is ready, then kicks off the *next* prefetch.

    Usage::

        producer = AsyncDataProducer(MyProducer(...))
        # or: set config.async_prefetch = True and the Trainer wraps it automatically
    """

    def __init__(self, inner: DataProducer):
        self._inner = inner
        self.config = inner.config
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_result: Dataset | None = None
        self._prefetch_error: BaseException | None = None

    def produce(
        self,
        model: nn.Module,
        global_step: int,
        processing_class: Any = None,
        accelerator: Any = None,
        args: Any = None,
        **kwargs,
    ) -> Dataset:
        # If there's a prefetched result, use it
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None
            if self._prefetch_error is not None:
                raise self._prefetch_error
            result = self._prefetch_result
            self._prefetch_result = None
        else:
            # First call — produce synchronously
            result = self._inner.produce(
                model=model,
                global_step=global_step,
                processing_class=processing_class,
                accelerator=accelerator,
                args=args,
                **kwargs,
            )

        # Start prefetching the next dataset
        self._start_prefetch(model, global_step, processing_class, accelerator, args, **kwargs)
        return result

    def _start_prefetch(self, model, global_step, processing_class, accelerator, args, **kwargs):
        def _worker():
            try:
                self._prefetch_result = self._inner.produce(
                    model=model,
                    global_step=global_step,
                    processing_class=processing_class,
                    accelerator=accelerator,
                    args=args,
                    **kwargs,
                )
            except BaseException as e:
                self._prefetch_error = e

        self._prefetch_error = None
        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    # Forward lifecycle hooks
    def on_rollout_begin(self, global_step: int) -> None:
        if hasattr(self._inner, "on_rollout_begin"):
            self._inner.on_rollout_begin(global_step=global_step)

    def on_rollout_end(self, dataset: Dataset, global_step: int) -> None:
        if hasattr(self._inner, "on_rollout_end"):
            self._inner.on_rollout_end(dataset=dataset, global_step=global_step)


# ======================================================================
# Callback
# ======================================================================


class DataProducerCallback(TrainerCallback):
    """Trainer callback that forwards lifecycle events to a DataProducer.

    Automatically added by the Trainer when a ``data_producer`` is set.
    """

    def __init__(self, data_producer: DataProducer):
        self.data_producer = data_producer

    def on_train_begin(self, args, state, control, **kwargs):
        """Log that online training is starting."""
        logger.info("DataProducerCallback: online training started.")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Clean up async resources if any."""
        if isinstance(self.data_producer, AsyncDataProducer):
            if self.data_producer._prefetch_thread is not None:
                self.data_producer._prefetch_thread.join(timeout=5.0)
        return control


# ======================================================================
# Reference datasets
# ======================================================================


class RolloutDataset(Dataset):
    """Simple dataset for RL rollout data (prompts, completions, rewards).

    Each item is a dict with keys ``"prompt"``, ``"completion"``,
    ``"reward"``, plus any extra fields.

    Usage::

        ds = RolloutDataset(
            prompts=prompt_ids,       # list[Tensor] or Tensor
            completions=comp_ids,     # list[Tensor] or Tensor
            rewards=reward_values,    # list[float] or Tensor
            extras={"advantages": adv_tensor},
        )
    """

    def __init__(
        self,
        prompts: list | torch.Tensor,
        completions: list | torch.Tensor,
        rewards: list | torch.Tensor,
        extras: dict[str, Any] | None = None,
    ):
        self.prompts = prompts
        self.completions = completions
        self.rewards = rewards
        self.extras = extras or {}
        assert len(prompts) == len(completions) == len(rewards)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "prompt": self.prompts[idx],
            "completion": self.completions[idx],
            "reward": self.rewards[idx],
        }
        for k, v in self.extras.items():
            item[k] = v[idx]
        return item


class PreferencePairDataset(Dataset):
    """Simple dataset for preference pair data (prompt, chosen, rejected).

    Each item is a dict with keys ``"prompt"``, ``"chosen"``, ``"rejected"``.

    Usage::

        ds = PreferencePairDataset(
            prompts=prompt_ids,
            chosen=chosen_ids,
            rejected=rejected_ids,
        )
    """

    def __init__(
        self,
        prompts: list | torch.Tensor,
        chosen: list | torch.Tensor,
        rejected: list | torch.Tensor,
    ):
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        assert len(prompts) == len(chosen) == len(rejected)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "prompt": self.prompts[idx],
            "chosen": self.chosen[idx],
            "rejected": self.rejected[idx],
        }
