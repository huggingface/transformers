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
Internal epoch-source abstraction for the Trainer.

These classes are private implementation details. They unify static-dataset
training and online-data-producer training behind a single iterator interface
so that ``_inner_training_loop`` has no ``if online: … else: …`` branching.
"""

from __future__ import annotations

import logging
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from .trainer_utils import has_length


if TYPE_CHECKING:
    from .data_producer import DataProducer


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TrainingPlan:
    """Immutable bag of training-level constants computed once before the loop.

    Matches the 7-tuple returned by ``Trainer.set_initial_training_values()``.
    """

    num_train_epochs: int
    num_update_steps_per_epoch: int
    num_examples: int
    num_train_samples: int
    total_train_batch_size: int
    steps_in_epoch: int
    max_steps: int


@dataclass
class _EpochSpec:
    """Per-epoch data bundle passed to ``Trainer._run_epoch()``.

    Replaces the 7 per-epoch keyword arguments that ``_run_epoch`` previously
    accepted as individual parameters.
    """

    epoch: float
    dataloader: DataLoader
    steps_in_epoch: int
    num_update_steps_per_epoch: int
    resume_from_checkpoint: str | None = None
    epochs_trained: int = 0
    steps_trained_in_current_epoch: int = 0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class _EpochSource(ABC):
    """Abstract source of training epochs.

    Two concrete implementations exist:

    * :class:`_StaticEpochSource` — wraps the traditional fixed
      ``train_dataset`` / ``DataLoader`` pipeline.
    * :class:`_OnlineEpochSource` — wraps a :class:`DataProducer` that
      generates fresh data each rollout round.
    """

    @abstractmethod
    def compute_plan(self, trainer: Any) -> _TrainingPlan:
        """Create dataloader(s) and compute training-level constants.

        Called once at the start of ``_inner_training_loop``.
        """
        ...

    @abstractmethod
    def iter_epochs(
        self,
        trainer: Any,
        plan: _TrainingPlan,
        epochs_trained: int,
        steps_trained_in_current_epoch: int,
        resume_from_checkpoint: str | None,
    ) -> Iterator[_EpochSpec]:
        """Yield one :class:`_EpochSpec` per training epoch."""
        ...

    def post_model_setup(self, trainer: Any, model: Any, dataloader: DataLoader) -> None:
        """Hook called after model wrapping (``accelerator.prepare``).

        Default implementation is a no-op.  ``_StaticEpochSource`` uses this
        to apply the SP (Sequence Parallelism) dataloader adapter.
        """

    @property
    @abstractmethod
    def initial_dataloader(self) -> DataLoader:
        """The dataloader to pass to ``_prepare_for_training``.

        For the static path this is the train dataloader; for the online path
        this is the dataloader created from the first ``produce()`` call.
        """
        ...


# ---------------------------------------------------------------------------
# Static (traditional dataset) source
# ---------------------------------------------------------------------------


class _StaticEpochSource(_EpochSource):
    """Epoch source for the standard ``train_dataset`` path.

    Behaviour is **identical** to the original ``_inner_training_loop``:
    delegates to ``get_train_dataloader()`` and ``set_initial_training_values()``
    (the override points that subclasses like GRPOTrainer rely on) and yields
    the same dataloader every epoch.
    """

    def __init__(self):
        self._train_dataloader: DataLoader | None = None

    def compute_plan(self, trainer: Any) -> _TrainingPlan:
        train_dataloader = trainer.get_train_dataloader()
        if trainer.is_fsdp_xla_v2_enabled:
            from .integrations.tpu import tpu_spmd_dataloader

            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            total_train_batch_size,
            steps_in_epoch,
            max_steps,
        ) = trainer.set_initial_training_values(trainer.args, train_dataloader)

        self._train_dataloader = train_dataloader

        return _TrainingPlan(
            num_train_epochs=num_train_epochs,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            num_examples=num_examples,
            num_train_samples=num_train_samples,
            total_train_batch_size=total_train_batch_size,
            steps_in_epoch=steps_in_epoch,
            max_steps=max_steps,
        )

    @property
    def initial_dataloader(self) -> DataLoader:
        assert self._train_dataloader is not None, "compute_plan() must be called first"
        return self._train_dataloader

    def post_model_setup(self, trainer: Any, model: Any, dataloader: DataLoader) -> None:
        # Apply the SP adapter and store back.  The adapter must run after
        # accelerator.prepare (which happens inside _prepare_for_training).
        self._train_dataloader = trainer._apply_sp_adapter(dataloader, model)

    def iter_epochs(
        self,
        trainer: Any,
        plan: _TrainingPlan,
        epochs_trained: int,
        steps_trained_in_current_epoch: int,
        resume_from_checkpoint: str | None,
    ) -> Iterator[_EpochSpec]:
        for epoch in range(epochs_trained, plan.num_train_epochs):
            yield _EpochSpec(
                epoch=epoch,
                dataloader=self._train_dataloader,
                steps_in_epoch=plan.steps_in_epoch,
                num_update_steps_per_epoch=plan.num_update_steps_per_epoch,
                resume_from_checkpoint=resume_from_checkpoint,
                epochs_trained=epochs_trained,
                steps_trained_in_current_epoch=steps_trained_in_current_epoch,
            )


# ---------------------------------------------------------------------------
# Online (DataProducer) source
# ---------------------------------------------------------------------------


class _OnlineEpochSource(_EpochSource):
    """Epoch source backed by a :class:`DataProducer`.

    Each rollout round calls ``produce(model)`` to get a fresh dataset, wraps
    it in a ``DataLoader``, and yields ``mini_epochs`` passes over it.
    """

    def __init__(self, data_producer: "DataProducer"):
        self._producer = data_producer
        self._initial_dataloader: DataLoader | None = None
        self._initial_dataset = None
        self._model = None

    def compute_plan(self, trainer: Any) -> _TrainingPlan:
        args = trainer.args
        max_steps = args.max_steps
        config = self._producer.config

        # Produce the initial dataset to establish dataloader shape
        self._initial_dataset = trainer._produce_data(trainer.model)
        dataloader = trainer._get_online_dataloader(self._initial_dataset)
        self._initial_dataloader = dataloader

        total_train_batch_size = trainer.get_total_train_batch_size(args)

        if has_length(dataloader):
            len_dataloader = len(dataloader)
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            steps_in_epoch = len_dataloader
        else:
            # IterableDataset — rely on max_steps
            num_update_steps_per_epoch = max_steps
            steps_in_epoch = max_steps * args.gradient_accumulation_steps

        # Compute num_train_epochs based on max_rollouts or max_steps
        if config.max_rollouts is not None:
            num_train_epochs = config.max_rollouts * config.mini_epochs
            if max_steps <= 0:
                max_steps = num_train_epochs * num_update_steps_per_epoch
        else:
            num_train_epochs = math.ceil(max_steps / num_update_steps_per_epoch) if num_update_steps_per_epoch > 0 else sys.maxsize

        num_examples = total_train_batch_size * max_steps
        num_train_samples = num_examples

        return _TrainingPlan(
            num_train_epochs=num_train_epochs,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            num_examples=num_examples,
            num_train_samples=num_train_samples,
            total_train_batch_size=total_train_batch_size,
            steps_in_epoch=steps_in_epoch,
            max_steps=max_steps,
        )

    @property
    def initial_dataloader(self) -> DataLoader:
        assert self._initial_dataloader is not None, "compute_plan() must be called first"
        return self._initial_dataloader

    def post_model_setup(self, trainer: Any, model: Any, dataloader: DataLoader) -> None:
        self._model = model
        # Apply SP adapter to the initial dataloader
        self._initial_dataloader = trainer._apply_sp_adapter(dataloader, model)

    def iter_epochs(
        self,
        trainer: Any,
        plan: _TrainingPlan,
        epochs_trained: int,
        steps_trained_in_current_epoch: int,
        resume_from_checkpoint: str | None,
    ) -> Iterator[_EpochSpec]:
        config = self._producer.config
        rollout = 0
        epoch_counter = 0

        while True:
            # Stop conditions
            if config.max_rollouts is not None and rollout >= config.max_rollouts:
                break

            # Get dataset for this rollout
            if rollout == 0:
                # Use the dataset produced during compute_plan()
                dataloader = self._initial_dataloader
            else:
                # Remove the previous dataloader from accelerator tracking
                # to avoid accumulating stale references (which would leak
                # memory and interfere with checkpoint save/load).
                prev_dl = dataloader  # noqa: F821 — always set on prior iteration
                acc_dls = trainer.accelerator._dataloaders
                if prev_dl in acc_dls:
                    acc_dls.remove(prev_dl)

                dataset = trainer._produce_data(self._model)
                dataloader = trainer._get_online_dataloader(dataset)
                dataloader = trainer._apply_sp_adapter(dataloader, self._model)
                # Update callback handler reference
                trainer.callback_handler.train_dataloader = dataloader

            # Recompute steps_in_epoch for this dataloader (may differ if
            # the produced dataset has a different size)
            if has_length(dataloader):
                steps_in_epoch = len(dataloader)
                num_update_steps_per_epoch = max(
                    steps_in_epoch // trainer.args.gradient_accumulation_steps
                    + int(steps_in_epoch % trainer.args.gradient_accumulation_steps > 0),
                    1,
                )
            else:
                steps_in_epoch = plan.steps_in_epoch
                num_update_steps_per_epoch = plan.num_update_steps_per_epoch

            # Yield mini_epochs passes over this rollout's data
            for mini in range(config.mini_epochs):
                if epoch_counter < epochs_trained:
                    # Skip epochs that were already trained (checkpoint resume)
                    epoch_counter += 1
                    continue

                epoch_idx = rollout + mini / config.mini_epochs

                yield _EpochSpec(
                    epoch=epoch_idx,
                    dataloader=dataloader,
                    steps_in_epoch=steps_in_epoch,
                    num_update_steps_per_epoch=num_update_steps_per_epoch,
                    resume_from_checkpoint=resume_from_checkpoint if epoch_counter == epochs_trained else None,
                    epochs_trained=epochs_trained,
                    steps_trained_in_current_epoch=steps_trained_in_current_epoch if epoch_counter == epochs_trained else 0,
                )

                epoch_counter += 1

                # Check if training should stop (the caller checks
                # control.should_training_stop, but we also need to break
                # out of mini_epochs if max_steps is reached)
                if trainer.control.should_training_stop:
                    return

            rollout += 1

            if trainer.control.should_training_stop:
                return
