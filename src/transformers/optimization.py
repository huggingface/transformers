# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""PyTorch optimization for BERT model."""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from .trainer_pt_utils import LayerWiseDummyOptimizer, LayerWiseDummyScheduler
from .trainer_utils import SchedulerType
from .utils import logging


logger = logging.get_logger(__name__)


def _get_constant_lambda(_=None):
    return 1


def get_constant_schedule(optimizer: Optimizer, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    return LambdaLR(optimizer, _get_constant_lambda, last_epoch=last_epoch)


def get_reduce_on_plateau_schedule(optimizer: Optimizer, **kwargs):
    """
    Create a schedule with a constant learning rate that decreases when a metric has stopped improving.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        kwargs (`dict`, *optional*):
            Extra parameters to be passed to the scheduler. See `torch.optim.lr_scheduler.ReduceLROnPlateau`
            for possible parameters.

    Return:
        `torch.optim.lr_scheduler.ReduceLROnPlateau` with the appropriate schedule.
    """

    return ReduceLROnPlateau(optimizer, **kwargs)


def _get_constant_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0


def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = -1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init


def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by *lr_end*, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        lr_end (`float`, *optional*, defaults to 1e-7):
            The end LR.
        power (`float`, *optional*, defaults to 1.0):
            Power factor.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Note: *power* defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_inverse_sqrt_schedule_lr_lambda(current_step: int, *, num_warmup_steps: int, timescale: int | None = None):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    shift = timescale - num_warmup_steps
    decay = 1.0 / math.sqrt((current_step + shift) / timescale)
    return decay


def get_inverse_sqrt_schedule(
    optimizer: Optimizer, num_warmup_steps: int, timescale: int | None = None, last_epoch: int = -1
):
    """
    Create a schedule with an inverse square-root learning rate, from the initial lr set in the optimizer, after a
    warmup period which increases lr linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        timescale (`int`, *optional*, defaults to `num_warmup_steps`):
            Time scale.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    # Note: this implementation is adapted from
    # https://github.com/google-research/big_vision/blob/f071ce68852d56099437004fd70057597a95f6ef/big_vision/utils.py#L930

    if timescale is None:
        timescale = num_warmup_steps or 10_000

    lr_lambda = partial(_get_inverse_sqrt_schedule_lr_lambda, num_warmup_steps=num_warmup_steps, timescale=timescale)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, min_lr_rate: float = 0.0
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float | None = None,
    min_lr_rate: float | None = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float = 0.0,
    warmup_lr_rate: float | None = None,
):
    current_step = float(current_step)
    num_warmup_steps = float(num_warmup_steps)
    num_training_steps = float(num_training_steps)

    if current_step < num_warmup_steps:
        if warmup_lr_rate is None:
            return (current_step + 1.0) / max(1.0, num_warmup_steps)
        else:
            warmup_lr_rate = float(warmup_lr_rate)
            return warmup_lr_rate + (1.0 - warmup_lr_rate) * (current_step) / (max(1, num_warmup_steps - 1))
    progress = (current_step - num_warmup_steps + 1.0) / (max(1.0, num_training_steps - num_warmup_steps))
    factor = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    factor = factor * (1 - min_lr_rate) + min_lr_rate
    return max(0, factor)


def get_cosine_with_min_lr_schedule_with_warmup_lr_rate(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float | None = None,
    min_lr_rate: float | None = None,
    warmup_lr_rate: float | None = None,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.
        warmup_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the start learning rate. If not set, `warmup_lr_rate` will be treated as float(1/num_warmup_steps).

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Only one of min_lr or min_lr_rate should be set")
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults["lr"]
    elif min_lr_rate is None:
        raise ValueError("One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`")

    lr_lambda = partial(
        _get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
        min_lr_rate=min_lr_rate,
        warmup_lr_rate=warmup_lr_rate,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_wsd_scheduler_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    warmup_type: str,
    decay_type: str,
    min_lr_ratio: float,
    num_cycles: float,
):
    if current_step < num_warmup_steps:
        progress = float(current_step) / float(max(1, num_warmup_steps))
        if warmup_type == "linear":
            factor = progress
        elif warmup_type == "cosine":
            factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif warmup_type == "1-sqrt":
            factor = 1.0 - math.sqrt(1.0 - progress)
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
        return max(0.0, factor)

    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0

    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
        if decay_type == "linear":
            factor = 1.0 - progress
        elif decay_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        elif decay_type == "1-sqrt":
            factor = 1.0 - math.sqrt(progress)
        factor = factor * (1.0 - min_lr_ratio) + min_lr_ratio
        return max(0.0, factor)
    return min_lr_ratio


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_decay_steps: int,
    num_training_steps: int | None = None,
    num_stable_steps: int | None = None,
    warmup_type: str = "linear",
    decay_type: str = "cosine",
    min_lr_ratio: float = 0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that has three stages:
    1. warmup: increase from min_lr_ratio times the initial learning rate to the initial learning rate following a warmup_type.
    2. stable: constant learning rate.
    3. decay: decrease from the initial learning rate to min_lr_ratio times the initial learning rate following a decay_type.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_decay_steps (`int`):
            The number of steps for the decay phase.
        num_training_steps (`int`, *optional*):
            The total number of training steps. This is the sum of the warmup, stable and decay steps. If `num_stable_steps` is not provided, the stable phase will be `num_training_steps - num_warmup_steps - num_decay_steps`.
        num_stable_steps (`int`, *optional*):
            The number of steps for the stable phase. Please ensure that `num_warmup_steps + num_stable_steps + num_decay_steps` equals `num_training_steps`, otherwise the other steps will default to the minimum learning rate.
        warmup_type (`str`, *optional*, defaults to "linear"):
            The type of warmup to use. Can be 'linear', 'cosine' or '1-sqrt'.
        decay_type (`str`, *optional*, defaults to "cosine"):
            The type of decay to use. Can be 'linear', 'cosine' or '1-sqrt'.
        min_lr_ratio (`float`, *optional*, defaults to 0):
            The minimum learning rate as a ratio of the initial learning rate.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    if num_training_steps is None and num_stable_steps is None:
        raise ValueError("Either num_training_steps or num_stable_steps must be specified.")

    if num_training_steps is not None and num_stable_steps is not None:
        warnings.warn("Both num_training_steps and num_stable_steps are specified. num_stable_steps will be used.")

    if warmup_type not in ["linear", "cosine", "1-sqrt"]:
        raise ValueError(f"Unknown warmup type: {warmup_type}, expected 'linear', 'cosine' or '1-sqrt'")

    if decay_type not in ["linear", "cosine", "1-sqrt"]:
        raise ValueError(f"Unknown decay type: {decay_type}, expected 'linear', 'cosine' or '1-sqrt'")

    if num_stable_steps is None:
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps

    lr_lambda = partial(
        _get_wsd_scheduler_lambda,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        warmup_type=warmup_type,
        decay_type=decay_type,
        min_lr_ratio=min_lr_ratio,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class StreamingAverage:
    """Rolling window average for smoothing metric values.

    Maintains a sliding window of values and computes their average,
    useful for smoothing noisy metric values before making learning rate decisions.

    Args:
        window_size (`int`):
            The maximum number of values to keep in the rolling window.
    """

    def __init__(self, window_size: int) -> None:
        self.window_size: int = window_size
        self.values: list[float] = []
        self.sum: float = 0.0

    def streamavg(self, value: float) -> float:
        """Add a value and return the current rolling average."""
        self.values.append(value)
        self.sum += value

        if len(self.values) > self.window_size:
            removed = self.values.pop(0)
            self.sum -= removed

        return self.sum / len(self.values)

    def state_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "values": self.values.copy(),
            "sum": self.sum,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.window_size = state_dict.get("window_size", self.window_size)
        self.values = state_dict.get("values", []).copy()
        self.sum = state_dict.get("sum", 0.0)


class GreedyLR:
    """Adaptive learning rate scheduler that responds to training metrics.

    GreedyLR dynamically adjusts the learning rate based on training performance:
    - Increases LR when metrics improve consistently (divides by factor)
    - Decreases LR when metrics plateau (multiplies by factor)

    This differs from traditional schedulers like cosine annealing by responding
    to actual training dynamics rather than following a predetermined schedule.

    Reference: `GreedyLR: A Novel Adaptive Learning Rate Scheduler <https://arxiv.org/abs/2512.14527>`_

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        mode (`str`, *optional*, defaults to `"min"`):
            One of 'min' or 'max'. In 'min' mode, LR will be reduced when the
            metric has stopped decreasing; in 'max' mode when it has stopped increasing.
        factor (`float`, *optional*, defaults to 0.95):
            Factor by which the learning rate will be adjusted. LR is multiplied by
            factor on plateau and divided by factor on improvement. Must be < 1.0.
        patience (`int`, *optional*, defaults to 10):
            Number of epochs with no improvement after which learning rate will be adjusted.
        threshold (`float`, *optional*, defaults to 1e-06):
            Threshold for measuring the new optimum.
        threshold_mode (`str`, *optional*, defaults to `"abs"`):
            One of 'rel' or 'abs'.
        cooldown (`int`, *optional*, defaults to 0):
            Number of epochs to wait before resuming normal operation after LR has been reduced.
        warmup (`int`, *optional*, defaults to 0):
            Number of epochs to wait before resuming normal operation after LR has been increased.
        min_lr (`float` or `list[float]`, *optional*, defaults to 0.001):
            A lower bound on the learning rate.
        max_lr (`float` or `list[float]`, *optional*, defaults to 1.0):
            An upper bound on the learning rate.
        eps (`float`, *optional*, defaults to 1e-08):
            Minimal decay applied to lr.
        verbose (`bool`, *optional*, defaults to `False`):
            If True, prints a message to stdout for each update.
        smooth (`bool`, *optional*, defaults to `False`):
            If True, applies streaming average smoothing to metrics.
        window_size (`int`, *optional*, defaults to 50):
            The window size for the streaming average when smooth=True.
        reset_start (`int`, *optional*, defaults to 500):
            Number of steps to wait at min_lr before resetting to initial state.

    Example:
        ```python
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = GreedyLR(optimizer, mode="min", patience=10)
        >>> for epoch in range(100):
        ...     train(...)
        ...     val_loss = validate(...)
        ...     scheduler.step(val_loss)
        ```
    """

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.95,
        patience: int = 10,
        threshold: float = 1e-6,
        threshold_mode: str = "abs",
        cooldown: int = 0,
        warmup: int = 0,
        min_lr: float | list[float] = 1e-3,
        max_lr: float | list[float] = 1.0,
        eps: float = 1e-8,
        verbose: bool = False,
        smooth: bool = False,
        window_size: int = 50,
        reset_start: int = 500,
    ) -> None:
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")

        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.warmup = warmup
        self.cooldown_counter = 0
        self.warmup_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.eps = eps
        self.smooth = smooth
        self.window_size = window_size
        self.reset_start = reset_start
        self.reset_start_original = reset_start
        self.last_epoch = 0

        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} max_lrs, got {len(max_lr)}")
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self._init_lrs = [group["lr"] for group in optimizer.param_groups]
        self._last_lr = self._init_lrs.copy()

        self.best: float = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.num_good_epochs = 0

        if mode not in ("min", "max"):
            raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in ("rel", "abs"):
            raise ValueError(f"threshold mode {threshold_mode} is unknown!")

        self._streaming_avg: StreamingAverage | None = None
        if smooth:
            self._streaming_avg = StreamingAverage(window_size)

    def step(self, metrics: float, epoch: int | None = None) -> None:
        """Perform a scheduler step based on the given metrics.

        Args:
            metrics (`float`):
                The metric value to use for LR adjustment decisions.
            epoch (`int`, *optional*):
                The current epoch number. If None, uses internal counter.
        """
        current = float(metrics)

        if self.smooth and self._streaming_avg is not None:
            current = self._streaming_avg.streamavg(current)

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            self.num_good_epochs = 0
        elif self.warmup_counter > 0:
            self.warmup_counter -= 1
            self.num_bad_epochs = 0
            self.num_good_epochs = 0
        else:
            if self.is_better(current, self.best):
                self.best = current
                self.num_bad_epochs = 0
                self.num_good_epochs += 1
            else:
                self.num_bad_epochs += 1
                self.num_good_epochs = 0

            if self.num_good_epochs > self.patience:
                self._increase_lr(epoch)
                self.warmup_counter = self.warmup
                self.num_good_epochs = 0
            elif self.num_bad_epochs > self.patience:
                self._reduce_lr(epoch)
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return current < best * (1.0 - self.threshold)
            else:
                return current < best - self.threshold
        else:
            if self.threshold_mode == "rel":
                return current > best * (1.0 + self.threshold)
            else:
                return current > best + self.threshold

    def _reduce_lr(self, epoch: int) -> None:
        all_at_min = True
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])

            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Epoch {epoch}: reducing learning rate of group {i} to {new_lr:.4e}.")

            if param_group["lr"] > self.min_lrs[i]:
                all_at_min = False

        if all_at_min:
            self.reset_start -= 1
            if self.reset_start <= 0:
                self._reset()

    def _increase_lr(self, epoch: int) -> None:
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = min(old_lr / self.factor, self.max_lrs[i])

            if new_lr - old_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Epoch {epoch}: increasing learning rate of group {i} to {new_lr:.4e}.")

        self.reset_start = self.reset_start_original

    def _reset(self) -> None:
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self._init_lrs[i]

        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.cooldown_counter = 0
        self.warmup_counter = 0
        self.reset_start = self.reset_start_original

        if self.smooth and self._streaming_avg is not None:
            self._streaming_avg = StreamingAverage(self.window_size)

        if self.verbose:
            print("Scheduler reset to initial state.")

    def get_last_lr(self) -> list[float]:
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the scheduler as a dictionary."""
        state = {
            "factor": self.factor,
            "min_lrs": self.min_lrs,
            "max_lrs": self.max_lrs,
            "patience": self.patience,
            "verbose": self.verbose,
            "cooldown": self.cooldown,
            "warmup": self.warmup,
            "cooldown_counter": self.cooldown_counter,
            "warmup_counter": self.warmup_counter,
            "mode": self.mode,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "num_good_epochs": self.num_good_epochs,
            "eps": self.eps,
            "last_epoch": self.last_epoch,
            "smooth": self.smooth,
            "window_size": self.window_size,
            "reset_start": self.reset_start,
            "reset_start_original": self.reset_start_original,
            "_last_lr": self._last_lr,
            "_init_lrs": self._init_lrs,
        }

        if self.smooth and self._streaming_avg is not None:
            state["_streaming_avg"] = self._streaming_avg.state_dict()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from a dictionary."""
        self.factor = state_dict.get("factor", self.factor)
        self.min_lrs = state_dict.get("min_lrs", self.min_lrs)
        self.max_lrs = state_dict.get("max_lrs", self.max_lrs)
        self.patience = state_dict.get("patience", self.patience)
        self.verbose = state_dict.get("verbose", self.verbose)
        self.cooldown = state_dict.get("cooldown", self.cooldown)
        self.warmup = state_dict.get("warmup", self.warmup)
        self.cooldown_counter = state_dict.get("cooldown_counter", self.cooldown_counter)
        self.warmup_counter = state_dict.get("warmup_counter", self.warmup_counter)
        self.mode = state_dict.get("mode", self.mode)
        self.threshold = state_dict.get("threshold", self.threshold)
        self.threshold_mode = state_dict.get("threshold_mode", self.threshold_mode)
        self.best = state_dict.get("best", self.best)
        self.num_bad_epochs = state_dict.get("num_bad_epochs", self.num_bad_epochs)
        self.num_good_epochs = state_dict.get("num_good_epochs", self.num_good_epochs)
        self.eps = state_dict.get("eps", self.eps)
        self.last_epoch = state_dict.get("last_epoch", self.last_epoch)
        self.smooth = state_dict.get("smooth", self.smooth)
        self.window_size = state_dict.get("window_size", self.window_size)
        self.reset_start = state_dict.get("reset_start", self.reset_start)
        self.reset_start_original = state_dict.get("reset_start_original", self.reset_start_original)
        self._last_lr = state_dict.get("_last_lr", self._last_lr)
        self._init_lrs = state_dict.get("_init_lrs", self._init_lrs)

        if "_streaming_avg" in state_dict:
            if self._streaming_avg is None:
                self._streaming_avg = StreamingAverage(self.window_size)
            self._streaming_avg.load_state_dict(state_dict["_streaming_avg"])

        if "_last_lr" in state_dict:
            for param_group, lr in zip(self.optimizer.param_groups, self._last_lr):
                param_group["lr"] = lr


def get_greedy_schedule(optimizer: Optimizer, **kwargs):
    """
    Create an adaptive learning rate scheduler that adjusts LR based on training metrics.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        kwargs (`dict`, *optional*):
            Extra parameters passed to the scheduler. See [`GreedyLR`] for possible parameters.

    Return:
        [`GreedyLR`] with the appropriate schedule.
    """
    return GreedyLR(optimizer, **kwargs)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.COSINE_WITH_RESTARTS: get_cosine_with_hard_restarts_schedule_with_warmup,
    SchedulerType.POLYNOMIAL: get_polynomial_decay_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
    SchedulerType.INVERSE_SQRT: get_inverse_sqrt_schedule,
    SchedulerType.REDUCE_ON_PLATEAU: get_reduce_on_plateau_schedule,
    SchedulerType.COSINE_WITH_MIN_LR: get_cosine_with_min_lr_schedule_with_warmup,
    SchedulerType.COSINE_WARMUP_WITH_MIN_LR: get_cosine_with_min_lr_schedule_with_warmup_lr_rate,
    SchedulerType.WARMUP_STABLE_DECAY: get_wsd_schedule,
    SchedulerType.GREEDY: get_greedy_schedule,
}


def get_scheduler(
    name: str | SchedulerType,
    optimizer: Optimizer,
    num_warmup_steps: int | None = None,
    num_training_steps: int | None = None,
    scheduler_specific_kwargs: dict | None = None,
):
    """
    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        scheduler_specific_kwargs (`dict`, *optional*):
            Extra parameters for schedulers such as cosine with restarts. Mismatched scheduler types and scheduler
            parameters will cause the scheduler function to raise a TypeError.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

    # If a `LayerWiseDummyOptimizer` is passed we extract the optimizer dict and
    # recursively call `get_scheduler` to get the proper schedulers on each parameter
    if optimizer is not None and isinstance(optimizer, LayerWiseDummyOptimizer):
        optimizer_dict = optimizer.optimizer_dict
        scheduler_dict = {}

        for param in optimizer_dict:
            scheduler_dict[param] = get_scheduler(
                name,
                optimizer=optimizer_dict[param],
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=scheduler_specific_kwargs,
            )

        def scheduler_hook(param):
            # Since the optimizer hook has been already attached we only need to
            # attach the scheduler hook, the gradients have been zeroed here
            scheduler_dict[param].step()

        for param in optimizer_dict:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(scheduler_hook)

        return LayerWiseDummyScheduler(optimizer_dict=optimizer_dict, lr=optimizer.defaults["lr"])

    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    if scheduler_specific_kwargs is None:
        scheduler_specific_kwargs = {}

    if name == SchedulerType.REDUCE_ON_PLATEAU:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    if name == SchedulerType.GREEDY:
        return schedule_func(optimizer, **scheduler_specific_kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    if name == SchedulerType.INVERSE_SQRT:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **scheduler_specific_kwargs)

    # wsd scheduler requires either num_training_steps or num_stable_steps
    if name == SchedulerType.WARMUP_STABLE_DECAY:
        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            **scheduler_specific_kwargs,
        )

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        **scheduler_specific_kwargs,
    )


class Adafactor(Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam original fairseq code:
    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost* https://huggingface.co/papers/1804.04235 Note that
    this optimizer internally adjusts the learning rate depending on the `scale_parameter`, `relative_step` and
    `warmup_init` options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and
    `relative_step=False`.

    Arguments:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*):
            The external learning rate.
        eps (`tuple[float, float]`, *optional*, defaults to `(1e-30, 0.001)`):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (`float`, *optional*, defaults to 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (`float`, *optional*, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (`float`, *optional*):
            Coefficient used for computing running averages of gradient
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay (L2 penalty)
        scale_parameter (`bool`, *optional*, defaults to `True`):
            If True, learning rate is scaled by root mean square
        relative_step (`bool`, *optional*, defaults to `True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (`bool`, *optional*, defaults to `False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings (https://discuss.huggingface.co/t/t5-finetuning-tips/684/3):

        - Training without LR warmup or clip_threshold is not recommended.

           - use scheduled LR warm-up to fixed LR
           - use clip_threshold=1.0 (https://huggingface.co/papers/1804.04235)
        - Disable relative updates
        - Use scale_parameter=False
        - Additional optimizer operations like gradient clipping should not be used alongside Adafactor

    Example:

    ```python
    Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=1e-3)
    ```

    Others reported the following combination to work well:

    ```python
    Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    ```

    When using `lr=None` with [`Trainer`] you will most likely need to use [`~optimization.AdafactorSchedule`]
    scheduler as following:

    ```python
    from transformers.optimization import Adafactor, AdafactorSchedule

    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)
    trainer = Trainer(..., optimizers=(optimizer, lr_scheduler))
    ```

    Usage:

    ```python
    # replace AdamW with Adafactor
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    ```"""

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` and `relative_step=True` options")
        if warmup_init and not relative_step:
            raise ValueError("`warmup_init=True` requires `relative_step=True`")

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
        }
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    @staticmethod
    def _get_options(param_group, param_shape):
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=(1.0 - beta2t))
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=(1.0 - beta2t))

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=(1.0 - beta2t))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=(1 - group["beta1"]))
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=(-group["weight_decay"] * lr))

                p_data_fp32.add_(-update)

                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)

        return loss


class AdafactorSchedule(LambdaLR):
    """
    Since [`~optimization.Adafactor`] performs its own scheduling, if the training loop relies on a scheduler (e.g.,
    for logging), this class creates a proxy object that retrieves the current lr values from the optimizer.

    It returns `initial_lr` during startup and the actual `lr` during stepping.
    """

    def __init__(self, optimizer, initial_lr=0.0):
        def lr_lambda(_):
            return initial_lr

        for group in optimizer.param_groups:
            group["initial_lr"] = initial_lr
        super().__init__(optimizer, lr_lambda)
        for group in optimizer.param_groups:
            del group["initial_lr"]

    def get_lr(self):
        opt = self.optimizer
        lrs = [
            opt._get_lr(group, opt.state[group["params"][0]])
            for group in opt.param_groups
            if group["params"][0].grad is not None
        ]
        if len(lrs) == 0:
            lrs = self.base_lrs  # if called before stepping
        return lrs


def get_adafactor_schedule(optimizer, initial_lr=0.0):
    """
    Get a proxy schedule for [`~optimization.Adafactor`]

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        initial_lr (`float`, *optional*, defaults to 0.0):
            Initial lr

    Return:
        [`~optimization.Adafactor`] proxy schedule object.


    """
    return AdafactorSchedule(optimizer, initial_lr)
