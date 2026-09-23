# Copyright 2026 Nathan. Apache-2.0.
"""
Autoregressive rollout: training the model on its own mistakes.

A model trained only to predict t+6h from observations has never seen its own output as input. At
inference it is then fed exactly that, nine times in a row to reach +120 h, and each small error becomes
the next step's starting condition. Errors compound, the forecast drifts toward something smooth and
wrong, and the failure appears only at long lead -- where the single-step loss curve said nothing.

DeepMind's answer, and the reason GraphCast holds up over ten days, is to close the loop during
training: predict, feed the prediction back in, predict again, and take the loss over the whole
trajectory. The model is then optimised for the distribution it will actually encounter rather than the
one it was handed. They ramp this to 12 autoregressive steps.

Two details decide whether this works.

**Feeding a prediction back is not free.** The model consumes ``analysis_channels`` and emits seven
surface fields, so the output is not shaped like the input. :func:`reinject` maps each predicted field
back into the channel it came from and **carries the rest forward unchanged** -- sea-surface temperature
moves on a timescale of weeks, land-sea mask never, incoming solar radiation is exactly known in advance.
Predicting those would be inventing work; freezing them is the physically honest choice, and it is what
operational models do with boundary conditions.

**Ramping matters more than the final number.** Starting at 12 steps from scratch trains on garbage: the
model's early outputs are noise, so steps 2 through 12 are noise-to-noise and the gradient is worthless.
:class:`RolloutSchedule` grows the horizon as training proceeds, which is DeepMind's curriculum and the
difference between this helping and hurting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .losses import masked_gaussian_nll


@dataclass
class RolloutSchedule:
    """
    How far to roll out, as a function of training step.

    Args:
        start_step: optimizer step at which rollout begins. Before it, training is single-step, because
            feeding back an untrained model's output teaches it to predict its own noise.
        ramp_steps: steps over which the horizon grows from 1 to ``max_steps``.
        max_steps: longest trajectory to train on. GraphCast ramps to 12.
        discount: weight on each successive step. Below 1 it leans on the near term, which is where the
            signal is cleanest; at 1.0 every step counts equally.
    """

    start_step: int = 2000
    ramp_steps: int = 8000
    max_steps: int = 12
    discount: float = 0.9

    def horizon(self, step: int) -> int:
        """Rollout length at this step: 1 until ``start_step``, then growing to ``max_steps``."""
        if step < self.start_step:
            return 1
        progress = (step - self.start_step) / max(self.ramp_steps, 1)
        return int(min(self.max_steps, 1 + progress * (self.max_steps - 1)))

    def weights(self, horizon: int, device=None) -> torch.Tensor:
        """Per-step loss weights, normalised so the total does not grow with the horizon."""
        powers = torch.arange(horizon, dtype=torch.float32, device=device)
        weights = self.discount**powers
        return weights / weights.sum()


def field_to_channel(variables: list[str], target_index: list[int]) -> dict[int, int]:
    """
    Invert the dataset's field map: which *input channel* each predicted surface field writes back to.

    :func:`naturev1.era5._target_index` says, for each of the model's surface fields, which input channel
    supervises it. Rolling out needs the other direction -- given a prediction, where does it go. Fields
    the dataset does not carry are simply absent, and are therefore carried forward rather than replaced.
    """
    return {field: channel for field, channel in enumerate(target_index) if channel >= 0}


def reinject(history: torch.Tensor, prediction: torch.Tensor, writeback: dict[int, int]) -> torch.Tensor:
    """
    Advance the input window by one step, substituting the model's own forecast for the newest frame.

    The forecast must be on the same points as the input -- it is being fed back in as input. Decoding
    onto a different mesh is a legitimate thing to do for *output*, but not for a rollout, so that is
    checked here rather than surfacing later as a broadcast error.

    Args:
        history: ``(B, T, P, C)`` the current input window.
        prediction: ``(B, P, F)`` predicted surface fields for the next step.
        writeback: field index -> input channel, from :func:`field_to_channel`.

    Returns:
        ``(B, T, P, C)`` with the oldest frame dropped and a new frame appended. The new frame starts as
        a copy of the last real one, so every channel the model does not predict -- sea-surface
        temperature, land-sea mask, solar radiation, the observation masks -- carries forward unchanged
        instead of being zeroed or invented.
    """
    if prediction.shape[1] != history.shape[2]:
        raise ValueError(
            f"Rollout needs the forecast on the same points as the input: got {prediction.shape[1]} "
            f"predicted points against {history.shape[2]} input points. Pass output_grid=analysis_grid "
            "(or leave it unset) -- a prediction decoded onto a different mesh cannot be fed back in."
        )
    latest = history[:, -1].clone()
    for field, channel in writeback.items():
        latest[:, :, channel] = prediction[:, :, field]
    return torch.cat([history[:, 1:], latest.unsqueeze(1)], dim=1)


#: Seconds per day and per year, matching :func:`naturev1.calendar_features` exactly. Using 365.2425
#: days here instead of the 365.25 that function uses would put the rollout clock a few minutes out per
#: step -- invisible for one step, a real phase error by step twelve.
DAY_SECONDS = 86_400.0
YEAR_SECONDS = 31_557_600.0
EPOCH_SECONDS = 3.15576e9


def advance_calendar(calendar: torch.Tensor, step_hours: float = 6.0) -> torch.Tensor:
    """
    Move the cyclic time features forward one step.

    :func:`naturev1.calendar_features` encodes hour-of-day and day-of-year as sine/cosine pairs, plus a
    slow linear epoch term. Rotating each pair by the right angle advances the clock *exactly*, with no
    timestamp needed -- which matters because a rollout is inventing steps that have no timestamp.

    The rotation is the angle-addition identity, so this is not an approximation: after four six-hour
    steps the day pair returns to precisely where it started.
    """
    advanced = calendar.clone()
    seconds = step_hours * 3600.0

    for offset, period in ((0, DAY_SECONDS), (2, YEAR_SECONDS)):
        if calendar.shape[-1] < offset + 2:
            continue
        angle = 2 * math.pi * seconds / period
        cos_a, sin_a = math.cos(angle), math.sin(angle)      # plain floats: no device to get wrong
        sin, cos = advanced[..., offset].clone(), advanced[..., offset + 1].clone()
        advanced[..., offset] = sin * cos_a + cos * sin_a
        advanced[..., offset + 1] = cos * cos_a - sin * sin_a

    if calendar.shape[-1] >= 5:
        advanced[..., 4] = advanced[..., 4] + seconds / EPOCH_SECONDS
    return advanced


def rollout_loss(
    model,
    batch: dict,
    horizon: int,
    schedule: RolloutSchedule,
    writeback: dict[int, int],
    analysis_grid,
    output_grid=None,
    step_hours: float = 6.0,
    lead_index: int = 0,
    accumulate: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Loss over a whole trajectory, with the model consuming its own forecasts.

    Each step predicts the ``lead_index``-th lead (the shortest, normally +6 h), is scored against the
    trajectory's truth at that horizon, and is then fed back in. Targets come from ``field_target``'s
    lead axis, so a window built with :func:`naturev1.lead_offsets` already carries everything needed --
    no new dataset.

    Args:
        accumulate: call ``backward()`` on each step as it is computed, freeing that step's graph before
            the next forward. **This is not an optimisation, it is what makes long rollouts possible.**
            Holding twelve forward graphs at once costs twelve times the activation memory of a normal
            step -- on a 96 GB card at batch 16 that is roughly 115 GB, and the process is killed. Since
            the trajectory is detached between steps (below), each step's gradient is independent, so
            per-step backward is *mathematically identical* to summing and calling backward once. Set
            ``False`` only for a short horizon where you want the graph kept, e.g. in a test.

    Returns:
        ``(loss, parts)``. With ``accumulate``, the loss is already backpropagated and comes back
        detached -- clip and step as usual, do not call ``backward()`` on it again. Per-step losses are
        in ``parts``, so drift is visible while training rather than only at evaluation.
    """
    output_grid = output_grid if output_grid is not None else analysis_grid
    history = batch["analysis"]
    calendar = batch["calendar"]
    target = batch["field_target"]                # (B, P, leads, fields)
    mask = batch.get("field_mask")

    horizon = max(1, min(horizon, target.shape[2]))
    weights = schedule.weights(horizon, device=history.device)
    total = torch.zeros((), device=history.device)
    parts: dict[str, float] = {}

    for step in range(horizon):
        outputs = model(analysis=history, analysis_grid=analysis_grid, calendar=calendar,
                        output_grid=output_grid)
        predicted = outputs["field_mean"][:, :, lead_index]
        log_var = outputs["field_log_var"][:, :, lead_index]

        truth = target[:, :, step]
        if mask is not None:
            truth = torch.where(mask[:, :, step] > 0, truth, torch.full_like(truth, float("nan")))
        step_loss = masked_gaussian_nll(predicted, log_var, truth)
        weighted = weights[step] * step_loss
        if accumulate:
            weighted.backward()
            total = total + weighted.detach()
        else:
            total = total + weighted
        parts[f"rollout_{step + 1}"] = float(step_loss.detach())

        if step + 1 < horizon:
            # Detaching between steps trains each prediction against its own input without
            # backpropagating through the whole trajectory, which would cost memory proportional to the
            # horizon and is not what makes this work -- seeing its own errors is.
            history = reinject(history, predicted.detach(), writeback)
            calendar = advance_calendar(calendar, step_hours)

    parts["rollout_horizon"] = float(horizon)
    parts["rollout_drift"] = parts.get(f"rollout_{horizon}", 0.0) - parts.get("rollout_1", 0.0)
    return total, parts


@torch.no_grad()
def rollout_forecast(
    model,
    history: torch.Tensor,
    calendar: torch.Tensor,
    analysis_grid,
    writeback: dict[int, int],
    steps: int = 20,
    output_grid=None,
    step_hours: float = 6.0,
    lead_index: int = 0,
) -> list[torch.Tensor]:
    """
    Run the model forward autoregressively at inference: 20 steps of 6 h is a five-day forecast.

    This is the free-running mode a rollout-trained model is meant for, and it is also the honest test of
    whether the training worked -- drift shows up here and nowhere else.

    Returns:
        One ``(B, P, fields)`` state per step.
    """
    output_grid = output_grid if output_grid is not None else analysis_grid
    was_training = model.training
    model.eval()
    trajectory = []

    for step in range(steps):
        outputs = model(analysis=history, analysis_grid=analysis_grid, calendar=calendar,
                        output_grid=output_grid)
        predicted = outputs["field_mean"][:, :, lead_index]
        trajectory.append(predicted)
        if step + 1 < steps:
            history = reinject(history, predicted, writeback)
            calendar = advance_calendar(calendar, step_hours)

    if was_training:
        model.train()
    return trajectory


def drift_report(trajectory: list[torch.Tensor], truth: torch.Tensor | None = None) -> str:
    """
    Does a free-running forecast stay alive, or collapse toward a smooth mean.

    The characteristic failure of a model never trained on its own output is not divergence -- it is the
    opposite. Variance decays, sharp features smear, and by day five it is forecasting a pleasant fog.
    Tracking per-step spatial standard deviation catches that directly, and no single-step loss will.
    """
    lines = [f"{'step':>5} {'spread':>10} {'vs step 1':>10}" + ("" if truth is None else f" {'RMSE':>10}")]
    first = float(trajectory[0].std())
    for step, state in enumerate(trajectory, start=1):
        spread = float(state.std())
        row = f"{step:>5} {spread:>10.4f} {spread / max(first, 1e-9):>9.1%}"
        if truth is not None and step <= truth.shape[2]:
            row += f" {float(((state - truth[:, :, step - 1]) ** 2).mean().sqrt()):>10.4f}"
        lines.append(row)
    lines.append("")
    lines.append("Spread collapsing toward zero is the signature of a model that was never trained on")
    lines.append("its own output: it hedges, the field smooths, and day five is a forecast of nothing.")
    return "\n".join(lines)
