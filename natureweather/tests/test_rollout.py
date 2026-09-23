# Copyright 2026 Nathan. Apache-2.0.
"""
Regression tests for autoregressive rollout.

A model trained only on observations has never seen its own output as input, and at inference that is
all it gets. Errors compound, the field smooths, and the failure appears only at long lead -- where the
single-step loss curve said nothing. Closing the loop during training is what GraphCast does and why it
holds up over ten days.

The subtle parts, and the ones tested here: a prediction is not shaped like an input, the clock has to
advance without a timestamp, and holding every step's graph at once runs a 96 GB card out of memory.
"""

from __future__ import annotations

import datetime as dt

import pytest
import torch
from naturev1 import (
    NatureConfig,
    NatureV1,
    RolloutSchedule,
    advance_calendar,
    calendar_features,
    field_to_channel,
    reinject,
    rollout_forecast,
    rollout_loss,
)
from naturev1.model import SURFACE_FIELDS

from ihelix import fibonacci_sphere


@pytest.fixture(scope="module")
def tiny():
    config = NatureConfig(latent_points=96, num_layers=2, hidden_size=64, history_frames=2,
                          lead_times_hours=(6, 12, 18))
    mesh = fibonacci_sphere(96, num_neighbours=8, cluster_size=24)
    grid = fibonacci_sphere(48, num_neighbours=8, cluster_size=16)
    return NatureV1(config, mesh), config, grid


def make_batch(config, grid, batch_size=1):
    leads = config.num_leads
    points = grid.num_points
    return {
        "analysis": torch.randn(batch_size, config.history_frames, points, config.analysis_channels),
        "calendar": calendar_features(torch.full((batch_size, config.history_frames), 1.7e9)),
        "field_target": torch.randn(batch_size, points, leads, len(SURFACE_FIELDS)),
        "field_mask": torch.ones(batch_size, points, leads, len(SURFACE_FIELDS)),
    }


# --------------------------------------------------------------------------------------------------
# Feeding a forecast back in
# --------------------------------------------------------------------------------------------------

def test_reinject_replaces_predicted_channels_and_keeps_the_rest():
    """
    Sea-surface temperature moves over weeks, a land mask never, solar radiation is known exactly.

    Predicting those would be inventing work. Freezing them is what operational models do with boundary
    conditions, and it is the difference between a rollout and a slow corruption of the input.
    """
    history = torch.randn(2, 3, 16, 10)
    prediction = torch.full((2, 16, 7), -99.0)
    writeback = {0: 0, 1: 3, 5: 7}

    rolled = reinject(history, prediction, writeback)

    assert rolled.shape == history.shape
    assert torch.equal(rolled[:, :-1], history[:, 1:]), "the window must shift, not shuffle"
    for channel in writeback.values():
        assert torch.allclose(rolled[:, -1, :, channel], torch.full((2, 16), -99.0))
    for channel in set(range(10)) - set(writeback.values()):
        assert torch.equal(rolled[:, -1, :, channel], history[:, -1, :, channel]), \
            f"channel {channel} is not predicted and must carry forward untouched"


def test_reinject_refuses_a_mismatched_mesh():
    """Decoding onto a different grid is fine for output and impossible for a rollout."""
    with pytest.raises(ValueError, match="same points"):
        reinject(torch.randn(1, 2, 16, 8), torch.randn(1, 9, 7), {0: 0})


def test_field_to_channel_inverts_the_dataset_map():
    """Unsupervised fields have no channel to write back to, and must simply be absent."""
    # t2m<-0, mslp<-1, u10 absent, v10<-3
    writeback = field_to_channel(["a", "b", "c", "d"], [0, 1, -1, 3, -1, -1, -1])
    assert writeback == {0: 0, 1: 1, 3: 3}


# --------------------------------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------------------------------

def test_calendar_advance_matches_a_real_timestamp():
    """A rollout invents steps that have no timestamp, so the clock is rotated rather than recomputed."""
    base = dt.datetime(2024, 7, 1, tzinfo=dt.timezone.utc)
    calendar = calendar_features(torch.tensor([base.timestamp()], dtype=torch.float64))

    rotated = calendar
    for step in range(1, 5):
        rotated = advance_calendar(rotated, 6.0)
        truth = calendar_features(
            torch.tensor([(base + dt.timedelta(hours=6 * step)).timestamp()], dtype=torch.float64)
        )
        assert float((rotated - truth).abs().max()) < 1e-6, f"clock drifted by step {step}"


def test_four_six_hour_steps_close_the_day():
    """The angle-addition identity is exact, so a full day must return the day pair to its start."""
    calendar = calendar_features(torch.tensor([1.7e9], dtype=torch.float64))
    rotated = calendar
    for _ in range(4):
        rotated = advance_calendar(rotated, 6.0)
    assert float((rotated[:, :2] - calendar[:, :2]).abs().max()) < 1e-6


def test_the_slow_epoch_term_advances_too():
    """
    The trend term is a running count, not a cycle, so it has to be added rather than rotated.

    Tolerance is loose on purpose: calendar_features returns float32 and the term sits near 0.54, so a
    day's increment of 2.7e-5 is close to what single precision can resolve there. Checking the
    magnitude is the honest test; demanding more would be testing float32, not this code.
    """
    calendar = calendar_features(torch.tensor([1.7e9], dtype=torch.float64))
    rotated = advance_calendar(calendar, 24.0)
    assert float(rotated[0, 4]) > float(calendar[0, 4])
    assert float(rotated[0, 4] - calendar[0, 4]) == pytest.approx(86_400.0 / 3.15576e9, rel=1e-2)


# --------------------------------------------------------------------------------------------------
# The schedule
# --------------------------------------------------------------------------------------------------

def test_the_horizon_ramps_rather_than_starting_long():
    """
    Twelve steps from scratch trains on noise: the model's early outputs are noise, so steps 2 through
    12 are noise-fed-noise and the gradient is worthless.
    """
    schedule = RolloutSchedule(start_step=100, ramp_steps=400, max_steps=12)
    assert schedule.horizon(0) == 1
    assert schedule.horizon(99) == 1
    assert schedule.horizon(100) == 1
    assert 1 < schedule.horizon(300) < 12
    assert schedule.horizon(500) == 12
    assert schedule.horizon(100_000) == 12, "the horizon must not run past its maximum"


def test_step_weights_do_not_grow_with_the_horizon():
    """Otherwise a longer rollout silently raises the learning rate."""
    schedule = RolloutSchedule(discount=0.9)
    for horizon in (1, 4, 12):
        weights = schedule.weights(horizon)
        assert len(weights) == horizon
        assert float(weights.sum()) == pytest.approx(1.0)
    assert schedule.weights(4)[0] > schedule.weights(4)[-1], "a discount must favour the near term"


# --------------------------------------------------------------------------------------------------
# The loss
# --------------------------------------------------------------------------------------------------

def test_rollout_loss_trains_every_step(tiny):
    model, config, grid = tiny
    batch = make_batch(config, grid)
    model.zero_grad(set_to_none=True)

    loss, parts = rollout_loss(model, batch, horizon=3, schedule=RolloutSchedule(),
                               writeback={0: 0, 1: 1}, analysis_grid=grid, output_grid=grid)

    assert torch.isfinite(loss)
    assert {"rollout_1", "rollout_2", "rollout_3"} <= set(parts)
    assert parts["rollout_horizon"] == 3
    trained = [p for p in model.parameters() if p.grad is not None]
    assert trained, "accumulate=True should have backpropagated already"
    assert all(torch.isfinite(p.grad).all() for p in trained)


def test_accumulating_matches_one_big_backward(tiny):
    """
    Per-step backward is what makes a 12-step rollout fit -- holding every graph at once costs about
    115 GB at batch 16 on the full grid. It is only legitimate because the trajectory is detached
    between steps, which makes the two paths mathematically identical. This checks that they are.
    """
    model, config, grid = tiny
    batch = make_batch(config, grid)
    schedule = RolloutSchedule()

    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    rollout_loss(model, batch, 3, schedule, {0: 0}, grid, grid, accumulate=True)
    accumulated = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    model.zero_grad(set_to_none=True)
    torch.manual_seed(0)
    loss, _ = rollout_loss(model, batch, 3, schedule, {0: 0}, grid, grid, accumulate=False)
    loss.backward()
    at_once = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    assert set(accumulated) == set(at_once)
    for name in accumulated:
        torch.testing.assert_close(accumulated[name], at_once[name], rtol=1e-4, atol=1e-6)


def test_rollout_forecast_runs_free(tiny):
    """The mode the training is for, and the only place drift is visible."""
    model, config, grid = tiny
    batch = make_batch(config, grid)

    trajectory = rollout_forecast(model, batch["analysis"], batch["calendar"], grid,
                                  {0: 0, 1: 1}, steps=5, output_grid=grid)
    assert len(trajectory) == 5
    assert all(state.shape == (1, grid.num_points, len(SURFACE_FIELDS)) for state in trajectory)
    assert all(torch.isfinite(state).all() for state in trajectory)
