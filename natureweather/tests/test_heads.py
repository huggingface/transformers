# Copyright 2026 Nathan. Apache-2.0.
"""
Regression tests for how the prediction heads start.

An untrained head that reports a physical quantity has two things to get right before it has learned
anything: where it points, and how sure it claims to be. Getting either wrong is not a slow start, it is
a loss term that drowns out every other head. These numbers were measured on a real fine-tuning step:

    before anchoring   total 60,565   intensity 119,191   eyewall 718   grad norm 207,018
    after  anchoring   total     58   intensity       4   eyewall   5   grad norm      13

Nothing about the architecture changed. The head was simply started at Atlantic climatology instead of
at zero, with the uncertainty that honestly goes with it.
"""

from __future__ import annotations

import math

import pytest
import torch
from naturev1 import NatureConfig, NatureV1
from naturev1.model import (
    PEAK_WIND_ANCHOR_KT,
    PRESSURE_ANCHOR_HPA,
    RI_DELTA_ANCHOR_KT,
    RMW_ANCHOR_NMI,
    TRACK_SPEED_DEG_PER_HOUR,
    WIND_ANCHOR_MS,
)

from ihelix import fibonacci_sphere


@pytest.fixture(scope="module")
def model():
    """A small model -- the heads are what is under test, and they do not depend on the backbone."""
    config = NatureConfig(latent_points=256, num_layers=2, hidden_size=128)
    mesh = fibonacci_sphere(256, num_neighbours=16, cluster_size=32)
    return NatureV1(config, mesh)


@pytest.fixture(scope="module")
def climatology(model):
    """What the heads say when the summary carries no information -- the model's prior."""
    summary = torch.zeros(1, model.config.hidden_size)
    with torch.no_grad():
        return {**model.eyewall_head(summary), **model.ri_head(summary), **model.track_head(summary)}


def test_eyewall_starts_at_atlantic_climatology(climatology):
    assert float(climatology["eyewall_peak_wind_kt"][0, 0]) == pytest.approx(PEAK_WIND_ANCHOR_KT[0], abs=1e-4)
    # Softplus and the +1 floor shift the radius slightly; it must still be in the right neighbourhood.
    assert RMW_ANCHOR_NMI[0] <= float(climatology["eyewall_rmw_nmi"][0, 0]) <= RMW_ANCHOR_NMI[0] + 2.0


def test_ri_change_starts_at_climatology(climatology):
    assert float(climatology["ri_delta_wind_kt"][0]) == pytest.approx(RI_DELTA_ANCHOR_KT[0], abs=1e-4)


def test_intensity_reports_physical_units_not_normalized_ones(model):
    """
    ``naturev1.forecast.decode_intensity`` reads m/s and absolute hPa. The anchoring changed the
    parameterization, and must not have changed that contract.
    """
    summary = torch.zeros(1, model.config.hidden_size)
    with torch.no_grad():
        intensity = model.intensity_head(summary).view(1, model.config.num_leads, 2, 2)
        anchor = torch.tensor([WIND_ANCHOR_MS[0], PRESSURE_ANCHOR_HPA[0]])
        scale = torch.tensor([WIND_ANCHOR_MS[1], PRESSURE_ANCHOR_HPA[1]])
        mean = intensity[..., 0] * scale + anchor

    assert 20.0 < float(mean[0, 0, 0]) < 60.0, "wind should open near 33 m/s, not near zero"
    assert 950.0 < float(mean[0, 0, 1]) < 1020.0, "pressure should open near 985 hPa, not near zero"


def test_uncertainty_starts_at_the_observed_spread(model, climatology):
    """
    A log-variance left at zero claims one knot of uncertainty about peak wind.

    That is what drove the eyewall term to 438 on the first step: the model was certain and wrong, and a
    Gaussian likelihood charges for exactly that.
    """
    sigma = float(climatology["eyewall_peak_wind_log_var"][0, 0].mul(0.5).exp())
    assert sigma == pytest.approx(PEAK_WIND_ANCHOR_KT[1], rel=0.05)
    assert sigma > 5.0, "an untrained head must not claim knife-edge certainty"

    rmw_sigma = float(climatology["eyewall_rmw_log_var"][0, 0].mul(0.5).exp())
    assert rmw_sigma == pytest.approx(RMW_ANCHOR_NMI[1], rel=0.05)

    delta_sigma = float(climatology["ri_delta_log_var"][0].mul(0.5).exp())
    assert delta_sigma == pytest.approx(RI_DELTA_ANCHOR_KT[1], rel=0.05)


def test_track_uncertainty_grows_with_lead_time(model, climatology):
    """
    A constant spread is wrong at both ends: overconfident at +120 h and slack at +6 h.

    The mean starts at zero displacement -- the storm is where it is -- so the spread that belongs with
    that prior is how far a storm typically travels by then.
    """
    scale = climatology["log_scale"][0, 0].exp()
    for lead, hours in enumerate(model.config.lead_times_hours):
        expected = TRACK_SPEED_DEG_PER_HOUR * hours
        assert float(scale[lead, 0]) == pytest.approx(expected, rel=0.02), f"+{hours}h latitude spread"
        assert float(scale[lead, 1]) == pytest.approx(expected, rel=0.02), f"+{hours}h longitude spread"

    assert scale[-1, 0] > scale[0, 0] * 5, "+120 h must be far less certain than +6 h"


def test_anchoring_touches_only_biases(model):
    """The weights still decide how a prediction varies with the input; only the starting point moved."""
    for head in (model.intensity_head, model.eyewall_head.peak_wind, model.ri_head.magnitude):
        assert head.weight.abs().max() < 1.0, "weights should still be the small random init"
        assert head.weight.std() > 0.0


def test_climatological_loss_is_sane(model):
    """
    The whole point, as one number: a Gaussian NLL against a real storm must open near single digits.

    A 100 kt storm against a head that starts at 65 +/- 30 kt costs about 4.8. Against one that starts
    at 0 +/- 1 kt it costs 5,000, and the eyewall term alone then dominates every other head.
    """
    from naturev1.losses import masked_gaussian_nll

    summary = torch.zeros(2, model.config.hidden_size)
    with torch.no_grad():
        out = model.eyewall_head(summary)
    target = torch.full_like(out["eyewall_peak_wind_kt"], 100.0)
    loss = masked_gaussian_nll(out["eyewall_peak_wind_kt"], out["eyewall_peak_wind_log_var"], target)
    assert float(loss) < 10.0, f"opening loss {float(loss):.1f} is too large to train alongside other heads"

    # And the same target against the unanchored parameterization, to show the gap is real.
    naive_mean = torch.zeros_like(target)
    naive_log_var = torch.zeros_like(target)
    naive = masked_gaussian_nll(naive_mean, naive_log_var, target)
    assert float(naive) > 100 * float(loss)
    assert float(naive) == pytest.approx(0.5 * (100.0**2 + math.log(2 * math.pi)), rel=1e-3)


def test_naturev1_carries_its_grids_to_the_device():
    """
    The bug a user hit in Colab: ``NatureV1(...).to("cuda")`` moved the weights and not the mesh.

    It surfaced as "mat1 is on cpu, different from other tensors on cuda:0" from inside a linear layer
    in RelativeEncoder -- four frames deep and with no mention of a grid anywhere in the traceback.
    """
    config = NatureConfig(latent_points=128, num_layers=2, hidden_size=96)
    model = NatureV1(config, fibonacci_sphere(128, num_neighbours=12, cluster_size=32)).to("meta")

    assert model.latent_grid.points.device.type == "meta"
    assert model.latent_grid.neighbour_offsets.device.type == "meta"

    supplied = fibonacci_sphere(64, num_neighbours=8, cluster_size=16)
    link = model.link(model.latent_grid, supplied, 8)
    assert link.offsets.device.type == "meta"
    assert link.alignment.device.type == "meta"
