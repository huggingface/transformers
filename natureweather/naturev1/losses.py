# Copyright 2026 Nathan. Apache-2.0.
"""
Losses that train the uncertainty, not just the answer.

Every head here is probabilistic, so every loss is a likelihood. That is deliberate: a model trained on
mean squared error will happily report a confident forecast it has no right to, because nothing in the
objective ever charged it for being wrong *and* sure. A likelihood does. When the model widens its
error bars it pays a fixed cost; when it narrows them it pays for every miss. The calibration is learned
rather than bolted on afterwards.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor,
                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """Negative log-likelihood of a diagonal Gaussian -- the honest version of MSE."""
    loss = 0.5 * (log_var + (target - mean) ** 2 / log_var.exp() + math.log(2 * math.pi))
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def track_mixture_nll(
    mode_logits: torch.Tensor,
    displacement: torch.Tensor,
    log_scale: torch.Tensor,
    correlation: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Likelihood of the observed track under a mixture of correlated 2-D Gaussians.

    This is what makes the scenarios real. Averaging several plausible tracks into one line would score
    terribly here, because the average of "recurves out to sea" and "hits the coast" is a track through
    somewhere the storm was never going. The mixture lets the model keep both branches and pay only for
    how well the *best-supported* one matched, which is how a forecaster thinks and how an ensemble
    behaves.

    Args:
        mode_logits: ``(B, M)`` unnormalized scenario weights.
        displacement: ``(B, M, L, 2)`` predicted offset from the current centre, degrees.
        log_scale: ``(B, M, L, 2)`` log standard deviations.
        correlation: ``(B, M, L)`` correlation in ``(-1, 1)``, letting the ellipse tilt.
        target: ``(B, L, 2)`` observed offsets.
        valid: ``(B, L)`` which lead times actually have a verification.

    Returns:
        Scalar loss.
    """
    target = target.unsqueeze(1)
    scale = log_scale.exp().clamp_min(1e-6)
    rho = correlation.clamp(-0.99, 0.99)

    z = (target - displacement) / scale
    zx, zy = z[..., 0], z[..., 1]
    one_minus = 1.0 - rho**2
    quadratic = (zx**2 - 2 * rho * zx * zy + zy**2) / one_minus
    log_det = 2 * log_scale.sum(-1) + torch.log(one_minus)
    log_prob = -0.5 * (quadratic + log_det + 2 * math.log(2 * math.pi))

    if valid is not None:
        log_prob = log_prob * valid.unsqueeze(1)
    # Sum over lead times inside each mode: a scenario is a whole trajectory, judged as one.
    per_mode = log_prob.sum(-1)
    return -(torch.logsumexp(F.log_softmax(mode_logits, -1) + per_mode, dim=-1)).mean()


def landfall_loss(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
                  positive_weight: float = 3.0) -> torch.Tensor:
    """
    Binary cross-entropy for "does it make landfall by this lead time".

    Landfall is rarer than not-landfall in any sampled window, and the asymmetry of being wrong is worse
    in one direction, so positives carry extra weight. Keep this modest: pushing it high buys recall by
    making the model cry wolf, and a forecast nobody trusts protects nobody.
    """
    weight = torch.where(target > 0.5, positive_weight, 1.0)
    loss = F.binary_cross_entropy_with_logits(logit, target, weight=weight, reduction="none")
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def weather_type_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None,
                      smoothing: float = 0.05) -> torch.Tensor:
    """Cross-entropy over weather categories, lightly smoothed because the class boundaries are fuzzy."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_target, reduction="none", label_smoothing=smoothing)
    if mask is not None:
        flat_mask = mask.reshape(-1).to(loss.dtype)
        return (loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
    return loss.mean()


def focal_bce(
    logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Focal binary cross-entropy, for a target that is almost always zero.

    Rapid intensification fires on about 4% of eligible track points, so plain cross-entropy is dominated
    by the easy negatives and the model learns to say "no" forever. Focal loss down-weights examples it
    already gets right by ``(1 - p)^gamma``, so the gradient keeps coming from the hard and the rare;
    ``alpha`` tilts the remaining weight toward positives.

    Accuracy is not the metric for this head. Precision, recall and Brier score are.
    """
    probability = torch.sigmoid(logits)
    p_t = probability * target + (1 - probability) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).clamp_min(1e-6) ** gamma * F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def masked_gaussian_nll(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Gaussian likelihood that simply skips missing targets, taking NaN as "not observed".

    Radius of maximum wind is reported on under 5% of best-track points and wind radii only since 2004.
    Treating those gaps as zeros would teach the model that storms routinely have no eyewall; masking
    them means the head learns from the points that were actually measured and stays silent elsewhere.
    """
    observed = torch.isfinite(target)
    if not bool(observed.any()):
        return torch.zeros((), device=mean.device, dtype=mean.dtype)
    safe = torch.where(observed, target, torch.zeros_like(mean))
    loss = 0.5 * (log_var + (safe - mean) ** 2 / log_var.exp() + math.log(2 * math.pi))
    return (loss * observed).sum() / observed.sum().clamp_min(1)


#: Per-field loss weights. Pressure and wind carry the storm's structure, so they lead; precipitation is
#: noisy and heavy-tailed, so it is damped to stop it dominating the gradient.
FIELD_WEIGHTS = {
    "t2m": 1.0, "mslp": 1.5, "u10": 1.5, "v10": 1.5, "rh": 0.8, "precip_rate": 0.5, "cloud": 0.5,
}
#: Longer leads are harder; down-weighting them stops day 5 from drowning out day 1, which is the lead
#: everyone actually acts on.
def lead_weights(num_leads: int, decay: float = 0.9) -> torch.Tensor:
    return torch.tensor([decay**i for i in range(num_leads)], dtype=torch.float32)


def total_loss(outputs: dict, batch: dict, field_names: tuple[str, ...], weights: dict | None = None) -> tuple[torch.Tensor, dict]:
    """
    Combine every head's likelihood into one scalar, and report the parts.

    Targets are all optional: a batch with no storm in it simply skips the track and landfall terms
    rather than inventing a label, so gridded and storm-centric data can be mixed in one run.

    Returns:
        ``(loss, parts)`` where ``parts`` holds detached scalars for logging.
    """
    weights = {"field": 1.0, "type": 0.3, "track": 1.0, "landfall": 0.5, "intensity": 0.5, "enso": 0.1,
               "ri": 2.0, "ri_delta": 0.5, "eyewall": 1.0, **(weights or {})}
    device = outputs["field_mean"].device
    parts: dict[str, float] = {}
    loss = torch.zeros((), device=device)

    if "field_target" in batch:
        lead_w = lead_weights(outputs["field_mean"].shape[2]).to(device).view(1, 1, -1, 1)
        field_w = torch.tensor([FIELD_WEIGHTS.get(n, 1.0) for n in field_names], device=device).view(1, 1, 1, -1)
        per = 0.5 * (
            outputs["field_log_var"]
            + (batch["field_target"] - outputs["field_mean"]) ** 2 / outputs["field_log_var"].exp()
            + math.log(2 * math.pi)
        )
        mask = batch.get("field_mask")
        combined = per * lead_w * field_w
        term = (combined * mask).sum() / mask.sum().clamp_min(1.0) if mask is not None else combined.mean()
        loss = loss + weights["field"] * term
        parts["field"] = float(term.detach())

    if "weather_type_target" in batch:
        term = weather_type_loss(outputs["weather_type_logits"], batch["weather_type_target"], batch.get("weather_type_mask"))
        loss = loss + weights["type"] * term
        parts["weather_type"] = float(term.detach())

    if "track_target" in batch:
        term = track_mixture_nll(
            outputs["mode_logits"], outputs["displacement"], outputs["log_scale"],
            outputs["correlation"], batch["track_target"], batch.get("track_valid"),
        )
        loss = loss + weights["track"] * term
        parts["track"] = float(term.detach())

    if "landfall_target" in batch:
        term = landfall_loss(outputs["landfall_logit"], batch["landfall_target"], batch.get("landfall_mask"))
        loss = loss + weights["landfall"] * term
        parts["landfall"] = float(term.detach())

    if "intensity_target" in batch:
        term = gaussian_nll(outputs["intensity_mean"], outputs["intensity_log_var"],
                            batch["intensity_target"], batch.get("intensity_mask"))
        loss = loss + weights["intensity"] * term
        parts["intensity"] = float(term.detach())

    if "ri_target" in batch:
        # Classification over thresholds, plus the denser regression on the actual 24-hour change, which
        # keeps the representation pointed at the physics rather than at the cut point.
        term = focal_bce(outputs["ri_logits"], batch["ri_target"], mask=batch.get("ri_mask"))
        loss = loss + weights["ri"] * term
        parts["ri"] = float(term.detach())
        if "ri_delta_target" in batch:
            delta = masked_gaussian_nll(
                outputs["ri_delta_wind_kt"], outputs["ri_delta_log_var"], batch["ri_delta_target"]
            )
            loss = loss + weights["ri_delta"] * delta
            parts["ri_delta"] = float(delta.detach())

    if "eyewall_target" in batch:
        # Peak wind is well observed; RMW is not; wind radii sit in between. All three are masked, so
        # each contributes exactly where it was measured.
        peak = masked_gaussian_nll(
            outputs["eyewall_peak_wind_kt"], outputs["eyewall_peak_wind_log_var"], batch["eyewall_target"]
        )
        term = peak
        parts["eyewall_peak"] = float(peak.detach())
        if "rmw_target" in batch:
            rmw = masked_gaussian_nll(
                outputs["eyewall_rmw_nmi"], outputs["eyewall_rmw_log_var"], batch["rmw_target"]
            )
            term = term + rmw
            parts["eyewall_rmw"] = float(rmw.detach())
        if "wind_radii_target" in batch:
            radii = masked_gaussian_nll(
                outputs["wind_radii_nmi"], outputs["wind_radii_log_var"], batch["wind_radii_target"]
            )
            term = term + radii
            parts["wind_radii"] = float(radii.detach())
        loss = loss + weights["eyewall"] * term

    if "enso_target" in batch:
        term = gaussian_nll(outputs["enso_mean"], outputs["enso_log_var"], batch["enso_target"])
        loss = loss + weights["enso"] * term
        parts["enso"] = float(term.detach())

    parts["total"] = float(loss.detach())
    return loss, parts
