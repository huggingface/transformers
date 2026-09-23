# Copyright 2026 Nathan. Apache-2.0.
"""
WeatherBench 2 scoring: the metrics the field is actually judged on.

Everything in this package so far reports a likelihood. A likelihood tells you the model is fitting
something; it does not tell you the model is any good, because it has no scale and nothing to compare
against. The weather community settled that argument years ago with a specific set of numbers, and this
module computes them:

* **Latitude-weighted RMSE** on Z500 (500 hPa geopotential height), T850 (850 hPa temperature) and the
  10 m wind vector. Z500 is the field medium-range forecasting lives or dies on -- it is the steering
  flow -- and it is the headline number in every comparison between GraphCast, Pangu, FourCastNet and
  the ECMWF operational model.
* **ACC**, the anomaly correlation coefficient, which asks whether the forecast gets the *pattern* right
  after the seasonal cycle is removed. A model can have respectable RMSE by regressing to climatology;
  ACC is what catches that.
* **Two baselines**, scored on the identical batches: **persistence** ("six hours from now looks like
  now") and **climatology** ("this date usually looks like this"). These are not decoration. A forecast
  that cannot beat persistence at +24 h is not a forecast, and one that ties climatology has learned
  nothing from the data it was given. Any RMSE quoted without them is uninterpretable.

The weighting is not optional either. Grid cells shrink as ``cos(latitude)``, so an unweighted mean over
a latitude-longitude grid counts a 165 km equatorial cell the same as a 29 km cell at 80 degrees, and
over-credits the poles by about a factor of six. Every operational score is area-weighted; these are too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


#: The WeatherBench 2 headline fields. Z500 and T850 need pressure-level data (see :mod:`naturev1.upper`);
#: the wind vector is available from surface variables alone.
HEADLINE = ("z500", "t850", "wind10m")

#: Typical magnitudes, used only to print a score next to something human-readable. Z500 is in metres of
#: geopotential height, T850 in kelvin, wind in m/s.
UNITS = {"z500": "m", "t850": "K", "wind10m": "m/s", "t2m": "K", "mslp": "hPa", "precip_rate": "mm/6h"}


def latitude_weights(latitudes, num_longitudes: int = 1, device=None, dtype=torch.float32) -> torch.Tensor:
    """
    Area weights for an equiangular grid, normalised to mean one.

    Normalised to *mean* one rather than sum one, so a weighted RMSE comes out in the field's own units
    and is directly comparable to an unweighted one -- which is the convention every published score
    uses, and the reason a number from here can sit in a table next to GraphCast's.
    """
    radians = torch.as_tensor(np.radians(np.asarray(latitudes, dtype=np.float64)))
    weights = torch.cos(radians).clamp_min(0.0)
    weights = weights / weights.mean()
    if num_longitudes > 1:
        weights = weights.repeat_interleave(num_longitudes)
    return weights.to(device=device, dtype=dtype)


def weighted_rmse(prediction: torch.Tensor, truth: torch.Tensor, weights: torch.Tensor,
                  mask: torch.Tensor | None = None) -> float:
    """
    Latitude-weighted root-mean-square error over ``(..., points)``.

    Weights broadcast against the point axis. A mask, where given, drops unobserved points entirely
    rather than scoring the model against a fill value.
    """
    error = (prediction - truth) ** 2
    shape = [1] * (error.ndim - 1) + [weights.shape[-1]]
    spread = weights.view(shape).to(error.dtype)
    if mask is not None:
        spread = spread * mask.to(error.dtype)
    total = (error * spread).sum()
    count = spread.sum().clamp_min(1e-12)
    return float(torch.sqrt(total / count))


def weighted_acc(prediction: torch.Tensor, truth: torch.Tensor, climatology: torch.Tensor,
                 weights: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """
    Anomaly correlation: does the forecast get the *pattern* right once climatology is removed.

    RMSE alone rewards a model that hedges toward the seasonal mean, because the mean is close to
    everything. ACC subtracts that mean from both sides first, so a model that only knows the season
    scores zero here however tidy its RMSE looks. Operationally, ACC 0.6 at a given lead is the usual
    line for "still useful".
    """
    pred_anomaly = prediction - climatology
    true_anomaly = truth - climatology
    shape = [1] * (prediction.ndim - 1) + [weights.shape[-1]]
    spread = weights.view(shape).to(prediction.dtype)
    if mask is not None:
        spread = spread * mask.to(prediction.dtype)

    covariance = (spread * pred_anomaly * true_anomaly).sum()
    pred_power = (spread * pred_anomaly**2).sum()
    true_power = (spread * true_anomaly**2).sum()
    return float(covariance / torch.sqrt(pred_power * true_power).clamp_min(1e-12))


@dataclass
class Score:
    """One field at one lead time, for the model and for the baselines it has to beat."""

    field: str
    lead_hours: int
    rmse: float
    acc: float | None = None
    persistence_rmse: float | None = None
    climatology_rmse: float | None = None

    @property
    def skill_vs_persistence(self) -> float | None:
        """Fraction of persistence's error removed. Zero means no better than doing nothing."""
        if not self.persistence_rmse:
            return None
        return 1.0 - self.rmse / self.persistence_rmse

    @property
    def skill_vs_climatology(self) -> float | None:
        """Fraction of climatology's error removed. Zero means the model learned nothing from the data."""
        if not self.climatology_rmse:
            return None
        return 1.0 - self.rmse / self.climatology_rmse

    def __repr__(self) -> str:
        unit = UNITS.get(self.field, "")
        parts = [f"{self.field} +{self.lead_hours}h  RMSE {self.rmse:.3f} {unit}"]
        if self.acc is not None:
            parts.append(f"ACC {self.acc:.3f}")
        if self.skill_vs_persistence is not None:
            parts.append(f"vs persistence {self.skill_vs_persistence:+.1%}")
        if self.skill_vs_climatology is not None:
            parts.append(f"vs climatology {self.skill_vs_climatology:+.1%}")
        return "  ".join(parts)


@dataclass
class Scorecard:
    """Every field at every lead, plus the printable table that goes in a README."""

    scores: list[Score] = field(default_factory=list)

    def add(self, score: Score) -> None:
        self.scores.append(score)

    def table(self) -> str:
        """The comparison table, laid out the way published scorecards are."""
        if not self.scores:
            return "no scores"
        header = (f"{'field':12} {'lead':>6} {'RMSE':>10} {'ACC':>7} {'persist':>10} "
                  f"{'climo':>10} {'skill_p':>9} {'skill_c':>9}")
        lines = [header, "-" * len(header)]
        for score in sorted(self.scores, key=lambda s: (s.field, s.lead_hours)):
            lines.append(
                f"{score.field:12} {score.lead_hours:>5}h "
                f"{score.rmse:>10.3f} "
                f"{'' if score.acc is None else f'{score.acc:>7.3f}'} "
                f"{'' if score.persistence_rmse is None else f'{score.persistence_rmse:>10.3f}'} "
                f"{'' if score.climatology_rmse is None else f'{score.climatology_rmse:>10.3f}'} "
                f"{'' if score.skill_vs_persistence is None else f'{score.skill_vs_persistence:>8.1%}'} "
                f"{'' if score.skill_vs_climatology is None else f'{score.skill_vs_climatology:>8.1%}'}"
            )
        return "\n".join(lines)

    def verdict(self) -> str:
        """The one line that says whether any of this was worth it."""
        beats_persistence = [s for s in self.scores if (s.skill_vs_persistence or 0) > 0]
        beats_climatology = [s for s in self.scores if (s.skill_vs_climatology or 0) > 0]
        total = len(self.scores)
        lines = [
            f"beats persistence on {len(beats_persistence)}/{total} field-lead pairs",
            f"beats climatology on {len(beats_climatology)}/{total}",
        ]
        if not beats_persistence:
            lines.append("-> not a forecast yet: doing nothing scores better at every lead.")
        elif not beats_climatology:
            lines.append("-> beats persistence but not climatology: it knows the season, not the weather.")
        else:
            longest = max(beats_climatology, key=lambda s: s.lead_hours)
            lines.append(f"-> genuine skill out to at least +{longest.lead_hours}h.")
        return "\n".join(lines)


def persistence_forecast(history: torch.Tensor) -> torch.Tensor:
    """
    The baseline every weather model must beat: the last observed state, unchanged.

    It is embarrassingly strong at short lead -- six hours is not long enough for much to happen -- which
    is exactly why it is the right floor. A model that cannot beat it at +24 h has learned nothing about
    how the atmosphere evolves, whatever its loss curve looked like.
    """
    return history[:, -1]


def climatology_forecast(climatology: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    """The other baseline: what this date usually looks like, ignoring today entirely."""
    return climatology.unsqueeze(0).expand(batch_size, *climatology.shape)


def build_climatology(dataset, indices, channels: int, points: int, samples: int = 200) -> torch.Tensor:
    """
    Average state over a sample of the record -- the "what this usually looks like" reference.

    A proper climatology is per calendar day and per hour. This is the flat version, which is the honest
    floor: if a model cannot beat the *flat* mean it certainly cannot beat the seasonal one.
    """
    stride = max(len(indices) // max(samples, 1), 1)
    total = torch.zeros(points, channels)
    seen = 0
    for position in range(0, len(indices), stride):
        item = dataset[int(position)]
        total += item["analysis"][-1, :, :channels]
        seen += 1
    return total / max(seen, 1)


@dataclass(frozen=True)
class ScoredField:
    """
    One field, and the three different places its numbers live.

    These are genuinely three index spaces and conflating them is silent. ``field_mean`` is indexed by
    the model's seven :data:`naturev1.SURFACE_FIELDS`; ``analysis`` is indexed by the dataset's input
    channels, of which there may be 24 or 89; and the normalizer is keyed by ERA5 variable name. They
    coincide for the first few entries by luck, not design, so scoring persistence with a field index --
    or decoding with one -- produces numbers that look plausible and are wrong.
    """

    name: str
    field_channel: int       # into field_mean / field_target, i.e. SURFACE_FIELDS
    input_channel: int       # into analysis, for the persistence baseline
    variable: str | None     # the normalizer's own name, for decoding to physical units


def scoring_fields(variables, target_index, names=None) -> list[ScoredField]:
    """
    Build the index triples from what the dataset already knows.

    ``target_index`` is :func:`naturev1.era5._target_index`'s output: for each model surface field, the
    input channel that supervises it, or -1 where ERA5 does not carry it. That is exactly the
    correspondence scoring needs, so it is read from there rather than restated by hand.

    Args:
        names: restrict to these field names. Defaults to every field the dataset actually supervises.
    """
    from .model import SURFACE_FIELDS

    fields = []
    for field_channel, input_channel in enumerate(target_index):
        if input_channel < 0:
            continue
        name = SURFACE_FIELDS[field_channel]
        if names is not None and name not in names:
            continue
        variable = variables[input_channel] if input_channel < len(variables) else None
        fields.append(ScoredField(name, field_channel, input_channel, variable))
    return fields


@torch.no_grad()
def score_model(
    model,
    loader,
    analysis_grid,
    latitudes,
    num_longitudes: int,
    lead_times_hours,
    fields: list[ScoredField],
    normalizer=None,
    climatology: torch.Tensor | None = None,
    max_batches: int = 32,
    device=None,
) -> Scorecard:
    """
    Score a model against persistence and climatology on identical batches.

    Args:
        fields: from :func:`scoring_fields`, which carries the three index spaces a score needs --
            where the prediction lives, where the persistence baseline lives, and what the normalizer
            calls it.
        normalizer: if given, predictions, truth and both baselines are returned to physical units
            before scoring, so the RMSE is in kelvin and hectopascals rather than in standard
            deviations -- which is the only form comparable to a published number.
        climatology: ``(points, channels)`` reference state, from :func:`build_climatology`.

    Returns:
        A :class:`Scorecard` holding every field at every lead, with both baselines.
    """
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    weights = latitude_weights(latitudes, num_longitudes, device=device)

    # Accumulate squared error rather than averaging per batch: a mean of RMSEs is not an RMSE.
    totals: dict[tuple[str, int], dict[str, float]] = {}

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        analysis = batch["analysis"].to(device)
        calendar = batch["calendar"].to(device)
        truth = batch["field_target"].to(device)
        mask = batch.get("field_mask")
        mask = mask.to(device) if mask is not None else None

        outputs = model(analysis=analysis, analysis_grid=analysis_grid, calendar=calendar,
                        output_grid=analysis_grid)
        prediction = outputs["field_mean"]                       # (B, P, leads, fields)

        for lead, hours in enumerate(lead_times_hours):
            for entry in fields:
                name, channel = entry.name, entry.field_channel
                predicted = prediction[:, :, lead, channel]
                actual = truth[:, :, lead, channel]
                spot = mask[:, :, lead, channel] if mask is not None else None
                if spot is not None and float(spot.sum()) == 0.0:
                    continue

                decodes = (normalizer is not None and entry.variable is not None
                           and entry.variable in getattr(normalizer, "variables", []))
                if decodes:
                    predicted = normalizer.decode(predicted, entry.variable)
                    actual = normalizer.decode(actual, entry.variable)

                key = (name, hours)
                bucket = totals.setdefault(key, {"model": 0.0, "persist": 0.0, "climo": 0.0, "count": 0.0})
                spread = weights.view(1, -1)
                if spot is not None:
                    spread = spread * spot
                bucket["model"] += float(((predicted - actual) ** 2 * spread).sum())
                bucket["count"] += float(spread.sum())

                # Persistence: the last input frame. Indexed by INPUT channel, not field channel --
                # they are different spaces, and using one for the other is the bug this class exists
                # to prevent.
                persisted = analysis[:, -1, :, entry.input_channel]
                if decodes:
                    persisted = normalizer.decode(persisted, entry.variable)
                bucket["persist"] += float(((persisted - actual) ** 2 * spread).sum())

                if climatology is not None:
                    reference = climatology[:, entry.input_channel].to(device).view(1, -1)
                    if decodes:
                        reference = normalizer.decode(reference, entry.variable)
                    bucket["climo"] += float(((reference - actual) ** 2 * spread).sum())

    if was_training:
        model.train()

    card = Scorecard()
    for (name, hours), bucket in totals.items():
        count = max(bucket["count"], 1e-12)
        card.add(Score(
            field=name, lead_hours=hours,
            rmse=math.sqrt(bucket["model"] / count),
            persistence_rmse=math.sqrt(bucket["persist"] / count),
            climatology_rmse=math.sqrt(bucket["climo"] / count) if climatology is not None else None,
        ))
    return card
