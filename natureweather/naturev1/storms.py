# Copyright 2026 Nathan. Apache-2.0.
"""
Stage two: best-track labels, paired with the reanalysis the backbone was pretrained on.

Stage one taught the backbone what the atmosphere does, from 1.6e10 free supervised values. This module
supplies the other half -- what a storm did -- and it is small on purpose, because it is small in
reality: 55,230 Atlantic track points, 1,167 of them landfalls, 1,839 rapid intensifications at the
30-knot threshold, 2,587 with a radius of maximum wind. That is the entire observational record, and no
amount of engineering makes it bigger.

What engineering can do is make sure those few thousand labels are spent on a few thousand parameters
instead of eighty-nine million. :meth:`naturev1.NatureV1.freeze_backbone` leaves 0.96M trainable, and
this module hands those heads real inputs -- the actual global atmosphere at the hour the storm was
observed -- rather than the random tensors a placeholder pipeline would feed them.

Three decisions here are the ones that decide whether the result means anything:

**Windows are paired by timestamp, not by index.** Every sample is a genuine (atmosphere, outcome) pair:
the ERA5 state at the synoptic hour the Hurricane Center recorded the storm at, and what that storm went
on to do at +6 through +120 hours. Points the reanalysis does not cover -- anything before 1959, and
HURDAT2's off-synoptic landfall specials -- are dropped rather than approximated.

**Scarce labels are masked, never imputed.** Radius of maximum wind exists on 4.7% of points. Filling
the other 95.3% with a plausible number teaches the head that plausible number. Every target here
arrives as NaN where it was not observed, and the losses skip it.

**Splits are by season.** Six-hourly points from one storm are near-duplicates of each other; split them
at random and the same hurricane lands on both sides of the split. Holding out whole seasons also keeps
that year's ENSO state and sea-surface temperatures out of training, which is the honest test.
"""

from __future__ import annotations

import numpy as np
import torch

from .besttrack import Track, rapid_intensification
from .model import RI_THRESHOLDS_KT, WIND_RADII_THRESHOLDS_KT


#: Knots to metres per second. :func:`naturev1.forecast.decode_intensity` reads the intensity head in
#: m/s and absolute hPa, so the targets are built in those units rather than in the archive's knots.
KT_TO_MS = 0.514444


def _displacement(track: Track, start: int, end: int) -> tuple[float, float]:
    """Latitude and longitude change between two track points, taking the short way around."""
    dlat = float(track.latitude[end] - track.latitude[start])
    dlon = float((track.longitude[end] - track.longitude[start] + 180.0) % 360.0 - 180.0)
    return dlat, dlon


def _nearest_points(times: np.ndarray, targets: np.ndarray, tolerance_hours: float) -> np.ndarray:
    """
    For each target time, the index of the nearest entry in ``times``, or -1 if none is close enough.

    Best tracks are six-hourly on the synoptic hours, which is exactly ERA5's cadence, so almost every
    point matches to the minute. The exceptions are HURDAT2's landfall specials, recorded at the hour
    the eye actually crossed the coast; those fall outside the tolerance and are dropped rather than
    snapped to a state up to three hours away from the event they describe.
    """
    order = np.searchsorted(times, targets)
    order = np.clip(order, 1, len(times) - 1)
    before, after = times[order - 1], times[order]
    nearest = np.where(targets - before <= after - targets, order - 1, order)
    gap = np.abs(times[nearest] - targets)
    return np.where(gap <= tolerance_hours * 3600, nearest, -1)


class StormTargets:
    """
    Best-track outcomes for one point in one storm, at every forecast lead.

    Built once per sample and cached, because the RI walk and the wind-radii lookups cost far more than
    the tensor construction and never change.
    """

    __slots__ = ("track", "point", "offsets", "delta", "leads")

    def __init__(self, track: Track, point: int, offsets: np.ndarray, delta: np.ndarray) -> None:
        self.track, self.point, self.offsets, self.delta = track, point, offsets, delta
        self.leads = len(offsets)

    def build(self) -> dict[str, torch.Tensor]:
        track, start, leads = self.track, self.point, self.leads
        nan = float("nan")

        displacement = torch.zeros(leads, 2)
        valid = torch.zeros(leads)
        intensity = torch.zeros(leads, 2)
        intensity_mask = torch.zeros(leads, 2)
        landfall = torch.zeros(leads)
        landfall_mask = torch.zeros(leads)
        peak_wind = torch.full((leads,), nan)
        rmw = torch.full((leads,), nan)
        radii = torch.full((leads, len(WIND_RADII_THRESHOLDS_KT), 4), nan)

        for lead, offset in enumerate(self.offsets):
            end = start + int(offset)
            if end >= len(track):
                continue                      # the storm ended before this lead; nothing to predict
            valid[lead] = 1.0
            displacement[lead, 0], displacement[lead, 1] = _displacement(track, start, end)

            wind_kt = float(track.max_wind_kt[end])
            if np.isfinite(wind_kt):
                intensity[lead, 0] = wind_kt * KT_TO_MS
                intensity_mask[lead, 0] = 1.0
                peak_wind[lead] = wind_kt
            pressure = float(track.min_pressure_hpa[end])
            if np.isfinite(pressure):
                intensity[lead, 1] = pressure
                intensity_mask[lead, 1] = 1.0

            # Landfall is cumulative: "will it make landfall by +48h", not "exactly at +48h".
            landfall[lead] = float(track.landfall[start : end + 1].any())
            landfall_mask[lead] = 1.0

            if np.isfinite(track.rmw_nmi[end]):
                rmw[lead] = float(track.rmw_nmi[end])
            observed = np.isfinite(track.wind_radii_nmi[end])
            if observed.any():
                radii[lead] = torch.tensor(
                    np.where(observed, track.wind_radii_nmi[end], np.nan), dtype=torch.float32
                )

        change = float(self.delta[start])
        eligible = np.isfinite(change)
        return {
            "track_target": displacement,
            "track_valid": valid,
            "intensity_target": intensity,
            "intensity_mask": intensity_mask,
            "landfall_target": landfall,
            "landfall_mask": landfall_mask,
            "eyewall_target": peak_wind,
            "rmw_target": rmw,
            "wind_radii_target": radii,
            "ri_target": torch.tensor(
                [float(eligible and change >= t) for t in RI_THRESHOLDS_KT], dtype=torch.float32
            ),
            "ri_mask": torch.full((len(RI_THRESHOLDS_KT),), float(eligible)),
            "ri_delta_target": torch.tensor(change if eligible else nan, dtype=torch.float32),
        }


class StormWindow(torch.utils.data.Dataset):
    """
    Reanalysis input at a storm's observed hour, with what that storm actually went on to do.

    Wraps an :class:`naturev1.ERA5Window` or :class:`naturev1.CachedERA5` whose ``indices`` were chosen
    to line up one-to-one with the track points, so the input half reuses machinery that is already
    tested rather than reimplementing the read.

    Build it with :func:`pair_tracks_with_reanalysis`, which does the alignment.
    """

    def __init__(self, base: torch.utils.data.Dataset, targets: list[StormTargets]) -> None:
        if len(base) != len(targets):
            raise ValueError(f"base has {len(base)} windows but {len(targets)} storm points were paired")
        self.base, self.targets = base, targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, item: int) -> dict:
        sample = dict(self.base[item])
        sample.update(self.targets[item].build())
        return sample

    def describe(self) -> str:
        """What the labels in this split actually contain -- worth printing before trusting a metric."""
        built = [target.build() for target in self.targets]
        total = len(built)
        if not total:
            return "no paired samples"

        def observed(key):
            stacked = torch.stack([item[key] for item in built])
            return 100.0 * torch.isfinite(stacked).float().mean()

        ri = torch.stack([item["ri_target"] for item in built])
        mask = torch.stack([item["ri_mask"] for item in built])
        lines = [f"{total:,} paired (reanalysis, outcome) samples"]
        for index, threshold in enumerate(RI_THRESHOLDS_KT):
            eligible = mask[:, index].sum()
            rate = 100.0 * ri[:, index].sum() / eligible.clamp_min(1.0)
            lines.append(f"  RI >= {threshold:.0f} kt / 24 h   {int(ri[:, index].sum()):>6,} of "
                         f"{int(eligible):>6,} eligible   base rate {rate:.2f}%")
        landfall = torch.stack([item["landfall_target"] for item in built])
        lines.append(f"  landfall by +120 h    {int(landfall[:, -1].sum()):>6,} of {total:,} "
                     f"({100.0 * landfall[:, -1].mean():.2f}%)")
        lines.append(f"  peak wind observed    {observed('eyewall_target'):.1f}% of lead slots")
        lines.append(f"  wind radii observed   {observed('wind_radii_target'):.1f}%")
        lines.append(f"  RMW observed          {observed('rmw_target'):.1f}%  <- the scarce one")
        return "\n".join(lines)


def pair_tracks_with_reanalysis(
    tracks: list[Track],
    store_times: np.ndarray,
    lead_offsets: tuple[int, ...],
    history: int = 6,
    cadence_hours: float = 6.0,
    tolerance_hours: float = 1.0,
    tropical_only: bool = True,
    ri_window_hours: int = 24,
) -> tuple[np.ndarray, list[StormTargets], dict]:
    """
    Line best-track points up with the reanalysis timesteps that cover them.

    Args:
        tracks: parsed best tracks.
        store_times: the reanalysis store's timestamps, as int64 seconds, ascending.
        lead_offsets: store-step offsets of the forecast leads, from :func:`naturev1.lead_offsets`.
        history: input frames each window needs before the storm's hour.
        tolerance_hours: how far a track point may sit from a reanalysis timestep and still pair.

    Returns:
        ``(window_starts, targets, report)``. Pass ``window_starts`` as the ``indices`` of an
        :class:`naturev1.ERA5Window`, and the two line up one-to-one.
    """
    store_times = np.asarray(store_times, dtype=np.int64)
    if not np.all(np.diff(store_times) > 0):
        raise ValueError("store_times must be strictly ascending")

    # The RI walk is over whole tracks and independent of the threshold -- it measures the 24-hour wind
    # change, and every threshold is a comparison against it -- so it runs once.
    walk = rapid_intensification(tracks, threshold_kt=min(RI_THRESHOLDS_KT),
                                 window_hours=ri_window_hours, tropical_only=tropical_only)
    deltas = walk["delta"]

    # Track points are spaced by the archive's own cadence, which must match the store's for a lead
    # offset in store steps to mean the same number of track points.
    steps = np.asarray(lead_offsets, dtype=np.int64)
    starts, targets = [], []
    considered = dropped_time = dropped_room = 0

    for track in tracks:
        seconds = track.time.astype("datetime64[s]").astype(np.int64)
        # Only points on the archive's regular grid: the off-synoptic landfall specials would otherwise
        # make a "lead offset" mean a different number of hours for different samples.
        regular = (seconds % int(cadence_hours * 3600)) == 0
        matched = _nearest_points(store_times, seconds, tolerance_hours)
        delta = deltas.get(track.storm_id)
        if delta is None:
            continue

        for point in range(len(track)):
            considered += 1
            if not regular[point] or matched[point] < 0:
                dropped_time += 1
                continue
            start = int(matched[point]) - history + 1
            if start < 0 or start + history - 1 + int(steps.max()) >= len(store_times):
                dropped_room += 1
                continue
            starts.append(start)
            targets.append(StormTargets(track, point, steps, delta))

    report = {
        "considered": considered,
        "paired": len(starts),
        "dropped_outside_reanalysis": dropped_time,
        "dropped_no_room": dropped_room,
        "storms": len({target.track.storm_id for target in targets}),
    }
    return np.asarray(starts, dtype=np.int64), targets, report


def format_pairing(report: dict) -> str:
    """A printable summary of what paired and what did not."""
    considered = max(report["considered"], 1)
    return "\n".join([
        f"  track points considered       {report['considered']:>8,}",
        f"  paired with reanalysis        {report['paired']:>8,}  "
        f"({100 * report['paired'] / considered:.1f}%)",
        f"  dropped, outside the record   {report['dropped_outside_reanalysis']:>8,}  "
        "(pre-1959, or off-synoptic landfall specials)",
        f"  dropped, no room for a window {report['dropped_no_room']:>8,}",
        f"  distinct storms               {report['storms']:>8,}",
    ])
