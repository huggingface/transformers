# Copyright 2026 Nathan. Apache-2.0.
"""
Regression tests for the stage-two pairing: best-track labels against reanalysis timesteps.

The failure mode this guards against is not a crash. It is a pipeline that runs, trains, and reports a
validation number, while the atmosphere it showed the model belongs to a different hour than the storm
outcome it graded the model on. Nothing detects that except asserting on the pairing itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from naturev1.besttrack import Track
from naturev1.model import RI_THRESHOLDS_KT, WIND_RADII_THRESHOLDS_KT
from naturev1.storms import (
    KT_TO_MS,
    StormTargets,
    StormWindow,
    format_pairing,
    pair_tracks_with_reanalysis,
)


SIX_HOURS = 6 * 3600
OFFSETS = (1, 2, 4)


def make_track(storm_id="AL012000", name="TEST", points=12, start="2000-08-01T00",
               landfall_at=None, rmw_at=None, winds=None):
    """A synthetic storm on the synoptic grid, moving steadily northwest and intensifying."""
    times = np.array([np.datetime64(start, "s") + np.timedelta64(i * SIX_HOURS, "s") for i in range(points)])
    records = np.array(["" for _ in range(points)], dtype=object)
    if landfall_at is not None:
        records[landfall_at] = "L"
    rmw = np.full(points, np.nan, dtype=np.float32)
    if rmw_at is not None:
        rmw[rmw_at] = 20.0
    return Track(
        storm_id=storm_id, name=name, time=times,
        latitude=np.array([15.0 + 0.5 * i for i in range(points)], dtype=np.float32),
        longitude=np.array([-40.0 - 0.7 * i for i in range(points)], dtype=np.float32),
        max_wind_kt=np.asarray(winds if winds is not None else [30.0 + 5 * i for i in range(points)],
                               dtype=np.float32),
        min_pressure_hpa=np.array([1005.0 - 3 * i for i in range(points)], dtype=np.float32),
        status=np.array(["TS"] * points),
        rmw_nmi=rmw,
        wind_radii_nmi=np.full((points, 3, 4), np.nan, dtype=np.float32),
        record=np.asarray(records),
    )


@pytest.fixture
def store_times():
    """Six-hourly reanalysis timestamps covering the synthetic storms, as int64 seconds."""
    base = np.datetime64("2000-07-01T00", "s").astype(np.int64)
    return base + np.arange(400, dtype=np.int64) * SIX_HOURS


# --------------------------------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------------------------------

def test_points_pair_with_the_matching_hour(store_times):
    track = make_track()
    starts, targets, report = pair_tracks_with_reanalysis([track], store_times, OFFSETS, history=2)
    assert report["paired"] == len(targets) == len(starts)

    for start, target in zip(starts, targets):
        now = start + 2 - 1                      # the last history frame is "now"
        assert store_times[now] == track.time[target.point].astype("datetime64[s]").astype(np.int64)


def test_points_outside_the_reanalysis_are_dropped_not_snapped(store_times):
    """A 1900 storm has no reanalysis; pairing it to the nearest available hour would be fiction."""
    old = make_track(storm_id="AL011900", start="1900-08-01T00")
    starts, targets, report = pair_tracks_with_reanalysis([old], store_times, OFFSETS, history=2)
    assert report["paired"] == 0
    assert report["dropped_outside_reanalysis"] == len(old)
    assert len(starts) == len(targets) == 0


def test_off_synoptic_landfall_specials_are_dropped(store_times):
    """
    HURDAT2 records landfalls at the hour the eye crossed, which is usually not a synoptic hour.

    Keeping them would make a fixed lead offset mean a different number of hours for different samples.
    """
    track = make_track()
    track.time[3] = track.time[3] + np.timedelta64(5000, "s")     # nudge off the six-hourly grid
    _, _, report = pair_tracks_with_reanalysis([track], store_times, OFFSETS, history=2)
    assert report["dropped_outside_reanalysis"] == 1


def test_windows_need_room_for_history_and_the_longest_lead(store_times):
    track = make_track(start="2000-07-01T00")      # begins at the very first reanalysis step
    starts, _, report = pair_tracks_with_reanalysis([track], store_times, OFFSETS, history=6)
    assert report["dropped_no_room"] >= 1, "the first points cannot have 6 frames of history"
    assert (starts >= 0).all()


def test_pairing_report_formats(store_times):
    _, _, report = pair_tracks_with_reanalysis([make_track()], store_times, OFFSETS, history=2)
    text = format_pairing(report)
    assert "paired with reanalysis" in text and "distinct storms" in text


# --------------------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------------------

def test_displacement_is_measured_from_the_current_centre(store_times):
    track = make_track()
    target = StormTargets(track, 2, np.array(OFFSETS), np.full(len(track), np.nan)).build()
    for lead, offset in enumerate(OFFSETS):
        assert float(target["track_target"][lead, 0]) == pytest.approx(0.5 * offset, abs=1e-4)
        assert float(target["track_target"][lead, 1]) == pytest.approx(-0.7 * offset, abs=1e-4)
    assert (target["track_valid"] == 1.0).all()


def test_longitude_displacement_takes_the_short_way(store_times):
    """A storm crossing the date line moves a few degrees, not 359."""
    track = make_track(points=4)
    track.longitude = np.array([179.0, -179.0, -177.0, -175.0], dtype=np.float32)
    target = StormTargets(track, 0, np.array([1]), np.full(4, np.nan)).build()
    assert float(target["track_target"][0, 1]) == pytest.approx(2.0, abs=1e-4)


def test_leads_past_the_end_of_the_storm_are_invalid(store_times):
    track = make_track(points=4)
    target = StormTargets(track, 2, np.array([1, 2, 4]), np.full(4, np.nan)).build()
    assert float(target["track_valid"][0]) == 1.0      # point 3 exists
    assert float(target["track_valid"][1]) == 0.0      # point 4 does not
    assert float(target["track_valid"][2]) == 0.0


def test_landfall_is_cumulative(store_times):
    """The question is "will it make landfall by +48h", not "exactly at +48h"."""
    track = make_track(points=12, landfall_at=3)
    target = StormTargets(track, 2, np.array([1, 2, 4]), np.full(12, np.nan)).build()
    assert float(target["landfall_target"][0]) == 1.0    # lands at +1 step
    assert float(target["landfall_target"][1]) == 1.0    # still true at +2
    assert float(target["landfall_target"][2]) == 1.0    # and at +4


def test_no_landfall_reads_as_zero_not_missing(store_times):
    track = make_track(points=12)
    target = StormTargets(track, 2, np.array(OFFSETS), np.full(12, np.nan)).build()
    assert (target["landfall_target"] == 0.0).all()
    assert (target["landfall_mask"] == 1.0).all(), "a storm that did not land is an observation"


def test_intensity_targets_are_in_the_units_the_decoder_reads(store_times):
    """``decode_intensity`` reports m/s and absolute hPa, so the targets must be in those units."""
    track = make_track(points=12)
    target = StormTargets(track, 2, np.array([1]), np.full(12, np.nan)).build()
    expected_wind = float(track.max_wind_kt[3]) * KT_TO_MS
    assert float(target["intensity_target"][0, 0]) == pytest.approx(expected_wind, abs=1e-3)
    assert float(target["intensity_target"][0, 1]) == pytest.approx(float(track.min_pressure_hpa[3]), abs=1e-3)
    assert float(target["eyewall_target"][0]) == pytest.approx(float(track.max_wind_kt[3]), abs=1e-3)


def test_scarce_labels_arrive_as_nan_not_as_a_plausible_number(store_times):
    """RMW exists on under 5% of the record. Filling the rest teaches the head the filler."""
    track = make_track(points=12, rmw_at=5)
    target = StormTargets(track, 4, np.array([1, 2, 4]), np.full(12, np.nan)).build()
    assert float(target["rmw_target"][0]) == pytest.approx(20.0)     # point 5, observed
    assert torch.isnan(target["rmw_target"][1]), "point 6 has no RMW and must say so"
    assert torch.isnan(target["wind_radii_target"]).all()
    assert target["wind_radii_target"].shape == (3, len(WIND_RADII_THRESHOLDS_KT), 4)


def test_ri_labels_come_from_one_walk_across_thresholds(store_times):
    """The 24-hour wind change is measured once; each threshold is a comparison against it."""
    delta = np.full(12, np.nan)
    delta[4] = 32.0
    target = StormTargets(make_track(points=12), 4, np.array(OFFSETS), delta).build()
    assert target["ri_target"].tolist() == [float(32.0 >= t) for t in RI_THRESHOLDS_KT]
    assert target["ri_target"].tolist() == [1.0, 1.0, 0.0]        # 25 and 30 yes, 35 no
    assert (target["ri_mask"] == 1.0).all()
    assert float(target["ri_delta_target"]) == pytest.approx(32.0)


def test_ineligible_ri_points_are_masked_out(store_times):
    """A point with no future within the window is not a negative example; it is no example."""
    target = StormTargets(make_track(points=12), 4, np.array(OFFSETS), np.full(12, np.nan)).build()
    assert (target["ri_target"] == 0.0).all()
    assert (target["ri_mask"] == 0.0).all(), "unlabelled must not be trained as 'no'"
    assert torch.isnan(target["ri_delta_target"])


# --------------------------------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------------------------------

class _FakeBase(torch.utils.data.Dataset):
    """Stands in for the ERA5 input half, which its own test module covers."""

    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, item):
        return {"analysis": torch.full((2, 4, 3), float(item)), "calendar": torch.zeros(2, 6)}


def test_storm_window_merges_inputs_and_outcomes(store_times):
    track = make_track()
    starts, targets, _ = pair_tracks_with_reanalysis([track], store_times, OFFSETS, history=2)
    window = StormWindow(_FakeBase(len(targets)), targets)

    sample = window[0]
    assert "analysis" in sample and "track_target" in sample
    assert sample["track_target"].shape == (len(OFFSETS), 2)
    assert len(window) == len(targets)


def test_storm_window_refuses_a_mismatched_base(store_times):
    """Silently zipping mismatched lengths would pair each storm with the wrong atmosphere."""
    _, targets, _ = pair_tracks_with_reanalysis([make_track()], store_times, OFFSETS, history=2)
    with pytest.raises(ValueError, match="paired"):
        StormWindow(_FakeBase(len(targets) - 1), targets)


def test_describe_reports_the_base_rates(store_times):
    track = make_track(points=12, landfall_at=6)
    _, targets, _ = pair_tracks_with_reanalysis([track], store_times, OFFSETS, history=2)
    text = StormWindow(_FakeBase(len(targets)), targets).describe()
    assert "base rate" in text and "RMW observed" in text
    for threshold in RI_THRESHOLDS_KT:
        assert f"{threshold:.0f} kt" in text
