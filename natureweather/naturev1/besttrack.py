# Copyright 2026 Nathan. Apache-2.0.
"""
Best-track labels: HURDAT2 and IBTrACS, and the rapid-intensification events derived from them.

This is the *label* source, not the training corpus, and the difference matters. HURDAT2 holds about
eighty thousand six-hourly rows across 174 years of Atlantic storms; IBTrACS about seven times that
globally. Against an 88M-parameter network those numbers are nothing, and a model fit to them directly
will memorise storms rather than learn weather.

What they are good for is supervising a small number of storm-specific heads on top of a backbone that
learned the atmosphere somewhere else -- from reanalysis, where the label is simply the next state and
the corpus is millions of times larger. See :mod:`naturev1.corpora` for that side.

Rapid intensification is scarcer still. The National Hurricane Center's threshold is a 30-knot increase
in maximum sustained wind within 24 hours; across the whole Atlantic record that fires on a low
single-digit percentage of track points. Any honest RI model has to start from that base rate, so
:func:`rapid_intensification` reports it rather than leaving you to assume.
"""

from __future__ import annotations

import datetime as dt
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HURDAT2_ATLANTIC = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"
HURDAT2_PACIFIC = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2024-040425.txt"
IBTRACS_ALL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)

#: HURDAT2 sentinel for "not reported".
MISSING = -999


@dataclass
class Track:
    """One storm's best track. Arrays are parallel, one entry per synoptic time."""

    storm_id: str
    name: str
    time: np.ndarray            # (T,) datetime64[s]
    latitude: np.ndarray        # (T,) degrees
    longitude: np.ndarray       # (T,) degrees, -180..180
    max_wind_kt: np.ndarray     # (T,) knots, 1-minute sustained
    min_pressure_hpa: np.ndarray
    status: np.ndarray          # (T,) two-letter classification
    rmw_nmi: np.ndarray         # (T,) radius of maximum wind -- the eyewall's radius
    wind_radii_nmi: np.ndarray  # (T, 3, 4) thresholds 34/50/64 kt by NE/SE/SW/NW quadrant

    def __len__(self) -> int:
        return self.time.shape[0]

    @property
    def year(self) -> int:
        return int(str(self.time[0])[:4])

    @property
    def peak_wind_kt(self) -> float:
        valid = self.max_wind_kt[self.max_wind_kt > 0]
        return float(valid.max()) if valid.size else float("nan")

    def __repr__(self) -> str:
        return f"Track({self.storm_id} {self.name} {self.year}, {len(self)} points, peak {self.peak_wind_kt:.0f} kt)"


def download(url: str, destination: str | Path) -> Path:
    """Fetch a best-track file, skipping the download when it is already on disk."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)
    return destination


def _parse_latitude(token: str) -> float:
    value = float(token[:-1])
    return value if token[-1] == "N" else -value


def _parse_longitude(token: str) -> float:
    value = float(token[:-1])
    value = value if token[-1] == "E" else -value
    return (value + 180.0) % 360.0 - 180.0


def parse_hurdat2(path: str | Path) -> list[Track]:
    """
    Parse a HURDAT2 file into :class:`Track` objects.

    The format is a header line naming the storm and how many rows follow, then that many data rows. The
    trailing column is the radius of maximum wind, which is the one direct observation of eyewall size in
    the archive -- and is ``-999`` for most of the record, so treat its absence as the norm.
    """
    tracks: list[Track] = []
    with open(path) as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]

    index = 0
    while index < len(lines):
        header = [part.strip() for part in lines[index].split(",")]
        storm_id, name, count = header[0], header[1], int(header[2])
        rows = lines[index + 1 : index + 1 + count]
        index += 1 + count

        times, lats, lons, winds, pressures, statuses, rmws, radii = [], [], [], [], [], [], [], []
        for row in rows:
            field = [part.strip() for part in row.split(",")]
            stamp = dt.datetime.strptime(field[0] + field[1], "%Y%m%d%H%M")
            times.append(np.datetime64(stamp, "s"))
            statuses.append(field[3])
            lats.append(_parse_latitude(field[4]))
            lons.append(_parse_longitude(field[5]))
            winds.append(float(field[6]))
            pressures.append(float(field[7]))
            radii.append([[float(field[8 + group * 4 + quadrant]) for quadrant in range(4)] for group in range(3)])
            rmws.append(float(field[20]) if len(field) > 20 else MISSING)

        as_nan = lambda values: np.where(np.asarray(values, dtype=np.float32) == MISSING, np.nan, values)
        tracks.append(
            Track(
                storm_id=storm_id, name=name, time=np.array(times),
                latitude=np.asarray(lats, dtype=np.float32), longitude=np.asarray(lons, dtype=np.float32),
                max_wind_kt=as_nan(winds), min_pressure_hpa=as_nan(pressures),
                status=np.asarray(statuses), rmw_nmi=as_nan(rmws),
                wind_radii_nmi=as_nan(np.asarray(radii, dtype=np.float32)),
            )
        )
    return tracks


def rapid_intensification(
    tracks: list[Track], threshold_kt: float = 30.0, window_hours: int = 24, tropical_only: bool = True
) -> dict:
    """
    Label rapid intensification and report how rare it actually is.

    The National Hurricane Center defines RI as an increase in maximum sustained wind of at least 30 kt
    in 24 hours. This walks every track, marks the points from which that happens, and returns both the
    labels and the base rate -- because the base rate is the number that decides how the head must be
    trained. A classifier that always says "no" will score in the high nineties on this task and be
    worthless, so accuracy is not the metric and class weighting is not optional.

    Args:
        tracks: parsed best tracks.
        threshold_kt: intensification that counts as rapid.
        window_hours: the window it must happen within.
        tropical_only: restrict to tropical classifications (TD/TS/HU), which is the usual convention --
            extratropical transition can show a wind rise that is not intensification in this sense.

    Returns:
        ``{"labels": {storm_id: bool array}, "base_rate": float, "positives": int, "eligible": int,
        "delta": {storm_id: float array}}``
    """
    tropical = {"TD", "TS", "HU"}
    labels, deltas, positives, eligible = {}, {}, 0, 0

    for track in tracks:
        wind = track.max_wind_kt
        seconds = track.time.astype("datetime64[s]").astype(np.int64)
        change = np.full(len(track), np.nan, dtype=np.float32)
        for i in range(len(track)):
            if tropical_only and track.status[i] not in tropical:
                continue
            if not np.isfinite(wind[i]):
                continue
            horizon = seconds[i] + window_hours * 3600
            reachable = np.where((seconds > seconds[i]) & (seconds <= horizon))[0]
            if reachable.size == 0:
                continue
            future = wind[reachable]
            if not np.isfinite(future).any():
                continue
            change[i] = float(np.nanmax(future) - wind[i])
        flag = np.isfinite(change) & (change >= threshold_kt)
        labels[track.storm_id] = flag
        deltas[track.storm_id] = change
        positives += int(flag.sum())
        eligible += int(np.isfinite(change).sum())

    return {
        "labels": labels, "delta": deltas, "positives": positives, "eligible": eligible,
        "base_rate": positives / max(eligible, 1),
        "threshold_kt": threshold_kt, "window_hours": window_hours,
    }


def split_by_storm(
    tracks: list[Track], validation_years: tuple[int, ...] = (2017, 2019, 2021, 2023),
    test_years: tuple[int, ...] = (2018, 2020, 2022, 2024),
) -> dict[str, list[Track]]:
    """
    Split by *season*, never by track point.

    This is the one that quietly ruins best-track models. Six-hourly points from the same storm are
    almost the same picture: split them at random and near-duplicates land on both sides, validation
    looks superb, and the model has memorised storms rather than learned anything. Holding out whole
    seasons also keeps a storm's environment -- that year's ENSO state, that year's sea-surface
    temperatures -- out of training entirely, which is the honest test.
    """
    groups = {"train": [], "validation": [], "test": []}
    for track in tracks:
        year = track.year
        key = "validation" if year in validation_years else "test" if year in test_years else "train"
        groups[key].append(track)
    return groups
