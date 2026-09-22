# Copyright 2026 Nathan. Apache-2.0.
"""
The corpora, and why there are two of them.

Best-track archives are labels, not a training set. HURDAT2 holds 55,230 Atlantic track points across
174 years; rapid intensification at the 30-knot threshold fires on 1,839 of them; radius of maximum wind
is reported on 2,587. Fitting 89M parameters to that would memorise storms and call it learning, and the
validation score would look wonderful right up until a storm it had not seen.

So the model is trained twice.

**Stage one is self-supervised on reanalysis**, where the label is free: given the atmosphere now,
predict it six hours from now. ERA5 covers 1959 to 2022 at hourly resolution, which is 561,024 timesteps
and, at a modest 240x121 grid with 84 variables, about 2.3e11 values. That is roughly three million times
the size of the best-track record, it needs no annotation, and it is where the backbone actually learns
what the atmosphere does.

**Stage two fine-tunes the storm heads** on best tracks with the backbone frozen, so the parameters
actually fitted to those scarce labels number under a million rather than eighty-nine.

There is also a free multiplier available here that most architectures do not get. This model is exactly
equivariant to rotating the globe about its axis -- verified to float64 round-off -- so shifting a storm
in longitude produces a genuinely different sample that is guaranteed physically consistent, rather than
the approximate augmentation a pixel-grid model would be forced to settle for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Corpus:
    """One public dataset, with the numbers that decide whether it is worth the bandwidth."""

    name: str
    uri: str
    span: str
    cadence: str
    approx_samples: int
    role: str
    notes: str


#: Everything below is public and needs no account except ERA5 via Copernicus (free registration).
CORPORA = (
    Corpus(
        "WeatherBench 2 (ERA5)", "gs://weatherbench2/datasets/era5/", "1959-2022", "1h and 6h",
        561_024, "pretraining",
        "Analysis-ready Zarr from 64x32 up to 1440x721. The 6-hourly 240x121 store holds 92,040 verified "
        "timesteps; the hourly stores hold six times that. Start here.",
    ),
    Corpus(
        "ARCO-ERA5", "gs://gcp-public-data-arco-era5/ar/", "1940-present", "1h",
        745_000, "pretraining",
        "Same data, longer span, chunked for random access. Use when you outgrow WeatherBench 2.",
    ),
    Corpus(
        "GOES-19 / GOES-18 ABI", "s3://noaa-goes19/, s3://noaa-goes18/", "2017-present", "5-10 min",
        1_800_000, "pretraining + live",
        "16 bands, both hemispheres of the Americas. GOES-16 is retired and its bucket returns nothing.",
    ),
    Corpus(
        "TC-PRIMED", "s3://noaa-nesdis-tcprimed-pds/v01r01/", "1998-present", "per overpass",
        180_000, "storm fine-tuning",
        "Satellite overpasses already paired with best-track position and intensity. Purpose-built for "
        "exactly this task and the single highest-value dataset for the storm heads.",
    ),
    Corpus(
        "IBTrACS v04r01",
        "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
        "v04r01/access/csv/ibtracs.ALL.list.v04r01.csv",
        "1842-present", "3-6h", 585_000, "labels",
        "Global, every basin. About seven times the Atlantic-only HURDAT2 record -- use this, not HURDAT2, "
        "unless you specifically want Atlantic.",
    ),
    Corpus(
        "HURDAT2 Atlantic", "https://www.nhc.noaa.gov/data/hurdat/", "1851-2024", "6h",
        55_230, "labels",
        "Highest-quality reanalysed Atlantic tracks. Carries radius of maximum wind and wind radii, which "
        "IBTrACS does less consistently -- but only on ~5% of points.",
    ),
    Corpus(
        "SHIPS developmental data", "https://rammb2.cira.colostate.edu/research/tropical-cyclones/ships/",
        "1982-present", "6h", 120_000, "storm fine-tuning",
        "The environmental predictors operational RI forecasting is built on: shear, ocean heat content, "
        "mid-level humidity, persistence. Precomputed per storm. Feed these as the environment channels.",
    ),
    Corpus(
        "OISST v2.1 sea-surface temperature", "s3://noaa-cdr-sea-surface-temp-optimum-interpolation-pds/",
        "1981-present", "daily", 16_000, "environment",
        "Sea-surface temperature drives intensification. Daily quarter-degree, the standard choice.",
    ),
)


def catalogue() -> str:
    """A printable summary of the corpora, their sizes and what each is for."""
    lines = [f"{'dataset':28} {'span':16} {'cadence':10} {'samples':>10}  role"]
    lines.append("-" * 88)
    for corpus in CORPORA:
        lines.append(
            f"{corpus.name:28} {corpus.span:16} {corpus.cadence:10} {corpus.approx_samples:>10,}  {corpus.role}"
        )
    return "\n".join(lines)


def open_weatherbench(
    resolution: str = "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
    variables: tuple[str, ...] = (
        "2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
        "10m_v_component_of_wind", "total_precipitation_6hr", "sea_surface_temperature",
    ),
):
    """
    Open the WeatherBench 2 ERA5 archive straight from cloud storage, lazily.

    Nothing downloads until a chunk is actually read, so a training run streams what it needs instead of
    staging terabytes first. ``pip install zarr gcsfs xarray`` and no credentials are required.

    Args:
        resolution: one of the WeatherBench 2 store names. 240x121 six-hourly is the sensible default;
            drop to 64x32 to debug a pipeline, rise to 1440x721 when you have the bandwidth.
        variables: which fields to keep.
    """
    import xarray as xr

    store = f"gs://weatherbench2/datasets/era5/{resolution}"
    dataset = xr.open_zarr(store, storage_options={"token": "anon"}, consolidated=True)
    keep = [name for name in variables if name in dataset]
    missing = set(variables) - set(keep)
    if missing:
        print(f"[corpora] not in this store, skipping: {sorted(missing)}")
    return dataset[keep]


def latlon_to_coords(latitudes: np.ndarray, longitudes: np.ndarray) -> torch.Tensor:
    """Turn a latitude/longitude mesh in degrees into the radian coordinate pairs a grid expects."""
    lat_mesh, lon_mesh = np.meshgrid(latitudes, longitudes, indexing="ij")
    return torch.tensor(
        np.stack([np.radians(lat_mesh.ravel()), np.radians(lon_mesh.ravel()) % (2 * np.pi)], -1),
        dtype=torch.float64,
    )


def rotate_longitude(coords: torch.Tensor, radians: float) -> torch.Tensor:
    """
    Shift every sample in longitude -- an augmentation that is exact for this architecture.

    The local strand reads displacements in each sample's own east/north/up frame, which turns with the
    sample, so a longitude shift produces identical arithmetic on genuinely different data. A pixel-grid
    model has to roll an array and accept the seam; here the operation is a symmetry of the model, which
    makes it free and physically honest. Because it preserves latitude it also preserves the Coriolis
    parameter and the climate zone, so the augmented sample is still a possible atmosphere.
    """
    shifted = coords.clone()
    shifted[:, 1] = torch.remainder(shifted[:, 1] + radians, 2 * torch.pi)
    return shifted


def augment_batch(coords: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply a random longitude rotation, the safe augmentation for a global field."""
    angle = float(torch.rand((), generator=generator) * 2 * torch.pi)
    return rotate_longitude(coords, angle)


def estimate_epoch(num_samples: int, samples_per_second: float) -> dict:
    """Wall time for one pass over a corpus at a measured throughput."""
    seconds = num_samples / max(samples_per_second, 1e-9)
    return {"samples": num_samples, "samples_per_second": samples_per_second,
            "hours": seconds / 3600, "days": seconds / 86400}
