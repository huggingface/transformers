# Copyright 2026 Nathan. Apache-2.0.
"""
The upper atmosphere: pressure levels, in two shapes.

Surface variables alone cannot forecast weather. Medium-range skill lives in the 500 hPa geopotential
height -- the steering flow -- and in the vertical shear between levels. A model given only 2 m
temperature and sea-level pressure is being asked to predict tomorrow's surface from today's surface,
which is persistence with extra arithmetic. GraphCast sees 227 variables: 5 at the surface and 6 at each
of 37 levels.

This module adds the levels, in two shapes, because they buy different things.

**Channel mode** stacks every level into the feature vector of a 2-D grid. 6 variables x 13 levels = 78
channels on the same 29,040 points. Measured cost: **+33,000 parameters and no extra compute**, because
the point count is unchanged and only the first projection gets wider. This is what GraphCast does -- its
icosahedral mesh is two-dimensional and levels are features on it -- and it is the pragmatic default.

**Volume mode** builds a genuine 3-D grid: 29,040 x 13 = 377,520 points in a spherical shell, with
pressure as a *logarithmic axis* so that 1000 -> 500 hPa is the same distance as 500 -> 250. Attention
then reaches vertically the same way it reaches horizontally, through the same geodesic window, and a
head's radius means something physical in all three directions at once.

This second mode is the one a mesh GNN cannot express. GraphCast's graph is a surface; verticality is a
feature vector, so "the layer above" is not a neighbour, it is a different column of the same node.
Here it is a neighbour, at a real distance, in the sample's own east/north/**up** frame. Whether that
matters for forecast skill is an open question nobody has answered -- but it is the question this
architecture exists to ask, and channel mode is the control it should be measured against.

Volume mode costs 13x the points, so start in channel mode, get a scorecard, then pay for the comparison.
"""

from __future__ import annotations

import numpy as np
import torch

from .era5 import DEFAULT_VARIABLES, FIELD_DIMS, Normalizer, era5_splits  # noqa: F401


#: The 13 standard levels the WeatherBench 2 six-hourly store carries, in hPa. GraphCast's full model
#: uses 37; its operational variant and GraphCast_small both use exactly these 13.
PRESSURE_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

#: Pressure-level variables worth having. ``wind_speed`` is left out because it is a function of u and v
#: -- handing a model a channel it can compute from two others it already has spends bandwidth on nothing.
UPPER_VARIABLES = (
    "geopotential",          # Z -- the steering flow. Z500 is the headline forecast metric.
    "temperature",           # T -- T850 is the other headline.
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",     # moisture aloft, which is what makes convection go
    "vertical_velocity",     # omega: rising air, and therefore weather
)

#: Surface variables worth pairing with them, beyond the six already in :data:`naturev1.era5.DEFAULT_VARIABLES`.
SURFACE_EXTRAS = (
    "surface_pressure",
    "total_column_water_vapour",
    "toa_incident_solar_radiation_6hr",   # the sun, which drives everything and is exactly known
    "land_sea_mask",                      # static, but the model has no other way to know where land is
    "geopotential_at_surface",            # orography: mountains steer weather
)

#: WeatherBench 2 scores these two above all else. Level is in hPa.
HEADLINE_LEVELS = {"z500": ("geopotential", 500), "t850": ("temperature", 850)}


def open_weatherbench_levels(
    resolution: str = "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
    upper: tuple[str, ...] = UPPER_VARIABLES,
    surface: tuple[str, ...] = DEFAULT_VARIABLES + SURFACE_EXTRAS,
    levels: tuple[int, ...] = PRESSURE_LEVELS,
):
    """
    Open the store and keep both the surface fields and the pressure-level ones.

    Returns:
        ``(dataset, surface_names, upper_names)`` with the dataset subset to what was actually found.
        Names are reported back because a store may not carry everything asked for.
    """
    import xarray as xr

    store = f"gs://weatherbench2/datasets/era5/{resolution}"
    dataset = xr.open_zarr(store, storage_options={"token": "anon"}, consolidated=True)

    surface_names = [name for name in surface if name in dataset]
    upper_names = [name for name in upper if name in dataset]
    missing = (set(surface) | set(upper)) - set(surface_names) - set(upper_names)
    if missing:
        print(f"[upper] not in this store, skipping: {sorted(missing)}")

    subset = dataset[surface_names + upper_names]
    if upper_names and "level" in subset.dims:
        available = [level for level in levels if level in set(int(x) for x in subset.level.values)]
        subset = subset.sel(level=available)
    return subset, surface_names, upper_names


def channel_names(surface_names, upper_names, levels) -> list[str]:
    """Flat channel ordering: every surface field, then every (variable, level) pair."""
    return list(surface_names) + [f"{name}@{level}" for name in upper_names for level in levels]


def read_levels(dataset, surface_names, upper_names, selector) -> np.ndarray:
    """
    Read a time selection as ``(T, lat, lon, C)`` with levels flattened into the channel axis.

    Dimension order is forced, as everywhere else in this package: the store is longitude-major and the
    coordinate mesh is latitude-major, and reading it untransposed binds every sample to the wrong place
    on Earth while looking entirely healthy.

    Static fields without a time axis are broadcast across the window rather than skipped.
    """
    block = dataset.isel(time=selector)
    steps = block.sizes["time"]

    pieces = []
    for name in surface_names:
        field = block[name]
        if "time" in field.dims:
            pieces.append(field.transpose(*FIELD_DIMS).values[..., None])
        else:
            # Static fields -- land-sea mask, orography, soil type -- carry no time axis, and they are
            # exactly the ones worth having: a model with no land mask cannot know a coastline is there,
            # and one with no orography cannot know the Rockies steer weather. Broadcast them across the
            # window rather than dropping them, which is how GraphCast supplies its constants too.
            spatial = field.transpose("latitude", "longitude").values
            pieces.append(np.broadcast_to(spatial[None, ..., None], (steps, *spatial.shape, 1)))

    for name in upper_names:
        stacked = block[name].transpose(*FIELD_DIMS, "level").values     # (T, lat, lon, level)
        pieces.append(stacked)
    return np.concatenate(pieces, axis=-1).astype(np.float32)


def headline_channels(surface_names, upper_names, levels) -> dict[str, int]:
    """
    Which flat channel holds Z500 and T850, so :mod:`naturev1.wb2` can score them by name.

    Returns only the ones actually present, so a surface-only run scores what it has rather than failing.
    """
    names = channel_names(surface_names, upper_names, levels)
    found = {}
    for label, (variable, level) in HEADLINE_LEVELS.items():
        key = f"{variable}@{level}"
        if key in names:
            found[label] = names.index(key)
    for label, variable in (("u10", "10m_u_component_of_wind"), ("v10", "10m_v_component_of_wind")):
        if variable in names:
            found[label] = names.index(variable)
    return found


def volume_coords(latitudes, longitudes, levels) -> torch.Tensor:
    """
    Coordinates for a genuine 3-D atmospheric grid: ``(lat*lon*level, 3)`` as (lat, lon, pressure).

    Ordering is level-slowest, matching :func:`read_levels`' channel layout flattened the same way, so a
    volume built here lines up with data read there without a reshape that could silently transpose.

    This is the representation a surface mesh cannot hold. Pair it with
    :meth:`ihelix.Geometry.atmosphere`, where pressure is a *logarithmic* axis: 1000 -> 500 hPa becomes
    the same distance as 500 -> 250, which is what makes a vertical neighbour mean the same thing at
    every altitude.
    """
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)

    lat_mesh, lon_mesh, level_mesh = np.meshgrid(latitudes, longitudes, levels, indexing="ij")
    return torch.tensor(
        np.stack([
            np.radians(lat_mesh.ravel()),
            np.radians(lon_mesh.ravel()) % (2 * np.pi),
            level_mesh.ravel(),
        ], -1),
        dtype=torch.float64,
    )


def volume_grid(latitudes, longitudes, levels, scale_height_km: float = 7.0):
    """
    The 3-D shell as a read-only :class:`ihelix.FieldGrid`, weighted by true cell volume.

    Weights are horizontal cell area times the layer's thickness in log-pressure, because that is the
    fraction of the atmosphere's *mass* each sample stands for. Weighting by area alone would count a
    50 hPa sample -- thin, and holding almost no air -- the same as a 1000 hPa one.
    """
    from ihelix import FieldGrid, Geometry

    from .era5 import equiangular_weights

    horizontal = equiangular_weights(latitudes, len(longitudes))          # (lat*lon,), sums to 1
    log_levels = np.log(np.asarray(levels, dtype=np.float64))
    edges = np.concatenate([
        [log_levels[0] - (log_levels[1] - log_levels[0]) / 2],
        (log_levels[:-1] + log_levels[1:]) / 2,
        [log_levels[-1] + (log_levels[-1] - log_levels[-2]) / 2],
    ])
    thickness = np.abs(np.diff(edges))
    thickness = thickness / thickness.sum()

    weights = np.outer(horizontal, thickness).ravel()                      # level-fastest, matching coords
    coords = volume_coords(latitudes, longitudes, levels)
    geometry = Geometry.atmosphere(scale_height_km=scale_height_km)
    return FieldGrid.source(coords, geometry, weights=torch.tensor(weights / weights.sum()))


def upper_air_report(surface_names, upper_names, levels, parameters: int = 89_033_475) -> str:
    """What adding the levels buys and what it costs, in the numbers that decide whether to do it."""
    channels = len(surface_names) + len(upper_names) * len(levels)
    points = 29_040
    per_step_mb = points * channels * 2 / 1e6
    values = 92_040 * points * channels
    return "\n".join([
        f"  surface variables      {len(surface_names):>8}",
        f"  upper variables        {len(upper_names):>8}  x {len(levels)} levels",
        f"  total channels         {channels:>8}   (was 6)",
        f"  supervised values      {values:>8.3e}   -> {values / parameters:,.0f} per parameter (was 180)",
        f"  staged size            {per_step_mb:>8.2f} MB/timestep, {per_step_mb * 1460 / 1000:.1f} GB/year",
        f"  volume-mode points     {points * len(levels):>8,}   (13x, for true 3-D attention)",
        "",
        "  Channel mode costs ~33k parameters and no extra compute. Volume mode costs 13x the points",
        "  and is the comparison worth running once channel mode has a scorecard.",
    ])
