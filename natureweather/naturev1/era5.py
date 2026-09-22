# Copyright 2026 Nathan. Apache-2.0.
"""
The pretraining corpus: ERA5 reanalysis, streamed.

This module exists because of an arithmetic problem. NatureV1 has 89 million parameters. The Atlantic
best-track archive has 55,230 rows, of which 1,839 are rapid-intensification events and 2,587 carry a
radius of maximum wind. Fitting 89M parameters to 55k rows does not produce a weather model, it produces
a very expensive lookup table for storms that have already happened.

ERA5 fixes the ratio rather than shrinking the model. The 6-hourly WeatherBench 2 store holds 92,040
verified timesteps from 1959 to 2021 on a 240x121 grid; the hourly stores hold six times that. At six
surface variables that is 1.6e10 supervised values -- 180 per parameter, where best tracks offer 0.0006 --
and every one is free, because the target for "the atmosphere at time t" is "the atmosphere at t+6h".
Nobody annotates anything. That is where the backbone learns what air does.

The storm heads are then fine-tuned on best tracks with the backbone frozen, so the parameter count
actually exposed to those scarce labels is 0.96M, not 89M. :meth:`naturev1.NatureV1.freeze_backbone`
does that half; this module does this one.

Four things here are worth knowing, because each one is a place where the obvious implementation is
quietly wrong.

**Cell weights are areas, not cosines.** The reflex on a latitude/longitude grid is to weight by
``cos(latitude)``. On an equiangular grid *with poles* that gives the pole row a weight of 6.1e-17 --
exactly zero in float32 -- so the Arctic and Antarctic disappear from every area-weighted sum the model
takes. The correct weight is the cell's spherical area, ``sin(upper edge) - sin(lower edge)``, which
makes the pole row a half-height cap: 76 times lighter than an equatorial cell, not 1.6e16 times.
See :func:`equiangular_weights`.

**Precipitation is not Gaussian and must not be z-scored raw.** 15% of the grid is exactly zero at any
moment and the maximum sits 30 standard deviations out. A network trained on that minimises its loss by
predicting zero everywhere and never recovers. :class:`Normalizer` puts precipitation through
``log1p(x/0.1mm)`` first, which turns a spike-and-slab into something a Gaussian head can actually fit,
and inverts it exactly on the way out.

**Missing is not average.** Sea-surface temperature has no value over land, which is 27.9% of the grid.
Filling those cells with the ocean mean tells the model there is 286 K water over Kansas. Each variable
with gaps gets a companion channel that is 1 where the value was measured and 0 where it was not, so
"unknown" is a thing the model can see rather than a lie it has to learn around.

**The augmentation is exact and free.** On an equiangular grid, rotating the globe by a whole number of
cells in longitude is the same operation as rolling the array -- the grid is unchanged, nothing is
rebuilt, nothing is resampled. The model is exactly equivariant to that rotation (verified to float64
round-off), so all 240 rolls are genuinely valid atmospheres rather than the approximation a pixel-grid
model settles for. The honest caveat: this holds because every channel rolls together. A channel that is
*not* a function of the atmosphere -- a land-sea mask, orography -- rolls too, and you are then training
on a planet whose continents moved. Fine for learning fluid dynamics, wrong for learning where Florida
is, so the fine-tuning stage turns augmentation off.

Streaming from cloud storage is the bottleneck, not the GPU: a window costs about 550 ms to fetch and
tens of milliseconds to train on. Use :func:`era5_loader` for workers, and for any real run call
:func:`materialise` first -- measured at 5 ms per window from local disk against 549 ms streamed, 105x.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .corpora import latlon_to_coords, open_weatherbench
from .model import SURFACE_FIELDS, calendar_features


#: Which ERA5 variable supervises which of the model's surface fields. Anything the model predicts but
#: ERA5 does not carry here (relative humidity, cloud fraction) is left unobserved rather than faked --
#: the target arrives as NaN and :func:`naturev1.losses.masked_gaussian_nll` skips it. A plausible-looking
#: stand-in would teach the head a number nobody measured, which is worse than teaching it nothing.
ERA5_TO_SURFACE = {
    "2m_temperature": "t2m",
    "mean_sea_level_pressure": "mslp",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation_6hr": "precip_rate",
}

#: Variables that are spike-and-slab rather than bell-shaped, with the scale of their log transform in
#: the variable's own units. 1e-4 m is 0.1 mm over six hours -- the drizzle threshold, below which the
#: distinction between "a little rain" and "no rain" is instrument noise.
LOG_SCALE = {"total_precipitation_6hr": 1e-4}

#: Default input variables. Sea-surface temperature is an input but not a target: it drives
#: intensification, and it is a boundary condition rather than something the atmosphere head forecasts.
DEFAULT_VARIABLES = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation_6hr",
    "sea_surface_temperature",
)


#: The dimension order this module works in. WeatherBench 2 stores its variables as
#: ``(time, longitude, latitude)``, which is the transpose of what the coordinate mesh from
#: :func:`naturev1.corpora.latlon_to_coords` expects. Reading the array without transposing it produces
#: the right *number* of points, sane-looking values and a falling loss, while every sample sits at the
#: wrong place on Earth -- so the order is forced on every read rather than assumed.
FIELD_DIMS = ("time", "latitude", "longitude")


def read_block(dataset, variables, selector) -> np.ndarray:
    """Read a time selection as ``(T, lat, lon, C)``, in that order, whatever the store's own layout."""
    block = dataset.isel(time=selector)
    stacked = [block[name].transpose(*FIELD_DIMS).values for name in variables]
    return np.stack(stacked, -1).astype(np.float32)


def equiangular_weights(latitudes: np.ndarray, num_longitudes: int) -> np.ndarray:
    """
    Quadrature weights for an equiangular latitude/longitude grid, normalised to sum to one.

    A cell spanning latitudes ``a`` to ``b`` covers a fraction ``(sin b - sin a) / 2`` of the sphere,
    which is the integral of the area element and not an approximation. Rows at the poles of a
    "with poles" grid are half-height caps whose 240 longitude entries are all the same physical point,
    so the cap's area is shared among them.

    This matters more than it looks. Weighted by ``cos(latitude)`` instead, the pole row gets 6.1e-17 --
    zero in float32 -- and every area-weighted quantity the model computes silently omits the poles. The
    two are within 0.03% of each other in the tropics and differ by sixteen orders of magnitude at 90
    degrees, which is exactly the region where the cheap version looks harmless in testing.
    """
    latitudes = np.asarray(latitudes, dtype=np.float64)
    radians = np.radians(latitudes)
    # Edges halfway to each neighbour, and at the pole for the outermost rows.
    interior = (radians[:-1] + radians[1:]) / 2.0
    top = np.clip(radians[0] + (radians[0] - interior[0]), -np.pi / 2, np.pi / 2)
    bottom = np.clip(radians[-1] + (radians[-1] - interior[-1]), -np.pi / 2, np.pi / 2)
    edges = np.concatenate([[top], interior, [bottom]])
    area = np.abs(np.sin(edges[:-1]) - np.sin(edges[1:])) / num_longitudes
    weights = np.repeat(area, num_longitudes)
    return weights / weights.sum()


def era5_source_grid(dataset, geometry=None):
    """
    Build the read-only :class:`ihelix.FieldGrid` for an ERA5 store, weighted by true cell area.

    ERA5 samples are never queried as neighbours of each other, only read onto the latent mesh, so
    :meth:`ihelix.FieldGrid.source` skips the kNN graph and the hierarchy: 29,040 points in about 5 ms
    instead of several seconds.

    Returns:
        ``(grid, latitudes, longitudes)`` with the axes in degrees, as stored.
    """
    from ihelix import FieldGrid, Geometry

    latitudes = np.asarray(dataset["latitude"].values, dtype=np.float64)
    longitudes = np.asarray(dataset["longitude"].values, dtype=np.float64)
    coords = latlon_to_coords(latitudes, longitudes)
    weights = equiangular_weights(latitudes, len(longitudes))
    grid = FieldGrid.source(coords, geometry or Geometry.globe(), weights=torch.tensor(weights))
    return grid, latitudes, longitudes


def split_prepared(prepared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the staging format into values and an observation mask. Unobserved becomes exactly zero --
    the variable's own mean in normalized space, contributing nothing to a linear read -- and is flagged
    in the mask so the model can tell "average" from "absent"."""
    observed = ~np.isnan(prepared)
    return np.where(observed, np.nan_to_num(prepared), 0.0).astype(np.float32), observed


@dataclass
class Normalizer:
    """
    Per-variable statistics, transforms and gap bookkeeping -- everything needed to make raw ERA5
    values something a network can be trained on, and to put predictions back into physical units.

    Args:
        variables: input variable names, in channel order.
        mean, std: computed on the *transformed* values, so they match what the model sees.
        missing: fraction of the grid where each variable has no value. Non-zero entries get a
            companion mask channel.
    """

    variables: list[str]
    mean: np.ndarray
    std: np.ndarray
    missing: np.ndarray

    @property
    def masked(self) -> list[int]:
        """Channel indices that carry gaps, and therefore get a companion observation channel."""
        return [index for index, fraction in enumerate(self.missing) if fraction > 0.0]

    @property
    def width(self) -> int:
        """Total input channels produced: one per variable, plus one mask per gappy variable."""
        return len(self.variables) + len(self.masked)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the per-variable shape transform (log for precipitation), leaving others alone."""
        values = values.copy()
        for index, name in enumerate(self.variables):
            if name in LOG_SCALE:
                values[..., index] = np.log1p(np.maximum(values[..., index], 0.0) / LOG_SCALE[name])
        return values

    def prepare(self, raw: np.ndarray) -> np.ndarray:
        """
        Normalized values with NaN kept where nothing was measured -- the staging format.

        Storing *raw* values as float16 does not work: sea-level pressure is about 101,325 Pa and the
        float16 ceiling is 65,504, so pressure overflows to infinity. Normalized values sit within a few
        units of zero, where float16 has a step of about 0.005 -- 0.1 K of temperature and 0.08 hPa of
        pressure, at or below ERA5's own precision. So normalize first, then narrow.
        """
        observed = ~np.isnan(raw)
        values = (self.transform(np.nan_to_num(raw).astype(np.float32)) - self.mean) / self.std
        return np.where(observed, values, np.nan).astype(np.float32)

    def encode(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Raw ERA5 values ``(..., C)`` to normalized values plus the observation mask.

        Unobserved cells come back as exactly zero -- the variable's own mean in normalized space, so
        they contribute nothing to a linear read -- and are flagged in the mask so the model can tell
        "average" from "absent".
        """
        return split_prepared(self.prepare(raw))

    def decode(self, values, variable: str):
        """Invert :meth:`encode` for one variable, transform included, back to physical units."""
        index = self.variables.index(variable)
        physical = values * float(self.std[index]) + float(self.mean[index])
        if variable in LOG_SCALE:
            if isinstance(physical, torch.Tensor):
                return torch.expm1(physical) * LOG_SCALE[variable]
            return np.expm1(physical) * LOG_SCALE[variable]
        return physical

    @classmethod
    def fit(cls, dataset, variables, samples: int = 40, cache: str | os.PathLike | None = None) -> Normalizer:
        """
        Compute statistics from a stride through the whole record, and cache them.

        A stride rather than the first N timesteps: taking the opening year would bake in that year's
        season and put the 1959 climate's bias into a model that has to forecast 2021. The read costs far
        more than the arithmetic, so the result is cached next to the data.
        """
        variables = list(variables)
        if cache is not None and Path(cache).exists():
            stored = json.loads(Path(cache).read_text())
            if stored.get("variables") == variables:
                return cls(variables, np.array(stored["mean"], np.float32),
                           np.array(stored["std"], np.float32), np.array(stored["missing"], np.float32))

        steps = dataset.sizes["time"]
        raw = read_block(dataset, variables, slice(0, steps, max(steps // max(samples, 1), 1)))
        missing = np.isnan(raw).mean(axis=tuple(range(raw.ndim - 1))).astype(np.float32)

        blank = cls(variables, np.zeros(len(variables), np.float32), np.ones(len(variables), np.float32), missing)
        shaped = blank.transform(np.nan_to_num(raw, nan=np.nan))
        axes = tuple(range(shaped.ndim - 1))
        mean = np.nanmean(shaped, axis=axes).astype(np.float32)
        std = (np.nanstd(shaped, axis=axes) + 1e-6).astype(np.float32)

        normalizer = cls(variables, mean, std, missing)
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            Path(cache).write_text(json.dumps(normalizer.to_dict(), indent=2))
        return normalizer

    def to_dict(self) -> dict:
        """Plain-JSON form, for the stats cache and for the training checkpoint."""
        return {"variables": self.variables, "mean": self.mean.tolist(),
                "std": self.std.tolist(), "missing": self.missing.tolist()}

    @classmethod
    def from_dict(cls, stored: dict) -> Normalizer:
        return cls(list(stored["variables"]), np.array(stored["mean"], np.float32),
                   np.array(stored["std"], np.float32), np.array(stored["missing"], np.float32))

    def report(self) -> str:
        """A printable table of what each channel became -- worth reading once before a long run."""
        lines = [f"{'variable':32} {'mean':>12} {'std':>12} {'missing':>9}  transform"]
        lines.append("-" * 84)
        for index, name in enumerate(self.variables):
            shape = f"log1p(x/{LOG_SCALE[name]:g})" if name in LOG_SCALE else "identity"
            lines.append(f"{name:32} {self.mean[index]:>12.4g} {self.std[index]:>12.4g} "
                         f"{100 * self.missing[index]:>8.1f}%  {shape}")
        if self.masked:
            names = ", ".join(self.variables[index] for index in self.masked)
            lines.append(f"\n+{len(self.masked)} observation-mask channel(s) for: {names}")
        return "\n".join(lines)


def normalization(dataset, variables, samples: int = 40, cache: str | os.PathLike | None = None):
    """Backwards-compatible shim: the mean and standard deviation alone. Prefer :meth:`Normalizer.fit`."""
    fitted = Normalizer.fit(dataset, variables, samples=samples, cache=cache)
    return fitted.mean, fitted.std


def era5_splits(dataset, val_years: int = 4, test_years: int = 2, steps_per_year: int = 1460):
    """
    Split the record by time, which is the only split that means anything here.

    Weather is autocorrelated for days. A random split puts 06:00 Tuesday in train and 12:00 Tuesday in
    validation, the model interpolates between two states it has already seen, and the curve reports a
    skill the model does not have. Holding out whole trailing years is the test the operational centres
    actually run: forecast a period that had not happened yet.

    Returns:
        ``{"train": idx, "val": idx, "test": idx}`` of window start indices, ordered in time.
    """
    steps = dataset.sizes["time"]
    test, val = test_years * steps_per_year, val_years * steps_per_year
    train_end = steps - val - test
    if train_end <= 0:
        raise ValueError(f"Record has {steps} steps, too short for {val_years}+{test_years} held-out years.")
    return {
        "train": np.arange(0, train_end),
        "val": np.arange(train_end, train_end + val),
        "test": np.arange(train_end + val, steps),
    }


def _target_index(variables) -> list[int]:
    """For each model surface field, which input channel supervises it, or -1 if ERA5 lacks it."""
    inverse = {target: source for source, target in ERA5_TO_SURFACE.items()}
    return [variables.index(inverse[field]) if inverse.get(field) in variables else -1
            for field in SURFACE_FIELDS]


def _build_window(
    prepared: np.ndarray,
    normalizer: Normalizer,
    target_index: list[int],
    history: int,
    channels: int | None,
    shift: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shared item construction: normalized ``(T, lat, lon, C)`` to ``(analysis, target)``.

    One implementation, taking the staging format, so the streamed and the staged dataset cannot drift
    apart in their logic. They are not bit-identical: staging at float16 quantizes, and the two agree to
    1.9e-3 in normalized units -- 0.04 K of temperature, 0.03 hPa of pressure. That is below ERA5's own
    precision, so a model pretrained on one is valid on the other. Stage at ``dtype=np.float32`` if you
    want them exact and can afford twice the disk.
    """
    values, observed = split_prepared(prepared)
    if normalizer.masked:
        values = np.concatenate([values, observed[..., normalizer.masked].astype(np.float32)], axis=-1)
    if shift is not None:
        values = np.roll(values, shift, axis=2)
        observed = np.roll(observed, shift, axis=2)

    span, points = values.shape[0], values.shape[1] * values.shape[2]
    flat = torch.from_numpy(np.ascontiguousarray(values.reshape(span, points, values.shape[-1])))

    analysis = flat[:history]
    if channels is not None and channels > analysis.shape[-1]:
        # Zeros, which after normalization is each variable's own mean: a channel that says nothing,
        # rather than one that says something false. The slots are there for real multi-level data.
        analysis = torch.nn.functional.pad(analysis, (0, channels - analysis.shape[-1]))

    target = torch.full((points, len(SURFACE_FIELDS)), float("nan"))
    final = flat[-1]
    final_observed = torch.from_numpy(observed[-1].reshape(points, -1))
    for field, source in enumerate(target_index):
        if source >= 0:
            column = final[:, source]
            target[:, field] = torch.where(final_observed[:, source], column, torch.full_like(column, float("nan")))
    return analysis, target


def _roll(augment: bool, seed: int, start: int, num_longitudes: int) -> int | None:
    """A deterministic longitude roll for this window, so a resumed run repeats its own augmentation."""
    if not augment:
        return None
    return int(np.random.default_rng(seed * 1_000_003 + start).integers(num_longitudes))


class ERA5Window(torch.utils.data.Dataset):
    """
    Windows of consecutive ERA5 states, streamed lazily from cloud storage.

    Each item is ``history`` frames of input and one target frame ``lead_steps`` later. Nothing downloads
    until an item is requested, so a run streams what it needs instead of staging terabytes first.

    Args:
        indices: window start positions -- pass one of :func:`era5_splits`' arrays.
        history: input frames. Six at 6-hourly cadence is a day and a half of context.
        lead_steps: how far ahead the target sits, in store cadence units.
        augment: roll longitude by a random whole number of cells. Exact, free, 240 distinct variants.
        channels: pad the input to this width so one config serves both training stages.
    """

    def __init__(
        self,
        dataset=None,
        indices: np.ndarray | None = None,
        history: int = 6,
        lead_steps: int = 1,
        augment: bool = True,
        channels: int | None = None,
        variables: tuple[str, ...] = DEFAULT_VARIABLES,
        resolution: str | None = None,
        stats_cache: str | os.PathLike | None = None,
        seed: int = 0,
    ) -> None:
        self._resolution, self._variables = resolution, tuple(variables)
        self._dataset = dataset if dataset is not None else self._open()
        self._pid = os.getpid()

        self.variables = [name for name in self._variables if name in self._dataset]
        self.history, self.lead_steps = history, lead_steps
        self.augment, self.channels, self.seed = augment, channels, seed

        steps, span = self._dataset.sizes["time"], history + lead_steps
        indices = np.arange(steps - span) if indices is None else np.asarray(indices)
        self.indices = indices[indices <= steps - span]

        self.normalizer = Normalizer.fit(self._dataset, self.variables, cache=stats_cache)
        self.num_longitudes = int(self._dataset.sizes["longitude"])
        self.num_latitudes = int(self._dataset.sizes["latitude"])
        self.target_index = _target_index(self.variables)

    @property
    def input_channels(self) -> int:
        """Channels an item actually carries, before any padding -- set ``NatureConfig`` at least this wide."""
        return self.normalizer.width

    def _open(self):
        kwargs = {"variables": self._variables}
        if self._resolution:
            kwargs["resolution"] = self._resolution
        return open_weatherbench(**kwargs)

    def __getstate__(self) -> dict:
        """
        Drop the open store before this object crosses a process boundary.

        The cloud client underneath zarr keeps a connection pool and an event loop that do not survive
        being duplicated into a child process -- inherited, they abort the worker outright
        (``Check failed: next_worker->state == KICKED``) rather than failing politely. Stripping the
        handle here means each worker opens its own, which costs one metadata read and is the only
        arrangement that actually works. Use :func:`era5_loader`, which pairs this with the ``spawn``
        start method; ``fork`` copies the parent's memory directly and never consults this.
        """
        state = dict(self.__dict__)
        state["_dataset"] = None
        return state

    @property
    def store(self):
        """The backing store, opened lazily and reopened if we have landed in another process."""
        if self._dataset is None or os.getpid() != self._pid:
            self._dataset = self._open()
            self._pid = os.getpid()
        return self._dataset

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        start = int(self.indices[item])
        span = self.history + self.lead_steps
        raw = read_block(self.store, self.variables, slice(start, start + span))

        analysis, target = _build_window(
            self.normalizer.prepare(raw), self.normalizer, self.target_index, self.history, self.channels,
            _roll(self.augment, self.seed, start, self.num_longitudes),
        )
        stamps = (self.store.time.values[start : start + self.history]
                  .astype("datetime64[s]").astype(np.int64))
        return {
            "analysis": analysis,
            "calendar": calendar_features(torch.tensor(stamps, dtype=torch.float64)),
            "field_target": target,
        }

    def denormalise(self, values, variable: str):
        """Put a normalized value back into the variable's own physical units."""
        return self.normalizer.decode(values, variable)


def era5_loader(dataset: torch.utils.data.Dataset, batch_size: int = 1, num_workers: int = 4,
                shuffle: bool = True, **kwargs) -> torch.utils.data.DataLoader:
    """
    A :class:`~torch.utils.data.DataLoader` configured for this data, which is not the default one.

    Two settings are not optional for a streamed :class:`ERA5Window` and are easy to get wrong:

    ``multiprocessing_context="spawn"`` -- the default on Linux is ``fork``, which duplicates the
    parent's memory including the cloud client's event loop, and that client aborts the child on sight.
    Spawn starts each worker clean and pickles the dataset across, which :meth:`ERA5Window.__getstate__`
    has already made safe.

    ``persistent_workers=True`` -- spawn costs a second or two per worker to start, which is nothing
    once but ruinous if it happens at every epoch boundary.

    A :class:`CachedERA5` reading a local memmap has neither problem, so it keeps the faster default --
    which is one more reason to :func:`materialise` before a real run.

    The one thing spawn asks of you: in a *script*, the code that builds the loader must sit behind
    ``if __name__ == "__main__":``, because each worker re-imports the main module and would otherwise
    build a second loader inside every worker. Notebooks and Colab are unaffected -- there is no main
    module file to re-import.
    """
    streamed = isinstance(dataset, ERA5Window)
    options = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if num_workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 4
        if streamed:
            options["multiprocessing_context"] = "spawn"
    options.update(kwargs)
    return torch.utils.data.DataLoader(dataset, **options)


def materialise(
    dataset,
    indices: np.ndarray,
    path: str | os.PathLike,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    normalizer: Normalizer | None = None,
    dtype=np.float16,
    chunk: int = 64,
    progress: bool = True,
) -> Path:
    """
    Stage a slice of the record onto local disk as a memory-mapped array.

    Streaming costs about 549 ms per window against tens of milliseconds of compute, so a cloud-backed
    epoch is network-bound by an order of magnitude and the GPU idles through most of it. Staging turns
    that into a 5 ms disk read -- measured 105x -- and every epoch after the first runs at the card's
    speed rather than the network's.

    What is stored is the *normalized* form, not raw values, for a reason worth stating: raw sea-level
    pressure is about 101,325 Pa and float16 tops out at 65,504, so staging raw data in half precision
    turns every pressure reading into infinity. Normalized values sit within a few units of zero where
    float16 resolves about 0.005 -- 0.1 K and 0.08 hPa, at or below ERA5's own precision. The statistics
    used are written into the sidecar so the staged copy stays self-describing, and NaN is preserved so
    land still reads as land.

    At 240x121 with six variables one timestep is 349 KB, so a decade of 6-hourly data is about 5 GB.

    Returns:
        The path to the ``.npy`` memmap. A sidecar ``.json`` records shape, variables, timestamps and
        the statistics the values were normalized with.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name in variables if name in dataset]
    normalizer = normalizer or Normalizer.fit(dataset, names)
    shape = (len(indices), dataset.sizes["latitude"], dataset.sizes["longitude"], len(names))
    gigabytes = float(np.prod(shape)) * np.dtype(dtype).itemsize / 1e9
    if progress:
        print(f"[era5] staging {len(indices):,} steps -> {path}  ({gigabytes:.2f} GB, {np.dtype(dtype).name})")

    store = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    ordered = np.sort(np.asarray(indices))
    for offset in range(0, len(ordered), chunk):
        take = ordered[offset : offset + chunk]
        store[offset : offset + len(take)] = normalizer.prepare(read_block(dataset, names, take)).astype(dtype)
        if progress:
            done = offset + len(take)
            print(f"\r[era5]   {done:,}/{len(ordered):,}  ({100 * done / len(ordered):.1f}%)", end="", flush=True)
    store.flush()
    if progress:
        print()

    stamps = dataset.time.values[ordered].astype("datetime64[s]").astype(np.int64)
    path.with_suffix(".json").write_text(json.dumps(
        {"variables": names, "shape": list(shape), "dtype": np.dtype(dtype).name, "normalized": True,
         "statistics": normalizer.to_dict(), "indices": ordered.tolist(), "time": stamps.tolist()},
        indent=2))
    return path


class CachedERA5(torch.utils.data.Dataset):
    """
    The same windows as :class:`ERA5Window`, read from a :func:`materialise` memmap.

    Same logic, no network: both go through :func:`_build_window`. Measured at 5 ms per window against
    549 ms streamed -- 105x -- so use this for any run longer than a smoke test. The values agree with
    the streamed path to 1.9e-3 in normalized units (float16 quantization, below ERA5's own precision);
    stage at ``dtype=np.float32`` if you need them exact.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        history: int = 6,
        lead_steps: int = 1,
        augment: bool = True,
        channels: int | None = None,
        stats_cache: str | os.PathLike | None = None,
        indices: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        self.variables = list(meta["variables"])
        self.values = np.load(path, mmap_mode="r")
        self.times = np.array(meta.get("time") or [], dtype=np.int64)
        self.history, self.lead_steps = history, lead_steps
        self.augment, self.channels, self.seed = augment, channels, seed
        self.num_longitudes = self.values.shape[2]

        span = history + lead_steps
        available = np.arange(max(len(self.values) - span, 0))
        self.indices = available if indices is None else np.asarray(indices)[np.asarray(indices) < len(available)]

        # The staged array is already normalized, and the statistics that did it travel with it.
        # Preferring the sidecar over any passed-in cache is deliberate: decoding a prediction with
        # statistics other than the ones the values were encoded with gives plausible, wrong numbers.
        if meta.get("statistics"):
            self.normalizer = Normalizer.from_dict(meta["statistics"])
        elif stats_cache is not None and Path(stats_cache).exists():
            self.normalizer = Normalizer.from_dict(json.loads(Path(stats_cache).read_text()))
        else:
            raise ValueError(
                f"{path.with_suffix('.json')} carries no statistics and no stats_cache was given. "
                "Re-stage with materialise(), which writes them."
            )
        self.target_index = _target_index(self.variables)

    @property
    def input_channels(self) -> int:
        return self.normalizer.width

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict:
        start = int(self.indices[item])
        span = self.history + self.lead_steps
        prepared = np.asarray(self.values[start : start + span], np.float32)
        analysis, target = _build_window(
            prepared, self.normalizer, self.target_index, self.history, self.channels,
            _roll(self.augment, self.seed, start, self.num_longitudes),
        )
        if len(self.times):
            stamps = torch.tensor(self.times[start : start + self.history], dtype=torch.float64)
        else:
            # Six-hourly cadence from an arbitrary epoch: the calendar head needs the phase of the year
            # and of the day, and a fixed offset preserves both.
            stamps = torch.tensor([(start + step) * 21600.0 for step in range(self.history)], dtype=torch.float64)
        return {
            "analysis": analysis,
            "calendar": calendar_features(stamps),
            "field_target": target,
        }

    def denormalise(self, values, variable: str):
        return self.normalizer.decode(values, variable)


def corpus_scale(dataset, variables: tuple[str, ...] = DEFAULT_VARIABLES, parameters: int = 89_000_000) -> str:
    """
    The ratio that decides whether a model this size can be trained without memorising.

    The rule of thumb everyone quotes is around ten samples per parameter for supervised fitting from
    scratch. That number is soft, but the gap it measures here is not: best tracks miss it by four orders
    of magnitude and reanalysis clears it by three.
    """
    names = [name for name in variables if name in dataset]
    steps = dataset.sizes["time"]
    per_step = dataset.sizes["latitude"] * dataset.sizes["longitude"] * len(names)
    values = steps * per_step
    return "\n".join([
        f"timesteps                 {steps:>18,}",
        f"values per timestep       {per_step:>18,}",
        f"supervised values         {values:>18,}",
        f"model parameters          {parameters:>18,}",
        f"values per parameter      {values / parameters:>18,.0f}",
        f"best-track rows, Atlantic {55_230:>18,}  ({55_230 / parameters:.4f} per parameter)",
        "",
        "Pretraining clears the rule of thumb by ~3 orders of magnitude; best tracks miss it by ~4.",
        "So the backbone learns here, and the storm heads -- 0.96M parameters -- learn from tracks.",
    ])
