# Copyright 2026 Nathan. Apache-2.0.
"""
Regression tests for the ERA5 pretraining pipeline.

Every test here exists because the corresponding bug shipped once and passed inspection. They all ran
clean on shapes, value ranges and a falling training loss while the data was wrong, which is the whole
point: these assert on *physics and geometry*, not on tensor shapes.

No network. The fixture builds a synthetic store that reproduces the real one's awkward properties --
dimensions in ``(time, longitude, latitude)`` order, pressure far outside float16's range, land gaps in
sea-surface temperature, and precipitation that is mostly exact zeros.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch


xr = pytest.importorskip("xarray")

from naturev1.era5 import (  # noqa: E402
    CachedERA5,
    ERA5Window,
    Normalizer,
    equiangular_weights,
    era5_source_grid,
    era5_splits,
    lead_offsets,
    materialise,
    read_block,
)


LATITUDES = np.linspace(90.0, -90.0, 25)
LONGITUDES = np.linspace(0.0, 360.0, 48, endpoint=False)


@pytest.fixture
def store():
    """
    A synthetic ERA5-shaped store, with the real one's traps built in.

    Temperature is a clean function of latitude alone, so any mix-up between the latitude and longitude
    axes shows up immediately as a broken zonal profile rather than as plausible noise.
    """
    steps = 12
    lat_mesh, lon_mesh = np.meshgrid(LATITUDES, LONGITUDES, indexing="ij")
    temperature = 300.0 - 80.0 * np.sin(np.radians(np.abs(lat_mesh)))
    pressure = 101_325.0 + 400.0 * np.cos(np.radians(lon_mesh))
    precipitation = np.where(lat_mesh > 45.0, 0.0, 0.004)
    sea_surface = np.where(np.abs(lat_mesh) > 60.0, np.nan, 290.0)   # "land" gap

    def stack(field):
        # Stored transposed, exactly as WeatherBench 2 does it.
        return (("time", "longitude", "latitude"),
                np.repeat(field.T[None], steps, 0).astype(np.float32))

    return xr.Dataset(
        {
            "2m_temperature": stack(temperature),
            "mean_sea_level_pressure": stack(pressure),
            "total_precipitation_6hr": stack(precipitation),
            "sea_surface_temperature": stack(sea_surface),
        },
        coords={
            "time": np.arange("1959-01-01", "1959-01-04", np.timedelta64(6, "h"), dtype="datetime64[ns]")[:steps],
            "latitude": LATITUDES,
            "longitude": LONGITUDES,
        },
    )


VARIABLES = ("2m_temperature", "mean_sea_level_pressure", "total_precipitation_6hr", "sea_surface_temperature")


# --------------------------------------------------------------------------------------------------
# Quadrature weights
# --------------------------------------------------------------------------------------------------

def test_weights_sum_to_one():
    weights = equiangular_weights(LATITUDES, len(LONGITUDES))
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_pole_cells_are_not_annihilated():
    """``cos(latitude)`` gives the pole row 6.1e-17 -- zero in float32 -- and silently deletes the poles."""
    weights = equiangular_weights(LATITUDES, len(LONGITUDES)).reshape(len(LATITUDES), len(LONGITUDES))
    pole, equator = weights[0, 0], weights[len(LATITUDES) // 2, 0]
    assert pole > 0.0
    assert equator / pole < 1e3, "pole cells must be merely small, not obliterated"
    assert np.cos(np.radians(LATITUDES[0])) < 1e-16, "the cosine trap this replaces"


def test_weights_reproduce_the_true_area_of_a_band():
    """A band from -30 to +30 degrees is exactly sin(30) = half the sphere."""
    latitudes = np.linspace(90.0, -90.0, 181)     # one-degree, so the band has clean edges
    weights = equiangular_weights(latitudes, 4).reshape(len(latitudes), 4)
    band = (latitudes >= -30.0) & (latitudes <= 30.0)
    # Edge cells are half in, half out, so subtract half of each boundary row.
    total = weights[band].sum() - 0.5 * (weights[latitudes == 30.0].sum() + weights[latitudes == -30.0].sum())
    assert total == pytest.approx(0.5, abs=1e-4)


# --------------------------------------------------------------------------------------------------
# Geographic binding -- the bug that looked fine in every other way
# --------------------------------------------------------------------------------------------------

def test_read_block_forces_latitude_major_order(store):
    """The store declares (time, longitude, latitude); everything downstream assumes the transpose."""
    assert store["2m_temperature"].dims == ("time", "longitude", "latitude")
    block = read_block(store, ["2m_temperature"], slice(0, 2))
    assert block.shape == (2, len(LATITUDES), len(LONGITUDES), 1)


def test_samples_land_where_the_grid_says_they_do(store):
    """
    Reading without transposing yields the right point count, sane values and a falling loss, while
    every sample sits at the wrong place on Earth. Only a physical check catches it.
    """
    grid, _, _ = era5_source_grid(store)
    window = ERA5Window(store, indices=np.arange(2), history=1, augment=False, variables=VARIABLES)
    values = window[0]["analysis"][0]
    latitudes = np.degrees(grid.coords[:, 0].numpy())
    temperature = window.denormalise(values[:, 0], "2m_temperature").numpy()

    # Every sample must carry the value its own coordinate implies, to within float16-free round-off.
    expected = 300.0 - 80.0 * np.sin(np.radians(np.abs(latitudes)))
    assert np.abs(temperature - expected).max() < 0.5
    # A transposed read scores about zero here; the profile is sin(|lat|), so even a perfect read
    # correlates about -0.97 against |lat| rather than -1.
    assert np.corrcoef(np.abs(latitudes), temperature)[0, 1] < -0.95


def test_augmentation_rolls_longitude_not_latitude(store):
    """
    A roll along the wrong axis preserves every value -- so a multiset check passes -- while turning the
    planet pole over pole. Zonal means are invariant to a longitude roll and destroyed by a latitude one.
    """
    grid, _, _ = era5_source_grid(store)
    latitudes = np.degrees(grid.coords[:, 0].numpy())
    plain = ERA5Window(store, indices=np.arange(2), history=1, augment=False, variables=VARIABLES)
    rolled = ERA5Window(store, indices=np.arange(2), history=1, augment=True, seed=5, variables=VARIABLES)

    a = plain[0]["analysis"][0, :, 1].numpy()      # pressure, which does vary with longitude
    b = rolled[0]["analysis"][0, :, 1].numpy()
    assert not np.allclose(a, b), "augmentation did nothing"

    for low in range(-90, 90, 30):
        band = (latitudes >= low) & (latitudes < low + 30)
        assert a[band].mean() == pytest.approx(b[band].mean(), abs=1e-4)


# --------------------------------------------------------------------------------------------------
# Normalization: transforms, gaps, and what float16 can hold
# --------------------------------------------------------------------------------------------------

def test_precipitation_is_log_transformed_and_inverts(store):
    normalizer = Normalizer.fit(store, list(VARIABLES))
    raw = read_block(store, list(VARIABLES), slice(0, 4))
    encoded, _ = normalizer.encode(raw)
    index = list(VARIABLES).index("total_precipitation_6hr")

    assert np.abs(encoded[..., index]).max() < 10.0, "a raw z-score put precipitation 30 sigma out"
    back = normalizer.decode(encoded[..., index], "total_precipitation_6hr")
    assert np.abs(back - raw[..., index]).max() < 1e-5


def test_gaps_get_a_mask_channel_rather_than_the_mean(store):
    """Filling absent sea-surface temperature with the ocean mean claims there is warm water over land."""
    window = ERA5Window(store, indices=np.arange(2), history=1, augment=False, variables=VARIABLES)
    sea = list(window.variables).index("sea_surface_temperature")
    assert window.normalizer.missing[sea] > 0.0
    assert window.input_channels == len(VARIABLES) + 1

    values = window[0]["analysis"][0]
    mask = values[:, len(window.variables) + window.normalizer.masked.index(sea)]
    assert set(np.unique(mask.numpy())) <= {0.0, 1.0}
    assert 0.0 < mask.mean() < 1.0
    # Where unobserved, the value channel is exactly zero -- contributing nothing to a linear read.
    assert values[mask == 0.0, sea].abs().max() == 0.0


def test_unmeasured_model_fields_are_masked_out(store):
    """
    ERA5 carries no relative humidity or cloud here, so the mask must zero them.

    The mask, not NaN: the loss multiplies target by mask, and NaN times zero is still NaN, which would
    poison every other field in the batch along with it.
    """
    from naturev1.model import SURFACE_FIELDS

    window = ERA5Window(store, indices=np.arange(2), history=1, augment=False, variables=VARIABLES)
    item = window[0]
    target, mask = item["field_target"], item["field_mask"]
    assert torch.isfinite(target).all(), "targets must never carry NaN"

    # The fixture supplies temperature, pressure and precipitation. Humidity, cloud and the wind
    # components are absent from it, and absent must read as absent.
    supplied = {"t2m", "mslp", "precip_rate"}
    for index, field in enumerate(SURFACE_FIELDS):
        observed = mask[:, :, index]
        if field in supplied:
            assert (observed == 1.0).all(), f"{field} is measured and must be unmasked"
        else:
            assert (observed == 0.0).all(), f"{field} is not measured and must be masked out"


def test_every_forecast_lead_is_supervised(store):
    """
    A 6-hourly store already holds every lead the model forecasts, so all of them are free.

    Training on t+6h alone and then asking for t+120h trains one lead and hopes for eight.
    """
    from naturev1.model import SURFACE_FIELDS

    offsets = lead_offsets((6, 12, 18, 24), cadence_hours=6.0)
    assert offsets == (1, 2, 3, 4)

    window = ERA5Window(store, indices=np.arange(2), history=2, lead_steps=offsets,
                        augment=False, variables=VARIABLES)
    item = window[0]
    assert item["field_target"].shape[1:] == (len(offsets), len(SURFACE_FIELDS))
    assert item["field_mask"].shape == item["field_target"].shape

    # Each lead must read the frame it names, not the same frame repeated.
    raw = read_block(store, list(VARIABLES), slice(0, window.history + max(offsets)))
    normalized = window.normalizer.prepare(raw)
    temperature = list(VARIABLES).index("2m_temperature")
    for lead, offset in enumerate(offsets):
        frame = normalized[window.history - 1 + offset, :, :, temperature].reshape(-1)
        assert np.allclose(item["field_target"][:, lead, 0].numpy(), frame, atol=1e-6)


def test_lead_times_must_land_on_store_steps():
    with pytest.raises(ValueError, match="not a whole number"):
        lead_offsets((6, 9), cadence_hours=6.0)


def test_window_reserves_room_for_the_longest_lead(store):
    """The span a window needs is set by its furthest target, not its nearest."""
    offsets = (1, 2, 4)
    window = ERA5Window(store, indices=np.arange(100), history=3, lead_steps=offsets,
                        variables=VARIABLES)
    assert window.indices.max() + 3 + max(offsets) <= store.sizes["time"]


# --------------------------------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------------------------------

def test_float16_staging_survives_sea_level_pressure(store, tmp_path):
    """
    Raw pressure is ~101,325 Pa and float16 tops out at 65,504, so staging raw values makes every
    pressure reading infinite. Staging the normalized form keeps it well inside range.
    """
    normalizer = Normalizer.fit(store, list(VARIABLES))
    raw = read_block(store, list(VARIABLES), slice(0, 2))
    assert raw[..., 1].max() > np.finfo(np.float16).max, "fixture must reproduce the overflow"
    with np.errstate(over="ignore"):      # the overflow is the point of this assertion
        assert np.isinf(raw[..., 1].astype(np.float16)).all()

    staged = normalizer.prepare(raw).astype(np.float16)
    assert np.isfinite(staged[..., 1]).all()
    back = normalizer.decode(staged[..., 1].astype(np.float32), "mean_sea_level_pressure")
    assert np.abs(back - raw[..., 1]).max() < 5.0     # < 0.05 hPa


def test_staged_and_streamed_windows_agree(store, tmp_path):
    path = materialise(store, np.arange(8), tmp_path / "cache.npy", variables=VARIABLES, progress=False)
    streamed = ERA5Window(store, indices=np.arange(6), history=2, augment=False, variables=VARIABLES)
    staged = CachedERA5(path, history=2, augment=False)

    for key in ("analysis", "calendar", "field_target"):
        a, b = streamed[0][key], staged[0][key]
        assert torch.equal(torch.isnan(a), torch.isnan(b)), f"{key}: NaN pattern differs"
        # float16 staging, so close rather than equal -- below ERA5's own precision.
        assert float((torch.nan_to_num(a) - torch.nan_to_num(b)).abs().max()) < 5e-3


def test_staged_copy_carries_its_own_statistics(store, tmp_path):
    """Decoding with statistics other than the encoding ones gives plausible, wrong numbers."""
    path = materialise(store, np.arange(8), tmp_path / "cache.npy", variables=VARIABLES, progress=False)
    staged = CachedERA5(path, history=2, augment=False)
    reference = Normalizer.fit(store, list(VARIABLES))
    assert staged.normalizer.variables == reference.variables
    assert np.allclose(staged.normalizer.mean, reference.mean)
    assert np.allclose(staged.normalizer.std, reference.std)


# --------------------------------------------------------------------------------------------------
# Splits and worker safety
# --------------------------------------------------------------------------------------------------

def test_splits_are_ordered_in_time_and_disjoint(store):
    """A random split lets the model interpolate between two states it has already seen."""
    splits = era5_splits(store, val_years=1, test_years=1, steps_per_year=3)
    assert splits["train"][-1] < splits["val"][0] < splits["test"][0]
    assert not (set(splits["train"]) & set(splits["val"]) & set(splits["test"]))


def test_split_rejects_a_record_too_short(store):
    with pytest.raises(ValueError, match="too short"):
        era5_splits(store, val_years=10, test_years=10, steps_per_year=1460)


def test_pickling_drops_the_store(store):
    """The cloud client aborts a worker outright if it is duplicated into one."""
    window = ERA5Window(store, indices=np.arange(4), history=1, variables=VARIABLES)
    restored = pickle.loads(pickle.dumps(window))
    assert restored._dataset is None
    assert restored.variables == window.variables
    assert np.allclose(restored.normalizer.mean, window.normalizer.mean)


def test_windows_never_run_off_the_end(store):
    """A window start must leave room for its history and its lead."""
    window = ERA5Window(store, indices=np.arange(100), history=4, lead_steps=2, variables=VARIABLES)
    assert window.indices.max() + 4 + 2 <= store.sizes["time"]
    assert window[len(window) - 1]["analysis"].shape[0] == 4


def test_augmentation_is_deterministic(store):
    """A resumed run must repeat its own augmentation, or the resume is not a resume."""
    window = ERA5Window(store, indices=np.arange(4), history=1, augment=True, seed=11, variables=VARIABLES)
    assert torch.equal(window[2]["analysis"], window[2]["analysis"])
    other = ERA5Window(store, indices=np.arange(4), history=1, augment=True, seed=12, variables=VARIABLES)
    assert not torch.equal(window[2]["analysis"], other[2]["analysis"])


def test_grid_collate_is_picklable(store):
    """
    The spawn start method pickles the collate function to every worker, and a local function or a
    lambda cannot be pickled -- the failure is late and reads ``Can't pickle local object``.
    """
    from naturev1.era5 import GridCollate, era5_loader

    grid, _, _ = era5_source_grid(store)
    collate = pickle.loads(pickle.dumps(GridCollate(grid)))
    window = ERA5Window(store, indices=np.arange(4), history=1, variables=VARIABLES)
    batch = collate([window[0], window[1]])

    assert batch["analysis"].shape[0] == 2
    assert batch["analysis_grid"].num_points == grid.num_points
    assert "output_grid" not in batch, "an output grid that was not asked for must not appear"

    loader = era5_loader(window, batch_size=2, num_workers=0, analysis_grid=grid, output_grid=grid)
    assert isinstance(loader.collate_fn, GridCollate)
    assert next(iter(loader))["output_grid"].num_points == grid.num_points
