# NatureV1

A probabilistic weather and hurricane-track model that treats satellite imagery as **geolocated samples on
a sphere**, not a rectangle of pixels.

```bash
pip install "naturev1[all]"
```

## Why not a ViT or CNN over the image

Three things break when you crop a GOES scene, divide by 255 and hand it to a vision backbone:

1. **A GOES pixel is not a fixed size.** ABI scans in fixed *angular* steps from geostationary orbit.
   Measured off a real scene, the limb pixel covers **64× the ground area** of the nadir pixel. A
   convolution weights them equally and silently over-counts the stretched edge of the disk.
2. **An array index is not a place.** Nothing in `image[400, 1200]` says 24.7°N 81.3°W.
3. **Normalizing to 0–1 destroys the physics.** Channel 13 is a brightness temperature: 180 K is an
   overshooting top, 300 K is warm ocean. Per-image rescaling erases exactly what was measured.

NatureV1 solves the fixed-grid projection for every pixel's latitude/longitude (validated against the
file's own metadata — it recovers 2.07 km at nadir where nominal C13 resolution is 2 km), computes each
pixel's true ground footprint and uses it as a quadrature weight, and keeps brightness temperature in
kelvin. Satellite and analysis data then fuse without either being resampled onto the other's grid.

## Every output is a distribution

Track comes out as weighted **scenarios**, each a full trajectory with a tilting uncertainty ellipse.
Collapsing them to one line scores 116 under the mixture likelihood where keeping both branches scores
8.2 — because the average of "recurves offshore" and "hits the coast" is a track through somewhere the
storm was never going. Fields carry per-point variance, landfall a probability per lead time, intensity a
Saffir-Simpson category with an interval.

## It trains twice, because the labels are 4 orders of magnitude too few

An 88M-parameter model fitted to the Atlantic best-track archive is not a weather model. The archive is
55,230 points — 1,167 landfalls, 1,839 rapid intensifications, 2,587 with a radius of maximum wind. That
is **0.0006 supervised values per parameter**, and what you get back is an expensive lookup table for
storms that already happened.

So the backbone never sees it.

**Stage one is self-supervised on ERA5 reanalysis**, where the label is free: given the atmosphere now,
predict it at +6 through +120 hours. The WeatherBench 2 six-hourly store is 92,040 timesteps from 1959 to
2021 — 1.6×10¹⁰ supervised values, **180 per parameter**, and nobody annotates anything.

**Stage two fine-tunes the storm heads with the backbone frozen.** 0.96M parameters of 89M — 1.1% — ever
see a best track. Those 24,585 paired storm points can fit 0.96M parameters; they cannot fit 89M.

```python
from naturev1 import ERA5Window, StormWindow, pair_tracks_with_reanalysis

pretrain = ERA5Window(era5, indices=splits["train"], lead_steps=offsets)   # free labels
starts, targets, _ = pair_tracks_with_reanalysis(tracks, store_times, offsets)
finetune = StormWindow(ERA5Window(era5, indices=starts), targets)          # real outcomes
model.freeze_backbone(True)
```

Longitude rotation is an **exact** augmentation here, not an approximation: on an equiangular grid,
rotating the globe by a whole number of cells is a roll of the array, and this architecture is exactly
equivariant to it. 240 valid atmospheres, free, nothing resampled.

## Four things the obvious implementation gets wrong

Each of these ran clean on shapes, ranges and a falling loss while the data was wrong, so the regression
tests assert on physics and geometry instead:

1. **Dimension order.** WeatherBench 2 stores `(time, longitude, latitude)`; the coordinate mesh is
   latitude-major. Read untransposed you get the right point count, sane values, and every sample bound
   to the wrong place on Earth. Correlation of temperature against its assigned latitude: ~0 → **−0.873**.
2. **Cell weights.** `cos(latitude)` gives the pole row 6.1e-17 — zero in float32 — deleting the poles
   from every area-weighted sum. True spherical cell area makes a pole cell 306× lighter, not 10¹⁶×.
3. **Precipitation.** 15% of the grid is exactly zero and the max sits 30σ out, so a z-score trains the
   model to predict zero everywhere. `log1p(x/0.1mm)` brings it to 4.5σ and inverts exactly.
4. **Missing is not average.** Sea-surface temperature is absent over 27.9% of the grid; filling it with
   the ocean mean asserts warm water over Kansas. Gappy variables get an observation channel.

## Heads start at climatology, not at zero

A linear head emits about zero, so asked for central pressure in hPa it opens 1000 off — a squared error
of a million. Measured on a real fine-tuning step, the intensity term was 119,191 of a 60,565 total, with
a gradient norm of 207,018. Anchoring each head at Atlantic climatology and starting each log-variance at
the observed spread (rather than claiming ±1 kt about peak wind) brings the same step to **total 58,
intensity 4, eyewall 5, gradient norm 13** — with no architecture change. The RI classifier opens at the
measured 4.11% base rate, not at even odds.

## Built for a runtime that dies

Checkpoints go down on a wall-clock interval, written to a temp file and renamed into place, so a killed
cell leaves the previous checkpoint intact. Resume restores optimizer moments, LR schedule, scaler and RNG
state — not just weights.

## Not trained

No weights here have seen real data. The architecture, losses, ingest and training loop are complete and
exercised end to end on real GOES-19 imagery, but **nothing here should inform a decision about a real
storm.** The National Hurricane Center is the authoritative source for tropical cyclone forecasts.

Made by Nathan.
