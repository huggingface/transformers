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

## Built for a runtime that dies

Checkpoints go down on a wall-clock interval, written to a temp file and renamed into place, so a killed
cell leaves the previous checkpoint intact. Resume restores optimizer moments, LR schedule, scaler and RNG
state — not just weights.

## Not trained

No weights here have seen real data. The architecture, losses, ingest and training loop are complete and
exercised end to end on real GOES-19 imagery, but **nothing here should inform a decision about a real
storm.** The National Hurricane Center is the authoritative source for tropical cyclone forecasts.

Made by Nathan.
