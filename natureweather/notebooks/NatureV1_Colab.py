"""
NatureV1 on Colab -- copy each block into its own cell.

    1  install and check the card
    2  build the 88M model
    3  the data argument: why this trains twice
    4  stage one data -- ERA5 reanalysis, 1.6e10 free labels
    5  fill the card: benchmark, autotune the batch, price the run
    6  stage one -- pretrain the backbone, checkpointing every minute
    7  stage two data -- HURDAT2 best tracks paired with the same reanalysis
    8  stage two -- fine-tune the storm heads with the backbone frozen
    9  live forecast from real GOES imagery
   10  hourly watcher

Cells 1-6 are the long pole and need no storm data at all. Cells 7-8 are quick -- there are only 24,585
paired storm points in the entire Atlantic record, which is the whole reason the backbone is frozen for
them. Cell 9 runs on an untrained model too; it will produce confident nonsense until 6 and 8 have run.
"""

# ══ CELL 1 ══ install and check the card ══════════════════════════════════════
# !pip install -q "naturev1[all]"
# !nvidia-smi --query-gpu=name,memory.total,power.max_limit --format=csv

from naturev1 import device_report


DEV = device_report()
print(DEV)
# On an RTX 6000 Blackwell you should see ~95.6 GB and bf16_supported=True. If bf16 is False you are on
# an older card: set precision="fp16" in cells 6 and 8, which needs loss scaling that the Trainer adds.

# ══ CELL 2 ══ build the 88M model ═════════════════════════════════════════════
import datetime as dt
import json
import os

import numpy as np
import torch
from naturev1 import (
    SURFACE_FIELDS,
    WEATHER_TYPES,
    NatureConfig,
    NatureV1,
    build_forecast,
    calendar_features,
    fetch_latest,
    normalize_channels,
    scene_from_netcdf,
    watch,
)

from ihelix import FieldGrid, Geometry, fibonacci_sphere


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CFG = NatureConfig(
    satellite_channels=6, analysis_channels=24, environment_channels=8,
    hidden_size=512, num_layers=17, num_heads=8, num_kv_heads=4,
    head_dim=64, intermediate_size=2048,
    latent_points=4096, min_radius_km=120.0, max_radius_km=1600.0,
    history_frames=6, lead_times_hours=(6, 12, 18, 24, 36, 48, 72, 96, 120), track_modes=6,
)
LATENT = fibonacci_sphere(CFG.latent_points, num_neighbours=CFG.latent_neighbours,
                          cluster_size=CFG.latent_cluster)
model = NatureV1(CFG, LATENT).to(DEVICE)
print(f"NatureV1: {model.num_parameters()/1e6:.2f}M parameters on {DEVICE}")

# The heads start at Atlantic climatology rather than at zero, with the uncertainty that goes with it.
with torch.no_grad():
    prior = model.eyewall_head(torch.zeros(1, CFG.hidden_size, device=DEVICE))
print(f"untrained prior: peak wind {float(prior['eyewall_peak_wind_kt'][0,0]):.0f} kt "
      f"+/- {float(prior['eyewall_peak_wind_log_var'][0,0].mul(0.5).exp()):.0f} kt")

# ══ CELL 3 ══ the data argument: why this trains twice ════════════════════════
from naturev1 import catalogue, corpus_scale, open_weatherbench


ERA5 = open_weatherbench()        # streams from cloud storage; nothing downloads yet
print(corpus_scale(ERA5, parameters=model.num_parameters()))
print()
print(catalogue())

# The number that matters is "values per parameter". Training an 88M model on the 55,230-row Atlantic
# best-track archive gives 0.0006 of them, and produces a lookup table for storms that already happened.
# Reanalysis gives 180, because the label is free: the target for the atmosphere now is the atmosphere
# six hours from now. So the backbone learns here, and only the 0.96M storm-head parameters ever see
# best tracks.

# ══ CELL 4 ══ stage one data -- ERA5 reanalysis ═══════════════════════════════
from naturev1 import ERA5Window, era5_source_grid, era5_splits, lead_offsets


SRC, LAT, LON = era5_source_grid(ERA5)      # read-only grid: 29,040 points in ~5 ms
SPLITS = era5_splits(ERA5, val_years=4, test_years=2)
OFFSETS = lead_offsets(CFG.lead_times_hours)     # (1,2,3,4,6,8,12,16,20) at 6-hourly cadence
STATS = "/content/era5_stats.json"

for name, index in SPLITS.items():
    print(f"  {name:5} {len(index):>7,} windows  "
          f"{str(ERA5.time.values[index[0]])[:10]} -> {str(ERA5.time.values[index[-1]])[:10]}")
print(f"\nlead times {CFG.lead_times_hours} h -> store offsets {OFFSETS}")
print("split by TIME, not at random: weather is autocorrelated for days, and a random split lets the")
print("model interpolate between two states it has already seen.")

train_stream = ERA5Window(ERA5, indices=SPLITS["train"], history=CFG.history_frames,
                          lead_steps=OFFSETS, channels=CFG.analysis_channels, stats_cache=STATS)
print(f"\n{train_stream.normalizer.report()}")
print(f"\nchannels produced {train_stream.input_channels} -> padded to {CFG.analysis_channels}")

# ── Staging. Streaming costs ~550 ms per window against tens of ms of compute, so a cloud-backed epoch
# ── is network-bound by an order of magnitude and the GPU idles through most of it. Staging is 105x.
from naturev1 import CachedERA5, materialise


STAGE_YEARS = 20                                 # ~10 GB at float16; raise it if you have the disk
STAGE_STEPS = STAGE_YEARS * 1460
CACHE = "/content/era5_cache.npy"

if not os.path.exists(CACHE):
    materialise(ERA5, SPLITS["train"][-STAGE_STEPS:], CACHE, normalizer=train_stream.normalizer)
train_ds = CachedERA5(CACHE, history=CFG.history_frames, lead_steps=OFFSETS,
                      channels=CFG.analysis_channels, augment=True)
print(f"staged {len(train_ds):,} windows")

# Augmentation is a longitude roll, which on an equiangular grid is exactly a rotation of the globe --
# and this model is exactly equivariant to that, so all 240 rolls are real atmospheres, not approximations.

# ══ CELL 5 ══ fill the card: benchmark, autotune, price the run ═══════════════
from naturev1 import autotune_batch_size, benchmark_steps, era5_loader, format_plan, masked_gaussian_nll, training_plan


GRAD_CKPT = True      # trades ~30% speed for a much larger batch; on 96 GB this is usually the win
model.gradient_checkpointing_enable(GRAD_CKPT)


def make_step(batch_size):
    """A closure that runs one full training step at this batch size, for the autotuner."""
    items = [train_ds[i] for i in range(batch_size)]
    batch = {k: torch.stack([x[k] for x in items]).to(DEVICE) for k in items[0]}

    def step():
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model(analysis=batch["analysis"], analysis_grid=SRC,
                        calendar=batch["calendar"], output_grid=SRC)
            loss = masked_gaussian_nll(out["field_mean"], out["field_log_var"],
                                       torch.where(batch["field_mask"] > 0, batch["field_target"],
                                                   torch.nan))
        loss.backward()
        model.zero_grad(set_to_none=True)
    return step


BATCH = autotune_batch_size(make_step, start=1, target_fraction=0.85)
print(f"largest batch that fits at 85% of VRAM: {BATCH}")

mark = benchmark_steps(make_step(BATCH), BATCH, gradient_checkpointing=GRAD_CKPT)
print(mark)

PLAN = training_plan(mark.samples_per_second, corpus_samples=len(train_ds), epochs=8,
                     watts=600.0, electricity_per_kwh=0.15, cloud_per_hour=2.50)
print("\nstage one, 8 epochs:")
print(format_plan(PLAN))
print(f"\nsteps for the plan: {int(len(train_ds) * 8 / BATCH):,}  <- use this as max_steps in cell 6")

# ══ CELL 6 ══ stage one -- pretrain, checkpointing every minute ═══════════════
from naturev1 import Trainer, TrainSettings


CKPT = "/content/drive/MyDrive/naturev1_ckpt"      # Drive outlives the VM
pretrain = TrainSettings(
    stage="pretrain",                    # everything trains; the label is the next state
    learning_rate=3e-4, warmup_steps=1000,
    max_steps=int(len(train_ds) * 8 / BATCH),
    grad_accum=1, precision="bf16",      # bf16 on Blackwell: no loss scaling needed
    checkpoint_dir=f"{CKPT}/stage1", checkpoint_seconds=60,
    hub_repo="Sigmandndnns/NatureV1-500", hub_push_seconds=900,
    ema_decay=0.999, log_every=25,
)

loader = era5_loader(train_ds, batch_size=BATCH, num_workers=4, shuffle=True,
                     analysis_grid=SRC, output_grid=SRC)
trainer = Trainer(model, pretrain, device=DEVICE)
trainer.resume()          # picks up wherever the last cell died; pulls from the Hub on a fresh VM
trainer.fit(loader, epochs=8)

# ══ CELL 7 ══ stage two data -- best tracks paired with the same reanalysis ═══
from naturev1 import (
    StormWindow,
    format_pairing,
    pair_tracks_with_reanalysis,
    parse_hurdat2,
    rapid_intensification,
    split_by_storm,
)
from naturev1.besttrack import download


HURDAT = download("https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt",
                  "/content/hurdat2.txt")
tracks = parse_hurdat2(HURDAT)
print(f"HURDAT2: {len(tracks):,} storms, {sum(len(t) for t in tracks):,} points, "
      f"{sum(int(t.landfall.sum()) for t in tracks):,} landfalls")

walk = rapid_intensification(tracks, threshold_kt=30.0)
print(f"rapid intensification: {walk['positives']:,} of {walk['eligible']:,} eligible points "
      f"= {100*walk['base_rate']:.2f}%")
print("  a classifier that always says 'no' scores 96% here and saves nobody, which is why the RI head")
print("  is trained with a focal loss and judged on precision, recall and Brier score -- not accuracy.")

GROUPS = split_by_storm(tracks)     # by SEASON: points from one storm are near-duplicates
STORE_TIMES = ERA5.time.values.astype("datetime64[s]").astype(np.int64)


def storm_split(which):
    starts, targets, report = pair_tracks_with_reanalysis(
        GROUPS[which], STORE_TIMES, OFFSETS, history=CFG.history_frames)
    base = ERA5Window(ERA5, indices=starts, history=CFG.history_frames, lead_steps=OFFSETS,
                      augment=False,        # a rolled globe would move the coastline the storm hit
                      channels=CFG.analysis_channels, stats_cache=STATS)
    return StormWindow(base, targets), report


storm_train, report = storm_split("train")
print(f"\npairing HURDAT2 against ERA5 1959-2021:\n{format_pairing(report)}")
print(f"\n{storm_train.describe()}")

# ══ CELL 8 ══ stage two -- fine-tune the heads, backbone frozen ═══════════════
trainable, total = model.freeze_backbone(True)
print(f"trainable: {trainable:,} of {total:,} ({100*trainable/total:.1f}%)")
print("This is the answer to the overfitting problem. 24,585 storm points cannot fit 89M parameters,")
print("but they can fit 0.96M -- and the backbone they sit on saw 1.6e10 values in stage one.")

finetune = TrainSettings(
    stage="finetune",
    learning_rate=1e-4, warmup_steps=200, max_steps=20_000,
    precision="bf16",
    checkpoint_dir=f"{CKPT}/stage2", checkpoint_seconds=60,
    hub_repo="Sigmandndnns/NatureV1-500", hub_push_seconds=900,
    ema_decay=0.999, early_stopping_patience=10, log_every=25,
)
storm_loader = era5_loader(storm_train, batch_size=max(BATCH // 2, 1), num_workers=4, shuffle=True,
                           analysis_grid=SRC, output_grid=SRC)
storm_trainer = Trainer(model, finetune, device=DEVICE)
storm_trainer.resume()
storm_trainer.fit(storm_loader, epochs=50)

# ══ CELL 9 ══ live forecast from real GOES imagery ════════════════════════════
CHANNELS = ("C13", "C09")     # clean IR window + mid-level water vapour
STORM     = (24.6, -78.2)     # current storm centre (lat, lon)
CITY      = (25.77, -80.19)   # somewhere you want a local forecast
DATA_DIR  = "./test_data"


def load_scene(paths, center, half_width_deg=9.0, stride=4):
    """Geolocated samples + true pixel footprints, cropped to a real box on the planet."""
    scene = scene_from_netcdf(
        {c: str(p) for c, p in paths.items()},
        bounds=(center[0]-half_width_deg, center[0]+half_width_deg,
                center[1]-half_width_deg, center[1]+half_width_deg),
        stride=stride,
    )
    coords = torch.tensor(np.stack([np.radians(scene.latitude),
                                    np.radians(scene.longitude) % (2*np.pi)], -1), dtype=torch.float64)
    grid = FieldGrid.from_points(coords, Geometry.globe(), num_neighbours=16, cluster_size=64,
                                 weights=torch.tensor(scene.area_km2, dtype=torch.float64))
    values = torch.tensor(normalize_channels(scene.values, scene.channels), dtype=torch.float32)
    pad = CFG.satellite_channels - values.shape[-1]
    if pad > 0:
        values = torch.cat([values, torch.zeros(values.shape[0], pad)], -1)
    return grid, values, scene


@torch.no_grad()
def forecast_now(storm=STORM, city=CITY):
    paths = fetch_latest(DATA_DIR, CHANNELS, satellite="east", product="conus")
    grid, values, scene = load_scene(paths, storm)
    print(scene)
    frames = CFG.history_frames
    sat = values[None, None].expand(1, frames, -1, -1).contiguous().to(DEVICE)
    cal = calendar_features(torch.full((1, frames), scene.timestamp.timestamp())).to(DEVICE)
    out_grid = fibonacci_sphere(2000, num_neighbours=16, cluster_size=50)
    model.eval()
    out = model(satellite=sat, satellite_grid=grid, calendar=cal,
                output_grid=out_grid, neighbours=12)
    return build_forecast(out, CFG.lead_times_hours, scene.timestamp,
                          storm_center=storm, output_grid=out_grid, point_of_interest=city)


fc = forecast_now()
print(json.dumps({k: fc[k] for k in ("issued", "enso")}, indent=2))
print("landfall:", {k: v for k, v in fc["landfall"].items() if k != "by_lead"})
for s in fc["track_scenarios"][:4]:
    p = s["track"][5]
    print(f"  {s['probability']:>5.0%}  +48h -> {p['latitude']:6.2f},{p['longitude']:7.2f}   "
          f"95% cone {p['cone_radius_km_95']:.0f} km")
for e in fc["eyewall"][:3]:
    print(f"  +{e['lead_hours']:3d}h  peak {e['peak_wind_kt']:.0f} kt  RMW {e['rmw_nmi']:.0f} nmi")
print(f"  rapid intensification (30 kt/24h): {fc['rapid_intensification']['probability_30kt']:.0%}")
pf = fc["point_forecast"]["forecast"][3]
print(f"  local +{pf['lead_hours']}h: {pf['weather']} ({pf['weather_confidence']:.0%})")

# ══ CELL 10 ══ hourly watcher ═════════════════════════════════════════════════
def on_new_scene(paths):
    result = forecast_now()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M")
    os.makedirs("./forecasts", exist_ok=True)
    with open(f"./forecasts/forecast_{stamp}.json", "w") as handle:
        json.dump(result, handle, indent=2)
    top = result["track_scenarios"][0]
    print(f"  most likely ({top['probability']:.0%}): +120h -> "
          f"{top['track'][-1]['latitude']:.2f},{top['track'][-1]['longitude']:.2f}")
    print(f"  landfall peak probability {result['landfall']['peak_probability']:.0%}")


# watch(on_new_scene, interval_seconds=3600, directory=DATA_DIR, channels=CHANNELS)
