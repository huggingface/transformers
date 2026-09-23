# ═══════════════════════════════════════════════════════════════════════════════════════════════════
#  NatureV1 — the whole thing in one cell. Paste and run; it installs what it needs.
#
#  Set the RUN_* switches below and execute. Everything resumes: if the cell dies, re-run it and it
#  picks up from the last checkpoint (written every 60 s, and mirrored to the Hub every 15 min).
#
#  Rough costs on an RTX 6000 Blackwell (96 GB). Cell 5 measures the real numbers on your card.
#    STAGE_YEARS=20 staging      ~10 GB disk, one-off download
#    RUN_PRETRAIN                the long pole — hours to days; watch the [val] lines
#    RUN_FINETUNE                ~24,585 samples, far quicker, early-stops on held-out loss
#    RUN_FORECAST                seconds
# ═══════════════════════════════════════════════════════════════════════════════════════════════════

RUN_PRETRAIN  = True      # stage one: self-supervised on ERA5 reanalysis
RUN_FINETUNE  = True      # stage two: storm heads on HURDAT2, backbone frozen
RUN_FORECAST  = True      # live forecast from the newest GOES scene
RUN_WATCHER   = False     # then keep re-forecasting every hour, forever

STAGE_YEARS   = 20        # years of ERA5 staged to local disk (~0.5 GB/year at float16)
EPOCHS        = 8         # passes over the staged corpus in stage one
GRAD_CKPT     = True      # recompute activations: ~30% slower, much bigger batch. Worth it on 96 GB.
CKPT_DIR      = "/content/drive/MyDrive/naturev1_ckpt"   # Drive outlives the VM
HUB_REPO      = "Sigmandndnns/NatureV1-500"
STORM         = (24.6, -78.2)      # current storm centre (lat, lon)
CITY          = (25.77, -80.19)    # somewhere you want a local forecast

# ═══ 0 ═══ bootstrap ═══════════════════════════════════════════════════════════════════════════════
# Standard library only, and it runs before numpy or torch are imported. That ordering is the whole
# point: pip may upgrade numpy while satisfying zarr or gcsfs, and a numpy that changes underneath an
# already-imported torch gives binary-incompatibility errors that look like a bug in this code.
import importlib
import importlib.util
import subprocess
import sys


_NEEDED = {                                    # import name -> pip requirement
    "naturev1":        "naturev1[all]>=0.4.0",
    "ihelix":          "ihelix>=0.3.0",
    "xarray":          "xarray>=2023.1",
    "zarr":            "zarr>=2.16",
    "gcsfs":           "gcsfs>=2023.1",
    "netCDF4":         "netCDF4>=1.6",
    "huggingface_hub": "huggingface_hub>=0.20",
}
_MIN_NATUREV1 = (0, 4, 0)


def _present(module: str) -> bool:
    """Is this importable right now? A broken install counts as absent."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _naturev1_too_old() -> bool:
    """The cell below uses APIs added in 0.4.0, so an older copy is as good as missing."""
    try:
        import naturev1
        parts = tuple(int(piece) for piece in naturev1.__version__.split(".")[:3])
        return parts < _MIN_NATUREV1
    except Exception:
        return True


def _bootstrap() -> bool:
    """Install whatever is missing. Returns True if anything was installed."""
    missing = [req for module, req in _NEEDED.items() if not _present(module)]
    if not missing and _naturev1_too_old():
        missing = [_NEEDED["naturev1"]]
    if not missing:
        return False

    print(f"installing {len(missing)} package(s): {', '.join(missing)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *missing])
    except subprocess.CalledProcessError as error:
        print(f"\n!! pip failed (exit {error.returncode}). Install by hand and re-run:")
        print(f"!!   !pip install --upgrade {' '.join(missing)}")
        raise SystemExit(1) from None
    importlib.invalidate_caches()
    return True


def _stale_after_install() -> list[str]:
    """
    Which already-imported packages did pip move out from under us.

    Colab usually has numpy loaded before any user cell runs, so "was it imported" is the wrong
    question -- it would demand a restart every single time. The right question is whether the copy in
    memory still matches the copy on disk, which is only false when pip actually upgraded it.
    """
    import importlib.metadata

    stale = []
    for name in ("numpy", "torch"):
        loaded = getattr(sys.modules.get(name), "__version__", None)
        if loaded is None:
            continue
        try:
            if importlib.metadata.version(name) != loaded:
                stale.append(f"{name} {loaded} -> {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return stale


_INSTALLED = _bootstrap()
_STALE = _stale_after_install() if _INSTALLED else []
if _STALE:
    # The version in memory no longer matches the one on disk. Carrying on gives binary-incompatibility
    # errors far from here; restarting is the only reliable fix.
    print(f"\n!! pip upgraded something already loaded: {', '.join(_STALE)}")
    print("!! Runtime -> Restart session, then run this cell again. Nothing is lost.")
    raise SystemExit(0)
print("dependencies ready\n")

# ───────────────────────────────────────────────────────────────────────────────────────────────────
import datetime as dt
import json
import os

import numpy as np
import torch
from naturev1 import (
    CachedERA5,
    ERA5Window,
    NatureConfig,
    NatureV1,
    StormWindow,
    Trainer,
    TrainSettings,
    autotune_batch_size,
    benchmark_steps,
    build_forecast,
    calendar_features,
    catalogue,
    corpus_scale,
    device_report,
    era5_loader,
    era5_source_grid,
    era5_splits,
    fetch_latest,
    format_pairing,
    format_plan,
    lead_offsets,
    masked_gaussian_nll,
    materialise,
    normalize_channels,
    open_weatherbench,
    pair_tracks_with_reanalysis,
    parse_hurdat2,
    rapid_intensification,
    scene_from_netcdf,
    split_by_storm,
    training_plan,
    watch,
)
from naturev1.besttrack import download

from ihelix import FieldGrid, Geometry, fibonacci_sphere


DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS = ("C13", "C09")          # clean IR window + mid-level water vapour
DATA_DIR = "./test_data"
STATS    = "/content/era5_stats.json"
CACHE    = "/content/era5_cache.npy"


def banner(text):
    print(f"\n{'═' * 99}\n  {text}\n{'═' * 99}")


# ═══ 1 ═══ hardware ════════════════════════════════════════════════════════════════════════════════
banner("HARDWARE")
print(device_report())

# ═══ 2 ═══ the model ═══════════════════════════════════════════════════════════════════════════════
banner("MODEL")
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
print(f"local head radii (km): {[round(float(r), 1) for r in model.blocks[0].local.radii.detach()]}")
with torch.no_grad():
    prior = model.eyewall_head(torch.zeros(1, CFG.hidden_size, device=DEVICE))
print(f"untrained prior: peak wind {float(prior['eyewall_peak_wind_kt'][0, 0]):.0f}"
      f" +/- {float(prior['eyewall_peak_wind_log_var'][0, 0].mul(0.5).exp()):.0f} kt  (Atlantic climatology)")

# ═══ 3 ═══ the corpora, and why this trains twice ══════════════════════════════════════════════════
banner("CORPORA")
ERA5 = open_weatherbench()          # streams from cloud storage; nothing downloads yet
print(corpus_scale(ERA5, parameters=model.num_parameters()))
print()
print(catalogue())

# ═══ 4 ═══ stage one data — ERA5 ═══════════════════════════════════════════════════════════════════
banner("STAGE ONE DATA — ERA5 reanalysis")
SRC, LAT, LON = era5_source_grid(ERA5)
SPLITS  = era5_splits(ERA5, val_years=4, test_years=2)
OFFSETS = lead_offsets(CFG.lead_times_hours)

for name, index in SPLITS.items():
    print(f"  {name:5} {len(index):>7,} windows  "
          f"{str(ERA5.time.values[index[0]])[:10]} -> {str(ERA5.time.values[index[-1]])[:10]}")
print(f"\nlead times {CFG.lead_times_hours} h -> store offsets {OFFSETS}")
print("split by TIME. Weather is autocorrelated for days, so a random split lets the model")
print("interpolate between two states it has already seen and report a skill it does not have.")

train_stream = ERA5Window(ERA5, indices=SPLITS["train"], history=CFG.history_frames,
                          lead_steps=OFFSETS, channels=CFG.analysis_channels, stats_cache=STATS)
print(f"\n{train_stream.normalizer.report()}")

if not os.path.exists(CACHE):
    print(f"\nstaging {STAGE_YEARS} years to local disk (one-off; streaming is 105x slower per window)")
    materialise(ERA5, SPLITS["train"][-STAGE_YEARS * 1460:], CACHE, normalizer=train_stream.normalizer)
train_ds = CachedERA5(CACHE, history=CFG.history_frames, lead_steps=OFFSETS,
                      channels=CFG.analysis_channels, augment=True)
val_ds = ERA5Window(ERA5, indices=SPLITS["val"], history=CFG.history_frames, lead_steps=OFFSETS,
                    augment=False, channels=CFG.analysis_channels, stats_cache=STATS)
print(f"\ntrain {len(train_ds):,} staged windows | validate on {len(val_ds):,} held-out windows")

# ═══ 5 ═══ fill the card, and price the run ════════════════════════════════════════════════════════
banner("BENCHMARK — largest batch that fits, and what the run costs")
model.gradient_checkpointing_enable(GRAD_CKPT)


def make_step(batch_size):
    """One full training step at this batch size, for the autotuner and the benchmark."""
    items = [train_ds[i] for i in range(batch_size)]
    batch = {k: torch.stack([x[k] for x in items]).to(DEVICE) for k in items[0]}

    def step():
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            out = model(analysis=batch["analysis"], analysis_grid=SRC,
                        calendar=batch["calendar"], output_grid=SRC)
            loss = masked_gaussian_nll(
                out["field_mean"], out["field_log_var"],
                torch.where(batch["field_mask"] > 0, batch["field_target"], torch.nan),
            )
        loss.backward()
        model.zero_grad(set_to_none=True)
    return step


BATCH = autotune_batch_size(make_step, start=1, target_fraction=0.85)
MARK  = benchmark_steps(make_step(BATCH), BATCH, gradient_checkpointing=GRAD_CKPT)
print(f"batch {BATCH} at 85% of VRAM | {MARK}")
PLAN = training_plan(MARK.samples_per_second, corpus_samples=len(train_ds), epochs=EPOCHS,
                     watts=600.0, electricity_per_kwh=0.15, cloud_per_hour=2.50)
print(f"\nstage one, {EPOCHS} epochs:")
print(format_plan(PLAN))
MAX_STEPS = int(len(train_ds) * EPOCHS / BATCH)
print(f"\n  optimizer steps {MAX_STEPS:,}")

# ═══ 6 ═══ stage one — pretrain the backbone ═══════════════════════════════════════════════════════
if RUN_PRETRAIN:
    banner("STAGE ONE — pretraining on reanalysis")
    pretrain = TrainSettings(
        stage="pretrain", learning_rate=3e-4, warmup_steps=1000, max_steps=MAX_STEPS,
        grad_accum=1, precision="bf16",
        checkpoint_dir=f"{CKPT_DIR}/stage1", checkpoint_seconds=60,
        hub_repo=HUB_REPO, hub_push_seconds=900,
        ema_decay=0.999, log_every=25, val_every=500, val_batches=32,
    )
    loader     = era5_loader(train_ds, batch_size=BATCH, num_workers=4, shuffle=True,
                             analysis_grid=SRC, output_grid=SRC)
    val_loader = era5_loader(val_ds, batch_size=BATCH, num_workers=2, shuffle=False,
                             analysis_grid=SRC, output_grid=SRC)
    trainer = Trainer(model, pretrain, device=DEVICE)
    trainer.resume()        # picks up wherever the last run died; pulls from the Hub on a fresh VM
    trainer.fit(loader, epochs=EPOCHS, val_loader=val_loader)
    print("\nRead the [val] lines: held-out falling = learning. Held-out rising while training")
    print("keeps falling = memorising, and the gap is how much.")

# ═══ 7 ═══ stage two data — best tracks paired with the same reanalysis ════════════════════════════
banner("STAGE TWO DATA — HURDAT2 best tracks")
HURDAT = download("https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt",
                  "/content/hurdat2.txt")
tracks = parse_hurdat2(HURDAT)
print(f"HURDAT2: {len(tracks):,} storms, {sum(len(t) for t in tracks):,} points, "
      f"{sum(int(t.landfall.sum()) for t in tracks):,} landfalls")

walk = rapid_intensification(tracks, threshold_kt=30.0)
print(f"rapid intensification: {walk['positives']:,} of {walk['eligible']:,} eligible = "
      f"{100 * walk['base_rate']:.2f}%")
print("  a classifier that always says 'no' scores 96% here and saves nobody, which is why the RI")
print("  head uses a focal loss and is judged on precision, recall and Brier score, not accuracy.")

GROUPS      = split_by_storm(tracks)       # by SEASON, never by point
STORE_TIMES = ERA5.time.values.astype("datetime64[s]").astype(np.int64)


def storm_split(which):
    """Pair one split's storms with the reanalysis hour each was observed at."""
    starts, targets, report = pair_tracks_with_reanalysis(
        GROUPS[which], STORE_TIMES, OFFSETS, history=CFG.history_frames)
    base = ERA5Window(ERA5, indices=starts, history=CFG.history_frames, lead_steps=OFFSETS,
                      augment=False,       # a rolled globe moves the coastline the storm hit
                      channels=CFG.analysis_channels, stats_cache=STATS)
    return StormWindow(base, targets), report


storm_train, train_report = storm_split("train")
storm_val,   val_report   = storm_split("validation")
print(f"\n{format_pairing(train_report)}")
print(f"\n{storm_train.describe()}")
print(f"\nvalidation: {val_report['paired']:,} samples from {val_report['storms']} unseen storms "
      f"(seasons 2017/2019/2021)")

# ═══ 8 ═══ stage two — fine-tune the heads, backbone frozen ════════════════════════════════════════
if RUN_FINETUNE:
    banner("STAGE TWO — fine-tuning the storm heads")

    # If stage one ran in a previous session, its weights are on disk but not in this process. Load
    # them before freezing, or the "frozen backbone" is a frozen *untrained* backbone -- which trains
    # without complaint and produces a model that has never seen the atmosphere.
    if not RUN_PRETRAIN:
        from naturev1 import CheckpointManager
        stage_one = CheckpointManager(f"{CKPT_DIR}/stage1", repo_id=HUB_REPO)
        stage_one.fetch_from_hub()
        restored = stage_one.load(model, map_location=DEVICE)
        if restored is None:
            print("!! no stage-one checkpoint found. The backbone is UNTRAINED, and freezing it now")
            print("!! would fine-tune storm heads on random features. Set RUN_PRETRAIN = True first.")
            raise SystemExit(1)
        print(f"loaded stage one from {CKPT_DIR}/stage1 (step {restored.step:,})")

    trainable, total = model.freeze_backbone(True)
    print(f"trainable {trainable:,} of {total:,} ({100 * trainable / total:.1f}%)")
    print(f"{train_report['paired']:,} storm points cannot fit {total/1e6:.0f}M parameters.")
    print(f"They can fit {trainable/1e6:.2f}M — on a backbone that saw 1.6e10 values in stage one.")

    finetune = TrainSettings(
        stage="finetune", learning_rate=1e-4, warmup_steps=200, max_steps=20_000,
        precision="bf16",
        checkpoint_dir=f"{CKPT_DIR}/stage2", checkpoint_seconds=60,
        hub_repo=HUB_REPO, hub_push_seconds=900,
        ema_decay=0.999, log_every=25, val_every=200, val_batches=32,
        early_stopping_patience=10,        # 10 held-out passes without improvement -> stop
    )
    storm_loader     = era5_loader(storm_train, batch_size=max(BATCH // 2, 1), num_workers=4,
                                   shuffle=True, analysis_grid=SRC, output_grid=SRC)
    storm_val_loader = era5_loader(storm_val, batch_size=max(BATCH // 2, 1), num_workers=2,
                                   shuffle=False, analysis_grid=SRC, output_grid=SRC)
    storm_trainer = Trainer(model, finetune, device=DEVICE)
    storm_trainer.resume()
    storm_trainer.fit(storm_loader, epochs=50, val_loader=storm_val_loader)

    # ─── the honest scoreboard ───
    banner("SCOREBOARD — is it generalising, or memorising?")
    held_out = storm_trainer.evaluate(storm_val_loader, max_batches=64)
    on_train = storm_trainer.evaluate(storm_loader, max_batches=64)
    gap = held_out["val_total"] - on_train["val_total"]
    print(f"held-out {held_out['val_total']:.4f}   training {on_train['val_total']:.4f}   gap {gap:+.4f}")
    print("  a large positive gap means it memorised the training storms\n")
    for key in sorted(held_out):
        if key != "val_total":
            print(f"  {key:24} {held_out[key]:.4f}")
    print("\nThese are likelihoods, not skill. Skill only means something against a baseline:")
    print("score persistence ('tomorrow = today') and climatology on these same batches before")
    print("trusting any forecast this model makes.")

# ═══ 9 ═══ live forecast from real GOES imagery ════════════════════════════════════════════════════
def load_scene(paths, center, half_width_deg=9.0, stride=4):
    """Geolocated samples plus true pixel footprints, cropped to a real box on the planet."""
    scene = scene_from_netcdf(
        {c: str(p) for c, p in paths.items()},
        bounds=(center[0] - half_width_deg, center[0] + half_width_deg,
                center[1] - half_width_deg, center[1] + half_width_deg),
        stride=stride,
    )
    coords = torch.tensor(
        np.stack([np.radians(scene.latitude), np.radians(scene.longitude) % (2 * np.pi)], -1),
        dtype=torch.float64,
    )
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


def show(fc):
    print(f"\nissued {fc['issued']}   ENSO {fc['enso']['phase']} ({fc['enso']['nino34_index']:+.2f})")
    print(f"landfall: {({k: v for k, v in fc['landfall'].items() if k != 'by_lead'})}")
    print("\ntrack scenarios at +48 h:")
    for s in fc["track_scenarios"][:4]:
        p = s["track"][5]
        print(f"  {s['probability']:>5.0%}  -> {p['latitude']:6.2f},{p['longitude']:7.2f}   "
              f"95% cone {p['cone_radius_km_95']:.0f} km")
    print("\neyewall:")
    for e in fc["eyewall"][:4]:
        print(f"  +{e['lead_hours']:3d}h  peak {e['peak_wind_kt']:5.0f} kt  {e['saffir_simpson']:<12} "
              f"RMW {e['rmw_nmi']:4.0f} nmi   34kt NE {e['wind_radii_nmi']['34kt']['NE']:.0f} nmi")
    ri = fc["rapid_intensification"]
    print(f"\nrapid intensification (24 h): 25kt {ri['probability_25kt']:.0%}  "
          f"30kt {ri['probability_30kt']:.0%}  35kt {ri['probability_35kt']:.0%}")
    print(f"  expected change {ri['expected_change_kt']:+.0f} kt {ri['expected_change_90pct']}  "
          f"likeliest onset +{ri['likeliest_onset_hours']}h")
    pf = fc["point_forecast"]["forecast"][3]
    print(f"\nlocal +{pf['lead_hours']}h: {pf['weather']} ({pf['weather_confidence']:.0%})")


if RUN_FORECAST:
    banner("LIVE FORECAST — newest GOES scene")
    show(forecast_now())

# ═══ 10 ═══ hourly watcher ═════════════════════════════════════════════════════════════════════════
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


if RUN_WATCHER:
    banner("WATCHING — a new forecast every hour")
    watch(on_new_scene, interval_seconds=3600, directory=DATA_DIR, channels=CHANNELS)
