"""
NatureV1 on Colab — copy each block into its own cell.

Cell 1 installs, Cell 2 builds the model, Cell 3 runs a live forecast from real GOES imagery,
Cell 4 is the data adapter you must fill in with real training data, Cell 5 trains with
minute checkpointing that survives a dead cell, Cell 6 is the hourly watcher.
"""

# ══ CELL 1 ══ install ═════════════════════════════════════════════════════════
# !pip install -q "naturev1[all]"
# !nvidia-smi --query-gpu=name,memory.total --format=csv

# ══ CELL 2 ══ build the model ═════════════════════════════════════════════════
import torch, numpy as np, datetime as dt, json, os
from ihelix import fibonacci_sphere, Geometry, FieldGrid
from naturev1 import (NatureConfig, NatureV1, calendar_features, build_forecast, fetch_latest,
                      scene_from_netcdf, normalize_channels, Trainer, TrainSettings,
                      SURFACE_FIELDS, WEATHER_TYPES, watch)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CFG = NatureConfig(
    satellite_channels=6, analysis_channels=24,
    hidden_size=512, num_layers=17, num_heads=8, num_kv_heads=4,
    head_dim=64, intermediate_size=2048,
    latent_points=4096, min_radius_km=120.0, max_radius_km=1600.0,
    history_frames=6, lead_times_hours=(6, 12, 18, 24, 36, 48, 72, 96, 120), track_modes=6,
)
LATENT = fibonacci_sphere(CFG.latent_points, num_neighbours=CFG.latent_neighbours,
                          cluster_size=CFG.latent_cluster)
model = NatureV1(CFG, LATENT).to(DEVICE)
print(f"NatureV1: {model.num_parameters()/1e6:.2f}M parameters on {DEVICE}")

# ══ CELL 3 ══ live forecast from real GOES imagery ════════════════════════════
CHANNELS = ("C13", "C09")     # clean IR window + mid-level water vapour
STORM    = (24.6, -78.2)      # current storm centre (lat, lon)
CITY     = (25.77, -80.19)    # somewhere you want a local forecast
DATA_DIR = "./test_data"

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
    T = CFG.history_frames
    sat = values[None, None].expand(1, T, -1, -1).contiguous().to(DEVICE)
    cal = calendar_features(torch.full((1, T), scene.timestamp.timestamp())).to(DEVICE)
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
    print(f"  {s['probability']:>5.0%}  +48h -> {p['latitude']:6.2f},{p['longitude']:7.2f}   95% cone {p['cone_radius_km_95']:.0f} km")
pf = fc["point_forecast"]["forecast"][3]
print(f"  local +{pf['lead_hours']}h: {pf['weather']} ({pf['weather_confidence']:.0%})")

# ══ CELL 4 ══ training data — YOU MUST FILL THIS IN ═══════════════════════════
# The model above is UNTRAINED: its output is noise. To make it forecast you need
# paired (inputs, verified outcome) samples. All of these are free:
#
#   HURDAT2 best tracks   https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt
#   ERA5 reanalysis       https://cds.climate.copernicus.eu  (ERA5 single+pressure levels)
#   GFS analysis          s3://noaa-gfs-bdp-pds/   (anonymous, no account)
#   GOES archive          s3://noaa-goes19/  s3://noaa-goes18/   (anonymous)
#   IBTrACS (global)      https://www.ncei.noaa.gov/products/international-best-track-archive
#
# One training sample = satellite + analysis at time t (and the 5 steps before it),
# with targets taken from what actually happened at t+6h ... t+120h.

class WeatherDataset(torch.utils.data.Dataset):
    """Replace the body with real loading. Shapes are what the model and losses expect."""
    def __init__(self, samples, analysis_grid, target_grid):
        self.samples, self.analysis_grid, self.target_grid = samples, analysis_grid, target_grid
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        s, T, L = self.samples[i], CFG.history_frames, CFG.num_leads
        N_ana, N_out = self.analysis_grid.num_points, self.target_grid.num_points
        return {
            "analysis":            torch.randn(T, N_ana, CFG.analysis_channels),   # <- real analysis
            "calendar":            calendar_features(torch.full((T,), s["t"])),
            "field_target":        torch.randn(N_out, L, len(SURFACE_FIELDS)),     # <- verified fields
            "weather_type_target": torch.randint(0, len(WEATHER_TYPES), (N_out, L)),
            "track_target":        torch.randn(L, 2),        # (dlat, dlon) from HURDAT2
            "landfall_target":     torch.randint(0, 2, (L,)).float(),
            "intensity_target":    torch.randn(L, 2),        # max wind m/s, min pressure hPa
        }

def collate(batch, analysis_grid, target_grid):
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    out["analysis_grid"], out["output_grid"] = analysis_grid, target_grid
    return out

# ══ CELL 5 ══ train, with checkpoints that survive the cell dying ═════════════
from functools import partial
ANALYSIS_GRID = fibonacci_sphere(2048, num_neighbours=32, cluster_size=64)
TARGET_GRID   = fibonacci_sphere(2048, num_neighbours=32, cluster_size=64)

settings = TrainSettings(
    learning_rate=3e-4, warmup_steps=500, max_steps=100_000,
    grad_accum=1, precision="bf16",                 # bf16: no loss scaling on Blackwell
    checkpoint_dir="/content/drive/MyDrive/naturev1_ckpt",   # Drive outlives the VM
    checkpoint_seconds=60,                          # a checkpoint every minute
    hub_repo="Sigmandndnns/NatureV1-500",           # mirrored to the Hub
    hub_push_seconds=900,
    log_every=10,
)

def train():
    ds = WeatherDataset([{"t": 1.7e9 + 3600*i} for i in range(512)], ANALYSIS_GRID, TARGET_GRID)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=partial(collate, analysis_grid=ANALYSIS_GRID, target_grid=TARGET_GRID))
    trainer = Trainer(model, settings, device=DEVICE)
    trainer.resume()          # picks up wherever the last cell died; pulls from the Hub on a fresh VM
    trainer.fit(loader, epochs=1000)

# train()   # <- uncomment once Cell 4 loads real data

# ══ CELL 6 ══ hourly watcher ══════════════════════════════════════════════════
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
