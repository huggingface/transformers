# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 — Build the model  (~88M parameters)
# ══════════════════════════════════════════════════════════════════════════════
import datetime as dt

import torch
from naturev1 import SURFACE_FIELDS, WEATHER_TYPES, NatureConfig, NatureV1, calendar_features

from ihelix import FieldGrid, Geometry, fibonacci_sphere


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CFG = NatureConfig(
    satellite_channels = 6,
    analysis_channels  = 24,
    hidden_size        = 512,
    num_layers         = 17,
    num_heads          = 8,
    num_kv_heads       = 4,
    head_dim           = 64,
    intermediate_size  = 2048,
    latent_points      = 4096,
    min_radius_km      = 120.0,
    max_radius_km      = 1600.0,
    history_frames     = 6,
    lead_times_hours   = (6, 12, 18, 24, 36, 48, 72, 96, 120),
    track_modes        = 6,
)

LATENT = fibonacci_sphere(CFG.latent_points,
                          num_neighbours=CFG.latent_neighbours,
                          cluster_size=CFG.latent_cluster)
model = NatureV1(CFG, LATENT).to(DEVICE)
print(f"NatureV1: {model.num_parameters():,} parameters ({model.num_parameters()/1e6:.2f}M)")
print(f"device: {DEVICE} | leads: {CFG.lead_times_hours} | scenarios: {CFG.track_modes}")
print(f"fields: {SURFACE_FIELDS}")
print(f"types : {WEATHER_TYPES}")
