# Copyright 2026 Nathan. Apache-2.0.
"""NatureV1 - a probabilistic weather model over geolocated fields."""

from .besttrack import Track, parse_hurdat2, rapid_intensification, split_by_storm
from .checkpoint import CheckpointManager, TrainingState
from .corpora import CORPORA, augment_batch, catalogue, open_weatherbench, rotate_longitude
from .era5 import (
                   FIELD_DIMS,
                   CachedERA5,
                   ERA5Window,
                   GridCollate,
                   Normalizer,
                   corpus_scale,
                   equiangular_weights,
                   era5_loader,
                   era5_source_grid,
                   era5_splits,
                   lead_offsets,
                   materialise,
                   normalization,
                   read_block,
)
from .forecast import (
                   build_forecast,
                   decode_eyewall,
                   decode_landfall,
                   decode_point_forecast,
                   decode_rapid_intensification,
                   decode_track,
                   nearest_point,
)
from .live import fetch_latest, latest_scene_keys, watch
from .losses import focal_bce, masked_gaussian_nll, total_loss, track_mixture_nll
from .model import (
                   RI_THRESHOLDS_KT,
                   SURFACE_FIELDS,
                   WEATHER_TYPES,
                   WIND_RADII_THRESHOLDS_KT,
                   EyewallHead,
                   NatureConfig,
                   NatureV1,
                   RapidIntensificationHead,
                   calendar_features,
)
from .planning import autotune_batch_size, benchmark_steps, device_report, format_plan, training_plan
from .satellite import SatelliteScene, fixed_grid_to_latlon, footprint_area_km2, normalize_channels, scene_from_netcdf
from .storms import StormTargets, StormWindow, format_pairing, pair_tracks_with_reanalysis
from .train import Trainer, TrainSettings, apply_ema, load_for_inference, next_state_targets


__version__ = "0.2.0"
__author__ = "Nathan"
__all__ = [
    "CORPORA", "RI_THRESHOLDS_KT", "WIND_RADII_THRESHOLDS_KT", "EyewallHead", "RapidIntensificationHead",
    "Track", "apply_ema", "augment_batch", "autotune_batch_size", "benchmark_steps", "catalogue",
    "device_report", "focal_bce", "format_plan", "masked_gaussian_nll", "next_state_targets",
    "open_weatherbench", "parse_hurdat2", "rapid_intensification", "rotate_longitude", "split_by_storm",
    "training_plan", "CachedERA5", "ERA5Window", "corpus_scale", "era5_source_grid", "era5_splits",
    "materialise", "normalization", "Normalizer", "equiangular_weights", "era5_loader", "FIELD_DIMS", "read_block", "lead_offsets", "GridCollate",
    "SURFACE_FIELDS", "WEATHER_TYPES", "CheckpointManager", "NatureConfig", "NatureV1", "SatelliteScene",
    "TrainSettings", "Trainer", "TrainingState", "build_forecast", "calendar_features", "decode_landfall",
    "decode_point_forecast", "decode_track", "decode_eyewall", "decode_rapid_intensification", "fetch_latest", "fixed_grid_to_latlon", "footprint_area_km2",
    "latest_scene_keys", "load_for_inference", "nearest_point", "normalize_channels", "scene_from_netcdf",
    "total_loss", "track_mixture_nll", "watch", "StormWindow", "StormTargets",
    "pair_tracks_with_reanalysis", "format_pairing",
]
