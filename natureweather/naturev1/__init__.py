# Copyright 2026 Nathan. Apache-2.0.
"""NatureV1 - a probabilistic weather model over geolocated fields."""

from .checkpoint import CheckpointManager, TrainingState
from .forecast import build_forecast, decode_landfall, decode_point_forecast, decode_track, nearest_point
from .live import fetch_latest, latest_scene_keys, watch
from .losses import total_loss, track_mixture_nll
from .model import SURFACE_FIELDS, WEATHER_TYPES, NatureConfig, NatureV1, calendar_features
from .satellite import SatelliteScene, fixed_grid_to_latlon, footprint_area_km2, normalize_channels, scene_from_netcdf
from .train import Trainer, TrainSettings, load_for_inference

__version__ = "0.1.0"
__author__ = "Nathan"
__all__ = [
    "SURFACE_FIELDS", "WEATHER_TYPES", "CheckpointManager", "NatureConfig", "NatureV1", "SatelliteScene",
    "TrainSettings", "Trainer", "TrainingState", "build_forecast", "calendar_features", "decode_landfall",
    "decode_point_forecast", "decode_track", "fetch_latest", "fixed_grid_to_latlon", "footprint_area_km2",
    "latest_scene_keys", "load_for_inference", "nearest_point", "normalize_channels", "scene_from_netcdf",
    "total_loss", "track_mixture_nll", "watch",
]
