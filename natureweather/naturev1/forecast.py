# Copyright 2026 Nathan. Apache-2.0.
"""
Turn raw model outputs into a forecast a person can read and act on.

The model emits distributions. This turns them into the things people actually ask: where will it go,
how sure are you, will it hit land, when, will it rain here. Every number that comes out of here carries
its uncertainty, because a track without a cone is a guess wearing a suit.
"""

from __future__ import annotations

import datetime as dt
import math

import torch

from .model import SURFACE_FIELDS, WEATHER_TYPES


def _ellipse(sigma_x: float, sigma_y: float, rho: float) -> tuple[float, float, float]:
    """
    Convert a 2-D Gaussian into the ellipse people draw: semi-major, semi-minor, bearing.

    Returns the 1-sigma axes in degrees and the bearing of the major axis in compass degrees, which is
    what a cone-of-uncertainty plot needs.
    """
    covariance_xy = rho * sigma_x * sigma_y
    trace = sigma_x**2 + sigma_y**2
    gap = math.sqrt(max((sigma_x**2 - sigma_y**2) ** 2 / 4 + covariance_xy**2, 0.0))
    major = math.sqrt(max(trace / 2 + gap, 0.0))
    minor = math.sqrt(max(trace / 2 - gap, 0.0))
    angle = 0.5 * math.atan2(2 * covariance_xy, sigma_x**2 - sigma_y**2)
    bearing = (90.0 - math.degrees(angle)) % 360.0
    return major, minor, bearing


def decode_track(
    outputs: dict,
    origin_lat: float,
    origin_lon: float,
    lead_times_hours: tuple[int, ...],
    issued: dt.datetime,
    batch_index: int = 0,
    max_scenarios: int | None = None,
) -> list[dict]:
    """
    Turn the mixture head into a ranked list of scenarios, each a full trajectory with a cone.

    Scenarios come back sorted by probability. Low-probability branches are kept rather than averaged
    away -- the whole reason for a mixture is that "62% offshore, 26% coast-hugging, 12% inland" is
    actionable and their average is not.
    """
    probabilities = outputs["mode_logits"][batch_index].softmax(-1)
    displacement = outputs["displacement"][batch_index]
    log_scale = outputs["log_scale"][batch_index]
    correlation = outputs["correlation"][batch_index]

    order = probabilities.argsort(descending=True)
    if max_scenarios:
        order = order[:max_scenarios]

    scenarios = []
    for rank, mode in enumerate(order.tolist()):
        points = []
        for lead_index, hours in enumerate(lead_times_hours):
            d_lat = float(displacement[mode, lead_index, 0])
            d_lon = float(displacement[mode, lead_index, 1])
            sigma_lat = float(log_scale[mode, lead_index, 0].exp())
            sigma_lon = float(log_scale[mode, lead_index, 1].exp())
            major, minor, bearing = _ellipse(sigma_lon, sigma_lat, float(correlation[mode, lead_index]))
            latitude = origin_lat + d_lat
            # Degrees of longitude shrink with latitude; the displacement is in degrees, so no conversion
            # is needed here, but the cone's east-west extent in kilometres does depend on it.
            longitude = (origin_lon + d_lon + 180.0) % 360.0 - 180.0
            points.append(
                {
                    "lead_hours": hours,
                    "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
                    "latitude": round(latitude, 3),
                    "longitude": round(longitude, 3),
                    "sigma_lat_deg": round(sigma_lat, 3),
                    "sigma_lon_deg": round(sigma_lon, 3),
                    "cone_semi_major_deg": round(major, 3),
                    "cone_semi_minor_deg": round(minor, 3),
                    "cone_bearing_deg": round(bearing, 1),
                    "cone_radius_km_68": round(major * 111.0, 1),
                    "cone_radius_km_95": round(major * 111.0 * 2.448, 1),
                }
            )
        scenarios.append(
            {"rank": rank + 1, "probability": round(float(probabilities[mode]), 4), "track": points}
        )
    return scenarios


def decode_landfall(
    outputs: dict, lead_times_hours: tuple[int, ...], issued: dt.datetime, batch_index: int = 0
) -> dict:
    """Landfall probability per lead time, plus the window it is most likely to happen in."""
    probability = outputs["landfall_logit"][batch_index].sigmoid()
    spread = outputs["landfall_time_log_var"][batch_index].mul(0.5).exp()
    by_lead = [
        {
            "lead_hours": hours,
            "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
            "probability": round(float(probability[i]), 4),
            "timing_sigma_hours": round(float(spread[i]), 2),
        }
        for i, hours in enumerate(lead_times_hours)
    ]
    peak = int(probability.argmax())
    # Probability of landfall by the final lead time is the largest cumulative value, since the head is
    # trained as "by this lead" rather than "at this lead".
    return {
        "by_lead": by_lead,
        "peak_probability": round(float(probability.max()), 4),
        "most_likely_lead_hours": lead_times_hours[peak],
        "most_likely_time": (issued + dt.timedelta(hours=lead_times_hours[peak])).isoformat(),
        "timing_sigma_hours": round(float(spread[peak]), 2),
        "makes_landfall_likely": bool(probability.max() > 0.5),
    }


def decode_intensity(
    outputs: dict, lead_times_hours: tuple[int, ...], issued: dt.datetime, batch_index: int = 0
) -> list[dict]:
    """Peak wind and minimum central pressure per lead, with 90% intervals."""
    mean = outputs["intensity_mean"][batch_index]
    sigma = outputs["intensity_log_var"][batch_index].mul(0.5).exp()
    rows = []
    for i, hours in enumerate(lead_times_hours):
        wind, pressure = float(mean[i, 0]), float(mean[i, 1])
        wind_sigma, pressure_sigma = float(sigma[i, 0]), float(sigma[i, 1])
        rows.append(
            {
                "lead_hours": hours,
                "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
                "max_wind_ms": round(wind, 1),
                "max_wind_90pct": [round(wind - 1.645 * wind_sigma, 1), round(wind + 1.645 * wind_sigma, 1)],
                "min_pressure_hpa": round(pressure, 1),
                "min_pressure_90pct": [
                    round(pressure - 1.645 * pressure_sigma, 1),
                    round(pressure + 1.645 * pressure_sigma, 1),
                ],
                "saffir_simpson": _category(wind),
            }
        )
    return rows


def _category(wind_ms: float) -> str:
    """Saffir-Simpson from 1-minute sustained wind in m/s."""
    knots = wind_ms * 1.94384
    for threshold, label in ((137, "Category 5"), (113, "Category 4"), (96, "Category 3"),
                             (83, "Category 2"), (64, "Category 1"), (34, "Tropical Storm")):
        if knots >= threshold:
            return label
    return "Tropical Depression"


def decode_point_forecast(
    outputs: dict,
    point_index: int,
    lead_times_hours: tuple[int, ...],
    issued: dt.datetime,
    batch_index: int = 0,
) -> list[dict]:
    """
    "Will it rain or be sunny here" -- the full per-location forecast, at one output point.

    Weather type comes back as a calibrated probability over categories rather than a single label, so
    "60% light rain, 30% overcast" stays visible instead of collapsing to "rain".
    """
    mean = outputs["field_mean"][batch_index, point_index]
    sigma = outputs["field_log_var"][batch_index, point_index].mul(0.5).exp()
    type_probabilities = outputs["weather_type_logits"][batch_index, point_index].softmax(-1)

    rows = []
    for i, hours in enumerate(lead_times_hours):
        fields = {}
        for j, name in enumerate(SURFACE_FIELDS):
            value, spread = float(mean[i, j]), float(sigma[i, j])
            fields[name] = {
                "value": round(value, 3),
                "sigma": round(spread, 3),
                "range_90pct": [round(value - 1.645 * spread, 3), round(value + 1.645 * spread, 3)],
            }
        ranked = sorted(
            ((WEATHER_TYPES[k], float(type_probabilities[i, k])) for k in range(len(WEATHER_TYPES))),
            key=lambda pair: -pair[1],
        )
        rows.append(
            {
                "lead_hours": hours,
                "valid_time": (issued + dt.timedelta(hours=hours)).isoformat(),
                "weather": ranked[0][0],
                "weather_confidence": round(ranked[0][1], 4),
                "weather_probabilities": {name: round(p, 4) for name, p in ranked},
                "fields": fields,
            }
        )
    return rows


def nearest_point(grid, latitude: float, longitude: float) -> int:
    """Index of the output-grid sample closest to a latitude/longitude, for a point forecast."""
    target = torch.tensor(
        [[math.radians(latitude), math.radians(longitude) % (2 * math.pi)]], dtype=grid.coords.dtype
    )
    return int((grid.points - grid.geometry.embed(target)).norm(dim=-1).argmin())


def build_forecast(
    outputs: dict,
    lead_times_hours: tuple[int, ...],
    issued: dt.datetime,
    storm_center: tuple[float, float] | None = None,
    output_grid=None,
    point_of_interest: tuple[float, float] | None = None,
    batch_index: int = 0,
) -> dict:
    """
    Assemble the whole forecast bundle: track scenarios, landfall, intensity, local weather, ENSO.

    Args:
        outputs: what :meth:`NatureV1.forward` returned.
        lead_times_hours: the model's lead times.
        issued: analysis time, UTC.
        storm_center: ``(lat, lon)`` of the current centre. Omit for a non-storm run and the track,
            landfall and intensity sections are left out rather than invented.
        output_grid: the grid gridded fields were decoded onto, needed for a point forecast.
        point_of_interest: ``(lat, lon)`` to produce a local forecast for.
    """
    bundle = {
        "issued": issued.isoformat(),
        "lead_times_hours": list(lead_times_hours),
        "enso": {
            "nino34_index": round(float(outputs["enso_mean"][batch_index]), 3),
            "sigma": round(float(outputs["enso_log_var"][batch_index].mul(0.5).exp()), 3),
            "phase": _enso_phase(float(outputs["enso_mean"][batch_index])),
        },
    }
    if storm_center is not None:
        bundle["storm_center"] = {"latitude": storm_center[0], "longitude": storm_center[1]}
        bundle["track_scenarios"] = decode_track(
            outputs, storm_center[0], storm_center[1], lead_times_hours, issued, batch_index
        )
        bundle["landfall"] = decode_landfall(outputs, lead_times_hours, issued, batch_index)
        bundle["intensity"] = decode_intensity(outputs, lead_times_hours, issued, batch_index)
    if point_of_interest is not None and output_grid is not None:
        index = nearest_point(output_grid, *point_of_interest)
        bundle["point_forecast"] = {
            "requested": {"latitude": point_of_interest[0], "longitude": point_of_interest[1]},
            "forecast": decode_point_forecast(outputs, index, lead_times_hours, issued, batch_index),
        }
    return bundle


def _enso_phase(index: float) -> str:
    """NOAA's convention: the Nino 3.4 anomaly thresholds at +/- 0.5 K."""
    if index >= 1.5:
        return "Strong El Nino"
    if index >= 0.5:
        return "El Nino"
    if index <= -1.5:
        return "Strong La Nina"
    if index <= -0.5:
        return "La Nina"
    return "Neutral"
