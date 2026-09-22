# Copyright 2026 Nathan. Apache-2.0.
"""
Satellite ingest: turn a GOES ABI scene into geolocated samples, not a rectangle of pixels.

The usual pipeline crops the array, divides by 255, and hands a CNN or a ViT a grid. Three things go
wrong when you do that, and all three matter for a storm:

1. **A GOES pixel is not a fixed size on the ground.** ABI scans in fixed angular steps from
   geostationary orbit, so the footprint is ~2 km at the sub-satellite point and grows without bound
   toward the limb -- past 5 km by the time you reach the Atlantic hurricane basin at high latitude. A
   convolution weights all of them equally, so it silently over-counts the stretched edge of the disk.
2. **The array index is not a place.** Nothing in `image[400, 1200]` says 24.7N 81.3W. The model has to
   infer the projection, and it cannot, because the projection is not in the data.
3. **Normalizing to 0-1 throws away the physics.** Channel 13 is a brightness temperature in kelvin.
   180 K is an overshooting top punching into the stratosphere; 300 K is warm ocean. That is a physical
   scale with physical thresholds, and rescaling it per-image destroys the one thing it was measuring.

So this module does the opposite of each. It solves the fixed-grid projection for the latitude and
longitude of every pixel, computes each pixel's real ground area, and keeps brightness temperature in
kelvin. What comes out is a point cloud on the sphere with physical values and quadrature weights --
exactly what `ihelix` consumes, and exactly what lets satellite and model data be fused without either
one being resampled onto the other's grid.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

import numpy as np


# GOES-16 was retired as GOES-East in 2025 and its bucket no longer receives data, which is a quiet
# failure: the listing succeeds and returns nothing. These are the buckets that are actually live.
GOES_BUCKETS = {"east": "noaa-goes19", "west": "noaa-goes18"}
#: Products, coarsest cadence first. CONUS refreshes every 5 minutes, full disk every 10.
GOES_PRODUCTS = {"conus": "ABI-L2-CMIPC", "fulldisk": "ABI-L2-CMIPF", "mesoscale1": "ABI-L2-CMIPM"}


@dataclass
class SatelliteScene:
    """A geolocated satellite scene: every sample carries a position, an area and a physical value."""

    latitude: np.ndarray          # (N,) degrees
    longitude: np.ndarray         # (N,) degrees, -180..180
    values: np.ndarray            # (N, C) physical units, one column per channel
    area_km2: np.ndarray          # (N,) true ground footprint
    channels: tuple[str, ...]
    timestamp: dt.datetime
    satellite: str

    def __len__(self) -> int:
        return self.latitude.shape[0]

    def __repr__(self) -> str:
        return (
            f"SatelliteScene({len(self)} samples, channels={self.channels}, "
            f"{self.timestamp:%Y-%m-%d %H:%M}Z, {self.satellite}, "
            f"footprint {self.area_km2.min():.1f}-{self.area_km2.max():.1f} km2)"
        )


def fixed_grid_to_latlon(
    scan_x: np.ndarray,
    scan_y: np.ndarray,
    satellite_longitude: float,
    perspective_height: float = 35786023.0,
    semi_major: float = 6378137.0,
    semi_minor: float = 6356752.31414,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the GOES fixed-grid projection for geodetic latitude and longitude.

    This is the intersection of the satellite's line of sight with the WGS84 ellipsoid, following the
    GOES-R Product User Guide (Vol. 3, section 5.1.2.8.1). Pixels whose ray misses the Earth -- the
    corners of a full-disk scene are mostly space -- come back as NaN rather than as a wrong answer.

    Args:
        scan_x, scan_y: fixed-grid scanning angles in radians, broadcast against each other.
        satellite_longitude: sub-satellite longitude in degrees.

    Returns:
        ``(latitude, longitude)`` in degrees, NaN off the disk.
    """
    height = perspective_height + semi_major
    ratio = semi_major**2 / semi_minor**2

    sin_x, cos_x = np.sin(scan_x), np.cos(scan_x)
    sin_y, cos_y = np.sin(scan_y), np.cos(scan_y)

    a = sin_x**2 + cos_x**2 * (cos_y**2 + ratio * sin_y**2)
    b = -2.0 * height * cos_x * cos_y
    c = height**2 - semi_major**2

    discriminant = b**2 - 4.0 * a * c
    # A negative discriminant means the ray never reaches the ellipsoid: that sample is deep space.
    off_disk = discriminant < 0
    discriminant = np.where(off_disk, np.nan, discriminant)

    distance = (-b - np.sqrt(discriminant)) / (2.0 * a)
    sx = distance * cos_x * cos_y
    sy = -distance * sin_x
    sz = distance * cos_x * sin_y

    latitude = np.degrees(np.arctan(ratio * sz / np.sqrt((height - sx) ** 2 + sy**2)))
    longitude = satellite_longitude - np.degrees(np.arctan2(sy, height - sx))
    longitude = (longitude + 180.0) % 360.0 - 180.0
    return latitude, longitude


def footprint_area_km2(
    latitude: np.ndarray, longitude: np.ndarray, earth_radius_km: float = 6371.0
) -> np.ndarray:
    """
    True ground area of each pixel, from the local stretch of the geolocation itself.

    Rather than assume a nominal resolution, this measures how far apart neighbouring pixel centres
    actually land on the ground and takes the cross product of the two spacings. At the sub-satellite
    point it recovers the nominal footprint; toward the limb it grows, which is the whole reason for
    computing it. These become the quadrature weights, so the stretched edge of the disk stops
    out-voting the centre.

    Args:
        latitude, longitude: ``(rows, cols)`` degrees, NaN allowed off the disk.

    Returns:
        ``(rows, cols)`` area in square kilometres.
    """
    lat_rad, lon_rad = np.radians(latitude), np.radians(longitude)
    cos_lat = np.cos(lat_rad)
    # Local east/north displacement per pixel step, in kilometres.
    east = earth_radius_km * cos_lat * _wrapped_gradient(lon_rad, axis=1)
    north = earth_radius_km * np.gradient(lat_rad, axis=1)
    east_row = earth_radius_km * cos_lat * _wrapped_gradient(lon_rad, axis=0)
    north_row = earth_radius_km * np.gradient(lat_rad, axis=0)
    # |a x b| for the two in-plane spacing vectors.
    return np.abs(east * north_row - north * east_row)


def _wrapped_gradient(angle: np.ndarray, axis: int) -> np.ndarray:
    """Central difference of an angle in radians, taking the shorter way around the circle."""
    gradient = np.gradient(angle, axis=axis)
    return (gradient + np.pi) % (2 * np.pi) - np.pi


def parse_goes_filename(key: str) -> dict:
    """Pull product, channel, satellite and scan start time out of an ABI object key."""
    name = key.rsplit("/", 1)[-1]
    match = re.match(
        r"OR_(?P<product>ABI-L2-\w+?)(?:-M\d)?(?:C(?P<channel>\d{2}))?_G(?P<sat>\d{2})_s(?P<start>\d{14})",
        name,
    )
    if not match:
        raise ValueError(f"Not an ABI filename: {name}")
    start = match.group("start")
    timestamp = dt.datetime.strptime(start[:11], "%Y%j%H%M").replace(tzinfo=dt.timezone.utc)
    timestamp += dt.timedelta(seconds=int(start[11:13]) + int(start[13]) / 10)
    return {
        "product": match.group("product"),
        "channel": int(match.group("channel")) if match.group("channel") else None,
        "satellite": f"G{match.group('sat')}",
        "timestamp": timestamp,
    }


def scene_from_netcdf(
    paths: dict[str, str],
    bounds: tuple[float, float, float, float] | None = None,
    stride: int = 1,
) -> SatelliteScene:
    """
    Build a :class:`SatelliteScene` from one or more ABI NetCDF files, one per channel.

    Args:
        paths: ``{channel_name: path}``, e.g. ``{"C13": "...", "C08": "..."}``. All files must share a
            grid, which they do within one product and scan.
        bounds: ``(lat_min, lat_max, lon_min, lon_max)`` to crop to, in degrees. Cropping happens in
            *geographic* space after geolocation, not in array space, so a box around a storm is a real
            box on the planet rather than a trapezoid that drifts with the projection.
        stride: take every n-th pixel, to thin a full-disk scene.

    Returns:
        A scene holding only samples that are on the Earth, inside ``bounds``, and not fill values.
    """
    import xarray as xr

    channels = tuple(sorted(paths))
    stacked, latitude, longitude, area, timestamp, satellite = None, None, None, None, None, None

    for index, channel in enumerate(channels):
        with xr.open_dataset(paths[channel]) as dataset:
            data = dataset["CMI"].values[::stride, ::stride].astype(np.float32)
            if latitude is None:
                projection = dataset["goes_imager_projection"].attrs
                scan_x = dataset["x"].values[::stride].astype(np.float64)
                scan_y = dataset["y"].values[::stride].astype(np.float64)
                mesh_x, mesh_y = np.meshgrid(scan_x, scan_y)
                latitude, longitude = fixed_grid_to_latlon(
                    mesh_x,
                    mesh_y,
                    float(projection["longitude_of_projection_origin"]),
                    float(projection["perspective_point_height"]),
                    float(projection["semi_major_axis"]),
                    float(projection["semi_minor_axis"]),
                )
                area = footprint_area_km2(latitude, longitude)
                timestamp = _as_datetime(dataset["t"].values)
                satellite = str(dataset.attrs.get("platform_ID", "unknown"))
                stacked = np.empty((*data.shape, len(channels)), dtype=np.float32)
            stacked[..., index] = data

    keep = np.isfinite(latitude) & np.isfinite(area) & np.isfinite(stacked).all(axis=-1)
    if bounds is not None:
        lat_min, lat_max, lon_min, lon_max = bounds
        keep &= (latitude >= lat_min) & (latitude <= lat_max)
        # Longitude comparison that survives a box straddling the date line.
        offset = (longitude - lon_min) % 360.0
        keep &= offset <= (lon_max - lon_min) % 360.0

    return SatelliteScene(
        latitude=latitude[keep].astype(np.float32),
        longitude=longitude[keep].astype(np.float32),
        values=stacked[keep],
        area_km2=area[keep].astype(np.float32),
        channels=channels,
        timestamp=timestamp,
        satellite=satellite,
    )


def _as_datetime(value) -> dt.datetime:
    seconds = np.datetime64(value, "s").astype("int64")
    return dt.datetime.fromtimestamp(int(seconds), tz=dt.timezone.utc)


#: Physical normalization for ABI channels: (offset K, scale K). Brightness temperature is a real
#: measurement on a real scale -- 180 K is an overshooting top, 300 K is warm ocean -- so it is centred
#: and scaled by fixed constants rather than rescaled per image, which would destroy exactly that.
ABI_NORMALIZATION = {
    "C08": (240.0, 30.0),   # upper-level water vapour
    "C09": (245.0, 30.0),   # mid-level water vapour
    "C10": (250.0, 30.0),   # lower-level water vapour
    "C13": (270.0, 40.0),   # clean infrared window, the cloud-top workhorse
    "C14": (270.0, 40.0),
    "C15": (270.0, 40.0),
}


def normalize_channels(values: np.ndarray, channels: tuple[str, ...]) -> np.ndarray:
    """Centre each channel on its physical range, keeping the scale comparable between scenes."""
    out = np.empty_like(values, dtype=np.float32)
    for index, channel in enumerate(channels):
        offset, scale = ABI_NORMALIZATION.get(channel, (270.0, 40.0))
        out[..., index] = (values[..., index] - offset) / scale
    return out
