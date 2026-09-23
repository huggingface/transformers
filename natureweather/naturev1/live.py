# Copyright 2026 Nathan. Apache-2.0.
"""
Live ingest: find the newest GOES scene, fetch it, run the model, repeat.

Two things that quietly break most versions of this:

* **GOES-16 is gone.** It was retired as GOES-East in 2025 and its bucket no longer receives data. The
  listing still succeeds and returns zero keys, so a poller written against it looks healthy and produces
  nothing forever. GOES-19 is East, GOES-18 is West.
* **The newest object is not the newest scene.** Files land out of order and an hour prefix only contains
  that hour, so "list the current hour and take the last key" returns nothing at ten past midnight and
  stale data whenever a file is late. This walks back through hours until it finds something and sorts by
  the scan start time parsed from the filename, not by key order.

No credentials are needed: the NOAA buckets are public, and the requests here are plain anonymous HTTPS.
"""

from __future__ import annotations

import datetime as dt
import shutil
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from .satellite import GOES_BUCKETS, GOES_PRODUCTS, parse_goes_filename


_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_objects(bucket: str, prefix: str, max_keys: int = 1000, timeout: float = 60.0) -> list[str]:
    """List keys under a prefix in a public S3 bucket, over anonymous HTTPS."""
    query = urllib.parse.urlencode({"list-type": "2", "prefix": prefix, "max-keys": str(max_keys)})
    url = f"https://{bucket}.s3.amazonaws.com/?{query}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        tree = ElementTree.fromstring(response.read())
    return [node.text for node in tree.findall("s3:Contents/s3:Key", _NAMESPACE)]


def latest_scene_keys(
    channels: tuple[str, ...] = ("C13",),
    satellite: str = "east",
    product: str = "conus",
    within_hours: int = 6,
    now: dt.datetime | None = None,
) -> dict[str, str]:
    """
    Newest available object per channel, walking back through hours until something turns up.

    Args:
        channels: ABI bands, e.g. ``("C13", "C09")``.
        satellite: ``"east"`` (GOES-19) or ``"west"`` (GOES-18).
        product: ``"conus"`` (5 min), ``"fulldisk"`` (10 min) or ``"mesoscale1"`` (1 min).
        within_hours: how far back to look before giving up.

    Returns:
        ``{channel: key}``. Channels with nothing recent are simply absent.
    """
    bucket = GOES_BUCKETS[satellite]
    product_code = GOES_PRODUCTS[product]
    now = now or dt.datetime.now(dt.timezone.utc)

    found: dict[str, tuple[dt.datetime, str]] = {}
    for hours_back in range(within_hours):
        moment = now - dt.timedelta(hours=hours_back)
        prefix = f"{product_code}/{moment:%Y}/{moment.timetuple().tm_yday:03d}/{moment:%H}/"
        try:
            keys = list_objects(bucket, prefix)
        except OSError:
            continue
        for key in keys:
            try:
                meta = parse_goes_filename(key)
            except ValueError:
                continue
            name = f"C{meta['channel']:02d}" if meta["channel"] else None
            if name in channels and (name not in found or meta["timestamp"] > found[name][0]):
                found[name] = (meta["timestamp"], key)
        if len(found) == len(channels):
            break
    return {name: key for name, (_, key) in found.items()}


def download(bucket: str, key: str, destination: str | Path, timeout: float = 600.0) -> Path:
    """
    Fetch one object, written atomically so a half-downloaded file never replaces a good one.

    This is the "save it over the old file" step: the destination is overwritten in place, but only once
    the download has completed, so a model reading the folder never sees a truncated NetCDF.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url, timeout=timeout) as response, open(temporary, "wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)
    return destination


def fetch_latest(
    directory: str | Path = "./test_data",
    channels: tuple[str, ...] = ("C13",),
    satellite: str = "east",
    product: str = "conus",
) -> dict[str, Path]:
    """
    Download the newest scene for each channel into ``directory``, overwriting the previous one.

    Files are named by channel only -- ``C13.nc``, not the full ABI key -- so the folder holds exactly
    one file per channel and the path your model reads never changes.
    """
    keys = latest_scene_keys(channels, satellite, product)
    if not keys:
        raise RuntimeError(
            f"No {product} data found for {channels} on GOES-{satellite} in the last few hours. "
            "Check the satellite is still operational and the product code is right."
        )
    bucket = GOES_BUCKETS[satellite]
    paths = {}
    for channel, key in keys.items():
        paths[channel] = download(bucket, key, Path(directory) / f"{channel}.nc")
        print(f"[live] {channel} <- {key.rsplit('/', 1)[-1]}", flush=True)
    return paths


def watch(
    run,
    interval_seconds: float = 3600.0,
    directory: str | Path = "./test_data",
    channels: tuple[str, ...] = ("C13",),
    satellite: str = "east",
    product: str = "conus",
    max_iterations: int | None = None,
) -> None:
    """
    Every ``interval_seconds``: download the newest scene, overwrite the old one, call ``run(paths)``.

    A failure in one cycle -- a network blip, a missing scene, a bad file -- is reported and skipped
    rather than ending the loop, because the next scene is only minutes away and an overnight watcher
    that dies at 3am on a transient error is worse than useless.

    Args:
        run: callable taking ``{channel: path}`` and doing whatever you want with the scene.
        interval_seconds: how long to wait between cycles.
        max_iterations: stop after this many cycles. Leave as None to run forever.
    """
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        started = time.time()
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        try:
            paths = fetch_latest(directory, channels, satellite, product)
            run(paths)
        except Exception as error:
            print(f"[live] {stamp} cycle failed: {type(error).__name__}: {error}", flush=True)
        iteration += 1
        if max_iterations is not None and iteration >= max_iterations:
            break
        sleep_for = max(0.0, interval_seconds - (time.time() - started))
        print(f"[live] {stamp} next cycle in {sleep_for / 60:.1f} min", flush=True)
        time.sleep(sleep_for)
