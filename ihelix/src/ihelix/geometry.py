# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Geometry: what replaces "position in a sequence".

The premise of this package is that flattening a field into a 1D sequence and hoping rotary embeddings
recover the geometry is backwards. A lat/lon grid flattened row-major puts two cells that are one row
apart ``n_lon`` positions apart in the sequence; the date line becomes a cliff; and the poles, where the
meridians converge, look to the model like ordinary interior points with ordinary neighbours. Nothing in
an index tells it otherwise, so it has to learn spherical topology from data.

Instead, hand it the geometry directly. Every axis declares what kind of thing it is, and the whole
coordinate system is mapped into a Euclidean space where the straight-line distance already *is* the
distance you care about:

* a **spherical** axis pair (latitude, longitude) becomes a point on a sphere in 3-D. Chordal distance
  there is a monotone function of great-circle distance, so the poles stop being special and the date line
  stops existing -- longitude 359.9 and 0.1 sit next to each other, because they are next to each other.
* a **periodic** axis becomes a circle, which is the same trick applied to a channel-flow box.
* **linear** and **log** axes pass through, scaled. Pressure is a log axis: the model should think a
  factor of two in pressure, not a hectopascal.

Each axis carries a ``scale`` in whatever unit you want distances measured in (kilometres, say), which is
what lets you say that a kilometre upwards is not meteorologically the same as a kilometre sideways.

The second thing geometry provides is a **local frame** at every point -- east, north, up on a sphere.
Neighbour offsets are expressed in that frame, so "300 km east" is the same input vector at the equator
and at 70 degrees north. That is what actually removes the polar distortion: not a correction factor, but
a representation in which the distortion was never introduced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

AxisKind = Literal["spherical", "periodic", "linear", "log"]

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Axis:
    """
    One axis (or, for ``spherical``, one pair of axes) of a coordinate system.

    Args:
        kind: ``"spherical"`` consumes ``(latitude, longitude)`` in radians and emits a 3-D point on a
            sphere of radius ``scale``. ``"periodic"`` consumes one coordinate and emits a 2-D point on a
            circle chosen so that arc length matches ``scale`` per coordinate unit. ``"linear"`` and
            ``"log"`` each consume and emit one coordinate.
        scale: Distance units per coordinate unit -- the sphere's radius for ``"spherical"``, and for the
            others the number of distance units one coordinate unit is worth. Setting these is how you
            declare the aspect ratio of your problem.
        period: Required for ``"periodic"``; the length after which the coordinate repeats.
        reference: Required for ``"log"``; the coordinate that maps to zero. For pressure levels in hPa,
            ``reference=1000.0`` puts the surface at the origin and altitude increasing upward.
    """

    kind: AxisKind
    scale: float = 1.0
    period: float | None = None
    reference: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("spherical", "periodic", "linear", "log"):
            raise ValueError(f"Unknown axis kind {self.kind!r}.")
        if self.scale <= 0:
            raise ValueError(f"Axis scale must be positive, got {self.scale}.")
        if self.kind == "periodic" and not (self.period and self.period > 0):
            raise ValueError("A periodic axis needs a positive `period`.")
        if self.kind == "log" and not (self.reference and self.reference > 0):
            raise ValueError("A log axis needs a positive `reference` coordinate.")

    @property
    def coord_dim(self) -> int:
        """Coordinate columns this axis consumes."""
        return 2 if self.kind == "spherical" else 1

    @property
    def embed_dim(self) -> int:
        """Embedding columns this axis produces."""
        return {"spherical": 3, "periodic": 2, "linear": 1, "log": 1}[self.kind]

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        """Map ``(..., coord_dim)`` coordinates to ``(..., embed_dim)`` metric-space positions."""
        if self.kind == "spherical":
            latitude, longitude = coords[..., 0], coords[..., 1]
            cos_lat = latitude.cos()
            return self.scale * torch.stack(
                [cos_lat * longitude.cos(), cos_lat * longitude.sin(), latitude.sin()], dim=-1
            )
        value = coords[..., 0]
        if self.kind == "periodic":
            # Radius chosen so that a step of `du` along the circle covers `scale * du` of arc.
            radius = self.scale * self.period / (2 * math.pi)
            angle = 2 * math.pi * value / self.period
            return radius * torch.stack([angle.cos(), angle.sin()], dim=-1)
        if self.kind == "log":
            return (self.scale * torch.log(self.reference / value.clamp_min(1e-12))).unsqueeze(-1)
        return (self.scale * value).unsqueeze(-1)

    def frame(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Orthonormal local frame at each point, ``(..., embed_dim, embed_dim)``, rows being basis vectors.

        On a sphere the rows are east, north and up, so expressing a neighbour offset in this frame gives
        the same numbers for the same physical displacement anywhere on the globe.
        """
        if self.kind == "spherical":
            latitude, longitude = coords[..., 0], coords[..., 1]
            sin_lat, cos_lat, sin_lon, cos_lon = latitude.sin(), latitude.cos(), longitude.sin(), longitude.cos()
            zero = torch.zeros_like(sin_lat)
            east = torch.stack([-sin_lon, cos_lon, zero], dim=-1)
            north = torch.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dim=-1)
            up = torch.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dim=-1)
            return torch.stack([east, north, up], dim=-2)
        if self.kind == "periodic":
            angle = 2 * math.pi * coords[..., 0] / self.period
            tangent = torch.stack([-angle.sin(), angle.cos()], dim=-1)
            radial = torch.stack([angle.cos(), angle.sin()], dim=-1)
            return torch.stack([tangent, radial], dim=-2)
        return torch.ones(*coords.shape[:-1], 1, 1, dtype=coords.dtype, device=coords.device)


class Geometry:
    """
    A coordinate system, built by composing :class:`Axis` objects.

    The same class covers every dimensionality the model might see::

        Geometry([Axis("linear", scale=1.0)])                                    # a 1-D signal
        Geometry([Axis("linear"), Axis("linear")])                               # an image
        Geometry([Axis("periodic", period=L), Axis("periodic", period=L),
                  Axis("linear")])                                               # a periodic box
        Geometry.globe()                                                         # the surface of the Earth
        Geometry.atmosphere()                                                    # the Earth plus pressure

    Nothing downstream of this class knows how many dimensions there are, or that any of them wrap.
    """

    def __init__(self, axes: Sequence[Axis]) -> None:
        if not axes:
            raise ValueError("A Geometry needs at least one axis.")
        self.axes = tuple(axes)
        self.coord_dim = sum(axis.coord_dim for axis in self.axes)
        self.embed_dim = sum(axis.embed_dim for axis in self.axes)

    def __repr__(self) -> str:
        parts = ", ".join(f"{a.kind}(scale={a.scale:g})" for a in self.axes)
        return f"Geometry([{parts}]) coord_dim={self.coord_dim} embed_dim={self.embed_dim}"

    @classmethod
    def globe(cls, radius: float = EARTH_RADIUS_KM) -> Geometry:
        """The surface of a sphere, coordinates ``(latitude, longitude)`` in radians."""
        return cls([Axis("spherical", scale=radius)])

    @classmethod
    def atmosphere(
        cls,
        radius: float = EARTH_RADIUS_KM,
        surface_pressure: float = 1000.0,
        scale_height_km: float = 7.0,
    ) -> Geometry:
        """
        A spherical shell: ``(latitude, longitude, pressure)``, angles in radians and pressure in hPa.

        Pressure enters logarithmically, which is what makes the vertical spacing physical -- the distance
        from 1000 hPa to 500 hPa equals the distance from 500 to 250. ``scale_height_km`` converts that
        into the same units as the horizontal, and is the knob that says how far "up" is compared to
        "along": the default puts one atmospheric scale height at 7 km.
        """
        return cls(
            [
                Axis("spherical", scale=radius),
                Axis("log", scale=scale_height_km, reference=surface_pressure),
            ]
        )

    def split(self, coords: torch.Tensor) -> list[torch.Tensor]:
        """Slice a coordinate tensor into per-axis pieces."""
        if coords.shape[-1] != self.coord_dim:
            raise ValueError(
                f"Expected coordinates with {self.coord_dim} columns for {self!r}, got {coords.shape[-1]}."
            )
        pieces, offset = [], 0
        for axis in self.axes:
            pieces.append(coords[..., offset : offset + axis.coord_dim])
            offset += axis.coord_dim
        return pieces

    def embed(self, coords: torch.Tensor) -> torch.Tensor:
        """Map ``(..., coord_dim)`` coordinates into the metric space, ``(..., embed_dim)``."""
        return torch.cat(
            [axis.embed(piece) for axis, piece in zip(self.axes, self.split(coords), strict=True)], dim=-1
        )

    def frames(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Block-diagonal orthonormal frames, ``(..., embed_dim, embed_dim)``.

        Each axis contributes its own block, so a neighbour offset expressed in this frame reads as
        "so far east, so far north, so far up" regardless of where on the manifold the query sits.
        """
        blocks = [axis.frame(piece) for axis, piece in zip(self.axes, self.split(coords), strict=True)]
        batch_shape = coords.shape[:-1]
        frame = torch.zeros(*batch_shape, self.embed_dim, self.embed_dim, dtype=coords.dtype, device=coords.device)
        offset = 0
        for block in blocks:
            size = block.shape[-1]
            frame[..., offset : offset + size, offset : offset + size] = block
            offset += size
        return frame

    def local_offsets(self, coords: torch.Tensor, neighbour_coords: torch.Tensor) -> torch.Tensor:
        """
        Displacement from each point to each of its neighbours, in the point's own local frame.

        Args:
            coords: ``(N, coord_dim)`` query coordinates.
            neighbour_coords: ``(N, K, coord_dim)`` neighbour coordinates.

        Returns:
            ``(N, K, embed_dim)`` offsets. On a sphere the columns are (east, north, up) in the query's
            frame -- the representation in which polar distortion never arises.
        """
        origin = self.embed(coords)
        target = self.embed(neighbour_coords)
        frame = self.frames(coords)
        return torch.einsum("nij,nkj->nki", frame, target - origin.unsqueeze(-2))
