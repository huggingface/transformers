# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Ready-made geometries for domains that are not the weather.

The point of declaring axes rather than flattening a grid is that nothing downstream is about weather.
A block does not know it is forecasting; it knows there are samples, distances, and quadrature weights.
Swap the geometry and the same weights, the same code and the same invariances apply to a different
physics.

This module is a shortcut, not a new capability -- every one of these is two lines of
:class:`~ihelix.geometry.Axis` -- but writing them down makes the range concrete and stops each new
domain from being rediscovered.

The property that carries across all of them is the one a fixed mesh cannot offer: **the grid is an
argument, not a shape baked into the weights.** Train on a coarse sampling, evaluate on a fine one, or
on scattered sensors that form no grid at all, without touching a parameter. For a simulation that is
the difference between one model and one model per mesh.
"""

from __future__ import annotations

from .geometry import EARTH_RADIUS_KM, Axis, Geometry


def ocean(radius: float = EARTH_RADIUS_KM, max_depth_m: float = 6000.0,
          depth_scale_m: float = 1000.0) -> Geometry:
    """
    Ocean: the sphere plus depth.

    Depth is linear rather than logarithmic, unlike atmospheric pressure -- seawater is very nearly
    incompressible, so a metre near the surface and a metre at 3 km are the same metre. ``depth_scale_m``
    sets how far "down" counts against "along": at the default, a kilometre of depth is worth a
    kilometre of horizontal distance, which makes a head's radius reach a sensible aspect ratio for
    mesoscale eddies.

    Fed satellite altimetry and sea-surface temperature, this is the geometry for eddy tracking. Feed it
    Argo floats too -- scattered, irregular, at wildly varying depths -- and they read onto the same mesh
    as the gridded fields, which is exactly the case a grid model has to interpolate away.
    """
    return Geometry([Axis("spherical", scale=radius), Axis("linear", scale=depth_scale_m / 1000.0)])


def seismic(extent_km: float = 1.0) -> Geometry:
    """
    A 3-D block of earth: x, y, depth, all in kilometres.

    Wave propagation through a velocity model is a field problem on an irregular domain, and seismic
    networks are the definition of scattered sampling -- stations sit where geology and politics allowed,
    not on a grid. Reading them directly, at their real coordinates, with each station weighted by the
    volume it stands for, is the natural formulation.
    """
    return Geometry([Axis("linear", scale=extent_km)] * 3)


def channel_flow(length: float, width: float, height_scale: float = 1.0) -> Geometry:
    """
    The canonical CFD box: periodic streamwise and spanwise, walls top and bottom.

    Periodicity here is not an approximation bolted on at the boundary -- a periodic axis embeds as a
    circle, so the last cell and the first are *neighbours in the metric*, at their true separation. A
    convolution on a flattened array has to pad or wrap and gets a seam either way.
    """
    return Geometry([
        Axis("periodic", period=length, scale=length / (2 * 3.141592653589793)),
        Axis("periodic", period=width, scale=width / (2 * 3.141592653589793)),
        Axis("linear", scale=height_scale),
    ])


def volume(extent: float = 1.0) -> Geometry:
    """A plain 3-D box: point clouds, medical volumes, molecular surfaces, anything with x/y/z."""
    return Geometry([Axis("linear", scale=extent)] * 3)


def image(pixel_scale: float = 1.0) -> Geometry:
    """
    A 2-D plane, for images treated as samples rather than as an array.

    Worth having even though it looks trivial: it is the control. Run the same architecture on an image
    as a plane and on the same image as a sphere patch, and the difference is the geometry, nothing else.
    """
    return Geometry([Axis("linear", scale=pixel_scale)] * 2)


def cylinder(radius: float, height_scale: float = 1.0) -> Geometry:
    """A pipe or an annulus: angle wraps, radius and axial distance do not."""
    return Geometry([
        Axis("periodic", period=2 * 3.141592653589793, scale=radius),
        Axis("linear", scale=1.0),
        Axis("linear", scale=height_scale),
    ])


#: Every domain in one place, with the radii that make sense for each. The radii matter more than the
#: axes: they are physical lengths, so they have to match the scale of the structures being modelled --
#: mesoscale eddies are tens of kilometres, boundary layers are millimetres.
CATALOGUE = {
    "globe": (Geometry.globe, {"min_radius": 120.0, "max_radius": 1600.0}),
    "atmosphere": (Geometry.atmosphere, {"min_radius": 120.0, "max_radius": 1600.0}),
    "ocean": (ocean, {"min_radius": 25.0, "max_radius": 400.0}),
    "seismic": (seismic, {"min_radius": 1.0, "max_radius": 50.0}),
    "channel_flow": (channel_flow, {"min_radius": 0.01, "max_radius": 0.5}),
    "volume": (volume, {"min_radius": 0.02, "max_radius": 0.4}),
    "image": (image, {"min_radius": 2.0, "max_radius": 64.0}),
    "cylinder": (cylinder, {"min_radius": 0.01, "max_radius": 0.5}),
}


def describe() -> str:
    """A printable list of the domains and the receptive radii each wants."""
    lines = [f"{'domain':14} {'coord dims':>10} {'min radius':>12} {'max radius':>12}  note"]
    lines.append("-" * 78)
    notes = {
        "globe": "Earth's surface",
        "atmosphere": "Earth + log-pressure",
        "ocean": "sphere + depth, km",
        "seismic": "3-D block, km",
        "channel_flow": "periodic x/y, wall-normal z",
        "volume": "point clouds, medical volumes",
        "image": "a plane -- the control case",
        "cylinder": "pipes and annuli",
    }
    for name, (factory, radii) in CATALOGUE.items():
        try:
            geometry = factory() if name in ("globe", "atmosphere", "seismic", "volume", "image") else None
            dims = geometry.coord_dim if geometry is not None else "-"
        except TypeError:
            dims = "-"
        lines.append(f"{name:14} {str(dims):>10} {radii['min_radius']:>12} {radii['max_radius']:>12}"
                     f"  {notes[name]}")
    return "\n".join(lines)
