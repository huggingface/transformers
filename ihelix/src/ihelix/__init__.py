# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
iHELIX -- one architecture for fields on any geometry.

Most sequence models reach a grid by flattening it and hoping a positional encoding puts the geometry
back. It does not: a flattened sphere has a seam at the date line, poles where the meridians pile up, and
rows whose vertical neighbours sit thousands of positions away. The model is left to learn the topology
of its own input from data.

iHELIX takes the opposite route. You declare what your axes *are* -- spherical, periodic, linear,
logarithmic -- and everything downstream works in a metric space where the geometry is already correct.
Neighbours are the nearest samples on the manifold. Receptive fields are lengths, not array steps.
Attention is weighted by how much of the domain each sample stands for, so refining the grid converges
instead of drifting.

The same code and the same class cover every shape of problem::

    Geometry([Axis("linear")])                                  a 1-D signal
    Geometry([Axis("linear"), Axis("linear")])                  an image
    Geometry([Axis("linear")] * 3)                              a volume or a point cloud
    Geometry([Axis("periodic", period=L)] * 2 + [Axis("linear")])   a channel-flow box
    Geometry.globe()                                            the surface of the Earth
    Geometry.atmosphere()                                       the Earth plus pressure

and time, when there is a time, is the one axis that gets a recurrence -- because it is the only axis with
an order. Space does not have one, and this architecture never pretends it does.

    >>> import torch
    >>> from ihelix import Axis, Geometry, FieldGrid, IHelixConfig, IHelixField
    >>> coords = torch.rand(256, 2, dtype=torch.float64)
    >>> grid = FieldGrid.from_points(coords, Geometry([Axis("linear"), Axis("linear")]),
    ...                              num_neighbours=8, cluster_size=32)
    >>> model = IHelixField(IHelixConfig(in_channels=3, hidden_size=32, num_layers=2, num_heads=2,
    ...                                  num_kv_heads=1, head_dim=16, min_radius=0.05, max_radius=0.4,
    ...                                  use_temporal=False), grid.geometry)
    >>> model(torch.randn(1, 256, 3), grid).shape
    torch.Size([1, 256, 3])

Made by Nathan.
"""

from .attention import CrossAttention, GeodesicAttention, GridLink, IndexAttention, RelativeEncoder
from .geometry import EARTH_RADIUS_KM, Axis, Geometry
from .grid import FieldGrid, fibonacci_sphere, latlon_grid
from .model import IHelixBlock, IHelixConfig, IHelixField, IHelixFieldModel, TemporalStrand

__version__ = "0.1.0"
__author__ = "Nathan"

__all__ = [
    "EARTH_RADIUS_KM",
    "Axis",
    "CrossAttention",
    "FieldGrid",
    "GeodesicAttention",
    "Geometry",
    "GridLink",
    "IHelixBlock",
    "IHelixConfig",
    "IHelixField",
    "IHelixFieldModel",
    "IndexAttention",
    "RelativeEncoder",
    "TemporalStrand",
    "__version__",
    "fibonacci_sphere",
    "latlon_grid",
]
