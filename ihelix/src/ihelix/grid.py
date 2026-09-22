# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The grid: everything about *where* the samples are, computed once and reused.

A :class:`FieldGrid` is data, not architecture. It holds the neighbour graph and the spatial hierarchy for
one set of sample points, and the model takes it as an argument. Swap in a grid built from a finer set of
points and the same weights run on it unchanged -- which is the whole answer to handling resolution. The
model never sees a grid shape, a resolution, or a dimensionality; it sees points, distances and weights.

Two structures are built here:

* a **neighbour graph** -- for each point, its nearest neighbours *in metric space*, so "nearby" means
  nearby on the manifold rather than nearby in a flattened index. On a sphere this is what makes a point
  next to the date line have neighbours on both sides of it, and a point near the pole have the neighbours
  that are physically close rather than the ones the grid happens to list adjacently.
* a **hierarchy** -- a tree of nested spatial clusters, built by recursive splitting along the widest axis.
  Contiguous runs of the resulting order are compact regions, which is what lets the long-range strand
  summarize and retrieve whole regions cheaply.

Both depend only on the sample positions, so for a fixed grid -- which is the normal case in forecasting --
they are computed once at startup and amortize to nothing.
"""

from __future__ import annotations

import torch

from .geometry import Geometry


def _knn(points: torch.Tensor, num_neighbours: int, chunk: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Brute-force nearest neighbours in the embedded space, computed in chunks to bound memory.

    Exact, and fine for the grids a forecast model runs on, since the result is cached per grid. For very
    large point sets, pass a precomputed neighbour index to :meth:`FieldGrid.from_points` instead.

    Returns ``(indices, distances)``, each ``(N, num_neighbours)``, nearest first and including self.
    """
    num_points = points.shape[0]
    num_neighbours = min(num_neighbours, num_points)
    indices, distances = [], []
    for start in range(0, num_points, chunk):
        block = points[start : start + chunk]
        # `cdist` rather than the expanded-square trick: slower, but it does not lose precision on
        # near-coincident points, which is exactly what a converging meridian produces.
        d = torch.cdist(block, points)
        near = d.topk(num_neighbours, dim=-1, largest=False)
        indices.append(near.indices)
        distances.append(near.values)
    return torch.cat(indices), torch.cat(distances)


def _knn_between(
    target: torch.Tensor, source: torch.Tensor, num_neighbours: int, chunk: int = 2048
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest source points for each target point, for reading one grid from another."""
    num_neighbours = min(num_neighbours, source.shape[0])
    indices, distances = [], []
    for start in range(0, target.shape[0], chunk):
        near = torch.cdist(target[start : start + chunk], source).topk(num_neighbours, dim=-1, largest=False)
        indices.append(near.indices)
        distances.append(near.values)
    return torch.cat(indices), torch.cat(distances)


def _spatial_order(points: torch.Tensor, leaf_size: int) -> torch.Tensor:
    """
    Order points so that contiguous runs are spatially compact, by recursively splitting the widest axis.

    This is the spatial analogue of a space-filling curve: a permutation, computed once, after which
    "chunk k of the array" means "region k of the manifold". It is used only to build the hierarchy -- the
    local strand never reads adjacency from it, because runs of a linear order always have seams and seams
    are precisely the artefact this package exists to avoid.
    """
    order = torch.arange(points.shape[0], device=points.device)
    stack, result = [order], []
    while stack:
        group = stack.pop()
        if group.numel() <= leaf_size:
            result.append(group)
            continue
        block = points[group]
        centred = block - block.mean(0, keepdim=True)
        # Split along the principal axis rather than the widest coordinate axis. Coordinate axes are an
        # arbitrary choice of frame, so splitting on them would make the hierarchy depend on how the
        # domain happens to be oriented; the principal axis turns with the data.
        #
        # Worth being straight about the limit here: where the covariance is isotropic -- a whole sphere,
        # exactly -- there is no principal axis to find, and the split direction is arbitrary. No finite
        # partition of a sphere into regions is equivariant under arbitrary rotation; that is a property
        # of partitions, not of this one, and a fixed icosahedral mesh has it just the same. Sub-regions
        # are not isotropic, so every split below the first is well determined.
        eigenvectors = torch.linalg.eigh(centred.T @ centred).eigenvectors
        direction = eigenvectors[:, -1]
        # Fix the sign so the ordering is reproducible rather than up to a coin flip.
        direction = direction * torch.sign(direction[direction.abs().argmax()])
        ranking = (centred @ direction).argsort()
        middle = group.numel() // 2
        # Pushed in reverse so the left half is popped first and the order stays deterministic.
        stack.append(group[ranking[middle:]])
        stack.append(group[ranking[:middle]])
    return torch.cat(result)


class FieldGrid:
    """
    Sample positions, their neighbour graph, and their spatial hierarchy.

    Build one per set of sample points and pass it to the model::

        grid = FieldGrid.from_points(coords, Geometry.globe(), num_neighbours=32)
        output = model(values, grid)

    Attributes:
        geometry: the coordinate system these points live in.
        coords: ``(N, coord_dim)`` the points themselves.
        points: ``(N, embed_dim)`` their positions in metric space.
        weights: ``(N,)`` quadrature weights -- the share of the domain each sample stands for. Attention
            is weighted by these, which is what makes the result converge as sampling density rises
            instead of drifting with it.
        neighbours: ``(N, K)`` nearest-neighbour indices, nearest first, self included.
        neighbour_offsets: ``(N, K, embed_dim)`` neighbour displacements in each query's local frame.
        neighbour_distances: ``(N, K)`` metric distances to those neighbours.
        neighbour_alignment: ``(N, K)`` how closely each neighbour's local frame lines up with the query's.
        order: ``(num_clusters * cluster_size,)`` the spatial ordering, padded to whole clusters.
        cluster_members: ``(num_clusters, cluster_size)`` point indices per leaf cluster.
        cluster_valid: ``(num_clusters, cluster_size)`` False where the padding starts.
        cluster_points: ``(num_clusters, embed_dim)`` weighted cluster centroids.
    """

    def __init__(
        self,
        geometry: Geometry,
        coords: torch.Tensor,
        weights: torch.Tensor,
        neighbours: torch.Tensor,
        neighbour_distances: torch.Tensor,
        cluster_size: int,
    ) -> None:
        self.geometry = geometry
        self.coords = coords
        self.points = geometry.embed(coords)
        self.weights = weights
        self.neighbours = neighbours
        self.neighbour_distances = neighbour_distances
        self.neighbour_offsets = geometry.local_offsets(coords, coords[neighbours])
        # How much each neighbour's frame is twisted relative to the query's: 1 when they are aligned,
        # less as the manifold curves between them. On a curved domain this is what a model needs in order
        # to parallel-transport a direction -- to know that "east" at the neighbour is not quite "east" here.
        frames = geometry.frames(coords)
        self.neighbour_alignment = torch.einsum("pij,pkij->pk", frames, frames[neighbours]) / geometry.embed_dim
        self.cluster_size = cluster_size

        num_points = coords.shape[0]
        num_clusters = max(1, -(-num_points // cluster_size))
        order = _spatial_order(self.points, cluster_size)
        padding = num_clusters * cluster_size - num_points
        if padding:
            # Pad by repeating the last point; `cluster_valid` keeps it out of every sum.
            order = torch.cat([order, order[-1:].expand(padding)])
        self.order = order
        self.cluster_members = order.view(num_clusters, cluster_size)
        valid = torch.ones(num_clusters * cluster_size, dtype=torch.bool, device=coords.device)
        if padding:
            valid[-padding:] = False
        self.cluster_valid = valid.view(num_clusters, cluster_size)

        member_weights = self.weights[self.cluster_members] * self.cluster_valid
        self.cluster_weights = member_weights.sum(-1)
        self.cluster_points = (self.points[self.cluster_members] * member_weights.unsqueeze(-1)).sum(-2) / (
            self.cluster_weights.clamp_min(1e-12).unsqueeze(-1)
        )

    @property
    def num_points(self) -> int:
        return self.coords.shape[0]

    @property
    def num_clusters(self) -> int:
        return self.cluster_members.shape[0]

    @property
    def num_neighbours(self) -> int:
        return self.neighbours.shape[1]

    def __repr__(self) -> str:
        return (
            f"FieldGrid({self.num_points} points, {self.num_neighbours} neighbours, "
            f"{self.num_clusters} clusters of {self.cluster_size}, {self.geometry!r})"
        )

    @classmethod
    def from_points(
        cls,
        coords: torch.Tensor,
        geometry: Geometry,
        num_neighbours: int = 32,
        cluster_size: int = 64,
        weights: torch.Tensor | None = None,
        neighbours: torch.Tensor | None = None,
    ) -> FieldGrid:
        """
        Build a grid from sample coordinates.

        Args:
            coords: ``(N, coord_dim)`` sample coordinates, in the units ``geometry`` expects.
            geometry: the coordinate system.
            num_neighbours: neighbours per point for the local strand.
            cluster_size: leaf size of the spatial hierarchy.
            weights: ``(N,)`` quadrature weights. Defaults to uniform, which is right for an equal-area
                mesh and wrong for a latitude-longitude grid -- use :func:`latlon_grid` for those, which
                fills in ``cos(latitude)``.
            neighbours: ``(N, K)`` precomputed neighbour indices, if you have a spatial index already.
        """
        coords = torch.as_tensor(coords)
        if coords.ndim != 2:
            raise ValueError(f"coords must be (N, coord_dim), got shape {tuple(coords.shape)}.")
        points = geometry.embed(coords)
        if weights is None:
            weights = torch.ones(coords.shape[0], dtype=coords.dtype, device=coords.device)
        weights = torch.as_tensor(weights, dtype=coords.dtype, device=coords.device)
        # Normalized so that the weights sum to one: attention then reads as an average over the domain
        # rather than a sum over however many samples happen to be present.
        weights = weights / weights.sum().clamp_min(1e-12)

        if neighbours is None:
            neighbours, distances = _knn(points, num_neighbours)
        else:
            neighbours = torch.as_tensor(neighbours, device=coords.device)
            distances = (points[neighbours] - points.unsqueeze(1)).norm(dim=-1)
        return cls(geometry, coords, weights, neighbours, distances, cluster_size)

    def suggest_neighbours(self, radius: float, sigmas: float = 3.0) -> int:
        """
        Neighbours needed for every sample to reach ``sigmas * radius``.

        Fixed-count neighbourhoods and fixed-length receptive fields pull against each other: refine the
        grid and a fixed count covers less ground. This reports the count that keeps the physical reach
        intact, which is what has to grow with resolution -- a preprocessing choice, not a change to the
        weights.
        """
        target = sigmas * radius
        within = (torch.cdist(self.points, self.points) <= target).sum(-1)
        # One past the fullest neighbourhood: the K-th neighbour then lies strictly outside the cutoff,
        # which is what guarantees nothing inside it was dropped.
        return min(int(within.max()) + 1, self.num_points)

    def to(self, device: torch.device | str) -> FieldGrid:
        """Move every cached tensor to ``device``."""
        for name, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self


def latlon_grid(
    num_lat: int,
    num_lon: int,
    geometry: Geometry | None = None,
    levels: torch.Tensor | None = None,
    **kwargs,
) -> FieldGrid:
    """
    A regular latitude-longitude grid, with the ``cos(latitude)`` quadrature weights it needs.

    This is the grid shape that causes all the trouble -- cells shrink toward the poles and the seam at the
    date line is an artefact of how the array was laid out. Neither survives contact with this package:
    the weights account for the shrinking cells, and the neighbour graph is built in metric space where
    the seam does not exist.

    Args:
        num_lat: latitude points, spanning the poles exclusive (a Gaussian-style offset grid).
        num_lon: longitude points, spanning 0 to 360 exclusive.
        geometry: defaults to :meth:`Geometry.globe`, or :meth:`Geometry.atmosphere` when ``levels`` is given.
        levels: optional pressure levels in hPa, which turns the surface into a 3-D shell.
    """
    latitude = torch.linspace(-torch.pi / 2, torch.pi / 2, num_lat + 2, dtype=torch.float64)[1:-1]
    longitude = torch.arange(num_lon, dtype=torch.float64) * (2 * torch.pi / num_lon)
    lat_mesh, lon_mesh = torch.meshgrid(latitude, longitude, indexing="ij")
    coords = torch.stack([lat_mesh.reshape(-1), lon_mesh.reshape(-1)], dim=-1)
    weights = lat_mesh.reshape(-1).cos()

    if levels is not None:
        levels = torch.as_tensor(levels, dtype=torch.float64).reshape(-1)
        coords = torch.cat(
            [coords.repeat_interleave(levels.numel(), 0), levels.repeat(coords.shape[0]).unsqueeze(-1)], dim=-1
        )
        weights = weights.repeat_interleave(levels.numel())
        geometry = geometry or Geometry.atmosphere()
    else:
        geometry = geometry or Geometry.globe()
    return FieldGrid.from_points(coords, geometry, weights=weights, **kwargs)


def fibonacci_sphere(num_points: int, geometry: Geometry | None = None, **kwargs) -> FieldGrid:
    """
    A near-equal-area spiral mesh on the sphere -- the natural home for the model's internal latent grid.

    Because the cells are all about the same size there is no pole to distort and no seam to cross, and the
    uniform quadrature weights are the correct ones. Encode onto this from whatever grid the data arrives
    on, run the blocks here, decode back out to whatever grid is wanted.
    """
    index = torch.arange(num_points, dtype=torch.float64)
    golden = (1 + 5**0.5) / 2
    latitude = torch.asin(1 - 2 * (index + 0.5) / num_points)
    longitude = torch.remainder(2 * torch.pi * index / golden, 2 * torch.pi)
    coords = torch.stack([latitude, longitude], dim=-1)
    return FieldGrid.from_points(coords, geometry or Geometry.globe(), **kwargs)
