# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
Attention that reads geometry instead of index order.

Three mixers, matching the three strands of `helix-lm` but with "position in a sequence" replaced
throughout by "position on a manifold":

* :class:`GeodesicAttention` -- the short range. Each sample attends to its nearest neighbours *in metric
  space*, with a bias computed from the displacement expressed in the sample's own local frame, and a
  per-head Gaussian window whose width is a physical length. Different heads see different physical
  radii, which is a receptive-field ladder in metres rather than in array steps.
* :class:`IndexAttention` -- the long range. The domain is summarized into a tree of nested regions and
  each query region retrieves a handful of others by content. This is what reaches a teleconnection, a
  distant vortex, or the other side of an image, without the hop count a message-passing mesh would need.
* :class:`CrossAttention` -- reading one point set from another, which is how the model gets on and off
  its internal mesh and thereby stops caring what resolution the data arrived at.

Every one of them weights by the quadrature weight of the sample being read. That single term is what
makes the result a discretization of a continuous operator rather than a sum over however many samples
happened to be present: attention becomes

    out(x) = integral a(x,y) v(y) dmu(y) / integral a(x,y) dmu(y)

estimated on the samples you have. Refine the grid and the estimate converges instead of drifting.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .grid import FieldGrid
from .kernels import NEG_SCORE


def _duplicate_mask(index: torch.Tensor) -> torch.Tensor:
    """Mark every repeat of a value along the last axis except its first occurrence."""
    order = index.argsort(dim=-1, stable=True)
    ordered = index.gather(-1, order)
    repeated = torch.zeros_like(ordered, dtype=torch.bool)
    repeated[..., 1:] = ordered[..., 1:] == ordered[..., :-1]
    return torch.zeros_like(repeated).scatter_(-1, order, repeated)


class RelativeEncoder(nn.Module):
    """
    Turns a neighbour displacement into a per-head attention bias.

    The input is the offset expressed in the query's own local frame, so the same physical displacement
    produces the same input anywhere on the manifold -- at the equator and at 80 degrees north, at the
    centre of an image and at its edge. That is the mechanism by which polar distortion is not corrected
    but simply never introduced.

    Alongside the offset it sees the distance (logarithmically, so near and far are both resolved) and how
    much the neighbour's frame is rotated relative to the query's, which on a curved manifold is the
    information a model needs to parallel-transport a direction.
    """

    def __init__(self, embed_dim: int, num_heads: int, hidden: int = 64, reference_length: float = 1.0) -> None:
        super().__init__()
        self.reference_length = reference_length
        self.net = nn.Sequential(nn.Linear(embed_dim + 2, hidden), nn.SiLU(), nn.Linear(hidden, num_heads, bias=False))

    def forward(self, offsets: torch.Tensor, distances: torch.Tensor, alignment: torch.Tensor) -> torch.Tensor:
        """``(P, K, embed_dim)``, ``(P, K)``, ``(P, K)`` -> ``(P, K, num_heads)``."""
        features = torch.cat(
            [
                offsets / self.reference_length,
                torch.log1p(distances / self.reference_length).unsqueeze(-1),
                alignment.unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.net(features)


class _HeadedProjections(nn.Module):
    """Query/key/value/output projections with grouped-query heads, shared by the attention modules."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        if num_heads % num_kv_heads:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}).")
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.groups, self.head_dim = num_heads // num_kv_heads, head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def queries(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, P, C)`` -> ``(B, P, kv_heads, groups, head_dim)``."""
        return self.q_proj(x).view(*x.shape[:-1], self.num_kv_heads, self.groups, self.head_dim)

    def keys_values(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(B, P, C)`` -> two ``(B, P, kv_heads, head_dim)``."""
        shape = (*x.shape[:-1], self.num_kv_heads, self.head_dim)
        return self.k_proj(x).view(shape), self.v_proj(x).view(shape)

    def merge(self, attended: torch.Tensor) -> torch.Tensor:
        """``(B, P, kv_heads, groups, head_dim)`` -> ``(B, P, hidden_size)``."""
        return self.o_proj(attended.reshape(*attended.shape[:2], self.num_heads * self.head_dim))


class GeodesicAttention(nn.Module):
    """
    The short-range strand: attention over nearest neighbours in metric space.

    Each head carries its own physical radius, applied as a Gaussian falloff on the logits, so a single
    layer sees several scales at once -- a few hundred kilometres and a few thousand, or a few pixels and
    a few dozen. Radii are learned but initialized across a geometric ladder, and they are lengths, not
    counts, so they mean the same thing when the sampling changes.

    The neighbour list has to be long enough to contain the widest radius; :meth:`coverage` reports
    whether it is, and the model warns once at construction if it is not.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        embed_dim: int,
        min_radius: float,
        max_radius: float,
        relative_hidden: int = 64,
        cutoff_sigmas: float = 3.0,
    ) -> None:
        super().__init__()
        self.cutoff_sigmas = cutoff_sigmas
        self.proj = _HeadedProjections(hidden_size, num_heads, num_kv_heads, head_dim)
        self.relative = RelativeEncoder(embed_dim, num_heads, relative_hidden, reference_length=min_radius)
        # A geometric ladder of physical receptive radii, one per head, stored in log space so training
        # cannot push one through zero.
        ladder = torch.logspace(math.log10(min_radius), math.log10(max_radius), num_heads, dtype=torch.float32)
        self.log_radius = nn.Parameter(ladder.log())

    @property
    def radii(self) -> torch.Tensor:
        return self.log_radius.exp()

    def forward(self, x: torch.Tensor, grid: FieldGrid) -> torch.Tensor:
        batch, num_points, _ = x.shape
        proj = self.proj
        neighbours = grid.neighbours

        queries = proj.queries(x)
        keys, values = proj.keys_values(x)
        # (B, P, K, kv_heads, head_dim) -> (B, P, kv_heads, K, head_dim)
        gathered_k = keys[:, neighbours].permute(0, 1, 3, 2, 4)
        gathered_v = values[:, neighbours].permute(0, 1, 3, 2, 4)

        logits = torch.einsum("bphgd,bphkd->bphgk", queries, gathered_k) * proj.scaling

        offsets = grid.neighbour_offsets.to(x.dtype)
        distances = grid.neighbour_distances.to(x.dtype)
        alignment = grid.neighbour_alignment.to(x.dtype)
        bias = self.relative(offsets, distances, alignment)  # (P, K, heads)
        bias = bias.permute(0, 2, 1).view(num_points, proj.num_kv_heads, proj.groups, -1)

        # Physical receptive field: a window in metres, not a cutoff in array steps. The window has
        # *compact support* -- it is exactly zero past `cutoff_sigmas` radii, not merely small. That is
        # what makes this attention exactly independent of which samples lie outside a head's reach,
        # rather than approximately so, and it is why the invariances below hold to the last bit rather
        # than to a few decimal places. `coverage` checks the one precondition: that the neighbour list
        # actually reaches the cutoff.
        radii = self.radii.to(x.dtype).view(proj.num_kv_heads, proj.groups, 1)
        spread = distances.view(num_points, 1, 1, -1) / radii
        window = -(spread**2) / 2
        window = window.masked_fill(spread > self.cutoff_sigmas, NEG_SCORE)
        # Quadrature: weight each sample by the share of the domain it stands for.
        quadrature = torch.log(grid.weights[neighbours].to(x.dtype).clamp_min(1e-30)).view(num_points, 1, 1, -1)

        logits = logits + (bias + window + quadrature).unsqueeze(0)
        attention = logits.softmax(-1)
        attended = torch.einsum("bphgk,bphkd->bphgd", attention, gathered_v)
        return proj.merge(attended)

    def coverage(self, grid: FieldGrid, sigmas: float | None = None) -> float:
        """
        Fraction of samples whose neighbour list reaches ``sigmas`` times the widest head radius.

        This is the architecture's one precondition, and it is worth stating plainly. At ``1.0`` every
        head's window is decided by its own radius, the samples beyond it contribute exactly nothing, and
        the model is exactly invariant to sample ordering, exactly equivariant to rotating the domain, and
        convergent under refinement. Below ``1.0`` the widest heads are being clipped by the neighbour
        count instead, those guarantees degrade smoothly, and refining the grid makes a fixed neighbour
        count cover less ground rather than more. Raise ``num_neighbours`` until this reaches 1.
        """
        sigmas = self.cutoff_sigmas if sigmas is None else sigmas
        if grid.num_neighbours >= grid.num_points:
            # The list already holds every sample, so nothing can have been excluded by truncation --
            # whatever the radii are, the cutoff is doing all the deciding.
            return 1.0
        reach = grid.neighbour_distances[:, -1]
        return float((reach >= sigmas * float(self.radii.max().detach())).to(torch.float32).mean())


class IndexAttention(nn.Module):
    """
    The long-range strand: retrieve whole regions of the domain by content.

    The domain is summarized bottom-up into a tree of nested regions. Each query region routes from its
    own summary down the tree, keeping a beam, and ends up attending over the samples of the
    ``index_topk`` regions it selected -- which may be anywhere, at any distance.

    This is the part a fixed mesh cannot do. Message passing on a multi-mesh couples regions by distance
    along fixed edges, so reaching the far side of the domain costs hops and the route is decided in
    advance. Here the route is content-addressed and learned: a ridge over the Atlantic can read the
    Pacific in one step if that is what the field calls for, and nothing in the wiring had to anticipate it.

    Selection is discrete, so the routing score is fed back as an additive bias on the retrieved logits,
    accumulated down the descent path -- which is what carries gradient into every level of the tree.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        landmark_dim: int,
        branching: int,
        beam_width: int,
        topk: int,
        num_distance_buckets: int,
        reference_length: float,
    ) -> None:
        super().__init__()
        from .kernels import IHelixLandmarkPooler

        self.proj = _HeadedProjections(hidden_size, num_heads, num_kv_heads, head_dim)
        self.branching, self.beam_width, self.topk = branching, beam_width, topk
        self.landmark_dim, self.reference_length = landmark_dim, reference_length
        self.num_distance_buckets = num_distance_buckets
        self.leaf_pooler = IHelixLandmarkPooler(head_dim, landmark_dim, 1e-5)
        # One pooler for every internal level: a summary of summaries is formed the same way at every
        # scale, which saves parameters and biases the tree toward scale invariance.
        self.node_pooler = IHelixLandmarkPooler(landmark_dim, landmark_dim, 1e-5)
        self.route_proj = nn.Linear(landmark_dim, landmark_dim, bias=False)
        self.distance_bias = nn.Parameter(torch.zeros(num_kv_heads, num_distance_buckets))
        self.level_bias = nn.Parameter(torch.zeros(num_kv_heads, 16))

    def _bucket(self, distance: torch.Tensor) -> torch.Tensor:
        """Logarithmic distance buckets, saturating -- so "very far" keeps meaning something."""
        scaled = (distance / self.reference_length).clamp_min(0)
        return torch.log2(scaled + 1.0).floor().long().clamp(max=self.num_distance_buckets - 1)

    def _build_tree(self, leaves: torch.Tensor) -> list[torch.Tensor]:
        """Leaves first; every level holds only nodes whose children are all present."""
        levels, count, level = [leaves], leaves.shape[2], 1
        while count // self.branching**level >= 1:
            nodes = count // self.branching**level
            children = levels[-1][:, :, : nodes * self.branching]
            children = children.reshape(*children.shape[:2], nodes, self.branching, self.landmark_dim)
            levels.append(self.node_pooler(children))
            level += 1
        return levels

    def _descend(self, levels: list[torch.Tensor], route: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Beam descent. Returns selected leaf indices and their accumulated path scores."""
        batch, kv_heads, num_queries, _ = route.shape
        device = route.device
        slots = self.beam_width * self.branching
        candidates = (
            torch.arange(slots, device=device).view(1, 1, 1, slots).expand(batch, kv_heads, num_queries, slots)
        )
        parent_path = route.new_zeros(candidates.shape)
        beam = beam_path = None

        for level in range(len(levels) - 1, -1, -1):
            if beam is not None:
                candidates = (
                    beam.unsqueeze(-1) * self.branching + torch.arange(self.branching, device=device)
                ).flatten(-2)
                parent_path = beam_path.unsqueeze(-1).expand(*beam.shape, self.branching).flatten(-2)
            landmarks = levels[level]
            count = landmarks.shape[2]
            valid = candidates < count
            safe = candidates.clamp(0, max(count - 1, 0))
            gathered = landmarks.gather(
                2, safe.reshape(batch, kv_heads, -1, 1).expand(-1, -1, -1, self.landmark_dim)
            ).view(batch, kv_heads, num_queries, -1, self.landmark_dim)
            score = (route.unsqueeze(3) * gathered).sum(-1) * (self.landmark_dim**-0.5)
            score = score + self.level_bias[:, min(level, self.level_bias.shape[1] - 1)].view(1, kv_heads, 1, 1)
            path = parent_path + score
            rank = path.masked_fill(~valid | _duplicate_mask(candidates), NEG_SCORE)
            width = self.topk if level == 0 else self.beam_width
            top = rank.topk(min(width, rank.shape[-1]), dim=-1)
            beam, beam_path = candidates.gather(-1, top.indices), path.gather(-1, top.indices)
        return beam, top.values

    def forward(self, x: torch.Tensor, grid: FieldGrid) -> torch.Tensor:
        batch = x.shape[0]
        proj = self.proj
        members, member_valid = grid.cluster_members, grid.cluster_valid
        num_clusters, cluster_size = members.shape

        queries = proj.queries(x)
        keys, values = proj.keys_values(x)
        # Regroup everything by cluster: (B, clusters, size, kv_heads, dim) -> (B, kv_heads, clusters, size, dim)
        cluster_k = keys[:, members].permute(0, 3, 1, 2, 4)
        cluster_v = values[:, members].permute(0, 3, 1, 2, 4)
        cluster_q = queries[:, members].permute(0, 3, 4, 1, 2, 5)  # (B, kv, groups, clusters, size, dim)

        weights = grid.weights[members].to(x.dtype) * member_valid
        leaves = self.leaf_pooler(cluster_k, member_valid.view(1, 1, num_clusters, cluster_size))
        levels = self._build_tree(leaves)
        route = self.route_proj(leaves)
        selected, path_score = self._descend(levels, route)

        usable = path_score > NEG_SCORE / 2
        flat = selected.clamp(0, num_clusters - 1).reshape(batch, proj.num_kv_heads, -1)
        picked_k = cluster_k.gather(2, flat[..., None, None].expand(-1, -1, -1, cluster_size, proj.head_dim)).view(
            batch, proj.num_kv_heads, num_clusters, -1, proj.head_dim
        )
        picked_v = cluster_v.gather(2, flat[..., None, None].expand(-1, -1, -1, cluster_size, proj.head_dim)).view(
            batch, proj.num_kv_heads, num_clusters, -1, proj.head_dim
        )
        picked_valid = (
            member_valid.view(1, 1, num_clusters, cluster_size)
            .expand(batch, proj.num_kv_heads, -1, -1)
            .gather(2, flat[..., None].expand(-1, -1, -1, cluster_size))
            .view(batch, proj.num_kv_heads, num_clusters, -1)
        )
        picked_weights = (
            weights.view(1, 1, num_clusters, cluster_size)
            .expand(batch, proj.num_kv_heads, -1, -1)
            .gather(2, flat[..., None].expand(-1, -1, -1, cluster_size))
            .view(batch, proj.num_kv_heads, num_clusters, -1)
        )

        logits = torch.einsum("bhgcsd,bhcmd->bhgcsm", cluster_q, picked_k) * proj.scaling

        # Coarse geometric prior at the region level: how far away the retrieved region is.
        separation = (grid.cluster_points.unsqueeze(1) - grid.cluster_points.unsqueeze(0)).norm(dim=-1)
        picked_distance = separation.unsqueeze(0).unsqueeze(0).expand(batch, proj.num_kv_heads, -1, -1)
        picked_distance = picked_distance.gather(3, selected.clamp(0, num_clusters - 1))  # (B, kv, clusters, topk)
        head_index = torch.arange(proj.num_kv_heads, device=x.device).view(1, -1, 1, 1)
        region_bias = self.distance_bias[head_index, self._bucket(picked_distance)]
        # Feeding the routing score back in is what makes the discrete top-k differentiable.
        region_bias = region_bias + F.logsigmoid(path_score.float()).to(x.dtype)
        region_bias = region_bias.repeat_interleave(cluster_size, dim=-1)

        mask = picked_valid & usable.repeat_interleave(cluster_size, dim=-1)
        additive = (region_bias + torch.log(picked_weights.clamp_min(1e-30))).masked_fill(
            mask.logical_not(), NEG_SCORE
        )
        logits = logits + additive.unsqueeze(2).unsqueeze(4)

        attention = logits.softmax(-1)
        attended = torch.einsum("bhgcsm,bhcmd->bhgcsd", attention, picked_v)
        # Back to point order: (B, kv, groups, clusters, size, dim) -> (B, N, kv, groups, dim)
        attended = attended.permute(0, 3, 4, 1, 2, 5).reshape(
            batch, num_clusters * cluster_size, proj.num_kv_heads, proj.groups, proj.head_dim
        )
        output = torch.zeros(
            batch, grid.num_points, proj.num_kv_heads, proj.groups, proj.head_dim, dtype=x.dtype, device=x.device
        )
        flat_order = grid.order
        keep = grid.cluster_valid.reshape(-1)
        output[:, flat_order[keep]] = attended[:, keep]
        return proj.merge(output)


class CrossAttention(nn.Module):
    """
    Read one point set from another, with the same geometric bias and quadrature weighting.

    This is how the model gets on and off its own internal mesh. Encode from whatever grid the data
    arrived on, run the blocks on a fixed mesh, decode to whatever grid is wanted. Because both directions
    are geometric rather than index-based, neither cares about resolution: the same weights read a
    one-degree grid, a quarter-degree grid, a rotated mesh, or a scatter of station observations.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        embed_dim: int,
        radius: float,
        relative_hidden: int = 64,
        cutoff_sigmas: float = 3.0,
    ) -> None:
        super().__init__()
        self.cutoff_sigmas = cutoff_sigmas
        self.proj = _HeadedProjections(hidden_size, num_heads, num_kv_heads, head_dim)
        self.relative = RelativeEncoder(embed_dim, num_heads, relative_hidden, reference_length=radius)
        self.log_radius = nn.Parameter(torch.full((num_heads,), math.log(radius)))

    def coverage(self, link: GridLink, sigmas: float | None = None) -> float:
        """Fraction of target points whose source list reaches the read radius. See `GeodesicAttention.coverage`."""
        sigmas = self.cutoff_sigmas if sigmas is None else sigmas
        if link.neighbours.shape[1] >= link.source_points:
            return 1.0
        reach = link.distances[:, -1]
        return float((reach >= sigmas * float(self.log_radius.exp().max().detach())).to(torch.float32).mean())

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        link: GridLink,
    ) -> torch.Tensor:
        """``target`` is ``(B, P, C)`` on the reading grid; ``source`` is ``(B, M, C)`` on the read grid."""
        batch, num_target, _ = target.shape
        proj = self.proj
        queries = proj.queries(target)
        keys, values = proj.keys_values(source)
        gathered_k = keys[:, link.neighbours].permute(0, 1, 3, 2, 4)
        gathered_v = values[:, link.neighbours].permute(0, 1, 3, 2, 4)

        logits = torch.einsum("bphgd,bphkd->bphgk", queries, gathered_k) * proj.scaling
        distances = link.distances.to(target.dtype)
        bias = self.relative(link.offsets.to(target.dtype), distances, link.alignment.to(target.dtype))
        bias = bias.permute(0, 2, 1).view(num_target, proj.num_kv_heads, proj.groups, -1)
        radii = self.log_radius.exp().to(target.dtype).view(proj.num_kv_heads, proj.groups, 1)
        spread = distances.view(num_target, 1, 1, -1) / radii
        window = -(spread**2) / 2
        # Compact support, as in `GeodesicAttention`: a read is decided by its radius, never by how many
        # samples happened to be within reach. The nearest source is always inside the cutoff, so no row
        # can be fully masked.
        window = window.masked_fill(
            (spread > self.cutoff_sigmas) & (distances > 0).view(num_target, 1, 1, -1), NEG_SCORE
        )
        quadrature = torch.log(link.weights.to(target.dtype).clamp_min(1e-30)).view(num_target, 1, 1, -1)

        attention = (logits + (bias + window + quadrature).unsqueeze(0)).softmax(-1)
        attended = torch.einsum("bphgk,bphkd->bphgd", attention, gathered_v)
        return proj.merge(attended)


class GridLink:
    """
    Precomputed correspondence from one grid to another: for each target point, its nearest sources.

    Like :class:`FieldGrid` this is data, built once for a pair of grids. Changing input resolution means
    building a new link, not touching the model.
    """

    def __init__(self, target: FieldGrid, source: FieldGrid, num_neighbours: int = 16) -> None:
        if target.geometry.embed_dim != source.geometry.embed_dim:
            raise ValueError("Linked grids must share a geometry.")
        from .grid import _knn_between

        self.neighbours, self.distances = _knn_between(target.points, source.points, num_neighbours)
        self.source_points = source.num_points
        self.offsets = target.geometry.local_offsets(target.coords, source.coords[self.neighbours])
        self.weights = source.weights[self.neighbours]
        target_frames = target.geometry.frames(target.coords)
        source_frames = source.geometry.frames(source.coords)[self.neighbours]
        self.alignment = torch.einsum("pij,pkij->pk", target_frames, source_frames) / target.geometry.embed_dim

    def to(self, device: torch.device | str) -> GridLink:
        for name, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self
