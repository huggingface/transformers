# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""
The claims, checked on untrained models.

Nothing here needs training. These are properties of the architecture -- that it is blind to the order
samples arrive in, that it does not care where the date line is, that refining the grid converges instead
of drifting -- and each of them is either true of the arithmetic or it is not.
"""

import math

import pytest
import torch

from ihelix import (
    Axis,
    FieldGrid,
    Geometry,
    IHelixConfig,
    IHelixField,
    IHelixFieldModel,
    fibonacci_sphere,
    latlon_grid,
)
from ihelix.geometry import EARTH_RADIUS_KM

DT = torch.float64
LOCAL_R = 1400.0


def sphere_grid(num_lat=16, num_lon=32, **kw):
    probe = latlon_grid(num_lat, num_lon, num_neighbours=8, cluster_size=32)
    return latlon_grid(num_lat, num_lon, num_neighbours=probe.suggest_neighbours(LOCAL_R), cluster_size=32, **kw)


def build(geometry, **over):
    settings = {
        "in_channels": 4,
        "hidden_size": 48,
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 12,
        "min_radius": 500.0,
        "max_radius": LOCAL_R,
        "landmark_dim": 16,
        "index_branching": 2,
        "index_beam_width": 2,
        "index_topk": 2,
        "use_temporal": False,
        **over,
    }
    torch.manual_seed(0)
    return IHelixField(IHelixConfig(**settings), geometry).to(DT).eval()


# --------------------------------------------------------------------------------------- geometry ---


def test_frames_are_orthonormal_including_at_the_poles():
    geometry = Geometry.globe()
    degrees = torch.tensor([-90.0, -89.999, -45.0, 0.0, 45.0, 89.999, 90.0], dtype=DT)
    longitudes = torch.tensor([0.0, 90.0, 180.0, 359.9], dtype=DT)
    lat, lon = torch.meshgrid(degrees, longitudes, indexing="ij")
    coords = torch.stack([lat.reshape(-1), lon.reshape(-1)], -1) * math.pi / 180
    frames = geometry.frames(coords)
    identity = torch.eye(3, dtype=DT).expand_as(frames)
    torch.testing.assert_close(frames @ frames.transpose(-1, -2), identity, rtol=0, atol=1e-12)


def test_the_date_line_is_not_a_cliff():
    """Flattening makes longitude 359.9 and 0.1 far apart. On the manifold they are neighbours."""
    geometry = Geometry.globe()

    def at(lon):
        return geometry.embed(torch.tensor([[0.0, lon * math.pi / 180]], dtype=DT))

    across_seam = (at(359.9) - at(0.1)).norm()
    half_world = (at(180.0) - at(0.1)).norm()
    assert across_seam < 30.0, across_seam
    assert half_world > 12000.0
    assert across_seam < half_world / 100


def test_meridians_converge_at_the_poles():
    """A degree of longitude is not a fixed distance, and the embedding knows it."""
    geometry = Geometry.globe()

    def step(lat):
        here = geometry.embed(torch.tensor([[lat * math.pi / 180, 0.0]], dtype=DT))
        along = geometry.embed(torch.tensor([[lat * math.pi / 180, 0.25 * math.pi / 180]], dtype=DT))
        return (here - along).norm().item()

    equator, midlat, polar = step(0.0), step(60.0), step(89.75)
    assert equator > midlat > polar
    assert midlat == pytest.approx(equator / 2, rel=1e-3)  # cos(60) = 1/2
    assert polar < equator / 100


def test_pressure_is_logarithmic():
    """Equal pressure ratios are equal vertical distances, which is what the atmosphere actually does."""
    geometry = Geometry.atmosphere()

    def gap(a, b):
        upper = geometry.embed(torch.tensor([[0.0, 0.0, a]], dtype=DT))
        lower = geometry.embed(torch.tensor([[0.0, 0.0, b]], dtype=DT))
        return (upper - lower).norm().item()

    assert gap(1000, 500) == pytest.approx(gap(500, 250), rel=1e-9)
    assert gap(500, 250) == pytest.approx(gap(250, 125), rel=1e-9)


def test_periodic_axes_wrap():
    geometry = Geometry([Axis("periodic", period=1.0)])

    def at(u):
        return geometry.embed(torch.tensor([[u]], dtype=DT))

    assert (at(0.99) - at(0.01)).norm() < (at(0.5) - at(0.01)).norm()


@pytest.mark.parametrize(
    "geometry,coord_dim,embed_dim",
    [
        (Geometry([Axis("linear")]), 1, 1),
        (Geometry([Axis("linear")] * 2), 2, 2),
        (Geometry([Axis("linear")] * 3), 3, 3),
        (Geometry([Axis("periodic", period=2.0)] * 2 + [Axis("linear")]), 3, 5),
        (Geometry.globe(), 2, 3),
        (Geometry.atmosphere(), 3, 4),
    ],
)
def test_every_geometry_reports_consistent_dimensions(geometry, coord_dim, embed_dim):
    assert (geometry.coord_dim, geometry.embed_dim) == (coord_dim, embed_dim)
    coords = torch.rand(7, coord_dim, dtype=DT) + 1.0
    assert geometry.embed(coords).shape == (7, embed_dim)
    frames = geometry.frames(coords)
    identity = torch.eye(embed_dim, dtype=DT).expand_as(frames)
    torch.testing.assert_close(frames @ frames.transpose(-1, -2), identity, rtol=0, atol=1e-12)


# ------------------------------------------------------------------------------------------- grid ---


def test_neighbours_cross_the_seam():
    """A sample beside the date line must draw neighbours from both sides of it."""
    num_lon = 32
    grid = latlon_grid(16, num_lon, num_neighbours=9, cluster_size=32)
    columns = (grid.neighbours[8 * num_lon] % num_lon).tolist()
    assert any(c > num_lon - 3 for c in columns), columns
    assert any(c < 3 for c in columns), columns


def test_quadrature_weights_follow_cell_area():
    """
    A latitude-longitude cell shrinks like ``cos(latitude)``, and the weights have to say so -- that is
    what stops the crowded polar samples from outvoting the sparse tropical ones.
    """
    grid = latlon_grid(16, 32, num_neighbours=9, cluster_size=32)
    # `.item()` first: comparing a torch tensor against `approx` goes through numpy, which is optional.
    assert grid.weights.sum().item() == pytest.approx(1.0)
    expected = grid.coords[:, 0].cos()
    torch.testing.assert_close(grid.weights, expected / expected.sum(), rtol=1e-12, atol=0)


def test_clusters_are_spatially_compact():
    grid = sphere_grid()
    members = grid.points[grid.cluster_members]
    radius = (members - members.mean(-2, keepdim=True)).norm(dim=-1).mean()
    assert radius < EARTH_RADIUS_KM / 2


def test_suggest_neighbours_actually_covers():
    grid = latlon_grid(16, 32, num_neighbours=8, cluster_size=32)
    needed = grid.suggest_neighbours(LOCAL_R)
    covered = latlon_grid(16, 32, num_neighbours=needed, cluster_size=32)
    assert (covered.neighbour_distances[:, -1] >= 3 * LOCAL_R).all()


# ------------------------------------------------------------------------------------- invariance ---


def test_sample_order_does_not_matter_at_all():
    """
    The formal statement of "no flattening artefact": the model has no notion of which sample came first.

    This is exact, not approximate. A sequence model reading a flattened grid cannot say the same.
    """
    grid = sphere_grid()
    model = build(grid.geometry)
    values = torch.randn(1, grid.num_points, 4, dtype=DT)
    reference = model(values, grid)

    permutation = torch.randperm(grid.num_points)
    shuffled = FieldGrid.from_points(
        grid.coords[permutation],
        grid.geometry,
        num_neighbours=grid.num_neighbours,
        cluster_size=grid.cluster_size,
        weights=grid.weights[permutation] * grid.weights.sum(),
    )
    torch.testing.assert_close(model(values[:, permutation], shuffled), reference[:, permutation], rtol=0, atol=0)


def test_local_strand_is_equivariant_to_spinning_the_globe():
    """
    Shifting every sample in longitude changes no physics, and changes no output.

    The local strand reads displacements in each sample's own east/north/up frame, which turns with the
    sample, so the arithmetic it performs is identical. Only the *index* strand moves, because carving the
    domain into regions is a discretization and no finite partition of a sphere commutes with rotation --
    the same is true of a fixed icosahedral mesh.
    """
    grid = sphere_grid()
    model = build(grid.geometry, index_layer_stride=10**6)
    values = torch.randn(1, grid.num_points, 4, dtype=DT)

    shifted_coords = grid.coords.clone()
    shifted_coords[:, 1] = torch.remainder(shifted_coords[:, 1] + 0.35, 2 * math.pi)
    shifted = FieldGrid.from_points(
        shifted_coords,
        grid.geometry,
        num_neighbours=grid.num_neighbours,
        cluster_size=grid.cluster_size,
        weights=grid.weights * grid.weights.sum(),
    )
    torch.testing.assert_close(model(values, shifted), model(values, grid), rtol=0, atol=1e-6)


def test_refining_the_grid_converges():
    """
    The resolution claim: read the same continuous field more finely and the answer settles down.

    It converges because attention is weighted by how much of the domain each sample stands for, which
    makes it a quadrature of a continuous operator rather than a sum over however many samples exist.
    """
    encode_radius = 900.0

    def field(coords):
        lat, lon = coords[:, 0], coords[:, 1]
        return torch.stack(
            [(3 * lon).sin() * (2 * lat).cos(), lat.sin(), lon.cos() * lat.cos(), (2 * lon + 1.0).sin()], -1
        )

    probe = fibonacci_sphere(180, num_neighbours=8, cluster_size=30)
    latent = fibonacci_sphere(180, num_neighbours=probe.suggest_neighbours(LOCAL_R), cluster_size=30)
    torch.manual_seed(0)
    model = (
        IHelixFieldModel(
            IHelixConfig(
                in_channels=4,
                hidden_size=48,
                num_layers=3,
                num_heads=4,
                num_kv_heads=2,
                head_dim=12,
                min_radius=700.0,
                max_radius=LOCAL_R,
                landmark_dim=16,
                index_branching=2,
                index_beam_width=2,
                index_topk=2,
                use_temporal=False,
                encode_radius=encode_radius,
            ),
            latent,
        )
        .to(DT)
        .eval()
    )

    readout = latlon_grid(16, 32, num_neighbours=8, cluster_size=32)
    readout = latlon_grid(16, 32, num_neighbours=readout.suggest_neighbours(encode_radius), cluster_size=32)

    outputs = {}
    for num_lat, num_lon in [(16, 32), (24, 48), (32, 64), (48, 96)]:
        probe = latlon_grid(num_lat, num_lon, num_neighbours=8, cluster_size=32)
        # The neighbour count has to rise with density to hold a fixed physical reach. That is a
        # preprocessing choice; the weights are untouched.
        needed = probe.suggest_neighbours(encode_radius)
        source = latlon_grid(num_lat, num_lon, num_neighbours=needed, cluster_size=32)
        model.config.encode_neighbours = model.config.decode_neighbours = needed
        model._links.clear()
        outputs[num_lat] = model(field(source.coords).unsqueeze(0), source, readout)

    finest = outputs.pop(48)
    errors = [((out - finest).norm() / finest.norm()).item() for _, out in sorted(outputs.items())]
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[-1] < 0.01, errors


# ------------------------------------------------------------------------------------------ model ---


@pytest.mark.parametrize(
    "name,geometry,num_points,steps,radii",
    [
        ("image", Geometry([Axis("linear")] * 2), 400, 1, (0.05, 0.15)),
        ("video", Geometry([Axis("linear")] * 2), 256, 6, (0.06, 0.18)),
        ("volume", Geometry([Axis("linear")] * 3), 512, 1, (0.15, 0.4)),
        ("periodic box", Geometry([Axis("periodic", period=1.0)] * 2 + [Axis("linear")]), 512, 4, (0.15, 0.4)),
        ("point cloud", Geometry([Axis("linear")] * 3), 300, 1, (0.2, 0.5)),
        ("1-D signal", Geometry([Axis("linear")]), 256, 5, (0.02, 0.08)),
    ],
)
def test_one_model_class_covers_every_domain(name, geometry, num_points, steps, radii):
    """Images, video, volumes, periodic boxes, particles, signals -- same class, same code path."""
    torch.manual_seed(0)
    coords = torch.rand(num_points, geometry.coord_dim, dtype=DT)
    grid = FieldGrid.from_points(coords, geometry, num_neighbours=48, cluster_size=32)
    model = build(
        geometry,
        in_channels=3,
        out_channels=2,
        hidden_size=32,
        num_layers=2,
        head_dim=8,
        landmark_dim=8,
        min_radius=radii[0],
        max_radius=radii[1],
        use_temporal=steps > 1,
        num_recurrent_heads=2,
        recurrent_head_dim=8,
        recurrent_value_head_dim=8,
        recurrent_chunk_size=4,
    )
    values = torch.randn(1, steps, num_points, 3, dtype=DT) if steps > 1 else torch.randn(1, num_points, 3, dtype=DT)
    with torch.no_grad():
        out = model(values, grid)
    assert out.shape == ((1, steps, num_points, 2) if steps > 1 else (1, num_points, 2))
    assert torch.isfinite(out).all()


def test_time_is_causal_and_space_is_not():
    """
    The sequential axis is time, and it runs one way. Space has no order and gets no recurrence.
    """
    grid = FieldGrid.from_points(
        torch.rand(128, 2, dtype=DT), Geometry([Axis("linear")] * 2), num_neighbours=32, cluster_size=32
    )
    model = build(
        grid.geometry,
        hidden_size=32,
        num_layers=2,
        head_dim=8,
        landmark_dim=8,
        min_radius=0.08,
        max_radius=0.25,
        use_temporal=True,
        num_recurrent_heads=2,
        recurrent_head_dim=8,
        recurrent_value_head_dim=8,
        recurrent_chunk_size=4,
    )
    values = torch.randn(1, 8, 128, 4, dtype=DT)
    reference = model(values, grid)
    edited = values.clone()
    edited[:, 5:] += 1.0
    torch.testing.assert_close(model(edited, grid)[:, :5], reference[:, :5], rtol=0, atol=0)


def test_temporal_state_does_not_grow_with_the_rollout():
    """A hundred frames cost the same state as one; that is what makes a long integration affordable."""
    from ihelix.model import TemporalStrand

    config = IHelixConfig(
        hidden_size=32,
        num_recurrent_heads=2,
        recurrent_head_dim=8,
        recurrent_value_head_dim=8,
        recurrent_chunk_size=4,
    )
    strand = TemporalStrand(config).to(DT)
    shapes = set()
    for steps in (4, 32, 128):
        _, state = strand(torch.randn(1, steps, 16, 32, dtype=DT), return_state=True)
        shapes.add(tuple(state.shape))
    assert len(shapes) == 1, shapes


def test_coverage_reports_honestly():
    """The one precondition has a diagnostic, and it tells the truth about short neighbour lists."""
    starved = latlon_grid(16, 32, num_neighbours=6, cluster_size=32)
    model = build(starved.geometry)
    assert model.blocks[0].local.coverage(starved) < 0.5
    ample = sphere_grid()
    assert model.blocks[0].local.coverage(ample) == 1.0


def test_gradients_reach_every_parameter():
    grid = sphere_grid()
    model = build(grid.geometry).to(torch.float32).train()
    values = torch.randn(1, grid.num_points, 4)
    model(values, grid).square().mean().backward()
    starved = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not starved, starved


def test_encode_process_decode_changes_grid_without_changing_weights():
    latent = fibonacci_sphere(120, num_neighbours=48, cluster_size=30)
    torch.manual_seed(0)
    model = (
        IHelixFieldModel(
            IHelixConfig(
                in_channels=4,
                hidden_size=32,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                min_radius=700.0,
                max_radius=LOCAL_R,
                landmark_dim=8,
                index_branching=2,
                index_beam_width=2,
                index_topk=2,
                use_temporal=False,
                encode_neighbours=32,
                decode_neighbours=32,
                encode_radius=900.0,
            ),
            latent,
        )
        .to(DT)
        .eval()
    )
    coarse, fine = sphere_grid(12, 24), sphere_grid(20, 40)
    values = torch.randn(1, coarse.num_points, 4, dtype=DT)
    with torch.no_grad():
        assert model(values, coarse, coarse).shape == (1, coarse.num_points, 4)
        assert model(values, coarse, fine).shape == (1, fine.num_points, 4)
        scattered = FieldGrid.from_points(
            torch.stack([torch.rand(50, dtype=DT) * 2 - 1, torch.rand(50, dtype=DT) * 6], -1),
            coarse.geometry,
            num_neighbours=32,
            cluster_size=25,
        )
        assert model(values, coarse, scattered).shape == (1, 50, 4)


@pytest.mark.parametrize(
    "bad",
    [
        {"min_radius": 0},
        {"max_radius": 1.0, "min_radius": 2.0},
        {"num_heads": 3, "num_kv_heads": 2},
        {"index_branching": 1},
        {"index_topk": 0},
    ],
)
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        IHelixConfig(**bad)


# --------------------------------------------------------------------------------------------------
# Device placement
# --------------------------------------------------------------------------------------------------

def test_grids_follow_the_weights_to_another_device():
    """
    ``model.to(device)`` walks parameters and buffers. A FieldGrid is neither.

    Left unhandled, ``model.to("cuda")`` moved every weight and left every grid on the CPU, and the
    failure surfaced deep inside cross-attention as "mat1 is on cpu, different from other tensors on
    cuda:0" -- pointing at a linear layer and saying nothing about grids. The meta device exercises the
    identical code path without needing a GPU.
    """
    grid = fibonacci_sphere(64, num_neighbours=8, cluster_size=16)
    config = IHelixConfig(in_channels=3, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=1,
                          head_dim=16, min_radius=200.0, max_radius=800.0, use_temporal=False,
                          latent_points=64)
    model = IHelixFieldModel(config, grid)
    assert model.latent_grid.points.device.type == "cpu"

    model = model.to("meta")
    for name in ("points", "coords", "weights", "neighbours", "neighbour_offsets", "neighbour_distances"):
        moved = getattr(model.latent_grid, name)
        assert moved.device.type == "meta", f"latent_grid.{name} stayed behind"


def test_a_caller_supplied_grid_is_pulled_to_the_model():
    """The source grid is built by the user on the CPU and handed in; it must be moved to meet them."""
    config = IHelixConfig(in_channels=3, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=1,
                          head_dim=16, min_radius=200.0, max_radius=800.0, use_temporal=False,
                          latent_points=64)
    model = IHelixFieldModel(config, fibonacci_sphere(64, num_neighbours=8, cluster_size=16)).to("meta")

    source = fibonacci_sphere(32, num_neighbours=8, cluster_size=16)
    assert source.points.device.type == "cpu"
    link = model.link(model.latent_grid, source, 8)
    assert link.offsets.device.type == "meta"
    assert source.points.device.type == "meta", "the caller's grid was not moved"


def test_link_cache_does_not_survive_a_device_change():
    """A correspondence built on one device serves tensors from that device, which is now the wrong one."""
    config = IHelixConfig(in_channels=3, hidden_size=32, num_layers=2, num_heads=2, num_kv_heads=1,
                          head_dim=16, min_radius=200.0, max_radius=800.0, use_temporal=False,
                          latent_points=64)
    model = IHelixFieldModel(config, fibonacci_sphere(64, num_neighbours=8, cluster_size=16))
    model.link(model.latent_grid, fibonacci_sphere(32, num_neighbours=8, cluster_size=16), 8)
    assert len(model._links) == 1

    model = model.to("meta")
    assert len(model._links) == 0
