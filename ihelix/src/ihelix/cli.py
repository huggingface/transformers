# Copyright 2026 Nathan. Licensed under the Apache License, Version 2.0.
"""Command line entry point: ``ihelix info | domains | converge | demo``."""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from . import __author__, __version__
from .geometry import Axis, Geometry
from .grid import FieldGrid, fibonacci_sphere, latlon_grid
from .model import IHelixConfig, IHelixField, IHelixFieldModel

DT = torch.float64


def _banner() -> str:
    return f"iHELIX {__version__} — fields on any geometry — made by {__author__}"


def cmd_info(args: argparse.Namespace) -> None:
    print(_banner())
    print("\nDeclare what your axes are; everything downstream works in a metric space where the")
    print("geometry is already correct. One class covers all of these:\n")
    rows = [
        ("1-D signal", Geometry([Axis("linear")])),
        ("image", Geometry([Axis("linear")] * 2)),
        ("volume / point cloud", Geometry([Axis("linear")] * 3)),
        ("periodic box (LES, CFD)", Geometry([Axis("periodic", period=1.0)] * 2 + [Axis("linear")])),
        ("globe", Geometry.globe()),
        ("atmosphere", Geometry.atmosphere()),
    ]
    for name, geometry in rows:
        axes = ", ".join(a.kind for a in geometry.axes)
        print(f"  {name:26} axes=({axes:36}) coords={geometry.coord_dim} -> metric space={geometry.embed_dim}")
    print("\nThree mixers per block:")
    print("  local   attention over nearest neighbours in metric space, per-head radii in real units")
    print("  index   content-addressed retrieval of distant regions through a hierarchy")
    print("  time    a gated delta rule along the one axis that has an order")
    print("\nTry `ihelix domains`, then `ihelix converge`.")


def cmd_domains(args: argparse.Namespace) -> None:
    print(_banner())
    print("\nThe same class and the same code path, over every shape of problem:\n")

    def mesh(*sizes):
        axes = [torch.linspace(0, 1, n, dtype=DT) for n in sizes]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(sizes))

    cases = [
        ("image 20x20", Geometry([Axis("linear")] * 2), mesh(20, 20), 1, (0.06, 0.2)),
        ("video 14x14, 6 frames", Geometry([Axis("linear")] * 2), mesh(14, 14), 6, (0.08, 0.25)),
        ("volume 8x8x8", Geometry([Axis("linear")] * 3), mesh(8, 8, 8), 1, (0.15, 0.45)),
        (
            "periodic box, 4 steps",
            Geometry([Axis("periodic", period=1.0)] * 2 + [Axis("linear")]),
            torch.rand(400, 3, dtype=DT),
            4,
            (0.15, 0.45),
        ),
        ("particle cloud", Geometry([Axis("linear")] * 3), torch.rand(300, 3, dtype=DT), 1, (0.2, 0.5)),
        (
            "1-D signal, 5 steps",
            Geometry([Axis("linear")]),
            torch.linspace(0, 1, 200, dtype=DT).unsqueeze(-1),
            5,
            (0.03, 0.1),
        ),
        ("globe", None, None, 4, (500.0, 1400.0)),
        ("atmosphere (3-D shell)", None, None, 3, (500.0, 1400.0)),
    ]
    print(f"  {'domain':24} {'coords':>7} {'metric':>7} {'samples':>8} {'frames':>7} {'cover':>6}  output")
    for name, geometry, points, steps, (rmin, rmax) in cases:
        if name.startswith("globe"):
            grid = latlon_grid(14, 28, num_neighbours=8, cluster_size=32)
            grid = latlon_grid(14, 28, num_neighbours=grid.suggest_neighbours(rmax), cluster_size=32)
        elif name.startswith("atmosphere"):
            levels = torch.tensor([1000.0, 700.0, 400.0], dtype=DT)
            grid = latlon_grid(10, 20, levels=levels, num_neighbours=8, cluster_size=32)
            grid = latlon_grid(10, 20, levels=levels, num_neighbours=grid.suggest_neighbours(rmax), cluster_size=32)
        else:
            probe = FieldGrid.from_points(points, geometry, num_neighbours=8, cluster_size=32)
            neighbours = min(probe.suggest_neighbours(rmax), points.shape[0])
            grid = FieldGrid.from_points(points, geometry, num_neighbours=neighbours, cluster_size=32)
        config = IHelixConfig(
            in_channels=3,
            out_channels=2,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            min_radius=rmin,
            max_radius=rmax,
            landmark_dim=8,
            index_branching=2,
            index_beam_width=2,
            index_topk=2,
            num_recurrent_heads=2,
            recurrent_head_dim=8,
            recurrent_value_head_dim=8,
            recurrent_chunk_size=4,
            use_temporal=steps > 1,
        )
        torch.manual_seed(0)
        model = IHelixField(config, grid.geometry).to(DT).eval()
        shape = (1, steps, grid.num_points, 3) if steps > 1 else (1, grid.num_points, 3)
        with torch.no_grad():
            out = model(torch.randn(*shape, dtype=DT), grid)
        print(
            f"  {name:24} {grid.geometry.coord_dim:>7} {grid.geometry.embed_dim:>7} {grid.num_points:>8} "
            f"{steps:>7} {model.blocks[0].local.coverage(grid):>6.2f}  {tuple(out.shape)}",
            flush=True,
        )


def cmd_converge(args: argparse.Namespace) -> None:
    """Read one continuous field at rising resolution; the answer should settle, not drift."""
    print(_banner())
    encode_radius, local_radius = 900.0, 1400.0

    def field(coords):
        lat, lon = coords[:, 0], coords[:, 1]
        return torch.stack(
            [(3 * lon).sin() * (2 * lat).cos(), lat.sin(), lon.cos() * lat.cos(), (2 * lon + 1.0).sin()], -1
        )

    probe = fibonacci_sphere(180, num_neighbours=8, cluster_size=30)
    latent = fibonacci_sphere(180, num_neighbours=probe.suggest_neighbours(local_radius), cluster_size=30)
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
                max_radius=local_radius,
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

    print("\nOne set of weights reading the same continuous field at rising resolution,")
    print("always writing to the same 16x32 points.\n")
    print(f"  {'read at':>10} {'samples':>8} {'neighbours':>11}  difference vs the finest read")
    outputs = {}
    for num_lat, num_lon in args.resolutions:
        probe = latlon_grid(num_lat, num_lon, num_neighbours=8, cluster_size=32)
        needed = probe.suggest_neighbours(encode_radius)
        source = latlon_grid(num_lat, num_lon, num_neighbours=needed, cluster_size=32)
        model.config.encode_neighbours = model.config.decode_neighbours = needed
        model._links.clear()
        with torch.no_grad():
            outputs[(num_lat, num_lon)] = (model(field(source.coords).unsqueeze(0), source, readout), needed)
    finest = outputs[args.resolutions[-1]][0]
    for key, (out, needed) in outputs.items():
        error = ((out - finest).norm() / finest.norm()).item()
        label = "(reference)" if key == args.resolutions[-1] else f"{error:.3%}"
        print(f"  {key[0]:4}x{key[1]:<5} {key[0] * key[1]:>8} {needed:>11}  {label}")
    print("\nThe neighbour count rises with density to hold a fixed physical reach. That is a")
    print("preprocessing choice; the weights are identical in every row.")


def cmd_demo(args: argparse.Namespace) -> None:
    print(_banner())
    grid = latlon_grid(16, 32, num_neighbours=8, cluster_size=32)
    grid = latlon_grid(16, 32, num_neighbours=grid.suggest_neighbours(1400.0), cluster_size=32)
    config = IHelixConfig(
        in_channels=5,
        hidden_size=64,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        min_radius=500.0,
        max_radius=1400.0,
        landmark_dim=16,
        index_branching=2,
        index_beam_width=2,
        index_topk=2,
        num_recurrent_heads=2,
        recurrent_head_dim=16,
        recurrent_value_head_dim=16,
        recurrent_chunk_size=4,
    )
    torch.manual_seed(0)
    model = IHelixField(config, grid.geometry).to(DT).eval()
    print(f"\n{grid}")
    print(f"parameters : {model.num_parameters():,}   blocks: {config.layer_types}")

    values = torch.randn(1, 6, grid.num_points, 5, dtype=DT)
    with torch.no_grad():
        out = model(values, grid)
    print(f"6 frames in -> {tuple(out.shape)} out\n")

    # The property that matters: sample order carries no information.
    permutation = torch.randperm(grid.num_points)
    shuffled = FieldGrid.from_points(
        grid.coords[permutation],
        grid.geometry,
        num_neighbours=grid.num_neighbours,
        cluster_size=grid.cluster_size,
        weights=grid.weights[permutation] * grid.weights.sum(),
    )
    with torch.no_grad():
        delta = (model(values[:, :, permutation], shuffled) - out[:, :, permutation]).abs().max().item()
    print(f"shuffle every sample's position in the array -> max difference {delta:.2e}")
    print("(a sequence model reading a flattened grid cannot say that)")

    shifted = grid.coords.clone()
    shifted[:, 1] = torch.remainder(shifted[:, 1] + 0.35, 2 * math.pi)
    spun = FieldGrid.from_points(
        shifted,
        grid.geometry,
        num_neighbours=grid.num_neighbours,
        cluster_size=grid.cluster_size,
        weights=grid.weights * grid.weights.sum(),
    )
    local_only = (
        IHelixField(IHelixConfig(**{**config.to_dict(), "index_layer_stride": 10**6}), grid.geometry).to(DT).eval()
    )
    with torch.no_grad():
        spun_delta = (local_only(values, spun) - local_only(values, grid)).abs().max().item()
    print(f"shift the whole globe 20 deg in longitude -> max difference {spun_delta:.2e} (local strand)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ihelix", description=_banner())
    parser.add_argument("--version", action="version", version=f"ihelix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="what it is and which geometries it covers").set_defaults(func=cmd_info)
    sub.add_parser("domains", help="run every shape of problem through one class").set_defaults(func=cmd_domains)
    sub.add_parser("demo", help="build a model, run it, check the invariances").set_defaults(func=cmd_demo)

    converge = sub.add_parser("converge", help="read one field at rising resolution")
    converge.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[16, 32, 24, 48, 32, 64, 48, 96],
        help="flat list of lat lon pairs",
    )
    converge.set_defaults(func=cmd_converge)

    args = parser.parse_args(argv)
    if getattr(args, "resolutions", None) is not None:
        flat = args.resolutions
        if len(flat) % 2:
            parser.error("--resolutions takes pairs of numbers")
        args.resolutions = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    try:
        args.func(args)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
