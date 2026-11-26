# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fusions of a decomposed `torch.export` graph into MLX's fused kernels (`mx.fast`), run by the MLX exporter before
it saves the graph."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ..utils.import_utils import is_torch_available
from .mlx_ops import rms_norm, rope, slice_write, swiglu


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from torch.fx import Graph, Node


def fuse_for_mlx(graph: Graph) -> None:
    """Rewrite `graph` in place to use MLX's fused kernels, then drop what no longer feeds an output."""
    aten = torch.ops.aten
    # Checks the runner ignores, which would keep the nodes a fusion replaces alive
    for node in [node for node in graph.nodes if node.target == aten._assert_tensor_metadata.default]:
        graph.erase_node(node)
    _drop_full_slices(graph)
    _fuse_cache_writes(graph)
    _fuse_rms_norm(graph)
    _fuse_swiglu(graph)
    _fuse_rope(graph)
    _drop_repeat_kv(graph)
    graph.eliminate_dead_code()


def _drop_full_slices(graph: Graph) -> None:
    """`x[..., 0:end]` over all of an axis (`end` past its size) as `x`: a no-op the other fusions would not see past."""
    aten = torch.ops.aten
    for node in [node for node in graph.nodes if node.target == aten.slice.Tensor]:
        x, dim, start, end, step = (list(node.args) + [0, None, None, 1][len(node.args) - 1 :])[:5]
        if start not in (None, 0) or step != 1:
            continue
        if end is None or (isinstance(end, int) and end >= sys.maxsize):
            node.replace_all_uses_with(x)


def _fuse_cache_writes(graph: Graph) -> None:
    """`x[:, :, arange(0, n) + start] = values` (a static cache write at consecutive positions) as one
    `mx.slice_update`, rather than a copy of `x` and a scatter."""
    aten = torch.ops.aten
    for write in [node for node in graph.nodes if node.target == aten.index_put.default]:
        x, indices, values = write.args[:3]
        if (write.args[3:] or [False])[0] or not isinstance(values, torch.fx.Node):
            continue
        axes = [axis for axis, index in enumerate(indices) if index is not None]
        if len(axes) != 1:
            continue
        positions = indices[axes[0]]
        if positions.target != aten.add.Tensor:
            continue
        steps = next((a for a in positions.args if getattr(a, "target", None) == aten.arange.start_step), None)
        start = next((a for a in positions.args if a is not steps), None)
        if steps is None or steps.args[0] != 0 or (steps.args[2:] or [1])[0] != 1:
            continue
        if not isinstance(start, torch.fx.Node) or start.meta["val"].numel() != 1:
            continue
        with graph.inserting_before(write):
            fused = graph.call_function(slice_write, (x, values, start, axes[0]))
        fused.meta = dict(write.meta)
        write.replace_all_uses_with(fused)


def _source(node: Node) -> Node:
    """`node` without the float32 upcast a norm computes in (`_to_copy` to float32)."""
    if node.target == torch.ops.aten._to_copy.default and node.kwargs.get("dtype") == torch.float32:
        return node.args[0]
    return node


def _fuse_rms_norm(graph: Graph) -> None:
    """`weight * (x * rsqrt(mean(x ** 2, -1, keepdim) + eps))`, upcast to float32 and back or not, as one
    `mx.fast.rms_norm(x, weight, eps)`."""
    aten = torch.ops.aten
    for rsqrt in [node for node in graph.nodes if node.target == aten.rsqrt.default]:
        add = rsqrt.args[0]
        if add.target != aten.add.Tensor or not isinstance(add.args[1], float):
            continue
        mean = add.args[0]
        if mean.target != aten.mean.dim or list(mean.args[1]) not in ([-1], [mean.args[0].meta["val"].dim() - 1]):
            continue
        square = mean.args[0]
        if square.target != aten.pow.Tensor_Scalar or square.args[1] != 2 or not mean.args[2]:
            continue
        x = square.args[0]
        for normed in list(rsqrt.users):
            if normed.target != aten.mul.Tensor or set(normed.args) != {x, rsqrt}:
                continue
            # Cast back to the input's dtype before the weight, as a norm computed in float32 does
            scaled = next(iter(normed.users)) if len(normed.users) == 1 else None
            if scaled is not None and scaled.target == aten._to_copy.default:
                scaled = next(iter(scaled.users)) if len(scaled.users) == 1 else None
            if scaled is None or scaled.target != aten.mul.Tensor:
                continue
            weight = scaled.args[0] if scaled.args[1] in (normed, *normed.users) else scaled.args[1]
            if weight.meta["val"].dim() != 1:
                continue
            with graph.inserting_before(scaled):
                fused = graph.call_function(rms_norm, (_source(x), weight, add.args[1], scaled.meta["val"].dtype))
            fused.meta = dict(scaled.meta)
            scaled.replace_all_uses_with(fused)


def _fuse_swiglu(graph: Graph) -> None:
    """`silu(gate) * up` (a SiLU-gated MLP) as one `swiglu(gate, up)`."""
    aten = torch.ops.aten
    for silu in [node for node in graph.nodes if node.target == aten.silu.default]:
        gate = silu.args[0]
        for gated in list(silu.users):
            if gated.target != aten.mul.Tensor or silu not in gated.args:
                continue
            up = gated.args[1] if gated.args[0] is silu else gated.args[0]
            if not isinstance(up, torch.fx.Node) or up.meta["val"].shape != gate.meta["val"].shape:
                continue
            with graph.inserting_before(gated):
                fused = graph.call_function(swiglu, (gate, up))
            fused.meta = dict(gated.meta)
            gated.replace_all_uses_with(fused)


def _drop_repeat_kv(graph: Graph) -> None:
    """Attention's keys and values without their heads repeated to the queries' (`repeat_kv`: unsqueeze, expand, copy,
    reshape): `mx.fast.scaled_dot_product_attention` takes fewer key/value heads than query heads, and the copy costs
    the whole cache on every decode step."""
    aten = torch.ops.aten
    for sdpa in [node for node in graph.nodes if node.target == aten.scaled_dot_product_attention.default]:
        args = list(sdpa.args)
        for index in (1, 2):
            repeated = args[index]
            if repeated.target not in (aten.view.default, aten._unsafe_view.default, aten.reshape.default):
                continue
            expanded = repeated.args[0]
            if expanded.target == aten.clone.default:
                expanded = expanded.args[0]
            if expanded.target != aten.expand.default:
                continue
            unsqueezed = expanded.args[0]
            if unsqueezed.target != aten.unsqueeze.default or unsqueezed.args[1] != 2:
                continue
            states = unsqueezed.args[0]
            if states.meta["val"].dim() != 4 or states.meta["val"].shape[-2:] != repeated.meta["val"].shape[-2:]:
                continue
            args[index] = states
        sdpa.args = tuple(args)


def _strip(node: Node, *targets) -> Node:
    """`node` without the leading ops in `targets` (an `unsqueeze`, a cast) between it and what it computes."""
    while isinstance(node, torch.fx.Node) and node.target in targets:
        node = node.args[0]
    return node


def _rotary_angle(table: Node, function) -> tuple | None:
    """`(angles, scaling)` for `table = function(angles) * scaling` (a rotary embedding's cos or sin), else `None`."""
    aten = torch.ops.aten
    table = _strip(table, aten.unsqueeze.default, aten._to_copy.default)
    scaling = 1.0
    if table.target == aten.mul.Tensor and isinstance(table.args[1], float):
        table, scaling = table.args[0], table.args[1]
    if table.target != function:
        return None
    return table.args[0], scaling


def _fuse_rope(graph: Graph) -> None:
    """`x * cos + rotate_half(x) * sin`, with `cos` and `sin` of `positions * inv_freq` (a rotary embedding over the
    whole head), as one `mx.fast.rope`."""
    aten = torch.ops.aten
    for add in [node for node in graph.nodes if node.target == aten.add.Tensor]:
        products = [arg for arg in add.args if isinstance(arg, torch.fx.Node) and arg.target == aten.mul.Tensor]
        if len(products) != 2:
            continue
        rotated = next(
            (p for p in products if any(getattr(a, "target", None) == aten.cat.default for a in p.args)), None
        )
        if rotated is None:
            continue
        direct = products[1] if rotated is products[0] else products[0]
        halves = next(a for a in rotated.args if getattr(a, "target", None) == aten.cat.default)
        sin_table = rotated.args[1] if rotated.args[0] is halves else rotated.args[0]
        # rotate_half(x): cat(-x[..., h:], x[..., :h])
        second, first = halves.args[0]
        if second.target != aten.neg.default:
            continue
        second = second.args[0]
        if (
            second.target != aten.slice.Tensor
            or first.target != aten.slice.Tensor
            or second.args[0] is not first.args[0]
        ):
            continue
        x = first.args[0]
        if x not in direct.args:
            continue
        cos_table = direct.args[1] if direct.args[0] is x else direct.args[0]
        cos_angle, sin_angle = _rotary_angle(cos_table, aten.cos.default), _rotary_angle(sin_table, aten.sin.default)
        if cos_angle is None or sin_angle is None or cos_angle != sin_angle:
            continue
        angles, scaling = cos_angle
        # angles = cat([f, f]), f = float(positions[..., None]) * inv_freq
        if angles.target != aten.cat.default or len(set(angles.args[0])) != 1:
            continue
        frequencies = angles.args[0][0]
        if frequencies.target != aten.mul.Tensor:
            continue
        inv_freq = next(
            (a for a in frequencies.args if isinstance(a, torch.fx.Node) and a.meta["val"].dim() == 1), None
        )
        positions = next((a for a in frequencies.args if a is not inv_freq), None)
        positions = _strip(positions, aten._to_copy.default)
        if inv_freq is None or positions.target != aten.unsqueeze.default:
            continue
        positions = positions.args[0]
        if positions.meta["val"].dim() != 2 or 2 * inv_freq.meta["val"].shape[0] != x.meta["val"].shape[-1]:
            continue
        with graph.inserting_before(add):
            fused = graph.call_function(rope, (x, positions, inv_freq, scaling))
        fused.meta = dict(add.meta)
        add.replace_all_uses_with(fused)
