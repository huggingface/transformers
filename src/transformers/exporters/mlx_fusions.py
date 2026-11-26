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

from typing import TYPE_CHECKING

from ..utils.import_utils import is_torch_available
from .mlx_ops import rms_norm, swiglu


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
    _fuse_rms_norm(graph)
    _fuse_swiglu(graph)
    graph.eliminate_dead_code()


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
