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
"""Fuses an exported ONNX graph into ONNX Runtime's own ops for one of its providers (`OnnxConfig(ort_fusions="cpu" |
"cuda")`): `fuse_for_onnxruntime`.

onnxscript's ORT fusions (`optimize_for_ort`) do most of it: RMS / layer norms and their skip variants, attention,
GELU. Around them, in order (`_STEPS`, which also says which provider each step is for):

1. **Prepare**: write the graph in the forms those fusions match (Gemma's `Pow(x, -0.5)` norms, grouped-query keys).
2. **Rotary embeddings**: fuse the ones onnxscript leaves (cos / sin folded to constants, or computed with a
   broadcast, or the `[batch, length, heads, head_dim]` layout), and attentions sharing a key (Gemma 4's).
3. **onnxscript's ORT fusions.**
4. **Repair**: undo the fusions of theirs ONNX Runtime rejects.
5. **More fusions**: norms without a scale, norms a float16 model computes in float32; `GroupQueryAttention` over a
   key-value cache (on CUDA, decode only, with the per-head q/k norms); on CUDA, `MoE`; on the CPU, the attentions
   left back to plain ops (its `MultiHeadAttention` is slower); the layout transposes around each attention
   cancelled; the projections reading one input as one `MatMul`, which `GroupQueryAttention` reads packed.
6. **CUDA provider**: rewrite what CUDA graphs can't replay.
7. **Clean up**: what every layer computes alike, computed once.

Every rewrite matches exactly what it replaces: the fused graph's outputs are the unfused one's.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import onnx_ir as ir
from onnx_ir.passes import common as common_passes
from onnxscript.rewriter import _ir_utils, pattern
from onnxscript.rewriter.ort_fusions import optimize_for_ort
from onnxscript.rewriter.ort_fusions.rotary_embedding import RotaryEmbeddingFusion, fuse_rotary_embedding
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa
from onnxscript.rewriter.ort_fusions.skip_normalization import fuse_skip_rms_normalization
from packaging import version

from ..utils import logging
from ..utils.import_utils import _is_package_available


logger = logging.get_logger(__name__)


# ── Graph helpers ─────────────────────────────────────────────────────────────


def _producer(value: ir.Value | None, op_type: str) -> ir.Node | None:
    """The node producing `value`, if it's an `op_type`."""
    node = value.producer() if value is not None else None
    return node if node is not None and node.op_type == op_type else None


def _perm(node: ir.Node) -> list[int]:
    return list(node.attributes.get_ints("perm", []))


def _static_dim(value: ir.Value, axis: int) -> int | None:
    """`value`'s size along `axis` if it's known: a dynamic export names some fixed sizes as symbols (`"2"`)."""
    if value.shape is None or len(value.shape) <= axis:
        return None
    dim = value.shape[axis]
    if isinstance(dim, int):
        return dim
    name = getattr(dim, "value", None)
    return int(name) if isinstance(name, str) and name.isdigit() else None


def _constant(value: ir.Value | None) -> np.ndarray | None:
    """`value` as an array if it's a constant."""
    if value is None or value.const_value is None:
        return None
    return value.const_value.numpy()


def _initializer(graph: ir.Graph, name: str, array: np.ndarray) -> ir.Value:
    value = ir.Value(name=name, const_value=ir.tensor(np.ascontiguousarray(array)))
    value.dtype, value.shape = ir.DataType.from_numpy(array.dtype), ir.Shape(array.shape)
    graph.register_initializer(value)
    return value


def _replace(old: ir.Value, new: ir.Value) -> None:
    """Every use of `old`, a graph output included (keeping its name), reads `new`."""
    name, is_output = old.name, old.is_graph_output()
    old.replace_all_uses_with(new, replace_graph_outputs=True)
    if is_output:
        old.name, new.name = f"{name}_unfused", name


def _add_node(graph: ir.Graph, before: ir.Node, op_type: str, inputs, shape, dtype, domain: str = "", **attributes):
    """A one-output `op_type` node inserted before `before`: its output, with `shape` and `dtype`."""
    node = ir.node(op_type, inputs, attributes=attributes, domain=domain)
    node.outputs[0].shape, node.outputs[0].dtype = (ir.Shape(shape) if shape is not None else None), dtype
    graph.insert_before(before, node)
    return node.outputs[0]


def _swapped_heads(value: ir.Value | None) -> ir.Value | None:
    """`x` if `value` is `Transpose(x, perm=[0, 2, 1, 3])`: `[batch, length, heads, head_dim]` and
    `[batch, heads, length, head_dim]` swapped."""
    transpose = _producer(value, "Transpose")
    return transpose.inputs[0] if transpose is not None and _perm(transpose) == [0, 2, 1, 3] else None


def _merged_heads(value: ir.Value | None) -> ir.Value | None:
    """`states`, `[batch, heads, length, head_dim]`, if `value` is their merge into `[batch, length, heads * head_dim]`
    (a transpose, then a reshape)."""
    reshape = _producer(value, "Reshape")
    return _swapped_heads(reshape.inputs[0]) if reshape is not None else None


# ── 1. Prepare ────────────────────────────────────────────────────────────────


def _pow(op, x, exponent):
    return op.Pow(x, exponent)


def _is_minus_half(context, x, exponent, **_) -> bool:
    value = _constant(exponent)
    return value is not None and value.size == 1 and float(value.reshape(())) == -0.5


def _reciprocal_sqrt(op, x, exponent):
    return op.Reciprocal(op.Sqrt(x))


# Gemma's RMSNorm writes `rsqrt` as `Pow(x, -0.5)`; the RMSNorm fusion matches `Reciprocal(Sqrt(x))`.
_RSQRT_FROM_POW = pattern.RewriteRule(_pow, _reciprocal_sqrt, _is_minus_half)


def _flattened_key_transpose(op, key, flat_shape, shape):
    return op.Reshape(op.Transpose(op.Reshape(key, flat_shape), perm=[0, 2, 1]), shape)


def _is_flattened_5d_key(context, key, flat_shape, shape, **_) -> bool:
    target = _constant(shape)
    if target is None or key.shape is None or len(key.shape) == 4:
        return False
    return len(target) == 4 and all(dim > 0 for dim in target.tolist())


def _key_transpose(op, key, flat_shape, shape):
    batch, heads, depth, length = _constant(shape).tolist()
    return op.Transpose(op.Reshape(key, op.Constant(value_ints=[batch, heads, length, depth])), perm=[0, 1, 3, 2])


def _shape_slice(value: ir.Value, source: ir.Value) -> tuple[int, int] | None:
    """`(start, end)` if `value` is `Slice(Shape(source), start, end)`, constant bounds on its only axis."""
    sliced = _producer(value, "Slice")
    shape = _producer(sliced.inputs[0], "Shape") if sliced is not None else None
    if shape is None or shape.inputs[0] is not source or shape.attributes.get_int("start", 0) != 0:
        return None
    if "end" in shape.attributes or len(sliced.inputs) > 4:
        return None
    bounds = [_constant(item) for item in sliced.inputs[1:3]]
    if None in bounds or any(bound.size != 1 for bound in bounds):
        return None
    return int(bounds[0].reshape(())), int(bounds[1].reshape(()))


def _is_dynamic_4d_key(context, key, flat_shape, shape, **_) -> bool:
    # Under dynamic shapes, both shapes are the key's own sizes: `[..., length, head_dim]` flattened to 3D, then
    # `[..., head_dim, length]` (the last two swapped).
    if key.shape is None or len(key.shape) != 4:
        return False
    target, flat = _producer(shape, "Concat"), _producer(flat_shape, "Concat")
    if target is None or flat is None or len(target.inputs) != 3 or len(flat.inputs) < 2:
        return False
    leading, last, second_last = (_shape_slice(value, key) for value in target.inputs)
    flat_tail = [_shape_slice(value, key) for value in flat.inputs[-2:]]
    return (
        leading is not None
        and leading[0] <= 0
        and leading[1] == -2
        and last is not None
        and last[0] == -1
        and last[1] >= 4
        and second_last == (-2, -1)
        and flat_tail[0] == (-2, -1)
        and flat_tail[1] is not None
        and flat_tail[1][0] == -1
        and flat_tail[1][1] >= 4
    )


def _transposed_key(op, key, flat_shape, shape):
    return op.Transpose(key, perm=[0, 1, 3, 2])


# torchlib transposes a key through 3D: a grouped-query key, repeated over its heads, is 5D, and its transpose folds
# into the repeat's reshape; under dynamic shapes, a 4D one's shapes are computed from its own. The attention fusion,
# which looks for a 4D transpose, matches neither: both are rewritten as one.
_KEY_TRANSPOSE = pattern.RewriteRuleSet(
    [
        pattern.RewriteRule(_flattened_key_transpose, _key_transpose, _is_flattened_5d_key),
        pattern.RewriteRule(_flattened_key_transpose, _transposed_key, _is_dynamic_4d_key),
    ]
)


# ── 2. Rotary embeddings ──────────────────────────────────────────────────────
# onnxscript first matches `x * cos + rotate_half(x) * sin` (`RotaryEmbedding` in its `ai.onnxruntime._fusion`
# domain), then turns a match into ONNX Runtime's op if `cos` / `sin` come from `MatMul(inv_freq, positions)`. Ours
# are constants (a static export) or computed with a broadcast `Mul` (a dynamic one), so the rules below turn its
# matches into ONNX Runtime's op instead, from a cache built from those `cos` / `sin`.


class _RotaryEmbeddingAnyLayoutFusion(RotaryEmbeddingFusion):
    """onnxscript's rotate-half match, `[batch, length, heads, head_dim]` (Gemma 4's) too: its check takes axis 1
    for the heads, which a dynamic export of that layout leaves symbolic."""

    def check(self, op, x, start1, end1, start2, end2, **_):
        result = pattern.MatchResult()
        if x is None or x.shape is None or len(x.shape) != 4 or not isinstance(x.shape[3], int):
            return result.fail("Input is not 4D with a known head size.", x)
        if not isinstance(x.shape[1], int) and not isinstance(x.shape[2], int):
            return result.fail("Input has no known heads axis.", x)
        half = x.shape[3] // 2
        halves = (
            _ir_utils.is_singleton_value(start1, 0)
            and _ir_utils.is_singleton_value(end1, half)
            and _ir_utils.is_singleton_value(start2, half)
            and _ir_utils.is_singleton_value(end2, lambda end: end >= x.shape[3])
        )
        return result if halves else result.fail("x is not split into two halves.")

    def rewrite(self, op, x, cos, sin, **_):
        heads = x.shape[1] if isinstance(x.shape[1], int) else x.shape[2]
        return op.RotaryEmbedding(x, cos, sin, interleaved=0, num_heads=heads, _domain="ai.onnxruntime._fusion")


def _rotary_layout(x: ir.Value, cos_shape) -> int | None:
    """The axis of `x` holding the sequence, if `cos` / `sin` of shape `cos_shape` rotate it whole: 2 for
    `[batch, heads, length, head_dim]` (they broadcast over the heads, axis 1), 1 for `[batch, length, heads,
    head_dim]` (axis 2). The length axis lining up with `x`'s tells the two apart at length 1."""
    if x.shape is None or len(x.shape) != 4 or len(cos_shape) != 4:
        return None
    if cos_shape[-1] != x.shape[-1] or not isinstance(cos_shape[-1], int) or cos_shape[-1] % 2:
        return None
    for length_axis, heads_axis in ((2, 1), (1, 2)):
        if cos_shape[heads_axis] == 1 and cos_shape[length_axis] == x.shape[length_axis]:
            if length_axis == 2 or isinstance(x.shape[2], int):
                return length_axis
    return None


def _rotate(op, x, layout, positions, caches, rotated):
    """ONNX Runtime's `RotaryEmbedding` on `x`: as is for `[batch, heads, length, head_dim]`; flattened to
    `[batch, length, heads * head_dim]` and back for `[batch, length, heads, head_dim]`, which it doesn't take."""
    interleaved = rotated.producer().attributes.get_int("interleaved", 0)
    if layout == 2:
        return op.RotaryEmbedding(x, positions, *caches, interleaved=interleaved, _domain="com.microsoft")
    flat = op.Reshape(x, op.Constant(value_ints=[0, 0, -1]))
    flat = op.RotaryEmbedding(
        flat, positions, *caches, interleaved=interleaved, num_heads=x.shape[2], _domain="com.microsoft"
    )
    return op.Reshape(flat, op.Shape(x))


def _rotary_match(op, x, cos, sin):
    return op.RotaryEmbedding(
        x, cos, sin, _domain="ai.onnxruntime._fusion", _allow_other_attributes=True, _outputs=["rotated"]
    )


def _has_constant_cache(context, x, cos, sin, rotated, **_) -> bool:
    # `cos` / `sin` folded to constants, `cat(freqs, freqs)` along the head and alike across the batch.
    if cos.const_value is None or sin.const_value is None:
        return False
    for values in (cos.const_value.numpy(), sin.const_value.numpy()):
        half = values.shape[-1] // 2
        if _rotary_layout(x, values.shape) is None or not np.array_equal(values[..., :half], values[..., half:]):
            return False
        if not (values == values[:1]).all():
            return False
    return True


def _rotate_with_constant_cache(op, x, cos, sin, rotated):
    # Their first half is the `[length, head_dim / 2]` cache, read at positions 0 to length - 1.
    layout = _rotary_layout(x, cos.const_value.numpy().shape)
    caches = []
    for value in (cos, sin):
        values = value.const_value.numpy()[0]
        values = values[0] if layout == 2 else values[:, 0]
        caches.append(op.Constant(value=ir.tensor(values[:, : values.shape[-1] // 2].astype(x.dtype.numpy()))))
    return _rotate(op, x, layout, op.Constant(value_ints=[0]), caches, rotated)


def _frequencies(cos_or_sin: ir.Value, function: str) -> tuple[ir.Value, int | None] | None:
    """`(freqs, cast)` if `cos_or_sin` is `Unsqueeze(function(Concat(freqs, freqs)))` over the heads, an optional
    `Cast` after the `function`: `freqs`, `[batch, length, head_dim / 2]`, the angles."""
    unsqueeze = _producer(cos_or_sin, "Unsqueeze")
    axes = _constant(unsqueeze.inputs[1]) if unsqueeze is not None and len(unsqueeze.inputs) > 1 else None
    if axes is None or axes.tolist() not in ([1], [2]):
        return None
    node, cast = unsqueeze.inputs[0].producer(), None
    if node is not None and node.op_type == "Cast":
        cast = node.attributes.get_int("to")
        node = node.inputs[0].producer()
    concat = _producer(node.inputs[0], "Concat") if node is not None and node.op_type == function else None
    if concat is None or len(concat.inputs) != 2 or concat.inputs[0] is not concat.inputs[1]:
        return None
    return concat.inputs[0], cast


def _has_computed_cache(context, x, cos, sin, rotated, **_) -> bool:
    # `cos` / `sin` computed from the position ids, `freqs = positions * inv_freq` broadcast.
    found = _frequencies(cos, "Cos"), _frequencies(sin, "Sin")
    if None in found or found[0] != found[1] or cos.shape is None or _rotary_layout(x, cos.shape) is None:
        return False
    freqs = found[0][0]
    return freqs.shape is not None and len(freqs.shape) == 3 and freqs.shape[-1] * 2 == x.shape[-1]


def _rotate_with_token_cache(op, x, cos, sin, rotated):
    # Each token's own `cos` / `sin` form the cache, `[batch * length, head_dim / 2]`, read at position
    # `b * length + s`: exact at any position, prefill or decode.
    layout = _rotary_layout(x, cos.shape)
    freqs, cast = _frequencies(cos, "Cos")
    batch, length = op.Shape(x, start=0, end=1), op.Shape(x, start=layout, end=layout + 1)
    half = op.Constant(value_ints=[freqs.shape[-1]])
    # Position ids shared across the batch give `freqs` a batch of 1: every row gets its own copy.
    freqs = op.Expand(freqs, op.Concat(batch, length, half, axis=0))
    caches = []
    for function in (op.Cos, op.Sin):
        cache = op.Reshape(function(freqs), op.Concat(op.Constant(value_ints=[-1]), half, axis=0))
        caches.append(op.Cast(cache, to=cast) if cast is not None else cache)
    tokens = op.Range(op.Constant(value_int=0), op.Squeeze(op.Mul(batch, length)), op.Constant(value_int=1))
    return _rotate(op, x, layout, op.Reshape(tokens, op.Concat(batch, length, axis=0)), caches, rotated)


_ROTARY_ANY_LAYOUT = pattern.RewriteRuleSet([_RotaryEmbeddingAnyLayoutFusion.rule()])
_ROTARY_CONSTANT_CACHE = pattern.RewriteRule(_rotary_match, _rotate_with_constant_cache, _has_constant_cache)
# `cos` / `sin` are every layer's: kept for the other rotary embeddings.
_ROTARY_TOKEN_CACHE = pattern.RewriteRule(
    _rotary_match, _rotate_with_token_cache, _has_computed_cache, remove_nodes=False
)


def _fuse_rotary_embeddings(model: ir.Model) -> int:
    fuse_rotary_embedding(model)
    _ROTARY_ANY_LAYOUT.apply_to_model(model)
    return _ROTARY_CONSTANT_CACHE.apply_to_model(model) + _ROTARY_TOKEN_CACHE.apply_to_model(model)


def _fuse_shared_key_attentions(model: ir.Model) -> int:
    """Fuse the attentions of layers reading another layer's cached keys (Gemma 4's), before onnxscript's ORT fusions,
    whose optimizer merges the copies made here back: each attention gets its own copy of the key transpose they
    share (torchlib's `Reshape` -> `Transpose` -> `Reshape`, or a 4D `Transpose`), since the attention fusion only
    replaces nodes nothing else reads. Returns how many attentions fused."""
    graph, copies = model.graph, 0
    for node in list(graph.all_nodes()):
        chain = None
        if node.op_type == "Reshape":
            transpose = _producer(node.inputs[0], "Transpose")
            flatten = _producer(transpose.inputs[0], "Reshape") if transpose is not None else None
            if flatten is not None and _perm(transpose) == [0, 2, 1]:
                chain = [flatten, transpose, node]
        elif node.op_type == "Transpose" and _perm(node) == [0, 1, 3, 2]:
            chain = [node]
        if chain is None:
            continue
        uses = [(user, index) for user, index in chain[-1].outputs[0].uses() if user.op_type == "MatMul"]
        for user, index in uses[1:]:
            value = chain[0].inputs[0]
            for original in chain:
                attributes = {key: attribute.value for key, attribute in original.attributes.items()}
                output = original.outputs[0]
                inputs = [value, *original.inputs[1:]]
                value = _add_node(graph, user, original.op_type, inputs, output.shape, output.dtype, **attributes)
            user.replace_input_with(index, value)
            copies += 1
    return fuse_sdpa(model, apply_shape_inference=True) if copies else 0


# ── 4. Repair ─────────────────────────────────────────────────────────────────


def _unfuse_mixed_dtype_skip_norms(model: ir.Model) -> int:
    """Turn each `SkipSimplifiedLayerNormalization` whose input, skip and scale dtypes differ (a norm computed in
    float32 on a float16 residual), which ONNX Runtime rejects, back into the `Add` and `SimplifiedLayerNormalization`
    it fused. Returns how many."""
    count = 0
    for node in list(model.graph.all_nodes()):
        if node.op_type != "SkipSimplifiedLayerNormalization":
            continue
        input, skip, scale = node.inputs[:3]
        if len({input.dtype, skip.dtype, scale.dtype}) == 1:
            continue
        graph, output = node.graph, node.outputs[0]
        total = _add_node(graph, node, "Add", [input, skip], input.shape, input.dtype)
        attributes = {"axis": -1, "stash_type": 1}
        if "epsilon" in node.attributes:
            attributes["epsilon"] = node.attributes["epsilon"].as_float()
        normalized = _add_node(
            graph, node, "SimplifiedLayerNormalization", [total, scale], output.shape, output.dtype, **attributes
        )
        ir.convenience.replace_all_uses_with(output, normalized)
        if len(node.outputs) > 3:
            ir.convenience.replace_all_uses_with(node.outputs[3], total)
        node.graph.remove(node, safe=True)
        count += 1
    return count


def _untranspose_batch_matmuls(model: ir.Model) -> int:
    """Put back, as a `Transpose`, each batch transpose (`transBatchA` / `transBatchB`) fused into a `FusedMatMul`
    whose inputs ONNX Runtime can't batch-transpose (they must share one rank, at least 3). Returns how many."""
    count = 0
    for node in list(model.graph.all_nodes()):
        if node.op_type != "FusedMatMul":
            continue
        ranks = [len(value.shape) if value.shape is not None else None for value in node.inputs[:2]]
        if ranks[0] == ranks[1] and ranks[0] is not None and ranks[0] >= 3:
            continue
        for index, side in enumerate("AB"):
            if not node.attributes.get_int(f"transBatch{side}", 0):
                continue
            rank, source = ranks[index], node.inputs[index]
            # onnxscript's definitions: the batch transpose moves the first dimension to the second-to-last
            # position, and flipping the last two too moves it last.
            perm = (
                [*range(1, rank), 0]
                if node.attributes.get_int(f"trans{side}", 0)
                else [*range(1, rank - 1), 0, rank - 1]
            )
            shape = [source.shape[axis] for axis in perm] if source.shape is not None else None
            transposed = _add_node(node.graph, node, "Transpose", [source], shape, source.dtype, perm=perm)
            node.replace_input_with(index, transposed)
            for attribute in (f"transBatch{side}", f"trans{side}"):
                node.attributes[attribute] = ir.Attr(attribute, ir.AttributeType.INT, 0)
            count += 1
    return count


# ── 5. More fusions ───────────────────────────────────────────────────────────


class _WeightlessRmsNormFusion(pattern.RewriteRuleClassBase):
    """onnxscript's RMSNorm fusion, for a norm without a scale (Gemma 4's value norm): a
    `SimplifiedLayerNormalization` scaling by ones, with or without its output cast back."""

    def __init__(self, name: str, cast_back: bool):
        super().__init__(name)
        self._cast_back = cast_back

    def pattern(self, op, x, epsilon, compute_dtype, target_dtype):
        computed = pattern.OrValue([op.Cast(x, to=compute_dtype), x])
        mean_square = op.ReduceMean(op.Pow(computed, 2.0), [-1], keepdims=1, noop_with_empty_axes=0)
        normalized = op.Mul(computed, op.Reciprocal(op.Sqrt(op.Add(mean_square, epsilon))))
        return op.Cast(normalized, to=target_dtype) if self._cast_back else normalized

    def check(self, op, x, epsilon, compute_dtype, target_dtype=None, **_):
        result = pattern.MatchResult()
        if not isinstance(_ir_utils.get_singleton_value(epsilon), float):
            return result.fail("Epsilon is not a float value.", epsilon)
        if x.shape is None or not isinstance(x.shape[-1], int):
            return result.fail("Normalized size is not known.", x)
        return result

    def rewrite(self, op, x, epsilon, compute_dtype, target_dtype=None, **_):
        stash = compute_dtype.as_int() if compute_dtype is not None else x.dtype.value
        if target_dtype is not None:
            dtype = ir.DataType(target_dtype.as_int())
        else:
            dtype = ir.DataType(stash) if compute_dtype is not None else x.dtype
        ones = op.Constant(value=ir.tensor(np.ones(x.shape[-1], dtype=dtype.numpy())))
        epsilon = _ir_utils.get_singleton_value(epsilon)
        return op.SimplifiedLayerNormalization(x, ones, axis=-1, epsilon=epsilon, stash_type=stash)


# After the RMSNorm fusion, which would otherwise lose the scale of a norm it matches part of.
_WEIGHTLESS_RMS_NORMS = pattern.RewriteRuleSet(
    [
        _WeightlessRmsNormFusion.rule("WeightlessRmsNorm", cast_back=False),
        _WeightlessRmsNormFusion.rule("WeightlessRmsNormCastBack", cast_back=True),
    ]
)


def _narrow_norms(model: ir.Model) -> int:
    """Run in its input's dtype each `SimplifiedLayerNormalization` whose output is cast straight back to it (a norm
    a float16 model computes in float32): it computes in float32 anyway and returns its scale's dtype, so a constant
    scale that dtype holds exactly goes in narrowed and the cast goes. Returns how many."""
    count = 0
    # Norms sharing a scale share its narrowed copy.
    narrowed = {}
    for node in list(model.graph.all_nodes()):
        if node.op_type != "SimplifiedLayerNormalization":
            continue
        (input, scale), output = node.inputs[:2], node.outputs[0]
        uses = list(output.uses())
        if input.dtype == scale.dtype or scale.const_value is None or len(uses) != 1:
            continue
        cast = uses[0][0]
        if cast.op_type != "Cast" or cast.attributes.get_int("to") != input.dtype.value:
            continue
        if cast.outputs[0].is_graph_output():
            continue
        key = (scale.name, input.dtype)
        if key not in narrowed:
            values = scale.const_value.numpy()
            narrow = values.astype(input.dtype.numpy())
            exact = np.array_equal(narrow.astype(values.dtype), values)
            name = f"{scale.name}_{input.dtype.name.lower()}"
            narrowed[key] = _initializer(model.graph, name, narrow) if exact else None
        if narrowed[key] is None:
            continue
        node.replace_input_with(1, narrowed[key])
        output.dtype = input.dtype
        ir.convenience.replace_all_uses_with(cast.outputs[0], output)
        cast.graph.remove(cast, safe=True)
        count += 1
    return count


class _HeadsSecond:
    """New cache states in `[batch, length, heads, head_dim]`, not the cache's `[batch, heads, length, head_dim]`."""

    def __init__(self, value: ir.Value):
        self.value = value


def _cache_update(value: ir.Value):
    """`(past, new, start, present)`, the key-value cache update a fused attention's key or value reads, through its
    merge into `[batch, length, heads * head_dim]` and, for grouped-query attention, its repeat over the heads. A
    dynamic cache appends (`Concat(past, new, axis=2)`, no `start`); a static one writes at `start` into its buffer
    (`ScatterND` on its `[length, batch, heads, head_dim]` view); a prefill, with no past, returns the new states as
    they are (no `past`)."""
    if (states := _merged_heads(value)) is None:
        return None
    if (repeat := _producer(states, "Reshape")) is not None:
        expand = _producer(repeat.inputs[0], "Expand")
        unsqueeze = _producer(expand.inputs[0], "Unsqueeze") if expand is not None else None
        if unsqueeze is None:
            return None
        states = unsqueeze.inputs[0]
    if (concat := _producer(states, "Concat")) is not None:
        if concat.attributes.get_int("axis") not in (2, -2) or len(concat.inputs) != 2:
            return None
        return concat.inputs[0], concat.inputs[1], None, concat.outputs[0]
    back = _producer(states, "Transpose")
    scatter = _producer(back.inputs[0], "ScatterND") if back is not None else None
    if scatter is None or _perm(back) != [1, 2, 0, 3] or scatter.attributes.get_string("reduction", "none") != "none":
        # No past: the new states are the present themselves.
        if states.is_graph_output() and states.shape is not None and len(states.shape) == 4:
            return None, states, None, states
        return None
    indices = scatter.inputs[1]
    past, new = (_producer(item, "Transpose") for item in (scatter.inputs[0], scatter.inputs[2]))
    if past is None or new is None or _perm(past) != [2, 0, 1, 3]:
        return None
    if indices.shape is None or len(indices.shape) != 2 or _static_dim(indices, 1) != 1:
        return None
    # A key comes rotated, `[batch, heads, length, head_dim]`; a value straight from its projection.
    if _perm(new) == [2, 0, 1, 3]:
        return past.inputs[0], new.inputs[0], indices, back.outputs[0]
    if _perm(new) == [1, 0, 2, 3]:
        return past.inputs[0], _HeadsSecond(new.inputs[0]), indices, back.outputs[0]
    return None


def _same_value(a: ir.Value | None, b: ir.Value | None, depth: int = 12) -> bool:
    """Whether `a` and `b` hold the same values: one value, equal constants, or the same op on the same inputs (the
    rotary caches each rotation computes its own copy of)."""
    if a is b:
        return True
    if a is None or b is None or depth == 0:
        return False
    if a.const_value is not None and b.const_value is not None:
        return np.array_equal(a.const_value.numpy(), b.const_value.numpy())
    x, y = a.producer(), b.producer()
    if x is None or y is None or (x.op_type, x.domain) != (y.op_type, y.domain) or len(x.inputs) != len(y.inputs):
        return False
    attributes = {k: v.value for k, v in x.attributes.items()}
    if attributes != {k: v.value for k, v in y.attributes.items()}:
        return False
    if x.op_type == "Shape":
        # The batch and length of the query and of the key, read off each: the same dims, known or the same symbol.
        a_shape, b_shape = x.inputs[0].shape, y.inputs[0].shape
        if a_shape is not None and b_shape is not None and len(a_shape) == len(b_shape):
            start, end = attributes.get("start", 0), attributes.get("end", len(a_shape))
            dims = list(zip(a_shape[start:end], b_shape[start:end]))
            if all(p == q and (isinstance(p, int) or getattr(p, "value", None) is not None) for p, q in dims):
                return True
    if a.index() != b.index():
        return False
    return all(_same_value(p, q, depth - 1) for p, q in zip(x.inputs, y.inputs))


def _query_states(query: ir.Value, kv_heads: int, num_heads: int):
    """`(states, groups)`: the attention's query heads, `[batch, heads, length, head_dim]`, behind its merge into
    `[batch, length, heads * head_dim]`; `groups` set when, for one new token, they were folded onto their key-value
    heads (`[batch, kv_heads, groups, head_dim]`, how transformers decodes grouped-query attention without repeating
    the key-value heads), the attention then running over the key-value heads."""
    if (states := _merged_heads(query)) is None:
        return None
    fold = _producer(states, "Reshape")
    if num_heads != kv_heads or fold is None:
        return states, None
    groups, heads = _static_dim(states, 2), _static_dim(fold.inputs[0], 1)
    if groups is None or heads != kv_heads * groups or _static_dim(fold.inputs[0], 2) != 1:
        return None
    return fold.inputs[0], groups


def _shared_rotation(query_states: ir.Value, new_key, head_size: int, prefill: bool):
    """`((query, key, (positions, cos_cache, sin_cache)), interleaved)`, the attention's query and new key before
    ONNX Runtime's `RotaryEmbedding` rotated them (`[batch, heads, length, head_dim]`), if one cache and the same
    positions rotate both whole; the positions None when they're the cache's default (0 on, no past)."""
    rotations = (
        _producer(query_states, "RotaryEmbedding"),
        _producer(new_key, "RotaryEmbedding") if isinstance(new_key, ir.Value) else None,
    )
    if None in rotations or any(node.domain != "com.microsoft" for node in rotations):
        return None
    (query_rotation, key_rotation) = rotations
    if len(query_rotation.inputs) != 4 or len(key_rotation.inputs) != 4:
        return None
    if not all(map(_same_value, query_rotation.inputs[1:], key_rotation.inputs[1:])):
        return None
    interleaved = query_rotation.attributes.get_int("interleaved", 0)
    if (
        key_rotation.attributes.get_int("interleaved", 0) != interleaved
        or "rotary_embedding_dim" in query_rotation.attributes
    ):
        return None
    positions, cos_cache, sin_cache = query_rotation.inputs[1:]
    # It rotates head sizes in multiples of 16 only.
    if head_size % 16 or _static_dim(cos_cache, 1) is None or _static_dim(cos_cache, 1) * 2 != head_size:
        return None
    if positions.shape is not None and len(positions.shape) == 1:
        # An offset: the cache's default positions only from 0 on, a prefill's.
        if not prefill or _constant(positions) is None or _constant(positions).tolist() != [0]:
            return None
        positions = None
    return (query_rotation.inputs[0], key_rotation.inputs[0], (positions, cos_cache, sin_cache)), interleaved


def _head_norm(states: ir.Value):
    """`(x, weight, epsilon)` if `states`, `[batch, heads, length, head_dim]`, are the heads of `x`, `[batch, length,
    heads * head_dim]`, each RMS-normalized with `weight` (Qwen3's query and key norms)."""
    norm = _producer(_swapped_heads(states), "SimplifiedLayerNormalization")
    if norm is None or len(norm.inputs) != 2 or _constant(norm.inputs[1]) is None:
        return None
    reshape = _producer(norm.inputs[0], "Reshape")
    x = reshape.inputs[0] if reshape is not None else None
    if x is None or x.shape is None or len(x.shape) != 3 or norm.attributes.get_int("axis", -1) not in (-1, 3):
        return None
    return x, norm.inputs[1], norm.attributes.get_float("epsilon", 1e-5)


def _fuse_group_query_attention(model: ir.Model, prefill: bool = True, head_norms: bool = False) -> int:
    """Turn each fused `MultiHeadAttention` reading a key-value cache's update into ONNX Runtime's
    `GroupQueryAttention`, which updates the cache itself (or, for a prefill, starts it) and reads its key-value heads
    without repeating them. Its mask stays the attention's own bias, so every row's `seqlens_k` is the whole length:
    exact whatever the padding. `prefill` sets whether a prefill, starting the cache, is fused too; `head_norms`
    whether per-head query and key norms are folded in too (CUDA only). Returns how many."""
    graph, count = model.graph, 0
    # A cache update several attentions read (layers sharing another layer's cache, Gemma 4's): its own layer's,
    # fused once; the others, reading it, stay as they are.
    readers = {}
    for attention in graph.all_nodes():
        if attention.op_type == "MultiHeadAttention" and len(attention.inputs) == 6:
            for update in (_cache_update(attention.inputs[1]), _cache_update(attention.inputs[2])):
                if update is not None:
                    readers[update[3]] = readers.get(update[3], 0) + 1
    for attention in list(graph.all_nodes()):
        if attention.op_type != "MultiHeadAttention" or len(attention.inputs) != 6:
            continue
        query, key, value, projection_bias, padding_mask, bias = attention.inputs
        updates = _cache_update(key), _cache_update(value)
        if projection_bias is not None or padding_mask is not None or None in updates:
            continue
        if any(readers[update[3]] > 1 for update in updates):
            continue
        (past_key, new_key, start, key_present), (past_value, new_value, _, value_present) = updates
        cached = [states for states in (past_key, new_key) if isinstance(states, ir.Value)]
        kv_heads = next((heads for states in cached if (heads := _static_dim(states, 1)) is not None), None)
        # Head sizes in multiples of 8, and self-attention only: the new key as long as the query.
        head_size = _static_dim(cached[0], 3)
        states, length_axis = (new_key.value, 1) if isinstance(new_key, _HeadsSecond) else (new_key, 2)
        new_length = states.shape[length_axis] if states.shape is not None else None
        if kv_heads is None or head_size is None or head_size % 8 or query.shape is None:
            continue
        num_heads = attention.attributes.get_int("num_heads")
        if (found := _query_states(query, kv_heads, num_heads)) is None:
            continue
        query_states, groups = found
        query_length = 1 if groups is not None else query.shape[1]
        if groups is not None:
            # The bias, expanded over the folded rows: its own `[batch, 1, 1, total]`.
            expand = _producer(bias, "Expand") if bias is not None else None
            if expand is None or _static_dim(expand.inputs[0], 2) != 1 or _static_dim(expand.inputs[0], 1) != 1:
                continue
            bias, num_heads = expand.inputs[0], kv_heads * groups
        if new_length != query_length:
            continue
        # After a past, it takes several new tokens for one sequence only (part of its contract): a decode graph that
        # takes any number of tokens is fused only for a batch of one.
        if past_key is not None and query_length != 1 and _static_dim(query, 0) != 1:
            continue
        if past_key is None and not prefill:
            continue
        tape = ir.tape.Tape(graph)

        def constant(values, dtype=np.int64, tape=tape, count=count):
            name = f"group_query_attention_{count}_{len(tape.initializers)}"
            return tape.initializer(ir.tensor(np.array(values, dtype=dtype)), name=name)

        def merged(states, tape=tape, merge=constant([0, 0, -1])):
            # To `[batch, length, heads * head_dim]`.
            if isinstance(states, _HeadsSecond):
                return tape.op("Reshape", [states.value, merge])
            return tape.op("Reshape", [tape.op("Transpose", [states], attributes={"perm": [0, 2, 1, 3]}), merge])

        # The total length as a constant when it's known: ONNX Runtime's shape inference reads one computed from shapes
        # as int64 (microsoft/onnxruntime#33865), and its CUDA kernel reads one computed on the GPU as 0
        # (microsoft/onnxruntime#33867). A static cache attends over its whole buffer, the bias masking the slots not
        # written yet; appended to a past, it's the past and the new tokens.
        past_length = 0 if past_key is None else _static_dim(past_key, 2)
        known_total = None
        if start is not None:
            known_total = past_length
        elif isinstance(query_length, int) and past_length is not None:
            known_total = query_length + past_length
        rotary, dtype, rotary_attributes = [None, None, None], query.dtype, {}
        if (rotation := _shared_rotation(query_states, new_key, head_size, past_key is None)) is not None:
            # Rotated by the attention itself, which stores the key rotated.
            (query_states, key_states, (positions, cos_cache, sin_cache)), interleaved = rotation
            new_key, rotary = key_states, [cos_cache, sin_cache, positions]
            rotary_attributes = {"do_rotary": 1, "rotary_interleaved": interleaved}
        if rotation is not None or groups is not None:
            query = merged(query_states)
        # The lengths are read off the query's heads, which keep their shape, rather than off their merge built here.
        sized, length_axis = query_states, 2
        key_states, norms = merged(new_key), []
        # CUDA's kernel takes half precision only: a float32 one runs on the CPU, which has no norms.
        half = dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
        if head_norms and half and isinstance(new_key, ir.Value):
            found = _head_norm(query_states), _head_norm(new_key)
            if None not in found and found[0][2] == found[1][2]:
                # Normalized by the attention itself, before its rotary.
                (query, query_norm, epsilon), (key_states, key_norm, _) = found
                weights = (constant(_constant(w).astype(dtype.numpy()), dtype.numpy()) for w in (query_norm, key_norm))
                norms = [None, None, None, *weights]
                rotary_attributes["qk_norm_epsilon"] = epsilon
                sized, length_axis = query, 1
        batch = tape.op("Shape", [sized], attributes={"start": 0, "end": 1})
        new_length = tape.op("Shape", [sized], attributes={"start": length_axis, "end": length_axis + 1})
        if past_key is None:
            length = new_length
        elif start is None:
            # Appended after the past.
            length = tape.op("Add", [tape.op("Shape", [past_key], attributes={"start": 2, "end": 3}), new_length])
        else:
            # Written in place from the first index on, and the mask, spanning the whole buffer, read up to there.
            offset = tape.op("Gather", [tape.op("Reshape", [start, constant([-1])]), constant([0])])
            length = tape.op("Add", [offset, new_length])
        # Each sequence's length after this step, the last key it attends to: where the new keys and values go.
        written = tape.op("Cast", [length], attributes={"to": ir.DataType.INT32})
        seqlens_k = tape.op("Expand", [tape.op("Sub", [written, constant([1], np.int32)]), batch])
        if known_total is not None:
            total_length = constant(known_total, np.int32)
        else:
            total_length = tape.op("Squeeze", [written])
        if rotary[2] is not None:
            # It takes a cache of at least the total rows; a per-token cache (`b * length + s`) has
            # `batch * length`, so zero rows, never read, are appended up to that (a `Concat`: there's no bfloat16
            # `Pad`, microsoft/onnxruntime#33874).
            total = constant([known_total]) if known_total is not None else length
            rows = tape.op("Shape", [rotary[0]], attributes={"start": 0, "end": 1})
            missing = tape.op("Max", [tape.op("Sub", [total, rows]), constant([0])])
            shape = tape.op(
                "Concat", [missing, tape.op("Shape", [rotary[0]], attributes={"start": 1})], attributes={"axis": 0}
            )
            zeros = tape.op("Expand", [constant(0, dtype.numpy()), shape])
            rotary[:2] = (tape.op("Concat", [cache, zeros], attributes={"axis": 0}) for cache in rotary[:2])
        attributes = {"num_heads": num_heads, "kv_num_heads": kv_heads}
        if "scale" in attention.attributes:
            attributes["scale"] = attention.attributes["scale"].as_float()
        attributes.update(rotary_attributes)
        outputs = tape.op_multi_out(
            "GroupQueryAttention",
            [query, key_states, merged(new_value), past_key, past_value, seqlens_k, total_length]
            + [*rotary, bias, *norms],
            attributes=attributes,
            domain="com.microsoft",
            num_outputs=3,
        )
        output, present_key, present_value = outputs
        if groups is not None:
            # Back to the folded attention's layout: each key-value head's query heads as its rows.
            folded = tape.op("Reshape", [output, constant([0, kv_heads, groups, head_size])])
            folded = tape.op("Transpose", [folded], attributes={"perm": [0, 2, 1, 3]})
            output = tape.op("Reshape", [folded, constant([0, groups, kv_heads * head_size])])
        for node in tape.nodes:
            graph.insert_before(attention, node)
        output.dtype, output.shape = dtype, attention.outputs[0].shape
        _replace(attention.outputs[0], output)
        for new, old in ((present_key, key_present), (present_value, value_present)):
            new.dtype, new.shape = old.dtype, old.shape
            if past_key is not None:
                _replace(old, new)
            else:
                # A prefill's present is the new states, which the attention's inputs still read: only the graph
                # output moves.
                name = old.name
                graph.outputs[graph.outputs.index(old)] = new
                old.name, new.name = f"{name}_unfused", name
        count += 1
    return count


def _uncast(value: ir.Value) -> ir.Value:
    """`value` before a `Cast`, if a `Cast` makes it."""
    cast = _producer(value, "Cast")
    return cast.inputs[0] if cast is not None else value


def _batched_matmul(value: ir.Value):
    """`(rows, weights, biases, ids)` if `value` is the `batched_mm` experts' projection of `rows`: each routed row
    times its expert's weight, `GatherND(weights, ids)`, plus its expert's bias, `GatherND(biases, ids)`, or not. The
    weights come back `[experts, out, in]`, ONNX Runtime's layout."""
    biases = bias_ids = None
    if (add := _producer(value, "Add")) is not None:
        for product, added in (add.inputs, add.inputs[::-1]):
            lookup = _producer(added, "GatherND")
            if lookup is not None and _constant(lookup.inputs[0]) is not None:
                value, biases, bias_ids = product, _constant(lookup.inputs[0]), lookup.inputs[1]
                break
        else:
            return None
    squeeze = _producer(value, "Squeeze")
    matmul = _producer(squeeze.inputs[0], "MatMul") if squeeze is not None else None
    if matmul is None:
        return None
    # `W[ids] @ x[..., None]` (`[experts, out, in]` weights) or `x[:, None] @ W[ids]` (`[experts, in, out]`).
    for gathered, vector, transposed in ((*matmul.inputs, False), (*matmul.inputs[::-1], True)):
        lookup, unsqueeze = _producer(gathered, "GatherND"), _producer(vector, "Unsqueeze")
        weights = _constant(lookup.inputs[0]) if lookup is not None else None
        if unsqueeze is None or weights is None or weights.ndim != 3:
            continue
        if bias_ids is not None and bias_ids is not lookup.inputs[1]:
            return None
        return unsqueeze.inputs[0], weights.transpose(0, 2, 1) if transposed else weights, biases, lookup.inputs[1]
    return None


def _gate(value: ir.Value):
    """`(projected, gate_first)` if `value` is `silu(gate) * up` (`gate * Sigmoid(gate)`) of `projected`'s two halves
    (its `Split`). Not a GELU gate (Gemma 4's): ONNX Runtime 1.29's CUDA `MoE` miscomputes gated GELU experts."""
    mul = _producer(value, "Mul")
    if mul is None:
        return None
    for activated, up in (mul.inputs, mul.inputs[::-1]):
        silu = _producer(activated, "Mul")
        if silu is None:
            continue
        gate = next((item for item in silu.inputs if _producer(item, "Split") is not None), None)
        sigmoid = next((_producer(item, "Sigmoid") for item in silu.inputs if _producer(item, "Sigmoid")), None)
        if gate is None or sigmoid is None or sigmoid.inputs[0] is not gate:
            continue
        split = _producer(gate, "Split")
        if split is None or len(split.outputs) != 2 or up not in split.outputs:
            continue
        if split.attributes.get_int("axis") in (1, -1):
            return split.inputs[0], split.outputs[0] is gate
    return None


def _router(weights: ir.Value):
    """`(logits, k, normalize, indices)` if the routing `weights` are `TopK(Softmax(logits))`'s values, renormalized (`Div`
    by their `ReduceSum`) or not, or `Softmax(TopK(logits))` (Granite's: the same experts and weights as the
    renormalized). A float16 model's router computes in float32, its softmax or its top-k: the casts around them are
    stepped over, the logits returned in the model's dtype."""
    if (downcast := _producer(weights, "Cast")) is not None:
        weights = downcast.inputs[0]
    normalize = 0
    division = _producer(weights, "Div")
    total = _producer(division.inputs[1], "ReduceSum") if division is not None else None
    if total is not None:
        weights, normalize = division.inputs[0], 1
    if (softmax_of_top := _producer(weights, "Softmax")) is not None and total is None:
        # Top-k logits, then their softmax.
        top_k = _producer(softmax_of_top.inputs[0], "TopK")
        if top_k is None or softmax_of_top.inputs[0] is not top_k.outputs[0]:
            return None
        if softmax_of_top.attributes.get_int("axis", -1) not in (-1, 1) or top_k.attributes.get_int("largest", 1) != 1:
            return None
        logits, normalize = top_k.inputs[0], 1
    else:
        top_k = _producer(weights, "TopK")
        softmax = _producer(_uncast(top_k.inputs[0]), "Softmax") if top_k is not None else None
        if softmax is None or weights is not top_k.outputs[0] or top_k.attributes.get_int("largest", 1) != 1:
            return None
        logits = softmax.inputs[0]
    if total is not None and total.inputs[0] is not top_k.outputs[0]:
        return None
    if (upcast := _producer(logits, "Cast")) is not None:
        logits = upcast.inputs[0]
    return logits, int(_constant(top_k.inputs[1]).reshape(())), normalize, top_k.outputs[1]


def _fuse_mixture_of_experts(model: ir.Model) -> int:
    """Turn each mixture-of-experts block (the `batched_mm` experts implementation behind a softmax / top-k router)
    into ONNX Runtime's CUDA `MoE`, which routes the tokens itself from the router's logits. Returns how many.

    Not for the CPU: ONNX Runtime's CPU `MoE` returns wrong values when its input comes from another node and its
    output feeds one (microsoft/onnxruntime#33866)."""
    graph, count = model.graph, 0
    for combine in list(graph.all_nodes()):
        # The routed rows' outputs summed per token, `ReduceSum(Reshape(weighted), axes=[1])`; a float16 model sums
        # them in float32.
        if combine.op_type != "ReduceSum" or combine.attributes.get_int("keepdims", 1) != 0 or len(combine.inputs) < 2:
            continue
        axes = _constant(combine.inputs[1])
        upcast = _producer(combine.inputs[0], "Cast")
        reshape = _producer(upcast.inputs[0] if upcast is not None else combine.inputs[0], "Reshape")
        weighted = _producer(reshape.inputs[0], "Mul") if reshape is not None else None
        if axes is None or axes.tolist() != [1] or weighted is None:
            continue
        for projected, scale in (weighted.inputs, weighted.inputs[::-1]):
            # A float16 model weights the experts' outputs in float32.
            down = _batched_matmul(_uncast(projected))
            if down is not None:
                break
        gate = _gate(down[0]) if down is not None else None
        up = _batched_matmul(gate[0]) if gate is not None else None
        # Each token repeated once per routed expert, `Reshape(Tile(Unsqueeze(tokens, 1)))`.
        repeated = _producer(up[0], "Reshape") if up is not None else None
        tile = _producer(repeated.inputs[0], "Tile") if repeated is not None else None
        tokens = _producer(tile.inputs[0], "Unsqueeze") if tile is not None else None
        # Each routed row's weight, the flattened routing weights.
        unsqueeze = _producer(_uncast(scale), "Unsqueeze") if down is not None else None
        flat = _producer(_uncast(unsqueeze.inputs[0]), "Reshape") if unsqueeze is not None else None
        router = _router(flat.inputs[0]) if flat is not None else None
        if tokens is None or router is None or up[3] is not down[3]:
            continue
        logits, k, normalize, indices = router
        ids = _producer(up[3], "Unsqueeze")
        flat_ids = _producer(ids.inputs[0], "Reshape") if ids is not None else None
        if flat_ids is None or flat_ids.inputs[0] is not indices:
            continue
        (_, fc1, up_biases, _), (_, fc2, down_biases, _), (_, gate_first) = up, down, gate
        tokens = tokens.inputs[0]
        if logits.shape is None or _static_dim(logits, len(logits.shape) - 1) != fc1.shape[0]:
            continue
        if logits.dtype != tokens.dtype:
            continue
        half = fc1.shape[1] // 2
        gate_rows, up_rows = (slice(0, half), slice(half, None)) if gate_first else (slice(half, None), slice(0, half))
        biases = (up_biases[:, gate_rows], up_biases[:, up_rows]) if up_biases is not None else (None, None)
        weights = [fc1[:, gate_rows], biases[0], fc2, down_biases, fc1[:, up_rows], biases[1]]
        weights = [
            _initializer(graph, f"mixture_of_experts_{count}_{index}", array) if array is not None else None
            for index, array in enumerate(weights)
        ]
        combined = combine.outputs[0]
        attributes = {"k": k, "normalize_routing_weights": normalize, "activation_type": "silu"}
        output = _add_node(
            graph,
            combine,
            "MoE",
            [tokens, logits, *weights],
            combined.shape,
            tokens.dtype,
            "com.microsoft",
            **attributes,
        )
        if upcast is not None:
            output = _add_node(graph, combine, "Cast", [output], combined.shape, combined.dtype, to=combined.dtype)
        _replace(combined, output)
        count += 1
    return count


def _unfuse_attentions(model: ir.Model) -> int:
    """Turn each `MultiHeadAttention` over `[batch, length, heads * head_dim]` queries, keys and values with an
    attention bias and no cache (an encoder's, or one `GroupQueryAttention` didn't take) back into `MatMul`s and a
    `Softmax`: ONNX Runtime's CPU kernel for it runs slower than the ops it fused (microsoft/onnxruntime#33873).
    Returns how many."""
    count = 0
    for node in list(model.graph.all_nodes()):
        if node.op_type != "MultiHeadAttention" or len(node.inputs) != 6 or len(node.outputs) != 1:
            continue
        # One `GroupQueryAttention` replaced, left for the clean-up to remove.
        if not node.outputs[0].uses() and not node.outputs[0].is_graph_output():
            continue
        query, key, value, projection_bias, padding_mask, bias = node.inputs
        if projection_bias is not None or padding_mask is not None or bias is None:
            continue
        if any(item is None or item.shape is None or len(item.shape) != 3 for item in (query, key, value)):
            continue
        heads = node.attributes.get_int("num_heads")
        head_size = _static_dim(query, 2) // heads if _static_dim(query, 2) is not None else None
        if head_size is None:
            continue
        scale = node.attributes["scale"].as_float() if "scale" in node.attributes else head_size**-0.5
        graph, tape = node.graph, ir.tape.Tape()
        split_heads = _initializer(graph, f"unfused_attention_{count}_heads", np.array([0, 0, heads, -1]))
        merge_heads = _initializer(graph, f"unfused_attention_{count}_merge", np.array([0, 0, -1]))
        factor = _initializer(graph, f"unfused_attention_{count}_scale", np.array(scale, dtype=query.dtype.numpy()))

        def heads_of(states, perm, tape=tape, split_heads=split_heads):
            return tape.op("Transpose", [tape.op("Reshape", [states, split_heads])], attributes={"perm": perm})

        scores = tape.op("MatMul", [heads_of(query, [0, 2, 1, 3]), heads_of(key, [0, 2, 3, 1])])
        scores = tape.op("Add", [tape.op("Mul", [scores, factor]), bias])
        probabilities = tape.op("Softmax", [scores], attributes={"axis": -1})
        context = tape.op("MatMul", [probabilities, heads_of(value, [0, 2, 1, 3])])
        context = tape.op("Transpose", [context], attributes={"perm": [0, 2, 1, 3]})
        output = tape.op("Reshape", [context, merge_heads])
        output.dtype = query.dtype
        graph.insert_before(node, tape.nodes)
        _replace(node.outputs[0], output)
        graph.remove(node, safe=True)
        count += 1
    return count


def _swap_heads(graph: ir.Graph, before: ir.Node, x: ir.Value, replacing: ir.Value | None = None) -> ir.Value:
    """`x` transposed to the other layout; in place of `replacing`, its shape."""
    shape = [x.shape[0], x.shape[2], x.shape[1], x.shape[3]] if x.shape is not None else None
    dtype = x.dtype
    if replacing is not None:
        shape, dtype = replacing.shape, replacing.dtype
    return _add_node(graph, before, "Transpose", [x], shape, dtype, perm=[0, 2, 1, 3])


def _sink_head_transposes(model: ir.Model) -> int:
    """Move the transposes between `[batch, length, heads, head_dim]` and `[batch, heads, length, head_dim]` past the
    ops that read either layout alike, the rotary embeddings, the repeat of the key-value heads and the zeroing of
    unattended rows, until they meet the transpose back and cancel: `MultiHeadAttention` reads and writes the first
    layout, so each attention copied its query, keys, values and output there and back. Returns how many pairs
    cancel."""
    graph = model.graph
    for index, node in enumerate(list(graph)):
        name = f"head_transposes_{index}"
        if node.op_type == "RotaryEmbedding" and node.domain == "com.microsoft":
            # Rotated in `[batch, length, heads * head_dim]`, which it takes given the heads.
            x = _swapped_heads(node.inputs[0])
            heads, head_size = (_static_dim(x, 2), _static_dim(x, 3)) if x is not None else (None, None)
            if heads is None or head_size is None:
                continue
            flat_shape = [x.shape[0], x.shape[1], heads * head_size]
            flat = _add_node(
                graph,
                node,
                "Reshape",
                [x, _initializer(graph, f"{name}_flat", np.array([0, 0, -1]))],
                flat_shape,
                x.dtype,
            )
            attributes = {key: attribute.value for key, attribute in node.attributes.items()} | {"num_heads": heads}
            rotated = _add_node(
                graph,
                node,
                "RotaryEmbedding",
                [flat, *node.inputs[1:]],
                flat_shape,
                x.dtype,
                "com.microsoft",
                **attributes,
            )
            split = _initializer(graph, f"{name}_heads", np.array([0, 0, heads, head_size]))
            rotated = _add_node(graph, node, "Reshape", [rotated, split], x.shape, x.dtype)
            _replace(node.outputs[0], _swap_heads(graph, node, rotated, node.outputs[0]))
        elif node.op_type == "Reshape" and (expand := _producer(node.inputs[0], "Expand")) is not None:
            # The key-value heads repeated, `[batch, kv_heads, groups, length, head_dim]` flattened: as
            # `[batch, length, kv_heads, groups, head_dim]` instead.
            unsqueeze = _producer(expand.inputs[0], "Unsqueeze")
            axes = _constant(unsqueeze.inputs[1]) if unsqueeze is not None and len(unsqueeze.inputs) == 2 else None
            x = _swapped_heads(unsqueeze.inputs[0]) if axes is not None and axes.tolist() == [2] else None
            if x is None or node.outputs[0].shape is None or expand.outputs[0].shape is None:
                continue
            kv_heads, groups, head_size = _static_dim(x, 2), _static_dim(expand.outputs[0], 2), _static_dim(x, 3)
            if None in (kv_heads, groups, head_size) or _static_dim(node.outputs[0], 1) != kv_heads * groups:
                continue
            batch, length = x.shape[0], x.shape[1]
            single = _initializer(graph, f"{name}_single", np.array([0, 0, kv_heads, 1, head_size]))
            states = _add_node(graph, node, "Reshape", [x, single], [batch, length, kv_heads, 1, head_size], x.dtype)
            # `Expand` broadcasts: ones keep the other axes.
            target = _initializer(graph, f"{name}_groups", np.array([1, 1, 1, groups, 1]))
            states = _add_node(
                graph, node, "Expand", [states, target], [batch, length, kv_heads, groups, head_size], x.dtype
            )
            heads = _initializer(graph, f"{name}_all_heads", np.array([0, 0, kv_heads * groups, head_size]))
            states = _add_node(
                graph, node, "Reshape", [states, heads], [batch, length, kv_heads * groups, head_size], x.dtype
            )
            _replace(node.outputs[0], _swap_heads(graph, node, states, node.outputs[0]))
        elif node.op_type == "Where" and (x := _swapped_heads(node.inputs[2])) is not None:
            # Rows zeroed by a condition over the batch and the queries, `[batch, 1, length, 1]`.
            condition, zero = node.inputs[0], _constant(node.inputs[1])
            if condition.shape is None or len(condition.shape) != 4 or zero is None or zero.size != 1:
                continue
            if _static_dim(condition, 1) != 1 or _static_dim(condition, 3) != 1:
                continue
            condition = _swap_heads(graph, node, condition)
            dtype = node.outputs[0].dtype
            zeroed = _add_node(graph, node, "Where", [condition, node.inputs[1], x], x.shape, dtype)
            _replace(node.outputs[0], _swap_heads(graph, node, zeroed, node.outputs[0]))
    count = 0
    for node in list(graph):
        # A swap back, read through a reshape to the same shape or not, is the states before the first.
        inner = node.inputs[0] if node.op_type == "Transpose" and _perm(node) == [0, 2, 1, 3] else None
        reshape = _producer(inner, "Reshape")
        if reshape is not None and inner.shape is not None and reshape.inputs[0].shape == inner.shape:
            inner = reshape.inputs[0]
        if (x := _swapped_heads(inner)) is not None:
            _replace(node.outputs[0], x)
            count += 1
    common_passes.RemoveUnusedNodesPass()(model)
    return count


def _pack_sibling_matmuls(model: ir.Model) -> int:
    """Turn the `MatMul`s reading one input with constant `[in, out]` weights (an attention's query, key and value
    projections; an MLP's gate and up ones) into one, their weights concatenated, its output split back: one kernel
    for several. Returns how many packs."""
    graph, count = model.graph, 0
    for value in [*graph.inputs, *(output for node in graph.all_nodes() for output in node.outputs)]:
        siblings = [
            user
            for user, index in value.uses()
            if user.op_type == "MatMul" and index == 0 and _constant(user.inputs[1]) is not None
        ]
        weights = [_constant(user.inputs[1]) for user in siblings]
        if len(siblings) < 2 or len({(weight.ndim, weight.shape[0], weight.dtype) for weight in weights}) != 1:
            continue
        if weights[0].ndim != 2 or len({sibling.graph for sibling in siblings}) != 1:
            continue
        packed = _initializer(graph, f"packed_matmul_{count}", np.concatenate(weights, axis=1))
        sizes = _initializer(graph, f"packed_matmul_{count}_sizes", np.array([w.shape[1] for w in weights], np.int64))
        matmul = ir.node("MatMul", [value, packed])
        split = ir.node("Split", [matmul.outputs[0], sizes], attributes={"axis": -1}, num_outputs=len(siblings))
        if value.shape is not None:
            matmul.outputs[0].shape = ir.Shape([*value.shape[:-1], packed.shape[1]])
        matmul.outputs[0].dtype = siblings[0].outputs[0].dtype
        # Before the first sibling: each reads the input there, so it's computed by then.
        first = min(siblings, key=list(graph).index)
        graph.insert_before(first, [matmul, split])
        for sibling, output in zip(siblings, split.outputs):
            output.dtype, output.shape = sibling.outputs[0].dtype, sibling.outputs[0].shape
            _replace(sibling.outputs[0], output)
        count += 1
    return count


def _unmerged(value: ir.Value) -> ir.Value:
    """`value` before a reshape splitting its last axis into heads and one merging them back, if that's what made it."""
    outer = _producer(value, "Reshape")
    inner = _producer(outer.inputs[0], "Reshape") if outer is not None else None
    target = _constant(outer.inputs[1]) if outer is not None else None
    if inner is None or target is None or target.tolist() != [0, 0, -1]:
        return value
    x, heads = inner.inputs[0], inner.outputs[0]
    if x.shape is None or heads.shape is None or len(x.shape) != 3 or len(heads.shape) != 4:
        return value
    size, split = _static_dim(x, 2), (_static_dim(heads, 2), _static_dim(heads, 3))
    if size is None or None in split or split[0] * split[1] != size or list(heads.shape[:2]) != list(x.shape[:2]):
        return value
    return x


def _pack_group_query_attention_inputs(model: ir.Model) -> int:
    """Hand each `GroupQueryAttention` whose query, key and value are the split of one packed projection that
    projection instead, which it reads packed: no split. Returns how many."""
    count = 0
    for node in model.graph:
        if node.op_type != "GroupQueryAttention" or node.domain != "com.microsoft" or None in node.inputs[:3]:
            continue
        parts = [_unmerged(value) for value in node.inputs[:3]]
        split = _producer(parts[0], "Split")
        if split is None or parts != list(split.outputs) or split.attributes.get_int("axis", 0) not in (-1, 2):
            continue
        node.replace_input_with(0, split.inputs[0])
        node.replace_input_with(1, None)
        node.replace_input_with(2, None)
        count += 1
    common_passes.RemoveUnusedNodesPass()(model)
    return count


# ── 6. CUDA provider ──────────────────────────────────────────────────────────


def _gather_nd(op, data, indices):
    return op.GatherND(data, indices, _allow_other_attributes=True, _outputs=["gathered"])


def _gathers_constant_indices(context, data, indices, gathered, **_) -> bool:
    if gathered.producer().attributes.get_int("batch_dims", 0) != 0 or indices.const_value is None:
        return False
    if data.shape is None or not all(isinstance(dim, int) for dim in data.shape):
        return False
    return 0 < indices.const_value.numpy().shape[-1] <= len(data.shape)


def _flat_gather(op, data, indices, gathered):
    # The same elements, read from the indexed dimensions flattened into one.
    values = indices.const_value.numpy()
    indexed, kept = list(data.shape[: values.shape[-1]]), list(data.shape[values.shape[-1] :])
    flat = np.ravel_multi_index(tuple(np.moveaxis(values % np.array(indexed), -1, 0)), indexed).astype(np.int64)
    rows = op.Reshape(data, op.Constant(value_ints=[int(np.prod(indexed)), *kept]))
    return op.Gather(rows, op.Constant(value=ir.tensor(flat)), axis=0)


# A mask indexed by constant grids (`attention_mask[batch_idx, kv_idx]`) exports as a `GatherND`, whose kernel
# fails CUDA graph replay (microsoft/onnxruntime#33875). A `Gather` reads the same elements.
_CONSTANT_GATHER_ND = pattern.RewriteRule(_gather_nd, _flat_gather, _gathers_constant_indices)


# ── Pipeline ──────────────────────────────────────────────────────────────────


def _merge_equal_computations(model: ir.Model) -> None:
    """Compute once what's computed alike. Equal constants and shapes first become one, so the nodes reading them
    compare equal: the `Shape` nodes reading one run of dims (known, or the same symbol) off different values all read
    it off the first."""
    first = {}
    for node in model.graph:
        if node.op_type != "Shape" or node.domain != "" or (shape := node.inputs[0].shape) is None:
            continue
        dims = shape[node.attributes.get_int("start", 0) : node.attributes.get_int("end", len(shape))]
        if not all(isinstance(dim, int) or dim.value is not None for dim in dims):
            continue
        key = tuple(dim if isinstance(dim, int) else dim.value for dim in dims)
        if key in first:
            _replace(node.outputs[0], first[key])
        else:
            first[key] = node.outputs[0]
    common_passes.DeduplicateInitializersPass()(model)
    common_passes.CommonSubexpressionEliminationPass()(model)
    common_passes.RemoveUnusedNodesPass()(model)


def _ort_fusions(model: ir.Model) -> int:
    _, fused = optimize_for_ort(model)
    return sum(fused.values())


_ANY, _CPU, _CUDA = frozenset({"cpu", "cuda"}), frozenset({"cpu"}), frozenset({"cuda"})

# The pipeline, step by step: `(name, step(model) -> how many times it applied, the providers it's for)`.
_STEPS = (
    # 1. Prepare.
    ("rsqrt_from_pow", _RSQRT_FROM_POW.apply_to_model, _ANY),
    ("key_transposes", _KEY_TRANSPOSE.apply_to_model, _ANY),
    # 2. Rotary embeddings, and attentions sharing a key.
    ("rotary_embeddings", _fuse_rotary_embeddings, _ANY),
    ("shared_key_attentions", _fuse_shared_key_attentions, _ANY),
    # 3. onnxscript's ORT fusions: norms, skip norms, attention, GELU.
    ("ort_fusions", _ort_fusions, _ANY),
    # 4. Repair.
    ("unfused_mixed_dtype_skip_norms", _unfuse_mixed_dtype_skip_norms, _ANY),
    ("untransposed_batch_matmuls", _untranspose_batch_matmuls, _ANY),
    # 5. More fusions.
    ("weightless_rms_norms", _WEIGHTLESS_RMS_NORMS.apply_to_model, _ANY),
    ("narrowed_norms", _narrow_norms, _ANY),
    # Narrowed, the norms whose skip fusion was undone take their residual's dtype: fused again.
    ("narrowed_skip_norms", fuse_skip_rms_normalization, _ANY),
    # CUDA's `GroupQueryAttention`, decode only: with a bias, it runs an unfused kernel, slower for a prefill than
    # `MultiHeadAttention`'s.
    ("group_query_attention", _fuse_group_query_attention, _CPU),
    ("group_query_attention_decode", partial(_fuse_group_query_attention, prefill=False, head_norms=True), _CUDA),
    ("mixture_of_experts", _fuse_mixture_of_experts, _CUDA),
    # ONNX Runtime's CPU `MultiHeadAttention` runs slower than the ops it fused: the ones left, unfused.
    ("unfused_attentions", _unfuse_attentions, _CPU),
    ("head_transposes", _sink_head_transposes, _ANY),
    ("packed_matmuls", _pack_sibling_matmuls, _ANY),
    ("packed_group_query_attentions", _pack_group_query_attention_inputs, _ANY),
    # 6. CUDA provider.
    ("constant_gather_nds", _CONSTANT_GATHER_ND.apply_to_model, _CUDA),
)


def fuse_for_onnxruntime(model: ir.Model, device: str = "cpu", steps: int | None = None) -> dict[str, int]:
    """Fuse `model` in place into ONNX Runtime's own ops, for its `"cpu"` or `"cuda"` provider (with the ops only
    the latter has, `MoE`). Returns how many times each step applied. `steps` runs only the first ones (to measure
    each step's share)."""
    if device == "cuda":
        installed, installed_version = _is_package_available("onnxruntime", return_version=True)
        if installed and version.parse(installed_version) < version.parse("1.28.0"):
            logger.warning(
                f"Graphs fused for ONNX Runtime's CUDA provider need onnxruntime>=1.28.0 (installed: "
                f"{installed_version}): before it, `GroupQueryAttention` has no query and key norms and float16 skip "
                "norms overflow on large activations."
            )
    counts = {name: step(model) if device in providers else 0 for name, step, providers in _STEPS[:steps]}
    # 7. What every layer computes alike (its mask's unattended rows), computed once.
    _merge_equal_computations(model)
    return counts
