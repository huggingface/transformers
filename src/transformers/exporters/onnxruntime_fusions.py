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
"""Fuses an exported ONNX graph into ONNX Runtime's own ops for one of its providers (`OnnxConfig(ort_fusions=...)`).

onnxscript's ORT fusions (`optimize_for_ort`) do most of it: norms, attention, GELU. `_STEPS` runs them, in order,
between the rewrites they need: the graph prepared in the forms they match, the fusions ONNX Runtime rejects undone,
and the fusions they don't cover (rotary embeddings, `GroupQueryAttention` over a cache, `MoE`, packed projections).
Every rewrite computes exactly what it replaces.
"""

from __future__ import annotations

import copy
from functools import partial

import numpy as np
import onnx_ir as ir
from onnx_ir.passes import common as common_passes
from onnxscript.rewriter import _ir_utils, pattern
from onnxscript.rewriter.ort_fusions import optimize_for_ort
from onnxscript.rewriter.ort_fusions.rotary_embedding import RotaryEmbeddingFusion, fuse_rotary_embedding
from onnxscript.rewriter.ort_fusions.sdpa import fuse_sdpa
from onnxscript.rewriter.ort_fusions.skip_normalization import fuse_skip_rms_normalization

from ..utils import logging
from ..utils.import_utils import is_onnxruntime_available


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


def _nodes(model: ir.Model, op_type: str, domain: str | None = None) -> list[ir.Node]:
    """The `op_type` nodes of `model` (in `domain`, if given), subgraphs included, listed up front so a step can edit
    the graph as it goes."""
    return [node for node in model.graph.all_nodes() if node.op_type == op_type and domain in (None, node.domain)]


def _replace_node(node: ir.Node, tape: ir.tape.Tape, output: ir.Value) -> None:
    """Replace `node` by the nodes `tape` recorded, `output` in place of its output."""
    node.graph.insert_before(node, tape.nodes)
    _replace(node.outputs[0], output)
    node.graph.remove(node, safe=True)


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


def _type_loop_sequences(model: ir.Model) -> None:
    """Type the tensor sequences a `Loop` carries as sequences: left untyped, onnx_ir's shape inference, which
    onnxscript's ORT fusions run, types them as tensors (onnx/ir-py#539)."""
    for node in _nodes(model, "Loop"):
        body = node.attributes["body"].value
        for index, initial in enumerate(node.inputs[2:]):
            if initial is None or not isinstance(initial.type, ir.SequenceType):
                continue
            for value in (node.outputs[index], body.inputs[index + 2], body.outputs[index + 1]):
                if value.type is None:
                    value.type = copy.deepcopy(initial.type)


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
    cos_frequencies, sin_frequencies = _frequencies(cos, "Cos"), _frequencies(sin, "Sin")
    if cos_frequencies is None or cos_frequencies != sin_frequencies:
        return False
    if cos.shape is None or _rotary_layout(x, cos.shape) is None:
        return False
    freqs, _ = cos_frequencies
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


def _fuse_rotary_embeddings(model: ir.Model) -> None:
    fuse_rotary_embedding(model)
    _ROTARY_ANY_LAYOUT.apply_to_model(model)
    _ROTARY_CONSTANT_CACHE.apply_to_model(model)
    _ROTARY_TOKEN_CACHE.apply_to_model(model)


def _fuse_shared_key_attentions(model: ir.Model) -> None:
    """Fuse the attentions of layers reading another layer's cached keys (Gemma 4's), before onnxscript's ORT fusions,
    whose optimizer merges the copies made here back: each attention gets its own copy of the key transpose they
    share (torchlib's `Reshape` -> `Transpose` -> `Reshape`, or a 4D `Transpose`), since the attention fusion only
    replaces nodes nothing else reads."""
    graph, copied = model.graph, False
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
            copied = True
    if copied:
        fuse_sdpa(model, apply_shape_inference=True)


# ── 4. Repair ─────────────────────────────────────────────────────────────────


def _unfuse_mixed_dtype_skip_norms(model: ir.Model) -> None:
    """Turn each `SkipSimplifiedLayerNormalization` whose input, skip and scale dtypes differ (a norm computed in
    float32 on a float16 residual), which ONNX Runtime rejects, back into the `Add` and `SimplifiedLayerNormalization`
    it fused (microsoft/onnxscript#3155)."""
    for node in _nodes(model, "SkipSimplifiedLayerNormalization"):
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
        _replace(output, normalized)
        if len(node.outputs) > 3:
            _replace(node.outputs[3], total)
        node.graph.remove(node, safe=True)


def _set_partial_rotary_heads(model: ir.Model) -> None:
    """Set `num_heads` on each `RotaryEmbedding` onnxscript made partial (`rotary_embedding_dim`) without it, which
    ONNX Runtime requires alongside it (microsoft/onnxscript#3185): a 4D input's axis 1."""
    for node in _nodes(model, "RotaryEmbedding", "com.microsoft"):
        if "num_heads" in node.attributes:
            continue
        x = node.inputs[0]
        if "rotary_embedding_dim" not in node.attributes or x.shape is None or len(x.shape) != 4:
            continue
        if (heads := _static_dim(x, 1)) is None:
            continue
        node.attributes["num_heads"] = ir.AttrInt64("num_heads", heads)


def _attention(tape: ir.tape.Tape, query, key_transposed, value, scale: ir.Value, bias) -> ir.Value:
    """`softmax(query @ key_transposed * scale + bias) @ value` over the heads, the attention a fused op stood for."""
    scores = tape.op("Mul", [tape.op("MatMul", [query, key_transposed]), scale])
    if bias is not None:
        scores = tape.op("Add", [scores, bias])
    return tape.op("MatMul", [tape.op("Softmax", [scores], attributes={"axis": -1}), value])


def _unfuse_leftover_attentions(model: ir.Model) -> None:
    """Turn each attention onnxscript fused into its intermediate `SDPA` op but couldn't lower (one whose head count
    isn't static), which ONNX Runtime doesn't have, back into `MatMul`s and a `Softmax` (microsoft/onnxscript#3184)."""
    for index, node in enumerate(_nodes(model, "SDPA", "ai.onnxruntime._fusion")):
        query, key, value = node.inputs[:3]
        mask = node.inputs[3] if len(node.inputs) > 3 else None
        scale = node.attributes.get_float("scale", None)
        if scale is None:
            head_size = _static_dim(query, 3)
            if head_size is None:
                continue
            scale = head_size**-0.5
        graph, tape, output = node.graph, ir.tape.Tape(), node.outputs[0]
        factor = _initializer(graph, f"leftover_attention_{index}_scale", np.array(scale, query.dtype.numpy()))
        # The key comes `[batch, heads, length, head_dim]` (`"BHSd"`) or `[batch, length, heads, head_dim]`.
        perm = [0, 1, 3, 2] if node.attributes.get_string("key_format", "BHSd") == "BHSd" else [0, 2, 3, 1]
        key_transposed = tape.op("Transpose", [key], attributes={"perm": perm})
        attended = _attention(tape, query, key_transposed, value, factor, mask)
        attended.dtype, attended.shape = query.dtype, output.shape
        _replace_node(node, tape, attended)


def _unfuse_attention_activation_biases(model: ir.Model) -> None:
    """Turn each `MultiHeadAttention` bias packed from values that aren't `[D]` biases (a position embedding added to the
    query, DETR's) back into the `Add`s it fused (microsoft/onnxscript#3181)."""
    for node in _nodes(model, "MultiHeadAttention", "com.microsoft"):
        if len(node.inputs) < 4:
            continue
        concat = _producer(node.inputs[3], "Concat")
        if concat is None or len(concat.inputs) != 3:
            continue
        if all(part.shape is not None and len(part.shape) == 1 for part in concat.inputs):
            continue
        for index, part in enumerate(concat.inputs):
            # A bias the fusion filled in with zeros for an input that had none.
            values = _constant(part)
            if values is not None and not values.any():
                continue
            states = node.inputs[index]
            node.replace_input_with(
                index, _add_node(node.graph, node, "Add", [states, part], states.shape, states.dtype)
            )
        node.replace_input_with(3, None)


def _unfuse_group_norms(model: ir.Model) -> None:
    """Turn each `GroupNorm` back into standard ops, the way torchlib exports a group norm: ONNX Runtime has no CPU
    kernel for it, and its CUDA kernel only takes some channel and group counts (microsoft/onnxscript#3182)."""
    for index, node in enumerate(_nodes(model, "GroupNorm", "com.microsoft")):
        x, gamma, beta = node.inputs[:3]
        if x.shape is None or len(x.shape) != 4:
            continue
        graph, tape, groups = node.graph, ir.tape.Tape(), node.attributes.get_int("groups")
        name = f"unfused_group_norm_{index}"
        to_groups = _initializer(graph, f"{name}_groups", np.array([0, groups, -1]))
        ones = _initializer(graph, f"{name}_ones", np.ones(groups, x.dtype.numpy()))
        zeros = _initializer(graph, f"{name}_zeros", np.zeros(groups, x.dtype.numpy()))
        # Channels first, each group's channels and positions on one axis, normalized there.
        channels_first = tape.op("Transpose", [x], attributes={"perm": [0, 3, 1, 2]})
        grouped = tape.op("Reshape", [channels_first, to_groups])
        epsilon = node.attributes.get_float("epsilon", 1e-5)
        normalized = tape.op("InstanceNormalization", [grouped, ones, zeros], attributes={"epsilon": epsilon})
        normalized = tape.op("Reshape", [normalized, tape.op("Shape", [channels_first])])
        normalized = tape.op("Transpose", [normalized], attributes={"perm": [0, 2, 3, 1]})
        scaled = tape.op("Mul", [normalized, tape.op("CastLike", [gamma, x])])
        shifted = tape.op("Add", [scaled, tape.op("CastLike", [beta, x])])
        shifted.dtype, shifted.shape = x.dtype, x.shape
        _replace_node(node, tape, shifted)


def _untranspose_batch_matmuls(model: ir.Model, matmuls: bool = False) -> None:
    """Put back, as a `Transpose`, each batch transpose (`transBatchA` / `transBatchB`) fused into a `FusedMatMul`
    whose inputs ONNX Runtime can't batch-transpose (they must share one rank, at least 3; microsoft/onnxscript#3156).
    With `matmuls`, every `FusedMatMul` goes back to a `MatMul` (WebGPU has none: it would run on the CPU)."""
    for node in _nodes(model, "FusedMatMul"):
        ranks = [len(value.shape) if value.shape is not None else None for value in node.inputs[:2]]
        if None in ranks or (not matmuls and ranks[0] == ranks[1] and ranks[0] >= 3):
            continue
        for index, side in enumerate("AB"):
            batch = node.attributes.get_int(f"transBatch{side}", 0)
            flipped = node.attributes.get_int(f"trans{side}", 0)
            if not batch and not (matmuls and flipped):
                continue
            rank, source = ranks[index], node.inputs[index]
            # onnxscript's definitions: the batch transpose moves the first dimension to the second-to-last
            # position, and flipping the last two too moves it last.
            if batch:
                perm = [*range(1, rank), 0] if flipped else [*range(1, rank - 1), 0, rank - 1]
            else:
                perm = [*range(rank - 2), rank - 1, rank - 2]
            shape = [source.shape[axis] for axis in perm] if source.shape is not None else None
            transposed = _add_node(node.graph, node, "Transpose", [source], shape, source.dtype, perm=perm)
            node.replace_input_with(index, transposed)
            node.attributes.pop(f"transBatch{side}", None)
            node.attributes.pop(f"trans{side}", None)
        if matmuls:
            output = node.outputs[0]
            product = _add_node(node.graph, node, "MatMul", node.inputs[:2], output.shape, output.dtype)
            alpha = node.attributes.get_float("alpha", 1.0)
            if alpha != 1.0:
                scale = _initializer(model.graph, f"{output.name}_alpha", np.array(alpha, output.dtype.numpy()))
                product = _add_node(node.graph, node, "Mul", [product, scale], output.shape, output.dtype)
            _replace(output, product)


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


def _narrow_norms(model: ir.Model) -> None:
    """Run in its input's dtype each `SimplifiedLayerNormalization` whose output is cast straight back to it (a norm
    a float16 model computes in float32): it computes in float32 anyway and returns its scale's dtype, so a constant
    scale that dtype holds exactly goes in narrowed and the cast goes."""
    # Norms sharing a scale share its narrowed copy.
    narrowed = {}
    for node in _nodes(model, "SimplifiedLayerNormalization"):
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


def _cache_update(value: ir.Value):
    """`(present, past, new, start, heads_second)`, the key-value cache update a fused attention's key or value reads, through its
    merge into `[batch, length, heads * head_dim]` and, for grouped-query attention, its repeat over the heads. A
    dynamic cache appends (`Concat(past, new, axis=2)`, no `start`); a static one writes at `start` into its buffer
    (`ScatterND` on its `[length, batch, heads, head_dim]` view); a prefill, with no past, returns the new states as
    they are (no `past`). `heads_second` when the new states are `[batch, length, heads, head_dim]`, not the cache's
    `[batch, heads, length, head_dim]`."""
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
        return concat.outputs[0], concat.inputs[0], concat.inputs[1], None, False
    back = _producer(states, "Transpose")
    scatter = _producer(back.inputs[0], "ScatterND") if back is not None else None
    if scatter is None or _perm(back) != [1, 2, 0, 3] or scatter.attributes.get_string("reduction", "none") != "none":
        # No past: the new states are the present themselves.
        if states.is_graph_output() and states.shape is not None and len(states.shape) == 4:
            return states, None, states, None, False
        return None
    indices = scatter.inputs[1]
    past, new = (_producer(item, "Transpose") for item in (scatter.inputs[0], scatter.inputs[2]))
    if past is None or new is None or _perm(past) != [2, 0, 1, 3]:
        return None
    if indices.shape is None or len(indices.shape) != 2 or _static_dim(indices, 1) != 1:
        return None
    # A key comes rotated, `[batch, heads, length, head_dim]`; a value straight from its projection.
    if _perm(new) == [2, 0, 1, 3]:
        return back.outputs[0], past.inputs[0], new.inputs[0], indices, False
    if _perm(new) == [1, 0, 2, 3]:
        return back.outputs[0], past.inputs[0], new.inputs[0], indices, True
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


def _shared_rotation(query_states: ir.Value, new_key: ir.Value | None, head_size: int, prefill: bool):
    """`(query, key, positions, cos_cache, sin_cache, interleaved)`, the attention's query and new key before
    ONNX Runtime's `RotaryEmbedding` rotated them (`[batch, heads, length, head_dim]`), if one cache and the same
    positions rotate both whole; the positions None when they're the cache's default (0 on, no past)."""
    rotations = (
        _producer(query_states, "RotaryEmbedding"),
        _producer(new_key, "RotaryEmbedding"),
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
        offset = _constant(positions)
        if not prefill or offset is None or offset.tolist() != [0]:
            return None
        positions = None
    return query_rotation.inputs[0], key_rotation.inputs[0], positions, cos_cache, sin_cache, interleaved


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


def _computed_from(value: ir.Value | None, sources: tuple[ir.Value, ...]) -> bool:
    """Whether `value` is computed from any of `sources`."""
    stack, seen = [value], set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        if any(current is source for source in sources):
            return True
        seen.add(id(current))
        if (node := current.producer()) is not None:
            stack.extend(node.inputs)
    return False


def _tape_constant(tape: ir.tape.Tape, prefix: str, values, dtype=np.int64) -> ir.Value:
    return tape.initializer(ir.tensor(np.array(values, dtype=dtype)), name=f"{prefix}_{len(tape.initializers)}")


def _merge_heads(tape: ir.tape.Tape, states: ir.Value, heads_second: bool, merge: ir.Value) -> ir.Value:
    """`states` to `[batch, length, heads * head_dim]`."""
    if not heads_second:
        states = tape.op("Transpose", [states], attributes={"perm": [0, 2, 1, 3]})
    return tape.op("Reshape", [states, merge])


def _sequence_lengths(tape, constant, sized, length_axis, past_key, start, known_total):
    """`(length, seqlens_k, total_length)`, `GroupQueryAttention`'s lengths: each sequence's length after this step
    (`seqlens_k` is its last key, where the new keys and values go), and the total length, a constant when it's known.
    ONNX Runtime's shape inference reads a total computed from shapes as int64 (microsoft/onnxruntime#33865), and its
    CUDA kernel reads one computed on the GPU as 0 (microsoft/onnxruntime#33867)."""
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
    written = tape.op("Cast", [length], attributes={"to": ir.DataType.INT32})
    seqlens_k = tape.op("Expand", [tape.op("Sub", [written, constant([1], np.int32)]), batch])
    if known_total is not None:
        total_length = constant(known_total, np.int32)
    else:
        total_length = tape.op("Squeeze", [written])
    return length, seqlens_k, total_length


def _padded_rotary_caches(tape, constant, caches, total, dtype):
    """The rotary `caches` with zero rows, never read, appended up to `total` rows: `GroupQueryAttention` takes a cache
    of at least the total length, and a per-token cache (`b * length + s`) has `batch * length` (a `Concat`: there's
    no bfloat16 `Pad`, microsoft/onnxruntime#33874)."""
    rows = tape.op("Shape", [caches[0]], attributes={"start": 0, "end": 1})
    missing = tape.op("Max", [tape.op("Sub", [total, rows]), constant([0])])
    rest = tape.op("Shape", [caches[0]], attributes={"start": 1})
    zeros = tape.op("Expand", [constant(0, dtype.numpy()), tape.op("Concat", [missing, rest], attributes={"axis": 0})])
    padded = []
    for cache in caches:
        padded.append(tape.op("Concat", [cache, zeros], attributes={"axis": 0}))
    return padded


def _fuse_group_query_attention(
    model: ir.Model, prefill: bool = True, head_norms: bool = False, rotary: bool = True
) -> None:
    """Turn each fused `MultiHeadAttention` reading a key-value cache's update into ONNX Runtime's
    `GroupQueryAttention`, which updates the cache itself (or, for a prefill, starts it) and reads its key-value heads
    without repeating them. Its mask stays the attention's own bias, so every row's `seqlens_k` is the whole length:
    exact whatever the padding. `prefill` sets whether a prefill, starting the cache, is fused too; `head_norms`
    whether per-head query and key norms are folded in too (CUDA only); `rotary` whether the rotary embedding is
    (WebGPU's reads its positions off `seqlens_k`, not `position_ids`)."""
    graph = model.graph
    # A cache update several attentions read (layers sharing another layer's cache, Gemma 4's): its own layer's,
    # fused once; the others, reading it, stay as they are.
    readers = {}
    for attention in _nodes(model, "MultiHeadAttention"):
        if len(attention.inputs) != 6:
            continue
        for value in attention.inputs[1:3]:
            if (update := _cache_update(value)) is not None:
                present = update[0]
                readers[present] = readers.get(present, 0) + 1
    for index, attention in enumerate(_nodes(model, "MultiHeadAttention")):
        if len(attention.inputs) != 6:
            continue
        query, key, value, projection_bias, padding_mask, bias = attention.inputs
        updates = _cache_update(key), _cache_update(value)
        if projection_bias is not None or padding_mask is not None or None in updates:
            continue
        (key_present, past_key, new_key, start, key_heads_second) = updates[0]
        (value_present, past_value, new_value, _, value_heads_second) = updates[1]
        if readers[key_present] > 1 or readers[value_present] > 1:
            continue
        cached = [states for states in (past_key, None if key_heads_second else new_key) if states is not None]
        kv_heads = next((heads for states in cached if (heads := _static_dim(states, 1)) is not None), None)
        # Head sizes in multiples of 8, and self-attention only: the new key as long as the query.
        head_size = _static_dim(cached[0], 3)
        length_axis = 1 if key_heads_second else 2
        new_length = new_key.shape[length_axis] if new_key.shape is not None else None
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
        # A bias computed from the updated cache (Doge's, from its values) would read the fused attention's own output.
        if past_key is not None and _computed_from(bias, (key_present, value_present)):
            continue
        tape = ir.tape.Tape(graph)
        constant = partial(_tape_constant, tape, f"group_query_attention_{index}")
        merge = constant([0, 0, -1])

        # A static cache attends over its whole buffer, the bias masking the slots not written yet; appended to a
        # past, it's the past and the new tokens.
        past_length = 0 if past_key is None else _static_dim(past_key, 2)
        known_total = None
        if start is not None:
            known_total = past_length
        elif isinstance(query_length, int) and past_length is not None:
            known_total = query_length + past_length
        rotary_inputs, dtype, attributes = [None, None, None], query.dtype, {}
        rotation = None
        if rotary and not key_heads_second:
            rotation = _shared_rotation(query_states, new_key, head_size, past_key is None)
        if rotation is not None:
            # Rotated by the attention itself, which stores the key rotated.
            query_states, new_key, positions, cos_cache, sin_cache, interleaved = rotation
            rotary_inputs = [cos_cache, sin_cache, positions]
            attributes = {"do_rotary": 1, "rotary_interleaved": interleaved}
        if rotation is not None or groups is not None:
            query = _merge_heads(tape, query_states, False, merge)
        # The lengths are read off the query's heads, which keep their shape, rather than off their merge built here.
        sized, length_axis = query_states, 2
        key_states, norms = _merge_heads(tape, new_key, key_heads_second, merge), []
        # CUDA's kernel takes half precision only: a float32 one runs on the CPU, which has no norms.
        half = dtype in (ir.DataType.FLOAT16, ir.DataType.BFLOAT16)
        if head_norms and half and not key_heads_second:
            found = _head_norm(query_states), _head_norm(new_key)
            if None not in found and found[0][2] == found[1][2]:
                # Normalized by the attention itself, before its rotary.
                (query, query_norm, epsilon), (key_states, key_norm, _) = found
                norms = [None, None, None]
                for weight in (query_norm, key_norm):
                    norms.append(constant(_constant(weight).astype(dtype.numpy()), dtype.numpy()))
                attributes["qk_norm_epsilon"] = epsilon
                sized, length_axis = query, 1
        length, seqlens_k, total_length = _sequence_lengths(
            tape, constant, sized, length_axis, past_key, start, known_total
        )
        if rotary_inputs[2] is not None:
            total = constant([known_total]) if known_total is not None else length
            rotary_inputs[:2] = _padded_rotary_caches(tape, constant, rotary_inputs[:2], total, dtype)
        attributes.update(num_heads=num_heads, kv_num_heads=kv_heads)
        if "scale" in attention.attributes:
            attributes["scale"] = attention.attributes["scale"].as_float()
        value_states = _merge_heads(tape, new_value, value_heads_second, merge)
        outputs = tape.op_multi_out(
            "GroupQueryAttention",
            [
                query,
                key_states,
                value_states,
                past_key,
                past_value,
                seqlens_k,
                total_length,
                *rotary_inputs,
                bias,
                *norms,
            ],
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


def _fuse_mixture_of_experts(model: ir.Model) -> None:
    """Turn each mixture-of-experts block (the `batched_mm` experts implementation behind a softmax / top-k router)
    into ONNX Runtime's CUDA `MoE`, which routes the tokens itself from the router's logits.

    Not for the CPU: ONNX Runtime's CPU `MoE` returns wrong values when its input comes from another node and its
    output feeds one (microsoft/onnxruntime#33866)."""
    graph = model.graph
    for index, combine in enumerate(_nodes(model, "ReduceSum")):
        # The routed rows' outputs summed per token, `ReduceSum(Reshape(weighted), axes=[1])`; a float16 model sums
        # them in float32.
        if combine.attributes.get_int("keepdims", 1) != 0 or len(combine.inputs) < 2:
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
        else:
            continue
        gated_rows, fc2, down_biases, down_ids = down
        if (gate := _gate(gated_rows)) is None:
            continue
        projected_rows, gate_first = gate
        if (up := _batched_matmul(projected_rows)) is None:
            continue
        repeated_rows, fc1, up_biases, up_ids = up
        # Each token repeated once per routed expert, `Reshape(Tile(Unsqueeze(tokens, 1)))`.
        repeated = _producer(repeated_rows, "Reshape")
        tile = _producer(repeated.inputs[0], "Tile") if repeated is not None else None
        tokens = _producer(tile.inputs[0], "Unsqueeze") if tile is not None else None
        # Each routed row's weight, the flattened routing weights.
        unsqueeze = _producer(_uncast(scale), "Unsqueeze")
        flat = _producer(_uncast(unsqueeze.inputs[0]), "Reshape") if unsqueeze is not None else None
        router = _router(flat.inputs[0]) if flat is not None else None
        if tokens is None or router is None or up_ids is not down_ids:
            continue
        logits, k, normalize, indices = router
        ids = _producer(up_ids, "Unsqueeze")
        flat_ids = _producer(ids.inputs[0], "Reshape") if ids is not None else None
        if flat_ids is None or flat_ids.inputs[0] is not indices:
            continue
        tokens = tokens.inputs[0]
        if logits.shape is None or _static_dim(logits, len(logits.shape) - 1) != fc1.shape[0]:
            continue
        if logits.dtype != tokens.dtype:
            continue
        half = fc1.shape[1] // 2
        gate_rows, up_rows = (slice(0, half), slice(half, None)) if gate_first else (slice(half, None), slice(0, half))
        biases = (up_biases[:, gate_rows], up_biases[:, up_rows]) if up_biases is not None else (None, None)
        weights = []
        for slot, array in enumerate([fc1[:, gate_rows], biases[0], fc2, down_biases, fc1[:, up_rows], biases[1]]):
            name = f"mixture_of_experts_{index}_{slot}"
            weights.append(_initializer(graph, name, array) if array is not None else None)
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


def _unfuse_attentions(model: ir.Model) -> None:
    """Turn each `MultiHeadAttention` over `[batch, length, heads * head_dim]` queries, keys and values with an
    attention bias and no cache (an encoder's, or one `GroupQueryAttention` didn't take) back into `MatMul`s and a
    `Softmax`: ONNX Runtime's CPU kernel for it runs slower than the ops it fused (microsoft/onnxruntime#33873)."""
    for index, node in enumerate(_nodes(model, "MultiHeadAttention")):
        if len(node.inputs) != 6 or len(node.outputs) != 1:
            continue
        # One `GroupQueryAttention` replaced, left for the clean-up to remove.
        if not node.outputs[0].uses() and not node.outputs[0].is_graph_output():
            continue
        query, key, value, projection_bias, padding_mask, bias = node.inputs
        if projection_bias is not None or padding_mask is not None or bias is None:
            continue
        if any(item is None or item.shape is None or len(item.shape) != 3 for item in (query, key, value)):
            continue
        if (hidden_size := _static_dim(query, 2)) is None:
            continue
        heads = node.attributes.get_int("num_heads")
        scale = node.attributes.get_float("scale", (hidden_size // heads) ** -0.5)
        graph, tape = node.graph, ir.tape.Tape()
        split_heads = _initializer(graph, f"unfused_attention_{index}_heads", np.array([0, 0, heads, -1]))
        merge_heads = _initializer(graph, f"unfused_attention_{index}_merge", np.array([0, 0, -1]))
        factor = _initializer(graph, f"unfused_attention_{index}_scale", np.array(scale, dtype=query.dtype.numpy()))
        # Split into heads, the key transposed for the product.
        split = []
        for states, perm in ((query, [0, 2, 1, 3]), (key, [0, 2, 3, 1]), (value, [0, 2, 1, 3])):
            split.append(tape.op("Transpose", [tape.op("Reshape", [states, split_heads])], attributes={"perm": perm}))
        context = tape.op("Transpose", [_attention(tape, *split, factor, bias)], attributes={"perm": [0, 2, 1, 3]})
        output = tape.op("Reshape", [context, merge_heads])
        output.dtype = query.dtype
        _replace_node(node, tape, output)


def _swap_heads(graph: ir.Graph, before: ir.Node, x: ir.Value) -> ir.Value:
    """`x` transposed to the other layout."""
    shape = [x.shape[0], x.shape[2], x.shape[1], x.shape[3]] if x.shape is not None else None
    return _add_node(graph, before, "Transpose", [x], shape, x.dtype, perm=[0, 2, 1, 3])


def _replace_by_swapped_heads(node: ir.Node, x: ir.Value) -> None:
    """Every use of `node`'s output reads `x` transposed to the other layout instead, which has that output's shape."""
    output = node.outputs[0]
    _replace(output, _add_node(node.graph, node, "Transpose", [x], output.shape, output.dtype, perm=[0, 2, 1, 3]))


def _sink_head_transposes(model: ir.Model) -> None:
    """Move the transposes between `[batch, length, heads, head_dim]` and `[batch, heads, length, head_dim]` past the
    ops that read either layout alike, the rotary embeddings, the repeat of the key-value heads and the zeroing of
    unattended rows, until they meet the transpose back and cancel: `MultiHeadAttention` reads and writes the first
    layout, so each attention copied its query, keys, values and output there and back."""
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
            _replace_by_swapped_heads(node, rotated)
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
            _replace_by_swapped_heads(node, states)
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
            _replace_by_swapped_heads(node, zeroed)
    for node in list(graph):
        # A swap back, read through a reshape to the same shape or not, is the states before the first.
        inner = node.inputs[0] if node.op_type == "Transpose" and _perm(node) == [0, 2, 1, 3] else None
        reshape = _producer(inner, "Reshape")
        if reshape is not None and inner.shape is not None and reshape.inputs[0].shape == inner.shape:
            inner = reshape.inputs[0]
        if (x := _swapped_heads(inner)) is not None:
            _replace(node.outputs[0], x)
    common_passes.RemoveUnusedNodesPass()(model)


def _pack_sibling_matmuls(model: ir.Model) -> None:
    """Turn the `MatMul`s reading one input with constant `[in, out]` weights (an attention's query, key and value
    projections; an MLP's gate and up ones) into one, their weights concatenated, its output split back: one kernel
    for several."""
    graph = model.graph
    values = list(graph.inputs)
    for node in graph.all_nodes():
        values.extend(node.outputs)
    for position, value in enumerate(values):
        siblings, weights = [], []
        for user, index in value.uses():
            weight = _constant(user.inputs[1]) if user.op_type == "MatMul" and index == 0 else None
            if weight is not None:
                siblings.append(user)
                weights.append(weight)
        if len(siblings) < 2 or len({(weight.ndim, weight.shape[0], weight.dtype) for weight in weights}) != 1:
            continue
        if weights[0].ndim != 2 or len({sibling.graph for sibling in siblings}) != 1:
            continue
        packed = _initializer(graph, f"packed_matmul_{position}", np.concatenate(weights, axis=1))
        sizes = _initializer(
            graph, f"packed_matmul_{position}_sizes", np.array([w.shape[1] for w in weights], np.int64)
        )
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


def _pack_group_query_attention_inputs(model: ir.Model) -> None:
    """Hand each `GroupQueryAttention` whose query, key and value are the split of one packed projection that
    projection instead, which it reads packed: no split."""
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
    common_passes.RemoveUnusedNodesPass()(model)


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


# The pipeline, step by step: `(step(model), the providers it's for, or None for all)`.
_STEPS = (
    # 1. Prepare.
    (_type_loop_sequences, None),
    (_RSQRT_FROM_POW.apply_to_model, None),
    (_KEY_TRANSPOSE.apply_to_model, None),
    # 2. Rotary embeddings, and attentions sharing a key.
    (_fuse_rotary_embeddings, None),
    (_fuse_shared_key_attentions, None),
    # 3. onnxscript's ORT fusions: norms, skip norms, attention, GELU.
    (optimize_for_ort, None),
    # 4. Repair.
    (_unfuse_mixed_dtype_skip_norms, None),
    (_set_partial_rotary_heads, None),
    (_unfuse_leftover_attentions, None),
    (_unfuse_attention_activation_biases, None),
    (_unfuse_group_norms, None),
    (_untranspose_batch_matmuls, ("cuda", "dml")),
    (partial(_untranspose_batch_matmuls, matmuls=True), ("webgpu",)),
    # 5. More fusions.
    (_WEIGHTLESS_RMS_NORMS.apply_to_model, None),
    (_narrow_norms, None),
    # Narrowed, the norms whose skip fusion was undone take their residual's dtype: fused again.
    (fuse_skip_rms_normalization, None),
    # CUDA's `GroupQueryAttention`, decode only: with a bias, it runs an unfused kernel, slower for a prefill than
    # `MultiHeadAttention`'s (microsoft/onnxruntime#33869).
    (_fuse_group_query_attention, ("cpu",)),
    (partial(_fuse_group_query_attention, prefill=False, head_norms=True), ("cuda",)),
    # WebGPU's rotates at the positions after the past, ignoring `position_ids`: rotary stays its own op.
    (partial(_fuse_group_query_attention, rotary=False), ("webgpu",)),
    # DirectML's ignores rotary and `attention_bias` and needs unpacked q/k/v: it keeps `MultiHeadAttention`.
    (_fuse_mixture_of_experts, ("cuda",)),
    # ONNX Runtime's CPU `MultiHeadAttention` runs slower than the ops it fused (microsoft/onnxruntime#33873): the ones
    # left, unfused.
    (_unfuse_attentions, ("cpu",)),
    (_sink_head_transposes, None),
    (_pack_sibling_matmuls, None),
    (_pack_group_query_attention_inputs, None),
    # 6. CUDA provider.
    (_CONSTANT_GATHER_ND.apply_to_model, ("cuda",)),
    # What every layer computes alike (its mask's unattended rows), computed once.
    (_merge_equal_computations, None),
)


def fuse_for_onnxruntime(model: ir.Model, ort_fusions: str) -> None:
    """Fuse `model` in place into ONNX Runtime's own ops for the provider it runs on, `ort_fusions` (`"cpu"`, `"cuda"`,
    `"webgpu"` or `"dml"`)."""
    if ort_fusions == "cuda" and not is_onnxruntime_available("1.28.0"):
        logger.warning(
            "Graphs fused for ONNX Runtime's CUDA provider need onnxruntime>=1.28.0: before it, `GroupQueryAttention` "
            "has no query and key norms and float16 skip norms overflow on large activations."
        )
    for step, providers in _STEPS:
        if providers is None or ort_fusions in providers:
            step(model)
