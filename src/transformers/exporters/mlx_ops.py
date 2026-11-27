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
"""MLX programs: an exported ATen graph and its weights, run by rebuilding each op as the MLX op computing it. The
exporter checks its graphs against `MLX_OPS` and the runner runs them."""

from __future__ import annotations

import contextlib
import functools
import itertools
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..utils.import_utils import is_mlx_available, is_torch_available


if is_torch_available():
    import torch

if is_mlx_available():
    import mlx.core as mx


@dataclass
class MLXProgram:
    """An MLX export: its graph (`{"inputs", "nodes", "outputs"}`, JSON) and the weights its nodes read."""

    graph: dict[str, Any]
    weights: dict[str, mx.array]


# ── Values ──────────────────────────────────────────────────────────────────
# A node's argument in the graph's JSON: `{"__node__": name}` for another node's value, `{"__dtype__": name}` for a dtype,
# lists, and plain scalars.


def mlx_dtype(name: str):
    return mx.bool_ if name == "bool" else getattr(mx, name)


def _decode(value, values: dict[str, Any]):
    if isinstance(value, dict):
        if "__node__" in value:
            return values[value["__node__"]]
        if "__dtype__" in value:
            return mlx_dtype(value["__dtype__"])
        return {key: _decode(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, values) for item in value]
    return value


def run_graph(graph: dict[str, Any], weights: dict[str, mx.array], arrays) -> tuple:
    """The graph's outputs for its tensor inputs `arrays`, as MLX's lazy arrays."""
    values = {**weights, **graph["constants"], **dict(zip(graph["inputs"], arrays))}
    for node in graph["nodes"]:
        args, kwargs = _decode(node["args"], values), _decode(node["kwargs"], values)
        values[node["name"]] = MLX_OPS[node["target"]](*args, **kwargs)
    return tuple(values[name] for name in graph["outputs"])


# ── Ops ─────────────────────────────────────────────────────────────────────
# Each takes the ATen op's arguments, with MLX arrays for its tensors and MLX dtypes, and returns what it returns.


def _astype(x, dtype=None):
    return x if dtype is None else x.astype(dtype)


def _axes(dim) -> list[int] | None:
    """Torch's `dim` argument of a reduction (`None` or empty for all axes) as MLX's `axis`."""
    if dim is None or (isinstance(dim, list) and len(dim) == 0):
        return None
    return dim if isinstance(dim, list) else [dim]


def _scalar_dtype(value):
    if isinstance(value, bool):
        return mx.bool_
    return mx.int64 if isinstance(value, int) else mx.float32


def _arange(start, end=None, step=1, dtype=None, **_):
    if end is None:
        start, end = 0, start
    if dtype is None:
        dtype = mx.int64 if all(isinstance(v, int) for v in (start, end, step)) else mx.float32
    return mx.arange(start, end, step, dtype=dtype)


def _expand(x, size, implicit=False):
    offset = len(size) - x.ndim
    return mx.broadcast_to(x, [x.shape[i - offset] if s == -1 else s for i, s in enumerate(size)])


def _slice(x, dim=0, start=None, end=None, step=1):
    index = [slice(None)] * x.ndim
    index[dim] = slice(start, None if end is not None and end >= x.shape[dim] else end, step)
    return x[tuple(index)]


def _split(x, split_size, dim=0):
    return mx.split(x, list(range(split_size, x.shape[dim], split_size)), axis=dim)


def _split_with_sizes(x, split_sizes, dim=0):
    return mx.split(x, list(itertools.accumulate(split_sizes))[:-1], axis=dim)


def _cat(tensors, dim=0):
    # Torch skips the legacy empty 1D tensors (an empty `DynamicCache` layer's keys)
    kept = [t for t in tensors if not (t.ndim == 1 and t.shape[0] == 0)]
    return mx.concatenate(kept, axis=dim) if len(kept) > 1 else kept[0]


def _index(x, indices):
    return x[tuple(slice(None) if i is None else i for i in indices)]


def _index_put(x, indices, values, accumulate=False):
    index = tuple(slice(None) if i is None else i for i in indices)
    # MLX's GPU scatter takes no 64-bit integers (positions, ids): those scatter on the CPU, which shares the memory
    with mx.stream(mx.cpu) if x.dtype in (mx.int64, mx.uint64) else contextlib.nullcontext():
        if accumulate:
            return x.at[index].add(values)
        # A copy: other nodes may still read `x`, which `__setitem__` would rewrite under them
        out = mx.array(x)
        out[index] = values
        return out


def _squeeze(x, dims=None):
    dims = range(x.ndim) if dims is None else [dims] if isinstance(dims, int) else dims
    dims = [d % x.ndim for d in dims if x.shape[d] == 1]
    return mx.squeeze(x, dims) if dims else x


def _constant_pad_nd(x, pad, value=0):
    widths = [(0, 0)] * x.ndim
    for i in range(len(pad) // 2):
        widths[x.ndim - 1 - i] = (pad[2 * i], pad[2 * i + 1])
    return mx.pad(x, widths, constant_values=value)


def _gelu(x, approximate="none"):
    if approximate == "tanh":
        return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    return 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))


def _div(a, b, rounding_mode=None):
    if rounding_mode == "floor":
        return mx.floor_divide(a, b)
    quotient = mx.divide(a, b)
    if rounding_mode == "trunc":
        quotient = mx.where(quotient < 0, mx.ceil(quotient), mx.floor(quotient))
        integer = isinstance(a, mx.array) and mx.issubdtype(a.dtype, mx.integer)
        return quotient.astype(a.dtype) if integer else quotient
    return quotient


def _max_dim(x, dim, keepdim=False):
    return mx.max(x, axis=dim, keepdims=keepdim), mx.argmax(x, axis=dim, keepdims=keepdim).astype(mx.int64)


def _min_dim(x, dim, keepdim=False):
    return mx.min(x, axis=dim, keepdims=keepdim), mx.argmin(x, axis=dim, keepdims=keepdim).astype(mx.int64)


def _sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    mask = attn_mask
    if is_causal:
        # Torch aligns the causal mask top-left, MLX's `"causal"` bottom-right: they agree on a square one
        q_len, kv_len = q.shape[-2], k.shape[-2]
        mask = "causal" if q_len == kv_len else mx.tril(mx.ones((q_len, kv_len), dtype=mx.bool_))
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def rms_norm(x, weight, eps, dtype):
    """`mx.fast.rms_norm`, which `mlx_fusions` writes in place of the ops of an RMS norm."""
    return mx.fast.rms_norm(x, weight.astype(x.dtype), eps).astype(dtype)


def swiglu(gate, up):
    """`silu(gate) * up`, which `mlx_fusions` writes in place of its three ops."""
    return gate * mx.sigmoid(gate) * up


def _native_layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    axes = list(range(-len(normalized_shape), 0))
    mean = mx.mean(x, axis=axes, keepdims=True)
    rstd = mx.rsqrt(mx.var(x, axis=axes, keepdims=True) + eps)
    if len(normalized_shape) == 1:
        out = mx.fast.layer_norm(x, weight, bias, eps)
    else:
        out = (x - mean) * rstd
        out = out if weight is None else out * weight
        out = out if bias is None else out + bias
    # The mean and inverse deviation, which MLX only computes if read (backward graphs)
    return out, mean, rstd


def _linear(x, weight, bias=None):
    # A matmul on the transposed view, rather than the permute and `addmm` the decomposition writes
    out = x @ weight.T
    return out if bias is None else out + bias


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True):
    return _native_layer_norm(x, normalized_shape, weight, bias, eps)[0]


def _rms_norm(x, normalized_shape, weight=None, eps=None):
    eps = mx.finfo(x.dtype).eps if eps is None else eps
    if len(normalized_shape) == 1:
        return mx.fast.rms_norm(x, weight, eps)
    axes = list(range(-len(normalized_shape), 0))
    out = x * mx.rsqrt(mx.mean(mx.square(x), axis=axes, keepdims=True) + eps)
    return out if weight is None else out * weight


def slice_write(x, values, start, axis):
    """`x` with `values` written from `start` on `axis`: `mx.slice_update`, which `mlx_fusions` writes in place of an
    `index_put` at consecutive positions (a static cache write). One position (a decode step) is one scatter, where a
    `slice_update` at a position read from an array takes two kernels."""
    if values.shape[axis] == 1:
        index = mx.broadcast_to(start.reshape([1] * x.ndim), values.shape)
        return mx.put_along_axis(x, index, values, axis=axis)
    return mx.slice_update(x, values, start.reshape(1), (axis,))


def rope(x, positions, inv_freq, scaling, base=None):
    """`mx.fast.rope`, which `mlx_fusions` writes in place of a rotary embedding's ops: on MLX's own frequencies when
    the embedding's are the standard `base ** (-2i / d)` (`base`), on its `inv_freq` otherwise. Each token is rotated
    at its own position: exact for any positions, padded or packed ones included."""
    batch, heads, length, head_dim = x.shape
    frequencies = {"base": base, "freqs": None if base is not None else 1 / inv_freq.astype(mx.float32)}
    offsets = mx.broadcast_to(positions, (batch, length)).reshape(-1).astype(mx.int32)
    if length == 1:
        # One token per row, at the row's offset
        out = mx.fast.rope(x, head_dim, traditional=False, scale=1.0, offset=offsets, **frequencies)
    else:
        rows = x.transpose(0, 2, 1, 3).reshape(batch * length, heads, 1, head_dim)
        rows = mx.fast.rope(rows, head_dim, traditional=False, scale=1.0, offset=offsets, **frequencies)
        out = rows.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
    return out if scaling == 1.0 else out * scaling


def _convolution(x, weight, bias, stride, padding, dilation, transposed, output_padding, groups):
    # MLX convolves channels-last, with `(out_channels, *kernel, in_channels / groups)` weights
    spatial = x.ndim - 2
    x = mx.moveaxis(x, 1, -1)
    if transposed:
        # A transposed convolution is a convolution of the stride-dilated input with the flipped kernel
        in_channels, group_out_channels, *kernel = weight.shape
        weight = weight.reshape(groups, in_channels // groups, group_out_channels, *kernel)
        weight = mx.moveaxis(weight, 1, -1).reshape(groups * group_out_channels, *kernel, in_channels // groups)
        low = [d * (k - 1) - p for d, k, p in zip(dilation, kernel, padding)]
        high = [lo + extra for lo, extra in zip(low, output_padding)]
        options = {"padding": (low, high), "kernel_dilation": dilation, "input_dilation": stride, "flip": True}
    else:
        weight = mx.moveaxis(weight, 1, -1)
        options = {"stride": stride, "padding": padding, "kernel_dilation": dilation}
    if groups == 1 or spatial < 3:
        out = mx.conv_general(x, weight, groups=groups, **options)
    else:
        # MLX groups only 1D and 2D convolutions
        pairs = zip(mx.split(x, groups, axis=-1), mx.split(weight, groups, axis=0))
        out = mx.concatenate([mx.conv_general(xs, ws, **options) for xs, ws in pairs], axis=-1)
    out = mx.moveaxis(out, -1, 1)
    return out if bias is None else out + bias.reshape(-1, *[1] * spatial)


def _pool_arguments(dims, kernel_size, stride, padding, dilation=1):
    def per_axis(value):
        return [value] * dims if isinstance(value, int) else list(value) * (dims if len(value) == 1 else 1)

    return per_axis(kernel_size), per_axis(stride or kernel_size), per_axis(padding), per_axis(dilation)


def _pool_padding(sizes, kernel, stride, padding, dilation, ceil_mode) -> list[tuple[int, int]]:
    """Each axis' `(low, high)` padding: `padding`, and on the high side what a `ceil_mode` window reads past it."""
    widths = []
    for size, k, s, p, d in zip(sizes, kernel, stride, padding, dilation):
        span = size + 2 * p - d * (k - 1) - 1
        windows = (-(-span // s) if ceil_mode else span // s) + 1
        # The last window starts inside the input or its low padding
        if ceil_mode and (windows - 1) * s >= size + p:
            windows -= 1
        widths.append((p, p + max(0, (windows - 1) * s + d * (k - 1) + 1 - (size + 2 * p))))
    return widths


def _pool_windows(x, kernel, stride, dilation):
    """The windows over `x`'s last `len(kernel)` axes, as `(*leading, *windows, *kernel)` (a strided view)."""
    dims = len(kernel)
    leading, sizes = x.shape[: x.ndim - dims], x.shape[x.ndim - dims :]
    strides = [1] * x.ndim
    for axis in range(x.ndim - 2, -1, -1):
        strides[axis] = strides[axis + 1] * x.shape[axis + 1]
    spatial = strides[x.ndim - dims :]
    windows = [(n - d * (k - 1) - 1) // s + 1 for n, k, s, d in zip(sizes, kernel, stride, dilation)]
    return mx.as_strided(
        x,
        (*leading, *windows, *kernel),
        (
            *strides[: x.ndim - dims],
            *[a * s for a, s in zip(spatial, stride)],
            *[a * d for a, d in zip(spatial, dilation)],
        ),
    )


def _avg_pool(
    dims, x, kernel_size, stride=(), padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None
):
    kernel, stride, padding, dilation = _pool_arguments(dims, kernel_size, stride, padding)
    widths = _pool_padding(x.shape[-dims:], kernel, stride, padding, dilation, ceil_mode)
    kernel_axes = tuple(range(-dims, 0))
    padded = mx.pad(x, [(0, 0)] * (x.ndim - dims) + widths)
    total = _pool_windows(padded, kernel, stride, dilation).sum(axis=kernel_axes)
    if divisor_override:
        return (total / divisor_override).astype(x.dtype)
    # Each window's divisor: the elements it covers of the input, and of its padding if counted, never of the high
    # side a `ceil_mode` window reads past it
    covered = mx.ones(x.shape[-dims:], dtype=mx.float32)
    if count_include_pad:
        covered = mx.pad(covered, [(p, p) for p in padding], constant_values=1)
        covered = mx.pad(covered, [(0, high - p) for (_, high), p in zip(widths, padding)])
    else:
        covered = mx.pad(covered, widths)
    count = _pool_windows(covered, kernel, stride, dilation).sum(axis=kernel_axes)
    return (total / count).astype(x.dtype)


def _max_pool_with_indices(dims, x, kernel_size, stride=(), padding=0, dilation=1, ceil_mode=False):
    kernel, stride, padding, dilation = _pool_arguments(dims, kernel_size, stride, padding, dilation)
    sizes = x.shape[-dims:]
    widths = _pool_padding(sizes, kernel, stride, padding, dilation, ceil_mode)
    padded = mx.pad(x, [(0, 0)] * (x.ndim - dims) + widths, constant_values=-float("inf"))
    windows = _pool_windows(padded, kernel, stride, dilation)
    windows = windows.reshape(*windows.shape[: x.ndim], -1)
    argmax = mx.argmax(windows, axis=-1, keepdims=True)
    # Indices into each input plane, as torch returns them (MLX computes them only if read)
    positions = mx.pad(mx.arange(math.prod(sizes)).reshape(sizes), widths, constant_values=-1)
    positions = _pool_windows(positions, kernel, stride, dilation)
    positions = mx.broadcast_to(positions.reshape(*positions.shape[:dims], -1), windows.shape)
    indices = mx.take_along_axis(positions, argmax, axis=-1).squeeze(-1).astype(mx.int64)
    return mx.max(windows, axis=-1), indices


def _vector_norm(x, ord=2, dim=None, keepdim=False, dtype=None):
    x = mx.abs(_astype(x, dtype))
    axes = _axes(dim)
    if ord == float("inf"):
        return mx.max(x, axis=axes, keepdims=keepdim)
    if ord == -float("inf"):
        return mx.min(x, axis=axes, keepdims=keepdim)
    if ord == 0:
        return mx.sum(x != 0, axis=axes, keepdims=keepdim).astype(x.dtype)
    return mx.power(mx.sum(mx.power(x, ord), axis=axes, keepdims=keepdim), 1 / ord)


def _topk(x, k, dim=-1, largest=True, sorted=True):
    indices = mx.argsort(-x if largest else x, axis=dim)
    indices = mx.take(indices, mx.arange(k), axis=dim)
    return mx.take_along_axis(x, indices, axis=dim), indices.astype(mx.int64)


def _sort(x, dim=-1, descending=False, stable=None):
    indices = mx.argsort(-x if descending else x, axis=dim)
    return mx.take_along_axis(x, indices, axis=dim), indices.astype(mx.int64)


def _histc(x, bins=100, min=0, max=0):
    # Each value's bin over `[min, max]`, the top edge in the last bin and values outside it in none
    low, high = (mx.min(x), mx.max(x)) if min == max else (min, max)
    index = mx.floor((x - low) * bins / (high - low)).astype(mx.int32)
    index = mx.where(x == high, bins - 1, index)
    inside = (x >= low) & (x <= high)
    hits = (index.reshape(-1, 1) == mx.arange(bins)) & inside.reshape(-1, 1)
    return mx.sum(hits, axis=0).astype(x.dtype)


def _grouped_mm(x, weight, offs=None, bias=None, out_dtype=None):
    """`(S, in) @ weight[expert]` for rows sorted by expert, `offs` their cumulative counts: one `mx.gather_mm`, rows
    past the last count (dropped by the router) zero."""
    experts = weight.shape[0]
    expert = mx.sum(mx.arange(x.shape[0])[:, None] >= offs[None, :], axis=-1)
    rows = mx.gather_mm(
        x[:, None, :], weight, rhs_indices=mx.minimum(expert, experts - 1).astype(mx.uint32), sorted_indices=True
    ).squeeze(1)
    rows = rows if bias is None else rows + bias[mx.minimum(expert, experts - 1)]
    return _astype(mx.where((expert < experts)[:, None], rows, 0), out_dtype or x.dtype)


def _identity(x, *_, **__):
    return x


def _ignored(*_, **__):
    return None


MLX_OPS: dict[str, Callable] = {}
if is_mlx_available():
    MLX_OPS = {
        # The graph's own arithmetic on sizes, and items of a multi-output op
        "operator.add": operator.add,
        "operator.sub": operator.sub,
        "operator.mul": operator.mul,
        "operator.floordiv": operator.floordiv,
        "operator.truediv": operator.truediv,
        "operator.mod": operator.mod,
        "operator.neg": operator.neg,
        "operator.getitem": operator.getitem,
        "torch.sym_max": max,
        "torch.sym_min": min,
        "operator.eq": operator.eq,
        "operator.ne": operator.ne,
        "operator.lt": operator.lt,
        "operator.le": operator.le,
        "operator.gt": operator.gt,
        "operator.ge": operator.ge,
        "torch.sym_ite": lambda condition, true, false: true if condition else false,
        # A value read on the host (`.item()`): the runner runs such a graph without `mx.compile`
        "aten._local_scalar_dense.default": lambda x: x.item(),
        "aten._is_all_true.default": mx.all,
        # Values and views
        "aten._assert_tensor_metadata.default": _ignored,
        "aten._assert_scalar.default": _ignored,
        "aten.sym_constrain_range_for_size.default": _ignored,
        "aten.alias.default": _identity,
        "aten.clone.default": _identity,
        "aten.lift_fresh_copy.default": _identity,
        "aten.detach.default": _identity,
        "aten._to_copy.default": lambda x, dtype=None, **_: _astype(x, dtype),
        "aten.sym_size.int": lambda x, dim: x.shape[dim],
        "aten.scalar_tensor.default": lambda value, dtype=None, **_: mx.array(value, dtype=dtype or mx.float32),
        "aten.full.default": lambda size, value, dtype=None, **_: mx.full(
            size, value, dtype=dtype or _scalar_dtype(value)
        ),
        "aten.full_like.default": lambda x, value, dtype=None, **_: mx.full(x.shape, value, dtype=dtype or x.dtype),
        "aten.zeros_like.default": lambda x, dtype=None, **_: mx.zeros(x.shape, dtype=dtype or x.dtype),
        # Uninitialized in torch: zeros are one of its possible values
        "aten.empty_permuted.default": lambda size, physical_layout, dtype=None, **_: mx.zeros(
            size, dtype=dtype or mx.float32
        ),
        "aten.copy.default": lambda x, src, non_blocking=False: mx.broadcast_to(src, x.shape).astype(x.dtype),
        "aten.repeat.default": mx.tile,
        "aten.ones_like.default": lambda x, dtype=None, **_: mx.ones(x.shape, dtype=dtype or x.dtype),
        "aten.arange.default": _arange,
        "aten.arange.start": _arange,
        "aten.arange.start_step": _arange,
        "aten.view.default": mx.reshape,
        "aten._unsafe_view.default": mx.reshape,
        "aten.reshape.default": mx.reshape,
        "aten.permute.default": mx.transpose,
        "aten.t.default": mx.transpose,
        "aten.unsqueeze.default": lambda x, dim: mx.expand_dims(x, dim if dim >= 0 else dim + x.ndim + 1),
        "aten.squeeze.default": _squeeze,
        "aten.squeeze.dim": _squeeze,
        "aten.squeeze.dims": _squeeze,
        "aten.expand.default": _expand,
        "aten.slice.Tensor": _slice,
        "aten.select.int": lambda x, dim, index: mx.take(x, index, axis=dim),
        "aten.cat.default": _cat,
        "aten.split.Tensor": _split,
        "aten.split_with_sizes.default": _split_with_sizes,
        "aten.constant_pad_nd.default": _constant_pad_nd,
        "aten.embedding.default": lambda weight, indices, *_: mx.take(weight, indices, axis=0),
        "aten.index.Tensor": _index,
        "aten.index_put.default": _index_put,
        "aten.index_select.default": lambda x, dim, index: mx.take(x, index, axis=dim),
        "aten.gather.default": lambda x, dim, index, **_: mx.take_along_axis(x, index, axis=dim),
        "aten.tril.default": lambda x, diagonal=0: mx.tril(x, diagonal),
        "aten.triu.default": lambda x, diagonal=0: mx.triu(x, diagonal),
        "aten.where.self": mx.where,
        # Elementwise
        "aten.add.Tensor": lambda a, b, alpha=1: a + (b if alpha == 1 else alpha * b),
        "aten.add.Scalar": lambda a, b, alpha=1: a + alpha * b,
        "aten.sub.Tensor": lambda a, b, alpha=1: a - (b if alpha == 1 else alpha * b),
        "aten.sub.Scalar": lambda a, b, alpha=1: a - alpha * b,
        "aten.rsub.Scalar": lambda a, b, alpha=1: b - alpha * a,
        "aten.mul.Tensor": operator.mul,
        "aten.mul.Scalar": operator.mul,
        "aten.div.Tensor": _div,
        "aten.div.Scalar": _div,
        "aten.div.Tensor_mode": _div,
        "aten.remainder.Tensor": mx.remainder,
        "aten.remainder.Scalar": mx.remainder,
        "aten.pow.Tensor_Scalar": lambda x, e: mx.square(x) if e == 2 else x**e,
        "aten.pow.Tensor_Tensor": mx.power,
        "aten.pow.Scalar": mx.power,
        "aten.neg.default": mx.negative,
        "aten.abs.default": mx.abs,
        "aten.relu.default": lambda x: mx.maximum(x, 0),
        "aten.log1p.default": mx.log1p,
        "aten.exp.default": mx.exp,
        "aten.log.default": mx.log,
        "aten.sqrt.default": mx.sqrt,
        "aten.rsqrt.default": mx.rsqrt,
        "aten.reciprocal.default": mx.reciprocal,
        "aten.sin.default": mx.sin,
        "aten.cos.default": mx.cos,
        "aten.tanh.default": mx.tanh,
        "aten.erf.default": mx.erf,
        "aten.sigmoid.default": mx.sigmoid,
        "aten.floor.default": mx.floor,
        "aten.ceil.default": mx.ceil,
        "aten.gelu.default": _gelu,
        "aten.maximum.default": mx.maximum,
        "aten.minimum.default": mx.minimum,
        "aten.clamp.default": lambda x, min=None, max=None: mx.clip(x, min, max),
        "aten.eq.Tensor": mx.equal,
        "aten.eq.Scalar": mx.equal,
        "aten.ne.Tensor": mx.not_equal,
        "aten.ne.Scalar": mx.not_equal,
        "aten.lt.Tensor": mx.less,
        "aten.lt.Scalar": mx.less,
        "aten.le.Tensor": mx.less_equal,
        "aten.le.Scalar": mx.less_equal,
        "aten.gt.Tensor": mx.greater,
        "aten.gt.Scalar": mx.greater,
        "aten.ge.Tensor": mx.greater_equal,
        "aten.ge.Scalar": mx.greater_equal,
        "aten.logical_not.default": mx.logical_not,
        "aten.logical_and.default": mx.logical_and,
        "aten.logical_or.default": mx.logical_or,
        "aten.bitwise_not.default": lambda x: mx.logical_not(x) if x.dtype == mx.bool_ else mx.bitwise_invert(x),
        "aten.bitwise_and.Tensor": lambda a, b: mx.logical_and(a, b) if a.dtype == mx.bool_ else mx.bitwise_and(a, b),
        "aten.bitwise_or.Tensor": lambda a, b: mx.logical_or(a, b) if a.dtype == mx.bool_ else mx.bitwise_or(a, b),
        # Reductions
        "aten.sum.dim_IntList": lambda x, dim=None, keepdim=False, dtype=None: _astype(
            mx.sum(x, axis=_axes(dim), keepdims=keepdim), dtype
        ),
        "aten.mean.dim": lambda x, dim=None, keepdim=False, dtype=None: _astype(
            mx.mean(x, axis=_axes(dim), keepdims=keepdim), dtype
        ),
        "aten.amax.default": lambda x, dim=(), keepdim=False: mx.max(x, axis=_axes(dim), keepdims=keepdim),
        "aten.amin.default": lambda x, dim=(), keepdim=False: mx.min(x, axis=_axes(dim), keepdims=keepdim),
        "aten.any.dim": lambda x, dim, keepdim=False: mx.any(x, axis=dim, keepdims=keepdim),
        "aten.any.dims": lambda x, dim=None, keepdim=False: mx.any(x, axis=_axes(dim), keepdims=keepdim),
        "aten.any.default": mx.any,
        "aten.all.dim": lambda x, dim, keepdim=False: mx.all(x, axis=dim, keepdims=keepdim),
        "aten.max.dim": _max_dim,
        "aten.min.dim": _min_dim,
        "aten.argmax.default": lambda x, dim=None, keepdim=False: mx.argmax(x, axis=dim, keepdims=keepdim).astype(
            mx.int64
        ),
        "aten.linalg_vector_norm.default": _vector_norm,
        "aten.topk.default": _topk,
        "aten.sort.default": _sort,
        "aten.sort.stable": lambda x, stable=None, dim=-1, descending=False: _sort(x, dim, descending),
        "aten.histc.default": _histc,
        "aten.cumsum.default": lambda x, dim, dtype=None: _astype(mx.cumsum(x, axis=dim), dtype),
        "aten._softmax.default": lambda x, dim, half_to_float=False: mx.softmax(x, axis=dim, precise=True),
        "aten._log_softmax.default": lambda x, dim, half_to_float=False: x - mx.logsumexp(x, axis=dim, keepdims=True),
        # Linear algebra
        "aten.mm.default": mx.matmul,
        "aten.bmm.default": mx.matmul,
        "aten.addmm.default": lambda bias, a, b, beta=1, alpha=1: mx.addmm(bias, a, b, alpha=alpha, beta=beta),
        "aten.scaled_dot_product_attention.default": _sdpa,
        "aten.native_layer_norm.default": _native_layer_norm,
        "aten.layer_norm.default": _layer_norm,
        "aten.rms_norm.default": _rms_norm,
        "aten.linear.default": _linear,
        "aten._grouped_mm.default": _grouped_mm,
        "transformers.grouped_mm_fallback.default": _grouped_mm,
        "aten.silu.default": lambda x: x * mx.sigmoid(x),
        # Convolutions and pooling
        "aten.convolution.default": _convolution,
        "aten.avg_pool1d.default": functools.partial(_avg_pool, 1),
        "aten.avg_pool2d.default": functools.partial(_avg_pool, 2),
        "aten.avg_pool3d.default": functools.partial(_avg_pool, 3),
        "aten.max_pool2d_with_indices.default": functools.partial(_max_pool_with_indices, 2),
        "aten.max_pool3d_with_indices.default": functools.partial(_max_pool_with_indices, 3),
        # Fusions (`mlx_fusions`)
        "transformers.exporters.mlx_ops.rms_norm": rms_norm,
        "transformers.exporters.mlx_ops.swiglu": swiglu,
        "transformers.exporters.mlx_ops.rope": rope,
        "transformers.exporters.mlx_ops.slice_write": slice_write,
    }


# ── Tensors ─────────────────────────────────────────────────────────────────


def to_mlx(tensor: torch.Tensor) -> mx.array:
    """A torch tensor as an MLX array, through host memory. NumPy has no bfloat16: its bits cross as int16."""
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        return mx.array(tensor.view(torch.int16).numpy()).view(mx.bfloat16)
    return mx.array(tensor.numpy())


def to_torch(array: mx.array) -> torch.Tensor:
    """An evaluated MLX array as a torch tensor sharing its memory, which keeps the array alive."""
    import numpy as np

    if array.dtype == mx.bfloat16:
        return torch.from_numpy(np.asarray(memoryview(array.view(mx.int16)))).view(torch.bfloat16)
    return torch.from_numpy(np.asarray(memoryview(array)))
