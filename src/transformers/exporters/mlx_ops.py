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


def rope(x, positions, inv_freq, scaling):
    """`mx.fast.rope`, which `mlx_fusions` writes in place of a rotary embedding's ops. Each token is its own row,
    rotated at its own position (an offset per row): exact for any positions, padded or packed ones included."""
    batch, heads, length, head_dim = x.shape
    offsets = mx.broadcast_to(positions, (batch, length)).reshape(-1).astype(mx.int32)
    rows = x.transpose(0, 2, 1, 3).reshape(batch * length, heads, 1, head_dim)
    rows = mx.fast.rope(
        rows, head_dim, traditional=False, base=None, scale=1.0, offset=offsets, freqs=1 / inv_freq.astype(mx.float32)
    )
    out = rows.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3)
    return out if scaling == 1.0 else out * scaling


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
        "aten.embedding.default": lambda weight, indices, *_: weight[indices],
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
        "aten.silu.default": lambda x: x * mx.sigmoid(x),
        # Fusions (`mlx_fusions`)
        "transformers.exporters.mlx_ops.rms_norm": rms_norm,
        "transformers.exporters.mlx_ops.swiglu": swiglu,
        "transformers.exporters.mlx_ops.rope": rope,
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
