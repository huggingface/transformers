# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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

"""Shared export utilities used by all exporter backends.

- **Patch and fix registries**: `@register_patch(backend, *paths)`, `@register_fx_node_fix` and
  `@register_fx_program_fix`, applied with `apply_patches` / `apply_fx_node_fixes` / `apply_fx_program_fixes`.
- **Cross-backend patches**: the ones more than one backend needs (`bucketize`, the cumulative reductions, the
  Mamba scan).
- **Tensor utilities**: `get_leaf_tensors`, `cast_leaf_tensors`, `duplicate_leaf_tensors`, `runner_feed`, and
  `prepare_for_export` (attention / experts implementation, output flags, precomputed inputs).

Taking a model apart is `decompose.py`'s, the modules it wraps a model in are `components.py`'s, and the inputs
a model would have computed for itself are `precompute.py`'s.
"""

from __future__ import annotations

import contextlib
import enum
import importlib
import inspect
from collections.abc import MutableMapping
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_torch_available
from .precompute import precompute_export_inputs


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

    from .. import masking_utils
    from ..modeling_utils import PreTrainedModel


# ── Patch and fix registries ────────────────────────────────────────────────
# `_PATCHES[backend]`: `(obj, attribute, factory)` triples installed reversibly.
# `_FX_NODE_FIXES[backend]`: `(gm, node) -> bool` fixers applied in place.

_PATCHES: dict[str, list[tuple[Any, str, callable]]] = {}
_FX_PROGRAM_FIXES: dict[str, list[callable]] = {}
_FX_NODE_FIXES: dict[str, list[callable]] = {}


@contextlib.contextmanager
def _patch_attribute(obj: Any, attribute: str, factory: Any):
    """Swap `obj.<attribute>` with `factory(original)` for the duration of the block."""
    original = getattr(obj, attribute)
    setattr(obj, attribute, factory(original))
    try:
        yield
    finally:
        setattr(obj, attribute, original)


@contextlib.contextmanager
def patch_attributes(patches: list[tuple[Any, str, callable]]):
    """Install `(obj, attribute, factory)` patches for the duration of the block, restoring on exit."""
    with contextlib.ExitStack() as stack:
        for obj, attribute, factory in patches:
            stack.enter_context(_patch_attribute(obj, attribute, factory))
        yield


@contextlib.contextmanager
def apply_patches(backend: str):
    """Install `_PATCHES[backend]` for the duration of the block."""
    with patch_attributes(_PATCHES.get(backend, [])):
        yield


def register_fx_node_fix(backend: str):
    """Append the decorated `(gm, node) -> bool` fix to `_FX_NODE_FIXES[backend]`."""

    def decorator(fn):
        _FX_NODE_FIXES.setdefault(backend, []).append(fn)
        return fn

    return decorator


def register_fx_program_fix(backend: str):
    """Append the decorated `(exported_program) -> None` fix to `_FX_PROGRAM_FIXES[backend]`.

    For fixes needing program-level context (range_constraints, graph_signature, state_dict).
    """

    def decorator(fn):
        _FX_PROGRAM_FIXES.setdefault(backend, []).append(fn)
        return fn

    return decorator


def apply_fx_program_fixes(backend: str, exported_program) -> None:
    """Apply `_FX_PROGRAM_FIXES[backend]` to `exported_program` (in place)."""
    for fix in _FX_PROGRAM_FIXES.get(backend, []):
        fix(exported_program)


def register_patch(backend: str, *paths: str):
    """Append the decorated `factory(original)` to `_PATCHES[backend]`, once per `path`.

    Each `path` is a dotted path like `"torch.Tensor.unsqueeze"`; the rightmost segment is the attribute
    to swap. Paths are resolved at decoration time, and one that fails to resolve (e.g. the backend isn't
    installed) is silently skipped.
    """

    def decorator(fn):
        for path in paths:
            obj_path, _, attribute = path.rpartition(".")
            obj = _resolve_dotted_path(obj_path)
            if obj is None:
                continue
            _PATCHES.setdefault(backend, []).append((obj, attribute, fn))
        return fn

    return decorator


def _resolve_dotted_path(path: str):
    """Resolve a dotted path, importing submodules where possible; `None` if it can't be resolved."""
    parts = path.split(".")
    try:
        obj = importlib.import_module(parts[0])
        for part in parts[1:]:
            try:
                obj = importlib.import_module(f"{obj.__name__}.{part}")
            except (ImportError, AttributeError):
                obj = getattr(obj, part)
        return obj
    except (ImportError, AttributeError):
        return None


def apply_fx_node_fixes(backend: str, graph_module) -> None:
    """Apply the first matching `_FX_NODE_FIXES[backend]` fix to every call_function node, then DCE.

    A fix returning `True` consumed the node. DCE can raise `SystemError` / `KeyError` on orphaned
    symbolic-size nodes; both are swallowed and the backend optimizer handles survivors.
    """
    fixes = _FX_NODE_FIXES.get(backend, [])
    for gm in graph_module.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue
            for fix in fixes:
                if fix(gm, node):
                    break
        try:
            gm.graph.eliminate_dead_code()
            gm.recompile()
        except (SystemError, KeyError):
            pass


def drop_runtime_asserts(graph_module) -> None:
    """Drop ``_assert_scalar`` / ``_assert_tensor_metadata`` runtime asserts, and the nodes only they used.

    Backend decomposition tracers can't proxy the ``Piecewise`` chain of ``_assert_scalar`` (``... is not
    tracked with proxy``), and ``_assert_tensor_metadata`` re-checks dtypes later stages change. The range facts
    survive in ``range_constraints``.
    """
    targets = (torch.ops.aten._assert_tensor_metadata.default, torch.ops.aten._assert_scalar.default)
    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        asserts = [node for node in module.graph.nodes if node.op == "call_function" and node.target in targets]
        # Targeted removal: a global `eliminate_dead_code` trips an fx `SystemError` on unrelated nodes.
        stack = [feeder for node in asserts for feeder in node.all_input_nodes]
        for node in asserts:
            module.graph.erase_node(node)
        # A feeder can be reached more than once; erasing it twice corrupts the graph.
        erased = set()
        while stack:
            feeder = stack.pop()
            if feeder in erased or feeder.op in ("placeholder", "output") or feeder.users or feeder.is_impure():
                continue
            # Erasing some dead `sym_size`s trips a C-level fx `SystemError`; leaving them is harmless.
            try:
                module.graph.erase_node(feeder)
            except SystemError:
                continue
            stack.extend(feeder.all_input_nodes)
            erased.add(feeder)
        module.recompile()


# ── Cross-backend patches ─────────────────────────────────────────────────────


def zero_fully_masked_rows(attn_output, attn_mask):
    """Zero the attention rows whose mask attends no key, as torch's fused SDPA kernels do."""
    if attn_mask.dtype == torch.bool:
        unattended = ~attn_mask.any(dim=-1, keepdim=True)
    else:
        unattended = attn_mask.amax(dim=-1, keepdim=True) <= torch.finfo(attn_mask.dtype).min
    return torch.where(unattended, attn_output.new_zeros(()), attn_output)


@register_patch("onnx", "transformers.models.blt.modeling_blt.byte_group_hash_function")
@register_patch("openvino", "transformers.models.blt.modeling_blt.byte_group_hash_function")
def _patch_byte_group_hash(original):
    """Evaluate BLT's rolling hash in base-256 limbs, folded into `% max_hash` as it goes.

    The int64 hash is wrong on both backends: ORT's `sum` reduces through fp32, and OV's CPU plugin runs
    internal nodes in i32 (`1000000007 ** 2` saturates). Limbs keep every intermediate exact.
    """
    limb_bits = 8
    limb = 1 << limb_bits
    limb_count = 64 // limb_bits

    def patch(token_ids, group_size: int = 2, prime: int = 1000000007, max_hash: int = 30000):
        # Beyond a 16-bit table the limb products leave the window where the plugin's `%` is exact
        if max_hash > (1 << 16):
            return original(token_ids, group_size=group_size, prime=prime, max_hash=max_hash)

        powers = [pow(prime, index, 1 << 64) for index in range(group_size)]
        limbs = torch.tensor(
            [[(power >> (limb_bits * position)) % limb for power in powers] for position in range(limb_count)],
            dtype=torch.int64,
            device=token_ids.device,
        )
        padding = torch.zeros(token_ids.shape[0], group_size - 1, dtype=torch.int64, device=token_ids.device)
        windows = torch.cat([padding, token_ids.to(torch.int64)], dim=1).unfold(1, group_size, 1)
        lanes = (windows.unsqueeze(-2) * limbs).sum(-1)

        value = carry = torch.zeros_like(lanes[..., 0])
        for position in range(limb_count):
            total = lanes[..., position] + carry
            digit = torch.bitwise_and(total, limb - 1)
            carry = (total - digit) // limb
            value = (value + digit * ((1 << (limb_bits * position)) % max_hash)) % max_hash
        # The limbs spell the unsigned value; eager's signed int64 is 2**64 lower when the top bit is set.
        negative = (digit >= limb // 2).to(torch.int64)
        return (value - negative * ((1 << 64) % max_hash) + max_hash) % max_hash

    return patch


@register_patch("onnx", "torch.histc")
@register_patch("openvino", "torch.histc")
@register_patch("executorch", "torch.histc")
def _patch_histc(original):
    """Replace `torch.histc` with a statically-shaped, deterministic `zeros` + `scatter_add_`.

    torchlib rejects integer input (and the float path is nondeterministic on CUDA); OV and ExecuTorch have
    no kernel. `bincount` has an unbacked output size. Out-of-range values stay uncounted, which MoE sentinel
    expert ids rely on.
    """

    def patch(input, bins=100, min=0, max=0, *, out=None):
        flat = input.reshape(-1)
        if max == min == 0:
            min_val = flat.min().float()
            max_val = flat.max().float()
        else:
            min_val = torch.tensor(float(min), device=flat.device)
            max_val = torch.tensor(float(max), device=flat.device)
        bin_width = (max_val - min_val) / bins
        values = flat.float()
        idx = ((values - min_val) / bin_width).long().clamp_(0, bins - 1)
        out_dtype = input.dtype if input.is_floating_point() else torch.float
        counted = ((values >= min_val) & (values <= max_val)).to(out_dtype)
        counts = torch.zeros(bins, dtype=out_dtype, device=input.device)
        return counts.scatter_add_(0, idx, counted)

    return patch


@register_fx_node_fix("onnx")
@register_fx_node_fix("openvino")
def _fix_scatter_reduce(gm, node):
    """Lower ``aten.scatter_reduce.two`` at the FX level; OV's frontend has no translation.

    Handles sum with ``include_self=True`` (scatter_add), and sum/mean/amax/amin with ``include_self=False``.
    """
    if node.target is not torch.ops.aten.scatter_reduce.two:
        return False
    if len(node.args) < 5:
        return False
    reduce = node.args[4]
    include_self = node.kwargs.get("include_self", True)
    self_arg, dim, index, src = node.args[0:4]

    if reduce == "sum" and include_self is True:
        with gm.graph.inserting_before(node):
            new = gm.graph.call_function(torch.ops.aten.scatter_add.default, args=(self_arg, dim, index, src))
            new.meta.update(node.meta)
        node.replace_all_uses_with(new)
        gm.graph.erase_node(node)
        return True

    if reduce in ("sum", "mean") and include_self is False:
        # Positions nothing scatters to keep ``self``; the scattered count finds them and divides the mean.
        self_val = self_arg.meta.get("val")
        src_val = src.meta.get("val")
        if self_val is None or src_val is None:
            return False
        with gm.graph.inserting_before(node):
            zeros = gm.graph.call_function(torch.ops.aten.zeros_like.default, args=(self_arg,))
            sums = gm.graph.call_function(torch.ops.aten.scatter_add.default, args=(zeros, dim, index, src))
            ones = gm.graph.call_function(torch.ops.aten.ones_like.default, args=(src,))
            counts = gm.graph.call_function(torch.ops.aten.scatter_add.default, args=(zeros, dim, index, ones))
            values = sums
            if reduce == "mean":
                divisor = gm.graph.call_function(torch.ops.aten.clamp_min.default, args=(counts, 1))
                values = gm.graph.call_function(torch.ops.aten.div.Tensor, args=(sums, divisor))
            # OV's frontend has no ``gt.Scalar`` translation, so compare against a 0-dim tensor
            zero_tensor = gm.graph.call_function(
                torch.ops.aten.scalar_tensor.default,
                args=(0,),
                kwargs={"dtype": src_val.dtype, "device": src_val.device},
            )
            touched = gm.graph.call_function(torch.ops.aten.gt.Tensor, args=(counts, zero_tensor))
            result = gm.graph.call_function(torch.ops.aten.where.self, args=(touched, values, self_arg))
            result.meta.update(node.meta)
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
        return True

    if reduce in ("amax", "amin") and include_self is False:
        # One-hot mask over `index`, reduce `src` where set, fall back to `self` where nothing scatters.
        self_val = self_arg.meta.get("val")
        src_val = src.meta.get("val")
        if self_val is None or src_val is None or not src_val.dtype.is_floating_point:
            return False
        ndim = self_val.ndim
        d = dim if dim >= 0 else dim + ndim
        k_size = self_val.shape[d]
        finfo = torch.finfo(src_val.dtype)
        fill_value = finfo.min if reduce == "amax" else finfo.max
        reduction = torch.ops.aten.amax.default if reduce == "amax" else torch.ops.aten.amin.default
        k_shape = [1] * (ndim + 1)
        k_shape[d] = -1
        with gm.graph.inserting_before(node):
            # A SymInt `arange` literal is decoded by OV as a malformed constant; feed it via `sym_size`.
            arange_size = (
                k_size
                if isinstance(k_size, int)
                else gm.graph.call_function(torch.ops.aten.sym_size.int, args=(self_arg, d))
            )
            arange = gm.graph.call_function(
                torch.ops.aten.arange.default, args=(arange_size,), kwargs={"device": self_val.device}
            )
            k_range = gm.graph.call_function(torch.ops.aten.view.default, args=(arange, k_shape))
            index_unsq = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(index, d))
            mask = gm.graph.call_function(torch.ops.aten.eq.Tensor, args=(index_unsq, k_range))
            src_unsq = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(src, d))
            # OV's frontend has no ``where.ScalarOther`` translation.
            scalar_kwargs = {"dtype": src_val.dtype, "device": src_val.device}
            fill_tensor = gm.graph.call_function(
                torch.ops.aten.scalar_tensor.default, args=(fill_value,), kwargs=scalar_kwargs
            )
            masked = gm.graph.call_function(torch.ops.aten.where.self, args=(mask, src_unsq, fill_tensor))
            extrema = gm.graph.call_function(reduction, args=(masked, [d + 1]))
            any_match = gm.graph.call_function(torch.ops.aten.any.dim, args=(mask, d + 1))
            result = gm.graph.call_function(torch.ops.aten.where.self, args=(any_match, extrema, self_arg))
            result.meta.update(node.meta)
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
        return True

    return False


@register_patch("onnx", "transformers.masking_utils._vmap_expansion_sdpa")
@register_patch("openvino", "transformers.masking_utils._vmap_expansion_sdpa")
@register_patch("executorch", "transformers.masking_utils._vmap_expansion_sdpa")
def _patch_broadcast_mask_expansion(_original):
    """Replace vmap-based mask expansion with broadcast expansion; no backend traces `torch.vmap`."""

    def patch(mask_function):
        def _expanded(batch_arange, head_arange, q_arange, kv_arange):
            broadcasted = masking_utils._non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange)
            return mask_function(*broadcasted).expand(
                batch_arange.shape[0], head_arange.shape[0], q_arange.shape[0], kv_arange.shape[0]
            )

        return _expanded

    return patch


@register_patch("onnx", "torch.reshape", "torch.Tensor.reshape", "torch.Tensor.view")
@register_patch("executorch", "torch.reshape", "torch.Tensor.reshape", "torch.Tensor.view")
def _patch_reshape(original):
    """Materialise a non-contiguous input before `reshape` / `view`.

    ONNX (`Cannot view a tensor with shape ... and strides ...`) and ExecuTorch's edge reshape both refuse
    that view. `is_contiguous_or_false` avoids a data-dependent guard under dynamic shapes.
    """
    from torch._prims_common import is_contiguous_or_false

    def patch(input, *shape, **kwargs):
        if isinstance(input, torch.Tensor) and not is_contiguous_or_false(input):
            input = input.clone(memory_format=torch.contiguous_format)
        return original(input, *shape, **kwargs)

    return patch


@register_patch("onnx", "torch.bucketize")
@register_patch("executorch", "torch.bucketize")
def _patch_bucketize(_original):
    """Decompose `bucketize` into a broadcast comparison and a sum.

    Neither ONNX nor ExecuTorch's portable runtime has a kernel (reached by e.g. idefics3 position ids).
    """

    def patch(input, boundaries, *, out_int32=False, right=False, out=None):
        if boundaries.numel() == 0:
            result = torch.zeros_like(input, dtype=torch.int64)
        else:
            below = boundaries <= input.unsqueeze(-1) if right else boundaries < input.unsqueeze(-1)
            result = below.sum(dim=-1)
        result = result.to(torch.int32) if out_int32 else result
        return out.copy_(result) if out is not None else result

    return patch


@register_patch("onnx", "torch.searchsorted")
@register_patch("openvino", "torch.searchsorted")
@register_patch("executorch", "torch.searchsorted")
def _patch_searchsorted(_original):
    """Decompose `searchsorted` like `bucketize`: count the sorted entries below each value (O(N*M))."""

    def patch(sorted_sequence, input, *, out_int32=False, right=False, side=None, out=None, sorter=None):
        if side is not None:
            right = side == "right"
        seq, val = sorted_sequence.unsqueeze(-2), input.unsqueeze(-1)
        below = seq <= val if right else seq < val
        result = below.sum(dim=-1)
        result = result.to(torch.int32) if out_int32 else result
        return out.copy_(result) if out is not None else result

    return patch


_MAMBA_SELECTIVE_SCANS = (
    "transformers.models.falcon_mamba.modeling_falcon_mamba.mamba_selective_scan",
    "transformers.models.jamba.modeling_jamba.mamba_selective_scan",
    "transformers.models.mamba.modeling_mamba.mamba_selective_scan",
    "transformers.models.zamba.modeling_zamba.mamba_selective_scan",
)


def _sequential_mamba_scan(original, associative_on: tuple[str, ...]):
    """Force the SSM scan's sequential path unless the input sits on a device in `associative_on`.

    The cpu `generic` combine mode runs under `vmap`, which `run_decompositions` cannot take apart; the
    cuda/xpu `pointwise` mode lowers on ONNX only.
    """

    def patch(hidden_states, *args, **kwargs):
        if hidden_states.device.type not in associative_on:
            kwargs["use_associative_scan"] = False
        return original(hidden_states, *args, **kwargs)

    return patch


@register_patch("onnx", *_MAMBA_SELECTIVE_SCANS)
def _patch_mamba_selective_scan_onnx(original):
    return _sequential_mamba_scan(original, associative_on=("cuda", "xpu"))


@register_patch("openvino", *_MAMBA_SELECTIVE_SCANS)
def _patch_mamba_selective_scan_openvino(original):
    return _sequential_mamba_scan(original, associative_on=())


@register_patch("onnx", "torch.cummax", "torch.Tensor.cummax")
@register_patch("openvino", "torch.cummax", "torch.Tensor.cummax")
@register_patch("executorch", "torch.cummax", "torch.Tensor.cummax")
def _patch_cummax(original):
    """`cummax` via a triangular-masked reduction — see `_cumulative_reduce`."""
    return _cumulative_reduce(mode="max")


@register_patch("onnx", "torch.cummin", "torch.Tensor.cummin")
@register_patch("openvino", "torch.cummin", "torch.Tensor.cummin")
@register_patch("executorch", "torch.cummin", "torch.Tensor.cummin")
def _patch_cummin(original):
    """`cummin` via a triangular-masked reduction — see `_cumulative_reduce`."""
    return _cumulative_reduce(mode="min")


def _cumulative_reduce(*, mode: str):
    """Replace `cummax` / `cummin` (no backend kernel) with a triangular-masked `max` / `min` over `j <= i`."""

    def patch(input, dim):
        sequence = input.movedim(dim, -1)
        positions = torch.arange(sequence.shape[-1], device=input.device)
        keep = positions.unsqueeze(0) <= positions.unsqueeze(1)
        if input.dtype == torch.bool:
            fill = mode != "max"
        else:
            info = torch.finfo if input.is_floating_point() else torch.iinfo
            fill = info(input.dtype).min if mode == "max" else info(input.dtype).max
        windows = torch.where(
            keep, sequence.unsqueeze(-2), torch.full((), fill, dtype=input.dtype, device=input.device)
        )
        reduced = windows.max(dim=-1) if mode == "max" else windows.min(dim=-1)
        return getattr(torch.return_types, f"cum{mode}")(
            (reduced.values.movedim(-1, dim), reduced.indices.movedim(-1, dim))
        )

    return patch


# ── Recursive structure traversal ──────────────────────────────────────────

# Not recursed into: Sym* types carry shape_env internals that recurse infinitely.
_LEAF_SKIP_TYPES: tuple[type, ...] = (type,)
if is_torch_available():
    _LEAF_SKIP_TYPES += (enum.Enum, torch.SymInt, torch.SymFloat, torch.SymBool)


def _map_leaf_tensors(obj: Any, fn: callable) -> Any:
    """Apply `fn` to every tensor in a nested structure, preserving container types.

    Dicts and `__dict__`-bearing objects are mutated in place (callers rely on the identity); sequences
    and sets are rebuilt.
    """
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return obj
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, (list, tuple, set, frozenset)):
        return type(obj)(_map_leaf_tensors(item, fn) for item in obj)
    if isinstance(obj, dict):
        for k in list(obj):
            obj[k] = _map_leaf_tensors(obj[k], fn)
        return obj
    if hasattr(obj, "__dict__"):
        for attr, attr_val in vars(obj).items():
            setattr(obj, attr, _map_leaf_tensors(attr_val, fn))
    return obj


def _iter_leaf_tensors(obj: Any, prefix: str = ""):
    """Yield `(dotted_path, tensor)` for every tensor in a nested structure."""
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return
    if isinstance(obj, torch.Tensor):
        # Not a truthiness test: an int-keyed dict (granite4_vision deepstack features) yields path `0`.
        yield prefix if prefix != "" else "output", obj
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for index, item in enumerate(obj):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_leaf_tensors(item, path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_tensors(value, path)
    elif hasattr(obj, "__dict__"):
        yield from _iter_leaf_tensors(vars(obj), prefix)


# ── Public tensor utilities ────────────────────────────────────────────────


def _class_to_path(cls: type) -> str:
    """A class as `module:qualname`, as pytree contexts and graph metadata name it."""
    return f"{cls.__module__}:{cls.__qualname__}"


def _path_to_class(path: str) -> type:
    """The class `_class_to_path` wrote. Importing its module also registers its `ModelOutput` pytree nodes,
    which a graph loaded from disk needs."""
    module_name, qualname = path.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def runner_feed(runner, kwargs: dict, *, warn_unused: bool = False) -> dict:
    """The subset of `kwargs` this graph takes, keyed the way it names them.

    A graph refuses a kwarg it was never traced with, while callers pass a processor's whole output.
    `warn_unused` logs what was dropped.
    """
    if not runner.input_names:
        return dict(kwargs)
    feed = {name: value for name, value in kwargs.items() if runner.declares(name, value)}
    if warn_unused and (unused := [name for name in kwargs if name not in feed]):
        logger.warning_once(
            f"Ignoring {unused}, which this graph was not traced with (it takes {sorted(runner.input_names)})."
        )
    return feed


# The inputs whose leading axis is the batch, in the order they are trusted to say it.
BATCH_INPUTS = ("input_ids", "inputs_embeds", "decoder_input_ids", "decoder_inputs_embeds", "attention_mask")


def leaf_name(name: str) -> str:
    """The kwarg-space name behind a graph port's name — `disambiguate_io_names` only ever prefixes."""
    for prefix in ("input.", "output."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def get_leaf_tensors(obj: Any) -> dict[str, torch.Tensor]:
    """Recursively retrieve all leaf tensors from a potentially nested structure.

    Args:
        obj (`Any`):
            A tensor, dataclass, dict, list, tuple, or any nesting thereof.

    Returns:
        `dict[str, torch.Tensor]`: Flat mapping from dotted path strings to tensors.
    """
    return dict(_iter_leaf_tensors(obj))


def duplicate_leaf_tensors(obj: Any, seen: set[int] | None = None) -> Any:
    """Clone tensors that appear more than once in an output structure.

    The ONNX optimizer merges outputs sharing a tensor and renames one, breaking the name mapping.
    `seen` pre-seeds taken identities: pass the input tensors so an input returned unmutated (prophetnet's
    `encoder_last_hidden_state`) is cloned too instead of collapsing onto the input's name.
    """
    seen = set() if seen is None else set(seen)

    def _dedup(tensor: torch.Tensor) -> torch.Tensor:
        if id(tensor) in seen:
            return tensor.clone()
        seen.add(id(tensor))
        return tensor

    return _map_leaf_tensors(obj, _dedup)


def cast_leaf_tensors(obj: Any, dtype: torch.dtype, device: torch.device) -> Any:
    """Recursively cast all floating-point tensors to the given dtype and device."""

    def _cast(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=dtype if tensor.is_floating_point() else None, device=device)

    return _map_leaf_tensors(obj, _cast)


def _module_attr(model: PreTrainedModel | torch.nn.Module, name: str):
    """`.device` / `.dtype` for any `nn.Module`, falling back to its first parameter; `None` if it has none."""
    if hasattr(model, name):
        return getattr(model, name)
    try:
        return getattr(next(model.parameters()), name)
    except StopIteration:
        return None


def module_device(model: PreTrainedModel | torch.nn.Module) -> torch.device | None:
    """Where this module's parameters live — see `_module_attr`."""
    return _module_attr(model, "device")


def module_dtype(model: PreTrainedModel | torch.nn.Module) -> torch.dtype | None:
    """The precision this module's parameters are in — see `_module_attr`."""
    return _module_attr(model, "dtype")


# Output flags that should be set on `model.config`, not passed as forward() kwargs.
_OUTPUT_FLAGS = ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss")


def prepare_for_export(
    model: PreTrainedModel | torch.nn.Module, inputs: MutableMapping[str, Any]
) -> tuple[PreTrainedModel | torch.nn.Module, MutableMapping[str, Any], dict[str, Any]]:
    """Configure model and inputs for export, mutating both in place.

    Rejects label inputs, pops output flags (`use_cache`, `return_dict`, ...) into the returned
    `output_flags` for the trace to apply onto `model.config`, precomputes data-dependent inputs, and
    moves input tensors to the model's device. Returns `(model, inputs, output_flags)`.
    """
    for label_key in ("labels", "future_values"):
        value = inputs.pop(label_key, None)
        if value is not None:
            raise ValueError(
                f"Found '{label_key}' in inputs. Loss computation is not supported during export. "
                f"Please remove '{label_key}' from your inputs before calling export()."
            )
    if hasattr(model, "config") and getattr(model.config, "return_loss", False):
        raise ValueError(
            "Found 'model.config.return_loss=True'. Loss computation is not supported during export. "
            "Please set 'model.config.return_loss=False' before calling export()."
        )
    if inputs.get("return_loss", False):
        raise ValueError(
            "Found 'return_loss=True' in inputs. Loss computation is not supported during export. "
            "Please remove 'return_loss' from your inputs or set it to False."
        )

    output_flags = {flag: inputs.pop(flag) for flag in _OUTPUT_FLAGS if flag in inputs}

    # `torch.export` records `None` kwargs as placeholders the caller must then pass back. Only dropped when
    # the default is `None` too, so omitting it can't switch the traced path.
    forward = getattr(model, "forward", None)
    if forward is not None:
        parameters = inspect.signature(forward).parameters
        for name in [name for name, value in inputs.items() if value is None]:
            parameter = parameters.get(name)
            if parameter is not None and parameter.default is None:
                inputs.pop(name)

    # Data-dependent vision/audio tensors dynamo can't trace.
    # TODO: use the collator API once it covers these cases.
    with torch.no_grad():
        # A decomposed component (e.g. `FSMTEncoder`) may carry no config.
        if (config := getattr(model, "config", None)) is not None:
            inputs.update(precompute_export_inputs(config, inputs))

    # Device only: SSM/recurrent states stay fp32 in a bf16 model, and a downcast would diverge from eager.
    device = module_device(model)
    if device is not None:
        inputs = cast_leaf_tensors(inputs, dtype=None, device=device)

    return model, inputs, output_flags
