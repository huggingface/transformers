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

Organised into four sections (search for the `# ── Name ──` banners):

- **Patch and fix registries** — backend-keyed `_PATCHES` / `_FX_NODE_FIXES` /
  `_FX_PROGRAM_FIXES` populated via `@register_patch(backend, *paths)` /
  `@register_fx_node_fix` / `@register_fx_program_fix`, applied via
  `apply_patches` / `apply_fx_node_fixes` / `apply_fx_program_fixes`.
- **Cross-backend patches** — the `@register_patch` replacements more than one
  backend needs (`torch.where` dtype mismatches, `bucketize`, the cumulative
  reductions).
- **Recursive structure traversal** — internal helpers (`_map_leaf_tensors`,
  `_iter_leaf_tensors`) that drive every other tensor utility.
- **Public tensor utilities** — `runner_feed`, `get_leaf_tensors`,
  `duplicate_leaf_tensors`, `cast_leaf_tensors`, and `prepare_for_export` (sets
  attention/experts impl, patches non-exportable patterns, strips output flags).

Taking a model apart is `decompose.py`'s, the modules a decomposition wraps it in are
`components.py`'s, and the inputs a model would have computed for itself are `precompute.py`'s.
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
# Single contract across exporters: `_PATCHES[backend]` lists `(obj, attribute, factory)` triples
# to install reversibly, and `_FX_NODE_FIXES[backend]` lists `(gm, node) -> bool` fixers to
# apply in place. Each exporter populates its slot at module load (via `@register_patch` /
# `@register_fx_node_fix` decorators, or direct list-append for cases that can't be expressed
# as dotted paths). The export pipeline drives them via the backend-keyed helpers below.

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
    """Install `(obj, attribute, factory)` patches for the duration of the block.

    Plural form of `_patch_attribute` — each `factory(original)` returns the replacement
    callable. Originals are restored on exit, even if the body raises.
    """
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

    Use this for fixes that need program-level context (range_constraints, graph_signature,
    state_dict) — the per-node `_FX_NODE_FIXES` shape only sees one node at a time.
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

    Each `path` is a dotted Python path like `"torch.where"`, `"torch.Tensor.unsqueeze"`,
    or `"transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router._cast_classifier"`.
    The rightmost segment is the attribute to swap; the rest is the object that owns it.
    Paths are resolved at decoration time — submodules are imported as needed, falling
    back to `getattr` for class attributes. A path that fails to resolve (e.g. the backend
    isn't installed) is silently skipped so the module still imports.

    Passing multiple paths registers the SAME factory against each — useful for swapping
    the same method or torch op across several call sites (e.g. ``torch.unsqueeze`` +
    ``torch.Tensor.unsqueeze``, or one vision-attention forward across N model classes).
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
    """Resolve a dotted Python path to the actual object — importing submodules where
    possible, falling back to `getattr` for class attributes (e.g. `torch.Tensor`).
    Returns `None` if the path can't be resolved (e.g. the backend isn't installed)."""
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
    """Walk every call_function node and apply the first matching `_FX_NODE_FIXES[backend]`
    fix, then DCE.

    Each fix has signature `(gm, node) -> bool`. Returning `True` means the fix consumed
    the node — no further fixes run against it. Fixes are expected to be disjoint by
    `node.target`; if multiple could apply, list order decides.

    After the walk, `Graph.eliminate_dead_code` runs on every sub-GraphModule and
    `gm.recompile()` is called once. PyTorch DCE occasionally raises `SystemError` /
    `KeyError` from `erase_node._update_args_kwargs` on orphaned symbolic-size nodes —
    we swallow both; any survivors are handled by the downstream backend optimizer.
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


# ── Cross-backend patches ─────────────────────────────────────────────────────
# Registered for more than one backend because the problem is the same one: these used to be two
# definitions apiece that had drifted in wording and in one case in behaviour.


@register_patch("onnx", "transformers.models.blt.modeling_blt.byte_group_hash_function")
@register_patch("openvino", "transformers.models.blt.modeling_blt.byte_group_hash_function")
def _patch_byte_group_hash(original):
    """Evaluate BLT's rolling hash in base-256 limbs, which both backends need for different reasons.

    The hash multiplies each byte of a group by ``prime ** k`` and relies on int64 semantics. ONNX
    Runtime multiplies int64 exactly but reduces through a float: ``sum`` turns `7000000049` into
    `7000000000`, losing everything below fp32's mantissa. OpenVINO's CPU plugin is narrower still —
    it executes every internal node in `i32`, so `1000000007 ** 2` saturates at `2147483647`. Either
    way the hash reads the wrong embedding rows.

    The powers are compile-time constants and the hash reaches the model only as ``hash % max_hash``,
    so it is computed here as a sum of 8-bit limbs: the single pass below carries each lane and folds
    it into the running remainder, keeping every intermediate small enough to be exact in both
    backends. Each limb is taken with `BitwiseAnd`, the only division is by 256 of a value already a
    multiple of it, and dropping the last carry is the int64 wraparound.
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
        # Those limbs spell the *unsigned* value; eager took the remainder of a signed int64, which
        # is 2**64 lower whenever the top bit -- the last digit's -- is set.
        negative = (digit >= limb // 2).to(torch.int64)
        return (value - negative * ((1 << 64) % max_hash) + max_hash) % max_hash

    return patch


@register_patch("onnx", "torch.histc")
@register_patch("openvino", "torch.histc")
def _patch_histc(original):
    """Replace `torch.histc` with a statically-shaped, deterministic `zeros` + `scatter_add_`.

    torchlib's `aten_histc` rejects integer input and casting to float calls the nondeterministic
    `_histc_cuda`; OV has no `aten.histc` lowering at all. `bincount`, the obvious replacement, has
    an unbacked SymInt output that trips downstream meta-shape guards (grouped_mm's `offs` check).
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
        idx = ((flat.float() - min_val) / bin_width).long().clamp_(0, bins - 1)
        out_dtype = input.dtype if input.is_floating_point() else torch.float
        counts = torch.zeros(bins, dtype=out_dtype, device=input.device)
        return counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=out_dtype))

    return patch


@register_fx_node_fix("onnx")
@register_fx_node_fix("openvino")
def _fix_scatter_reduce(gm, node):
    """Lower ``aten.scatter_reduce.two`` at the FX level — OV's frontend has no translation,
    and its ``ScatterElementsUpdate`` op can't accept the ``reduce`` string as a constant input.

    Handles two patterns the MoE/SSM models use:
      * ``reduce="sum", include_self=True`` → ``aten.scatter_add`` (BLT/JetMoe/NemotronH router).
      * ``reduce="amax"/"amin", include_self=False`` → masked extremum over a one-hot expansion of
        ``index`` (BLT byte-pooling, tapas segment reduction).
      * ``reduce="sum"/"mean", include_self=False`` → ``scatter_add`` onto zeros, divided by a
        scattered count for the mean (tapas segment reductions).

    Other combinations fall through to the generic OpConversionFailure.
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
        # ``include_self=False``: a position that receives at least one source element reduces over
        # *only* those elements, while a position nothing scatters to keeps ``self``. Scattering onto
        # zeros gives the former, and scattering ones alongside counts the contributors — which both
        # divides the mean and says which positions were touched at all.
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
                # clamped so untouched positions divide by 1 instead of 0 — `where` discards them anyway
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
        # ``amax``/``amin`` with ``include_self=False``: each source element competes for the extremum
        # at ``index[j]``; positions no source scatters to keep ``self``'s original value. Decompose to
        # a broadcast comparison + reduction: build a one-hot mask ``(index.unsqueeze(dim) ==
        # arange(K))``, reduce ``src`` where the mask is set (the opposite extreme elsewhere, so it
        # never wins), then fall back to ``self`` for positions with no scatter.
        self_val = self_arg.meta.get("val")
        src_val = src.meta.get("val")
        if self_val is None or src_val is None or not src_val.dtype.is_floating_point:
            return False
        ndim = self_val.ndim
        d = dim if dim >= 0 else dim + ndim
        k_size = self_val.shape[d]
        # the identity for the reduction: an element that never wins
        finfo = torch.finfo(src_val.dtype)
        fill_value = finfo.min if reduce == "amax" else finfo.max
        reduction = torch.ops.aten.amax.default if reduce == "amax" else torch.ops.aten.amin.default
        k_shape = [1] * (ndim + 1)
        k_shape[d] = -1
        with gm.graph.inserting_before(node):
            # ``k_size`` is symbolic under dynamic shapes (e.g. BLT's ``max_num_patches``); baking
            # the ``SymInt`` as an ``arange`` literal makes OV decode it as a malformed inlined
            # constant. Feed the dimension through a ``sym_size`` node so it stays a real Range input.
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
            # OV's frontend has no ``where.ScalarOther`` translation, so materialise the scalar
            # branches as 0-dim tensors and use ``where.self`` (broadcasts the same way).
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
    """Replace vmap-based mask expansion with broadcast expansion.

    No backend traces `torch.vmap`: OV's frontend sees inputs that "escaped" the vmap context,
    and `aot_autograd`/`gen_vmap_plumbing` reject vmap-built masks under ExecuTorch's lowering.
    """

    def patch(mask_function):
        def _expanded(batch_arange, head_arange, q_arange, kv_arange):
            broadcasted = masking_utils._non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange)
            return mask_function(*broadcasted).expand(
                batch_arange.shape[0], head_arange.shape[0], q_arange.shape[0], kv_arange.shape[0]
            )

        return _expanded

    return patch


@register_patch("executorch", "torch.nn.attention.varlen.varlen_attn")
def _patch_varlen_attn(original):
    """Lower `varlen_attn` to the block-diagonal masked SDPA it stands for.

    The chunked vision/audio attention patch calls it to express packed sequences as one op, and
    ExecuTorch cannot take it from there: the edge-dialect verifier trips on the CUDA flash op's aux
    outputs. The masked form is core-aten. Returns just the output tensor, which is `varlen_attn`'s
    contract — the underlying op's is `(output, *aux)`. OpenVINO keeps the op and converts it instead
    (`_convert_varlen_attn`), which is where a packed lowering belongs.
    """
    from .exporter_dynamo import varlen_attn_masked_sdpa

    def varlen_attn(*args, **kwargs):
        return varlen_attn_masked_sdpa(*args, **kwargs)

    return varlen_attn


@register_patch("onnx", "torch.reshape", "torch.Tensor.reshape", "torch.Tensor.view")
@register_patch("executorch", "torch.reshape", "torch.Tensor.reshape", "torch.Tensor.view")
def _patch_reshape(original):
    """Materialise a non-contiguous input before `reshape` / `view`.

    Both lowerings refuse the view a non-contiguous tensor would need: `torch.export` raises `Cannot view
    a tensor with shape ... and strides ...` for ONNX (whose optimizer folds a plain `aten.contiguous`
    away), and ExecuTorch's edge reshape reference refuses it outright. Cloning first is semantically a
    no-op -- eager `reshape` already copies in this case -- and only copies when the view would fail.

    `is_contiguous_or_false`, not `is_contiguous()`: under dynamic shapes contiguity is a data-dependent
    question, and the guard-free form answers "don't know" as "not contiguous", which is the safe side.
    The clone must force `contiguous_format`, since a bare `.clone()` preserves the input's layout.
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

    Neither backend has a kernel for it (ONNX has no op; the portable runtime ships no
    `bucketize.Tensor_out`, which VLM vision position ids reach — idefics2/3, smolvlm, phi4_multimodal).
    `boundaries` is 1-D and sorted, so the bucket index is the count of boundaries below each value.
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
    """Decompose `searchsorted` the same way as `bucketize`: for a sorted sequence the insertion index is
    the count of entries below each value. O(N*M) rather than a real binary search, but only ops both
    backends can lower."""

    def patch(sorted_sequence, input, *, out_int32=False, right=False, side=None, out=None, sorter=None):
        if side is not None:
            right = side == "right"
        seq, val = sorted_sequence.unsqueeze(-2), input.unsqueeze(-1)
        below = seq <= val if right else seq < val
        result = below.sum(dim=-1)
        result = result.to(torch.int32) if out_int32 else result
        return out.copy_(result) if out is not None else result

    return patch


@register_patch("onnx", "torch.cummax", "torch.Tensor.cummax")
@register_patch("executorch", "torch.cummax", "torch.Tensor.cummax")
def _patch_cummax(original):
    """`cummax` via a triangular-masked reduction — see `_cumulative_reduce`."""
    return _cumulative_reduce(mode="max")


@register_patch("onnx", "torch.cummin", "torch.Tensor.cummin")
@register_patch("executorch", "torch.cummin", "torch.Tensor.cummin")
def _patch_cummin(original):
    """`cummin` via a triangular-masked reduction — see `_cumulative_reduce`."""
    return _cumulative_reduce(mode="min")


def _cumulative_reduce(*, mode: str):
    """Replace `cummax` / `cummin` with a triangular-masked reduction: neither backend has a
    cumulative-scan kernel. Output `[..., i]` reduces over `j <= i`.

    The reduction is the two-output `max`/`min` so the real argmax comes back with it — a caller reading
    `.indices` gets the same answer it would from the op.
    """

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
        # The op returns a named tuple, and callers read it by field (`torch.cummax(x, -1).values`).
        return getattr(torch.return_types, f"cum{mode}")(
            (reduced.values.movedim(-1, dim), reduced.indices.movedim(-1, dim))
        )

    return patch


# ── Recursive structure traversal ──────────────────────────────────────────
# All tensor utilities share this traversal. _map_leaf_tensors applies a function
# to every tensor leaf; _iter_leaf_tensors yields (path, tensor) pairs.

# Types that should not be recursed into when extracting leaf tensors. Sym* types
# carry PyTorch shape_env internals that cause infinite recursion; Enums are scalars
# with no tensor fields.
_LEAF_SKIP_TYPES: tuple[type, ...] = (type,)
if is_torch_available():
    _LEAF_SKIP_TYPES += (enum.Enum, torch.SymInt, torch.SymFloat, torch.SymBool)


def _map_leaf_tensors(obj: Any, fn: callable) -> Any:
    """Apply `fn` to every tensor in a nested structure, preserving container types.

    Mutates dicts and `__dict__`-bearing objects in place (preserving identity — callers
    rely on this so downstream pops/mutations propagate back to the original mapping);
    rebuilds lists/tuples/sets/frozensets (immutable or order-sensitive containers).
    Skips non-traversable leaf types (enum, SymInt, etc.).
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
        # `!= ""` rather than falsiness, and `str(key)` below: a dict keyed by *integers* (granite4_vision
        # keys its deepstack features by layer index) hands down a path of `0` for the first entry, which is
        # falsy — so a truthiness test renamed that leaf "output", a name the graph never declared, and its
        # real one went missing from the feed ("Required inputs (['deepstack_features.0']) are missing").
        # Only bites a container flattened at the top level: nested under a name the path is already a
        # non-empty string.
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
# Extract or cast tensors from nested model outputs.


def _class_to_path(cls: type) -> str:
    """A class as `module:qualname` — how a graph's pytree contexts and its recorded metadata both name a
    type they cannot hold a reference to."""
    return f"{cls.__module__}:{cls.__qualname__}"


def _path_to_class(path: str) -> type:
    """The class `_class_to_path` wrote, importing its module. That import is the point as much as the
    class is: importing a modeling module is what registers its `ModelOutput` types as pytree nodes, which
    a graph loaded from disk needs before it can be called with one."""
    module_name, qualname = path.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def runner_feed(runner, kwargs: dict, *, warn_unused: bool = False) -> dict:
    """The subset of `kwargs` this graph takes, keyed the way it names them.

    The rule is the same wherever a graph is called -- an encoder component, the decode step, a
    single-graph `ExportedModel` -- so it is written once: a graph refuses a kwarg it was never traced
    with, and a caller should be free to pass a processor's whole output. `warn_unused` says so out loud,
    which only the single-graph case wants (the loop drops `generate`'s bookkeeping every step).
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

    When a model returns the same tensor under two output names (e.g. `last_hidden_state`
    and `hidden_states[0]`), the ONNX optimizer deduplicates the two output nodes and
    renames one, breaking the expected name mapping. Cloning duplicates gives each output
    leaf a distinct identity so the optimizer has nothing to merge.

    `seen` pre-seeds the identities that already count as taken — pass the *input* tensors so an output the
    model hands straight back is cloned too. Returned unmutated, an input is the very value the graph's
    placeholder holds, so the pair collapses to one name and the output's wins: prophetnet's decode returns
    its `encoder_outputs.last_hidden_state` as `encoder_last_hidden_state`, and its cross-attention cache
    (filled at prefill, untouched at decode) comes back under the name it went in with — both leaving the
    input name the runtime feeds absent from the session. A *mutated* input is a distinct value by then, so
    this only adds a copy where the graph would otherwise have lost a name.
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
    """`.device` / `.dtype` for any `nn.Module`.

    `PreTrainedModel` exposes both directly via `ModuleUtilsMixin`; a plain submodule (a `Linear` or a
    `MultiModalProjector` split out of a multi-modal model) does not, so fall back to its first parameter.
    `None` when the module has no parameters at all.
    """
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
    """Configure model and inputs for export. Mutates both `model` and `inputs` in place,
    returning `(model, inputs, output_flags)` where `output_flags` holds the values popped
    from `inputs` for `use_cache`, `return_dict`, etc. (to be applied reversibly onto
    `model.config` by `patch_model_config` during the trace).

    - Strips label inputs (`labels`, `future_values`) — loss computation is unsupported.
    - Pops output flags (`use_cache`, `return_dict`, …) from `inputs` so they don't appear
      as traced kwargs; the values are returned for the trace block to apply onto
      `model.config`.
    - Pre-computes data-dependent vision/audio kwargs registered via
      `@register_export_input_preparer` and writes them into `inputs`.
    - Casts input tensors to match the model's `dtype` / `device`.
    """
    # Strip label inputs — loss computation is not supported during export.
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

    # Pop output flags from `inputs` and return them so the caller can decide how to
    # honour them during the trace (we don't want them as traced kwargs).
    output_flags = {flag: inputs.pop(flag) for flag in _OUTPUT_FLAGS if flag in inputs}

    # Drop kwargs that are `None`: `torch.export` still records them as placeholders carrying no value, so
    # the graph declares an "input" there and dynamo then demands the key back on every call — a hole the
    # runtime has to fill with `None` for no benefit. Only when the parameter's default is `None` too, or
    # omitting it would switch the traced path (a `use_cache=True` default handed `None` would flip to True).
    forward = getattr(model, "forward", None)
    if forward is not None:
        parameters = inspect.signature(forward).parameters
        for name in [name for name, value in inputs.items() if value is None]:
            parameter = parameters.get(name)
            if parameter is not None and parameter.default is None:
                inputs.pop(name)

    # Pre-compute data-dependent vision/audio tensors that use loops, .tolist(),
    # repeat_interleave, or itertools.groupby — untraceable by dynamo.
    # TODO: use the collator API once it covers these cases.
    with torch.no_grad():
        # A decomposed component is a plain `nn.Module` and need not carry a config: an encoder-decoder's
        # `FSMTEncoder`, an RNN-T's `ParakeetRNNTDecoder`. Nothing to precompute from, so skip it — the
        # data-dependent tensors below are all config-derived.
        if (config := getattr(model, "config", None)) is not None:
            inputs.update(precompute_export_inputs(config, inputs))

    # Move input tensors onto the model's device (e.g. a cache built on CPU before a backend moved the
    # model). Dtypes are left as-is on purpose: inputs already carry the caller's/model's dtype, and cache
    # entries keep the dtype the model allocated them at — notably SSM/recurrent states the mixer holds in
    # fp32 for scan stability even in a bf16 model — which a blanket downcast would corrupt, making an
    # exported decode step diverge from eager.
    device = module_device(model)
    if device is not None:
        inputs = cast_leaf_tensors(inputs, dtype=None, device=device)

    return model, inputs, output_flags
