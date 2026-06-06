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

"""ONNX exporter.

Extends `DynamoExporter` with five extra stages that convert an `ExportedProgram`
into an ONNX model via `torch.onnx.export`:

1. **Torch patches** (`_PATCHES["onnx"]` via `apply_patches("onnx")`): reversibly
   monkey-patch `torch` ops at tracing time so `torch.export` and `torch.onnx.export`
   emit ONNX-lowerable patterns. Reverted on exit.
2. **ONNX patches** (`_PATCHES["onnx"]` via `apply_patches("onnx")`): reversibly
   hook `torch.onnx` internals — specifically `_prepare_exported_program_for_export`,
   so the FX node fixes (stage 3) run again right after `run_decompositions`.
   Same registry as stage 1, installed by the same `apply_patches` call.
3. **FX node fixes** (`_FX_NODE_FIXES["onnx"]` via `apply_fx_node_fixes("onnx", gm)`):
   per-node in-place rewrites on the `GraphModule` to drop or replace nodes ONNX
   can't lower (alias, in-place ops, dead comparisons, `_assert_*`, …). Triggered
   both directly after `torch.export` and indirectly via the stage 2 hook.
4. **ONNX translations** (`_ONNX_TRANSLATION_TABLE`): custom onnxscript functions
   passed as `custom_translation_table` that override the default torchlib
   lowering for specific aten ops where it's buggy or missing.
5. **ONNX IR fixes** (`_IR_FIXES` via `apply_onnx_ir_fixes`): post-export in-place
   fixes on the `ONNXProgram` IR for ORT compatibility.
"""

from __future__ import annotations

import copy
import functools
import operator
from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_onnxscript_available, is_torch_available
from .exporter_dynamo import DynamoExporter
from .utils import (
    apply_fx_node_fixes,
    apply_patches,
    duplicate_leaf_tensors,
    get_leaf_tensors,
    register_fx_node_fix,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.onnx import ONNXProgram

    from .. import masking_utils


if is_onnxscript_available():
    import onnx_ir
    from onnxscript.function_libs.torch_lib.ops.core import aten_index_put
    from onnxscript.onnx_opset import opset18 as op

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

    if is_onnxscript_available():
        from onnxscript.function_libs.torch_lib.ops.core import BOOL, INT64, TReal


logger = logging.get_logger(__file__)


class OnnxExporter(DynamoExporter):
    """Exporter that converts a [`PreTrainedModel`] to an ONNX `ONNXProgram`.

    Example:

    ```python
    >>> from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

    >>> exporter = OnnxExporter(export_config=OnnxConfig(dynamic=True))
    >>> onnx_program = exporter.export(model, inputs)
    >>> outputs = onnx_program(**inputs)  # run in-memory
    >>> OnnxExporter(export_config=OnnxConfig(f="model.onnx")).export(model, inputs)  # save to disk
    ```
    """

    export_config: OnnxConfig

    required_packages = ["torch", "onnx", "onnxscript"]

    def export(self, model: PreTrainedModel, sample_inputs: dict[str, Any]) -> ONNXProgram:
        with patch_model_outputs(model) as (inputs_names, outputs_names), apply_patches("onnx"):
            exported_program: ExportedProgram = super().export(model, sample_inputs)
            inputs_names, outputs_names = disambiguate_io_names(inputs_names, outputs_names)
            apply_fx_node_fixes("onnx", exported_program.graph_module)
            onnx_program: ONNXProgram = torch.onnx.export(
                exported_program,
                args=(),
                f=self.export_config.f,
                input_names=inputs_names,
                output_names=outputs_names,
                kwargs=copy.deepcopy(sample_inputs),
                custom_translation_table=_ONNX_TRANSLATION_TABLE,
                opset_version=self.export_config.opset_version,
                external_data=self.export_config.external_data,
                export_params=self.export_config.export_params,
                optimize=self.export_config.optimize,
            )

        apply_onnx_ir_fixes(onnx_program)
        return onnx_program


# ── ONNX helpers ────────────────────────────────────────────────────────────
# Model forward wrapper and I/O naming used by OnnxExporter.export.


@contextmanager
def patch_model_outputs(model):
    """Wrap `model.forward` to return a flat `dict[str, Tensor]` with duplicated outputs,
    and capture the input/output tensor names from the traced forward in the yielded
    `(inputs_names, outputs_names)` lists.
    """

    inputs_names: list[str] = []
    outputs_names: list[str] = []
    original_forward = model.forward

    @functools.wraps(original_forward)
    def patched_forward(*args, **kwargs):
        outputs = get_leaf_tensors(duplicate_leaf_tensors(original_forward(*args, **kwargs)))
        inputs_names.extend(get_leaf_tensors(kwargs).keys())
        outputs_names.extend(outputs.keys())
        return outputs

    try:
        model.forward = patched_forward
        yield inputs_names, outputs_names
    finally:
        model.forward = original_forward


def disambiguate_io_names(inputs_names: list[str], outputs_names: list[str]) -> tuple[list[str], list[str]]:
    """Prefix any name that appears in both lists with `input.` / `output.`."""
    for name in set(inputs_names).intersection(set(outputs_names)):
        inputs_names[inputs_names.index(name)] = f"input.{name}"
        outputs_names[outputs_names.index(name)] = f"output.{name}"
    return inputs_names, outputs_names


# ── Stage 1: Torch patches ─────────────────────────────────────────────────────
# Each `_patch_*(original)` factory is registered via `@register_patch("onnx", path)`,
# where `path` is the dotted Python path of the attribute to swap (e.g. `"torch.where"`,
# `"torch.Tensor.unsqueeze"`). Installation and restoration go through `apply_patches`.
#
# To add a new patch: define a `_patch_*` factory and decorate it.


@register_patch("onnx", "torch.where")
def _patch_where(original):
    """Normalize dtypes and scalars in torch.where."""

    def patch(condition, x=None, y=None):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype != y.dtype:
            y = y.to(x.dtype)
        elif isinstance(x, torch.Tensor) and isinstance(y, (int, float, bool)):
            y = torch.tensor(y, dtype=x.dtype, device=x.device)
        elif isinstance(y, torch.Tensor) and isinstance(x, (int, float, bool)):
            x = torch.tensor(x, dtype=y.dtype, device=y.device)
        if x is None and y is None:
            return original(condition)
        elif y is None:
            return original(condition, x)
        else:
            return original(condition, x, y)

    return patch


@register_patch("onnx", "torch.unsqueeze")
@register_patch("onnx", "torch.Tensor.unsqueeze")
def _patch_unsqueeze(original):
    """Support complex tensors in torch.unsqueeze."""

    def patch(self_or_input, dim):
        if torch.is_complex(self_or_input):
            real = original(self_or_input.real, dim)
            imag = original(self_or_input.imag, dim)
            return torch.complex(real, imag)
        return original(self_or_input, dim)

    return patch


@register_patch("onnx", "torch.nn.functional.scaled_dot_product_attention")
def _patch_scaled_dot_product_attention(original):
    """Handle GQA/MHA head mismatch and 5D blocked attention tensors."""

    def patch(query, key, *args, enable_gqa: bool = False, **kwargs):
        if enable_gqa and query.shape[1] == key.shape[1]:
            enable_gqa = False

        if query.dim() == 5:
            B, G = query.shape[0], query.shape[1]
            query = query.flatten(0, 1)
            key = key.flatten(0, 1)
            value = args[0].flatten(0, 1)
            args = (value,) + args[1:]
            if kwargs.get("attn_mask") is not None and kwargs["attn_mask"].dim() == 5:
                kwargs["attn_mask"] = kwargs["attn_mask"].flatten(0, 1)
            out = original(query, key, *args, enable_gqa=enable_gqa, **kwargs)
            return out.unflatten(0, (B, G))

        return original(query, key, *args, enable_gqa=enable_gqa, **kwargs)

    return patch


@register_patch("onnx", "transformers.masking_utils._vmap_expansion_sdpa")
def _patch_broadcast_mask_expansion(_original):
    """Replace vmap-based mask expansion with broadcast expansion."""

    def patch(mask_function):
        def _expanded(batch_arange, head_arange, q_arange, kv_arange):
            brodcasted = masking_utils._non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange)
            result = mask_function(*brodcasted).expand(
                batch_arange.shape[0], head_arange.shape[0], q_arange.shape[0], kv_arange.shape[0]
            )
            return result

        return _expanded

    return patch


@register_patch("onnx", "torch.nn.RMSNorm.forward")
def _patch_rms_norm_forward(original):
    """Use non-fused RMS normalization when elementwise_affine is False."""

    def patch(self, x):
        if not self.elementwise_affine:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return original(self, x)

    return patch


@register_patch("onnx", "torch.randperm")
def _patch_randperm(original):
    """Implement randperm via argsort(rand(n)) — no ONNX decomposition for aten.randperm."""

    def patch(n, *, dtype=torch.int64, layout=torch.strided, device=None, pin_memory=False, generator=None):
        return torch.argsort(torch.rand(n, device=device)).to(dtype)

    return patch


def _patch_cummax_or_cummin(original, *, mode: str):
    """Decompose cummax/cummin via triangular-mask reduction (O(N^2) memory)."""

    def patch(input, dim):
        n = input.shape[dim]
        x = input.movedim(dim, -1)  # (..., n)
        x_grid = x.unsqueeze(-2).expand(*x.shape[:-1], n, n)  # (..., n, n)
        include = torch.ones(n, n, dtype=torch.bool, device=input.device).tril()
        if input.dtype == torch.bool:
            fill_val = mode != "max"
        elif input.is_floating_point():
            fill_val = torch.finfo(input.dtype).min if mode == "max" else torch.finfo(input.dtype).max
        else:
            fill_val = torch.iinfo(input.dtype).min if mode == "max" else torch.iinfo(input.dtype).max
        fill = torch.full((), fill_val, dtype=input.dtype, device=input.device)
        masked = torch.where(include, x_grid, fill)
        out = masked.max(dim=-1) if mode == "max" else masked.min(dim=-1)
        return out.values.movedim(-1, dim), out.indices.movedim(-1, dim)

    return patch


@register_patch("onnx", "torch.cummax")
@register_patch("onnx", "torch.Tensor.cummax")
def _patch_cummax(original):
    return _patch_cummax_or_cummin(original, mode="max")


@register_patch("onnx", "torch.cummin")
@register_patch("onnx", "torch.Tensor.cummin")
def _patch_cummin(original):
    return _patch_cummax_or_cummin(original, mode="min")


@register_patch("onnx", "torch.bucketize")
def _patch_bucketize(original):
    """Vectorized bucketize avoiding scalar-constant tensors that cause alias/detach issues."""

    def patch(input, boundaries, *, out_int32=False, right=False):
        if boundaries.numel() == 0:
            result = torch.zeros_like(input, dtype=torch.int64)
            return result.to(torch.int32) if out_int32 else result
        if right:
            mask = boundaries <= input.unsqueeze(-1)
        else:
            mask = boundaries < input.unsqueeze(-1)
        result = mask.sum(-1)
        return result.to(torch.int32) if out_int32 else result

    return patch


@register_patch("onnx", "torch.searchsorted")
def _patch_searchsorted(original):
    """Decompose searchsorted via broadcast comparison + sum — no ONNX op for searchsorted.

    For sorted inputs the insertion index equals the count of elements satisfying
    the comparison (< for left, <= for right). This is O(N*M) instead of the
    real binary-search O(M log N) but only uses ops with ONNX translations.
    """

    def patch(sorted_sequence, values, *, out_int32=False, right=False, side=None, out=None, sorter=None):
        if side is not None:
            right = side == "right"
        if right:
            mask = sorted_sequence.unsqueeze(-1) <= values.unsqueeze(-2)
        else:
            mask = sorted_sequence.unsqueeze(-1) < values.unsqueeze(-2)
        result = mask.sum(-2)
        return result.to(torch.int32) if out_int32 else result

    return patch


@register_patch("onnx", "torch.full")
def _patch_full(original):
    """Force dtype=torch.long when fill_value is int and no dtype specified (ONNX defaults to float32)."""

    def patch(*args, dtype=None, **kwargs):
        if dtype is None:
            # find fill_value: positional arg or kwarg
            fill_value = kwargs.get("fill_value", args[1] if len(args) > 1 else None)
            if isinstance(fill_value, int):
                dtype = torch.long
        return original(*args, dtype=dtype, **kwargs)

    return patch


@register_patch("onnx", "torch.masked.mean")
def _patch_masked_mean(original):
    """Manual masked mean: avoids sum/int_count Div type mismatch in ONNX."""

    def patch(input, *, mask, dim=None, keepdim=False, dtype=None):
        mask_float = mask.float()
        n = mask_float.sum(dim=dim, keepdim=True).clamp(min=1.0)
        result = (input * mask_float).sum(dim=dim, keepdim=keepdim) / (n if keepdim else n.squeeze())
        return result.to(dtype) if dtype is not None else result

    return patch


@register_patch("onnx", "torch.masked.var")
def _patch_masked_var(original):
    """Manual masked var: avoids sum/int_count Div type mismatch in ONNX."""

    def patch(input, *, mask, dim=None, keepdim=False, unbiased=True):
        mask_float = mask.float()
        n = mask_float.sum(dim=dim, keepdim=True).clamp(min=1.0)
        mean = (input * mask_float).sum(dim=dim, keepdim=True) / n
        var = ((input - mean).pow(2) * mask_float).sum(dim=dim, keepdim=keepdim)
        denom = (n - 1.0) if unbiased else n
        if not keepdim:
            denom = denom.squeeze()
        return var / denom.clamp(min=1.0)

    return patch


@register_patch("onnx", "torch.Tensor.masked_scatter")
def _patch_masked_scatter(original):
    """Cumsum-gather-where strategy for masked_scatter (avoids ScatterND ORT failures)."""

    def patch(self, mask, source):
        mask = mask.expand_as(self)
        flat_mask = mask.reshape(-1)
        positions = (flat_mask.to(torch.int64).cumsum(0) - 1).clamp(min=0)
        gathered = source.reshape(-1)[positions]
        return torch.where(flat_mask, gathered, self.reshape(-1)).reshape(self.shape)

    return patch


# ── Stage 2: ONNX patches ──────────────────────────────────────────────────────
# Reversible swaps of `torch.onnx` internals via `@register_patch("onnx", path)`.
# Currently a single hook that intercepts the private `_prepare_exported_program_for_export`
# step so the FX node fixes (stage 3) run immediately after `run_decompositions` —
# any new symbolic-guard nodes the ONNX decomposition introduces get repaired before
# the FX → ONNX lowering picks them up.


@register_patch("onnx", "torch.onnx._internal.exporter._core._prepare_exported_program_for_export")
def _patch_prepare_for_export(original):
    """Run the FX node fixes immediately after the ONNX internal decomposition step.

    `torch.onnx.export` internally calls `run_decompositions` with the ONNX
    decomposition table, which can introduce new symbolic-guard nodes (e.g.
    `operator.le(sym_size, int_oo)`). These overflow during ONNX translation.
    Wrapping the prepare step lets us apply our FX fixes immediately after.

    <Tip warning={true}>

    This hooks `torch.onnx._internal.exporter._core._prepare_exported_program_for_export`,
    a private PyTorch API. It may break on PyTorch version upgrades. If it does,
    find the new entry point in `torch/onnx/_internal/exporter/_core.py`
    where `ExportedProgram.run_decompositions` is called and hook there instead.

    </Tip>
    """

    def patch(ep, *, registry):
        result = original(ep, registry=registry)
        apply_fx_node_fixes("onnx", result.graph_module)
        return result

    return patch


# ── Stage 3: FX node fixes ───────────────────────────────────────────────────
# `@register_fx_node_fix("onnx")` on `(gm, node) -> bool` per-node fixers, applied
# in place by `apply_fx_node_fixes("onnx", gm)`. Return `True` to consume the node;
# DCE runs at the end of the walk. Triggered twice in the pipeline: once explicitly
# after `torch.export`, once via the stage 2 patch after `run_decompositions`.


_COMPARISON_OPS = frozenset({operator.le, operator.lt, operator.ge, operator.gt, operator.eq, operator.ne})


@register_fx_node_fix("onnx")
def _fix_dead_comparison(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Erase or constant-fold comparison nodes involving symbolic infinities.

    torch.export emits guards like ``%le_3 = operator.le(sym_size, int_oo)`` where
    ``int_oo`` is a sympy ``IntInfinity`` object.  The ONNX translator tries to lower it
    to a C long and overflows.  Two cases handled:

    * No users → erase the node outright (PyTorch DCE skips Python callables).
    * Any arg is a non-FX-Node constant (e.g. ``int_oo``) → evaluate the comparison at
      graph-construction time, replace all uses with the Python bool result, and erase.
    """
    if node.target not in _COMPARISON_OPS:
        return False
    if len(node.users) == 0:
        gm.graph.erase_node(node)
        return True
    # Check if any arg is a compile-time constant (not a graph Node).
    if any(not isinstance(a, torch.fx.Node) for a in node.args):
        try:
            result = node.target(*node.args)
        except Exception:
            return False
        node.replace_all_uses_with(result)
        gm.graph.erase_node(node)
        return True
    return False


@register_fx_node_fix("onnx")
def _fix_alias(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace alias(x) -> x to break the alias -> detach_ -> index_put_ chain."""
    if node.target is not torch.ops.aten.alias.default:
        return False
    node.replace_all_uses_with(node.args[0])
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("onnx")
def _fix_detach_inplace(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace in-place detach_ with out-of-place detach."""
    if node.target is not torch.ops.aten.detach_.default:
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.detach.default, args=node.args, kwargs=node.kwargs)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("onnx")
def _fix_index_put_inplace(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace in-place index_put_ with out-of-place index_put."""
    if node.target is not torch.ops.aten.index_put_.default:
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.index_put.default, args=node.args, kwargs=node.kwargs)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


_ASSERTION_OPS = set()
if is_torch_available():
    _ASSERTION_OPS.update(
        {
            torch.ops.aten._assert_async.default,
            torch.ops.aten._assert_async.msg,
            torch.ops.aten._assert_scalar.default,
            torch.ops.aten._assert_tensor_metadata.default,
            torch.ops.aten.sym_constrain_range_for_size.default,
        }
    )


@register_fx_node_fix("onnx")
def _fix_assertion(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Erase assertion / shape-constraint nodes that have no ONNX equivalent."""
    if node.target not in _ASSERTION_OPS:
        return False
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("onnx")
def _fix_fill_diagonal_inplace(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace in-place fill_diagonal_ with out-of-place equivalent."""
    if node.target is not torch.ops.aten.fill_diagonal_.default:
        return False
    with gm.graph.inserting_before(node):
        tensor_arg = node.args[0]
        fill_value = node.args[1]
        # Build diagonal mask and use where
        rows = gm.graph.call_function(torch.ops.aten.sym_size.int, args=(tensor_arg, 0))
        cols = gm.graph.call_function(torch.ops.aten.sym_size.int, args=(tensor_arg, 1))
        eye = gm.graph.call_function(torch.ops.aten.eye.default, args=(rows, cols))
        eye_bool = gm.graph.call_function(torch.ops.aten.to.dtype, args=(eye, torch.bool))
        fill_tensor = gm.graph.call_function(torch.ops.aten.full_like.default, args=(tensor_arg, fill_value))
        new = gm.graph.call_function(torch.ops.aten.where.self, args=(eye_bool, fill_tensor, tensor_arg))
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("onnx")
def _fix_triu_inplace(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace in-place triu_ with out-of-place triu."""
    if node.target is not torch.ops.aten.triu_.default:
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.triu.default, args=node.args, kwargs=node.kwargs)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("onnx")
def _fix_sort_stable(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace aten.sort.stable with aten.sort.default (which has ONNX translation)."""
    if node.target is not torch.ops.aten.sort.stable:
        return False
    self_arg = node.args[0]
    dim = node.args[2] if len(node.args) > 2 else -1
    descending = node.args[3] if len(node.args) > 3 else False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.sort.default, args=(self_arg, dim, descending))
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("onnx")
def _fix_remainder_scalar(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite remainder.Scalar to remainder.Tensor when the 'scalar' arg is actually a tensor.

    After decomposition the second operand of ``aten.remainder.Scalar`` can be a graph
    node (SymbolicTensor) rather than a Python scalar.  The ONNX torchlib translation for
    ``remainder.Scalar`` calls ``int()`` on it and crashes.  Rewriting to
    ``remainder.Tensor`` uses the two-tensor ONNX translation which handles this correctly.
    """
    if node.target is not torch.ops.aten.remainder.Scalar:
        return False
    if len(node.args) < 2 or not isinstance(node.args[1], torch.fx.Node):
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.remainder.Tensor, args=node.args)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


# ── Stage 4: ONNX translations ────────────────────────────────────────────────
# Custom onnxscript `_aten_*` functions registered in `_ONNX_TRANSLATION_TABLE`
# that override `torchlib`'s default lowering for specific aten ops where the
# default is buggy or missing. Passed to `torch.onnx.export` as `custom_translation_table`.


def _values_broadcast_to_self(values: TReal, self: TReal) -> bool:
    """Static-shape check: does ``values.shape`` broadcast against ``self.shape``?

    Returns ``True`` only when every dim of ``values`` is statically known and either
    equals the corresponding (right-aligned) dim of ``self`` or is ``1``. Used to dispatch
    `_aten_index_put` between the broadcast and flat-gather paths — bailing on dynamic /
    unknown dims keeps us on the safe flat-gather fallback.
    """
    if values.shape is None or self.shape is None or len(values.shape) > len(self.shape):
        return False
    offset = len(self.shape) - len(values.shape)
    for v_dim, s_dim in zip(values.shape, self.shape[offset:]):
        try:
            v_dim, s_dim = int(v_dim), int(s_dim)
        except (TypeError, ValueError):
            return False
        if v_dim != 1 and v_dim != s_dim:
            return False
    return True


def _aten_index_put(
    self: TReal,
    indices: Sequence[INT64 | BOOL | None],
    values: TReal,
    accumulate: bool = False,
) -> TReal:
    """Bool-mask index_put with two paths; delegates non-bool-mask cases to torchlib.

    For `self[bool_mask] = values`, PyTorch supports two distinct shapes for ``values``:
    1. Broadcasts against ``self.shape`` (e.g. scalar `tensor[~mask] = 0`) — handled by
       `Expand(values, Shape(self)) + Where(mask, expanded, self)`.
    2. Equals ``bool_mask.sum()`` along its first dim, with remaining dims matching
       ``self`` (e.g. `inputs_embeds[image_mask] = image_features_flat`) — handled by
       the flat cumulative-count-Gather + Where trick.

    Path 1 is correct only when broadcast-compatibility can be statically verified — for
    dynamic shapes we fall through to path 2, which is also torchlib's default behaviour.
    """
    bool_mask = indices[0]
    is_bool = (
        bool_mask is not None and getattr(getattr(bool_mask, "type", None), "dtype", None) == onnx_ir.DataType.BOOL
    )
    if not is_bool:
        return aten_index_put(self, indices, values, accumulate)
    for _ in range(len(self.shape) - len(bool_mask.shape)):
        bool_mask = op.Unsqueeze(bool_mask, op.Constant(value_ints=[-1]))
    expanded_mask = op.Expand(bool_mask, op.Shape(self))
    if _values_broadcast_to_self(values, self):
        expanded_values = op.Expand(values, op.Shape(self))
        return op.Where(expanded_mask, expanded_values, self)
    flat_mask = op.Reshape(expanded_mask, op.Constant(value_ints=[-1]))
    flat_mask_int = op.Cast(flat_mask, to=7)  # INT64
    cs = op.CumSum(flat_mask_int, op.Constant(value_ints=[0]))
    positions = op.Clip(op.Sub(cs, op.Constant(value_ints=[1])), op.Constant(value_ints=[0]))
    flat_values = op.Reshape(values, op.Constant(value_ints=[-1]))
    gathered = op.Gather(flat_values, positions)
    flat_self = op.Reshape(self, op.Constant(value_ints=[-1]))
    result = op.Where(flat_mask, gathered, flat_self)
    return op.Reshape(result, op.Shape(self))


def _aten_bincount(self: INT64, weights=None, minlength: int = 0) -> INT64:
    """ONNX implementation of `torch.bincount`: count occurrences of non-negative ints.

    No native ONNX op. We use `OneHot(self, depth=max+1, values=[0,1])` then `ReduceSum`
    along the input axis. Weights are unused (splinter's only caller passes none).
    """
    one = op.Constant(value_ints=[1])
    max_val = op.Unsqueeze(op.ReduceMax(self, keepdims=0), op.Constant(value_ints=[0]))
    depth = op.Add(max_val, one)
    if minlength > 0:
        depth = op.Max(depth, op.Constant(value_ints=[minlength]))
    one_hot = op.OneHot(self, depth, op.Constant(value_ints=[0, 1]), axis=-1)
    return op.ReduceSum(one_hot, op.Constant(value_ints=[0]), keepdims=0)


_ONNX_TRANSLATION_TABLE: dict[Any, Any] = {}
if is_onnxscript_available():
    _ONNX_TRANSLATION_TABLE.update(
        {
            torch.ops.aten.index_put.default: _aten_index_put,
            torch.ops.aten.bincount.default: _aten_bincount,
        }
    )


# ── Stage 5: ONNX IR fixes ────────────────────────────────────────────────────
# Post-export in-place fixes to the `ONNXProgram` IR for ORT compatibility. Each
# fix has signature `(graph_like) -> None` and is applied to both the top-level
# graph and every function via `apply_onnx_ir_fixes`.
#
# Unlike the other stages, this one is a plain `_IR_FIXES` list rather than a
# decorator-driven registry — there's currently only one entry and we expect ORT
# to fix the underlying bug upstream soon, so the registry boilerplate isn't worth it.
#
# To add a new fix: implement `_fix_ir_*` and append to `_IR_FIXES`.


def _fix_ir_topk_sorted(graph_like: onnx_ir.Graph) -> None:
    """Set sorted=1 on TopK nodes (ORT CUDA EP rejects TopK without it)."""
    for ir_node in list(graph_like.all_nodes()):
        if ir_node.op_type == "TopK":
            ir_node.attributes["sorted"] = onnx_ir.Attr("sorted", onnx_ir.AttributeType.INT, 1)


_IR_FIXES = [
    _fix_ir_topk_sorted,
]


def apply_onnx_ir_fixes(onnx_program: ONNXProgram) -> None:
    """Apply each `(graph_like) -> None` IR fix to the main graph and every function."""
    graphs = [onnx_program.model.graph, *onnx_program.model.functions.values()]
    for fix in _IR_FIXES:
        for graph in graphs:
            fix(graph)
