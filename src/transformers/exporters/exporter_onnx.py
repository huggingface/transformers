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

"""ONNX exporter utilities.

This module provides the `OnnxExporter` class and helper functions used to
export PyTorch models to ONNX via TorchDynamo and `torch.onnx.export`.

The export pipeline has four stages, each with its own set of patches/fixes:

1. **Torch patches** (`patch_torch_ops`): monkey-patch PyTorch ops at tracing
   time to avoid problematic decompositions or unsupported patterns.
2. **FX graph patches** (`patch_fx_graph`): rewrite the FX graph produced by
   `torch.export` before `run_decompositions` to remove or replace nodes
   that cannot be lowered to ONNX.
3. **ONNX translations** (`_ONNX_TRANSLATION_TABLE`): custom onnxscript
   functions registered via `custom_translation_table` to override the
   default torchlib lowering for specific aten ops.
4. **ONNX IR patches** (`patch_onnx_ir`): post-export fixes to the ONNX IR
   for ORT compatibility.
"""

import copy
import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import OnnxConfig
from ..utils.import_utils import is_onnxscript_available, is_torch_available
from .exporter_dynamo import DynamoExporter
from .utils import dedup_output_tensors, get_inputs_outputs_names, get_leaf_tensors, prepare_for_export


if is_torch_available():
    import torch

    from .. import masking_utils

if is_onnxscript_available():
    import onnx_ir
    from onnxscript.function_libs.torch_lib.ops.core import aten_index_put
    from onnxscript.onnx_opset import opset18 as op

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

    if is_torch_available():
        from torch.export import ExportedProgram
        from torch.onnx import ONNXProgram


logger = logging.get_logger(__file__)


class OnnxExporter(DynamoExporter):
    """Exporter that converts `PreTrainedModel` instances to ONNX.

    Orchestrates the four-stage export pipeline: torch patches, FX graph
    patches, ONNX translation overrides, and ONNX IR patches.
    """

    export_config: OnnxConfig

    required_packages = ["torch", "onnx", "onnxscript"]

    def export(self, model: "PreTrainedModel", sample_inputs: dict[str, Any]) -> "ONNXProgram":
        """Export a model to ONNX using TorchDynamo."""
        inputs = copy.deepcopy(sample_inputs)
        model, inputs = prepare_for_export(model, inputs)

        with patch_model(model), patch_torch_ops():
            inputs_names, outputs_names = get_inputs_outputs_names(model, inputs)
            exported_program: ExportedProgram = super().export(model, inputs)
            patch_fx_graph(exported_program.graph_module)
            onnx_program: ONNXProgram = torch.onnx.export(
                exported_program,
                args=(),
                kwargs=inputs,
                f=self.export_config.f,
                input_names=inputs_names,
                output_names=outputs_names,
                custom_translation_table=_ONNX_TRANSLATION_TABLE,
                opset_version=self.export_config.opset_version,
                external_data=self.export_config.external_data,
                export_params=self.export_config.export_params,
                optimize=self.export_config.optimize,
            )

        patch_onnx_ir(onnx_program)
        return onnx_program


@contextmanager
def patch_model(model):
    """Temporarily wrap model.forward to return flat dict[str, Tensor] with deduped outputs."""

    original_forward = model.forward

    @functools.wraps(original_forward)
    def patched_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        return get_leaf_tensors(dedup_output_tensors(outputs), default="output")

    try:
        model.forward = patched_forward
        yield
    finally:
        model.forward = original_forward


# ── Stage 1: Torch patches ─────────────────────────────────────────────────────
# Monkey-patches applied during torch.export / Dynamo tracing.
# Each _patch_* function is a factory: receives the original op and returns the
# replacement, closing over the original.
#
# _TORCH_PATCH_TABLE is a list of (object, attr, factory) triples.
# patch_torch_ops installs them and restores on exit.
#
# To add a new patch: define a _patch_* factory and append to _TORCH_PATCH_TABLE.


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


def _patch_unsqueeze(original):
    """Support complex tensors in torch.unsqueeze."""

    def patch(self_or_input, dim):
        if torch.is_complex(self_or_input):
            real = original(self_or_input.real, dim)
            imag = original(self_or_input.imag, dim)
            return torch.complex(real, imag)
        return original(self_or_input, dim)

    return patch


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


def _patch_rms_norm_forward(original):
    """Use non-fused RMS normalization when elementwise_affine is False."""

    def patch(self, x):
        if not self.elementwise_affine:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return original(self, x)

    return patch


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


def _patch_masked_mean(original):
    """Manual masked mean: avoids sum/int_count Div type mismatch in ONNX."""

    def patch(input, *, mask, dim=None, keepdim=False, dtype=None):
        mask_float = mask.float()
        n = mask_float.sum(dim=dim, keepdim=True).clamp(min=1.0)
        result = (input * mask_float).sum(dim=dim, keepdim=keepdim) / (n if keepdim else n.squeeze())
        return result.to(dtype) if dtype is not None else result

    return patch


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


def _patch_masked_scatter(original):
    """Cumsum-gather-where strategy for masked_scatter (avoids ScatterND ORT failures)."""

    def patch(self, mask, source):
        mask = mask.expand_as(self)
        flat_mask = mask.reshape(-1)
        positions = (flat_mask.to(torch.int64).cumsum(0) - 1).clamp(min=0)
        gathered = source.reshape(-1)[positions]
        return torch.where(flat_mask, gathered, self.reshape(-1)).reshape(self.shape)

    return patch


# (object, attribute, factory) triples installed by patch_torch_ops.
_TORCH_PATCH_TABLE = [
    (torch, "where", _patch_where),
    (torch, "unsqueeze", _patch_unsqueeze),
    (torch.Tensor, "unsqueeze", _patch_unsqueeze),
    (torch.nn.functional, "scaled_dot_product_attention", _patch_scaled_dot_product_attention),
    (masking_utils, "_vmap_expansion_sdpa", _patch_broadcast_mask_expansion),
    (torch.nn.RMSNorm, "forward", _patch_rms_norm_forward),
    (torch, "randperm", _patch_randperm),
    (torch, "cummax", lambda orig: _patch_cummax_or_cummin(orig, mode="max")),
    (torch, "cummin", lambda orig: _patch_cummax_or_cummin(orig, mode="min")),
    (torch.Tensor, "cummax", lambda orig: _patch_cummax_or_cummin(orig, mode="max")),
    (torch.Tensor, "cummin", lambda orig: _patch_cummax_or_cummin(orig, mode="min")),
    (torch, "bucketize", _patch_bucketize),
    (torch.Tensor, "masked_scatter", _patch_masked_scatter),
    (torch, "full", _patch_full),
    (torch.masked, "mean", _patch_masked_mean),
    (torch.masked, "var", _patch_masked_var),
]


@contextmanager
def patch_torch_ops():
    """Context manager: install torch patches for ONNX export."""
    originals = []
    for obj, attr, factory in _TORCH_PATCH_TABLE:
        original = getattr(obj, attr)
        originals.append((obj, attr, original))
        setattr(obj, attr, factory(original))

    try:
        yield
    finally:
        for obj, attr, original in originals:
            setattr(obj, attr, original)


# ── Stage 2: FX graph patches ──────────────────────────────────────────────────
# Rewrite FX nodes between torch.export (stage 1) and run_decompositions.
# Each fixer: (gm, node) -> bool. Return True = node consumed, stop.
#
# To add a new fix: define _fix_* and append to _FX_NODE_FIXES.


def _fix_alias(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Replace alias(x) -> x to break the alias -> detach_ -> index_put_ chain."""
    if node.target is not torch.ops.aten.alias.default:
        return False
    node.replace_all_uses_with(node.args[0])
    gm.graph.erase_node(node)
    return True


def _fix_detach_inplace(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Replace in-place detach_ with out-of-place detach."""
    if node.target is not torch.ops.aten.detach_.default:
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.detach.default, args=node.args, kwargs=node.kwargs)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


def _fix_index_put_inplace(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Replace in-place index_put_ with out-of-place index_put."""
    if node.target is not torch.ops.aten.index_put_.default:
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.index_put.default, args=node.args, kwargs=node.kwargs)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


_ASSERTION_OPS = frozenset(
    {
        torch.ops.aten._assert_async.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten._assert_tensor_metadata.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
    }
)


def _fix_assertion(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Erase assertion / shape-constraint nodes that have no ONNX equivalent."""
    if node.target not in _ASSERTION_OPS:
        return False
    gm.graph.erase_node(node)
    return True


def _fix_fill_diagonal_inplace(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
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


def _fix_triu_inplace(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
    """Replace in-place triu_ with out-of-place triu."""
    if node.target is not torch.ops.aten.triu_.default:
        return False
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.triu.default, args=node.args, kwargs=node.kwargs)
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


def _fix_sort_stable(gm: "torch.fx.GraphModule", node: "torch.fx.Node") -> bool:
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


_FX_NODE_FIXES = [
    _fix_alias,
    _fix_assertion,
    _fix_detach_inplace,
    _fix_fill_diagonal_inplace,
    _fix_index_put_inplace,
    _fix_sort_stable,
    _fix_triu_inplace,
]


def patch_fx_graph(graph_module: "torch.fx.GraphModule") -> None:
    """Apply FX node fixes to all sub-GraphModules, then eliminate dead code."""
    for gm in graph_module.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue
            for fix in _FX_NODE_FIXES:
                if fix(gm, node):
                    break
        gm.graph.eliminate_dead_code()
        gm.recompile()


# ── Stage 3: Custom ONNX translations ─────────────────────────────────────────
# Override the default torchlib lowering for specific aten ops.
#
# To add a new translation: implement an _aten_* function and add to
# _ONNX_TRANSLATION_TABLE.


def _aten_index_put(self, indices, values, accumulate=False):
    """Bool-mask index_put via cumsum-gather-where; delegates other cases to torchlib."""
    bool_mask = indices[0]
    is_bool = (
        bool_mask is not None and getattr(getattr(bool_mask, "type", None), "dtype", None) == onnx_ir.DataType.BOOL
    )
    if not is_bool:
        return aten_index_put(self, indices, values, accumulate)
    for _ in range(len(self.shape) - len(bool_mask.shape)):
        bool_mask = op.Unsqueeze(bool_mask, op.Constant(value_ints=[-1]))
    expanded_mask = op.Expand(bool_mask, op.Shape(self))
    flat_mask = op.Reshape(expanded_mask, op.Constant(value_ints=[-1]))
    flat_mask_int = op.Cast(flat_mask, to=7)  # INT64
    cs = op.CumSum(flat_mask_int, op.Constant(value_ints=[0]))
    positions = op.Clip(op.Sub(cs, op.Constant(value_ints=[1])), op.Constant(value_ints=[0]))
    flat_values = op.Reshape(values, op.Constant(value_ints=[-1]))
    gathered = op.Gather(flat_values, positions)
    flat_self = op.Reshape(self, op.Constant(value_ints=[-1]))
    result = op.Where(flat_mask, gathered, flat_self)
    return op.Reshape(result, op.Shape(self))


_ONNX_TRANSLATION_TABLE = {
    torch.ops.aten.index_put.default: _aten_index_put,
}


# ── Stage 4: ONNX IR patches ──────────────────────────────────────────────────
# Post-export fixes to the ONNX IR for ORT compatibility.
#
# To add a new fix: implement _fix_ir_* and append to _IR_FIXES.


def _fix_ir_topk_sorted(graph_like: "onnx_ir.Graph") -> None:
    """Set sorted=1 on TopK nodes (ORT CUDA EP rejects TopK without it)."""
    for ir_node in list(graph_like.all_nodes()):
        if ir_node.op_type == "TopK":
            ir_node.attributes["sorted"] = onnx_ir.Attr("sorted", onnx_ir.AttributeType.INT, 1)


_IR_FIXES = [
    _fix_ir_topk_sorted,
]


def patch_onnx_ir(onnx_program: "ONNXProgram") -> None:
    """Apply ONNX IR fixes to the exported program for ORT compatibility."""
    for fix in _IR_FIXES:
        fix(onnx_program.model.graph)
        for func in onnx_program.model.functions.values():
            fix(func)


# Models that export but produce extremely inaccurate outputs.
ONNX_EXTREMELY_INACCURATE_MODEL_TYPES: set[str] = {
    "blt",  # 94.3% mismatch in last_hidden_state
    "flaubert",  # 40% mismatch in end_top_index (top-k beam search non-determinism)
    "parakeet_ctc",  # 100% NaN in logits
    "parakeet_encoder",  # 100% NaN in last_hidden_state
    "patchtst",  # NaN loss output
    "pp_doclayout_v2",  # 68.3% mismatch in enc_topk_bboxes (non-deterministic top-k selection)
    "pp_doclayout_v3",  # 68.3% mismatch in enc_topk_bboxes
    "d_fine",  # 43.3% mismatch in enc_topk_bboxes (non-deterministic top-k)
    "mm-grounding-dino",  # 25% mismatch in encoder_pred_boxes (non-deterministic top-k selection)
    "rt_detr",  # 43.3% mismatch in enc_topk_bboxes
    "rt_detr_v2",  # 43.3% mismatch in enc_topk_bboxes (non-deterministic top-k)
    "siglip2",  # 100% mismatch in logits
    "siglip2_vision_model",  # 73.8% mismatch in last_hidden_state
    "vit_mae",  # 99.3% mismatch in ids_restore (random masking)
    "xlm",  # 6.2% mismatch in end_top_index
}
