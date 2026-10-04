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

"""ONNX exporter: `DynamoExporter` followed by `torch.onnx.export`.

1. **Patches** (`apply_patches("onnx")`): reversible swaps of `torch` ops into ONNX-lowerable forms, plus a hook
   on `torch.onnx`'s `_prepare_exported_program_for_export` so the FX fixes re-run after its decompositions.
2. **FX node fixes** (`apply_fx_node_fixes("onnx", gm)`): per-node rewrites of what ONNX cannot lower (aliases,
   dead comparisons, `_assert_*`, ...).
3. **Translations** (`_get_onnx_translation_table`): onnxscript functions overriding torchlib where its lowering is
   missing or wrong (grouped matmul, varlen attention, SSM scans, ...).
4. **IR fixes** (`apply_onnx_ir_fixes`): in-place fixes on the exported IR for ONNX Runtime.
"""

from __future__ import annotations

import copy
import functools
import json
import operator
from collections.abc import MutableMapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from typing import Sequence as TypingSequence  # noqa: UP035  # onnxscript.script classifies attrs via typing.Sequence

import numpy as np

from ..utils import logging
from ..utils.import_utils import is_onnxscript_available, is_torch_available
from .configs import ExportFormat, OnnxConfig
from .exporter_dynamo import DynamoExporter
from .metadata import (
    EXPORT_METADATA_KEY,
)
from .utils import (
    _resolve_dotted_path,
    apply_fx_node_fixes,
    apply_patches,
    duplicate_leaf_tensors,
    get_leaf_tensors,
    register_fx_node_fix,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.onnx import ONNXProgram


if is_onnxscript_available():
    import onnx_ir
    from onnxscript import FLOAT, INT64, script  # runtime: `@script` evaluates these annotations at import
    from onnxscript.onnx_opset import opset18 as op

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

    if is_onnxscript_available():
        from onnxscript.function_libs.torch_lib.ops.core import TReal


logger = logging.get_logger(__file__)


class OnnxExporter(DynamoExporter):
    """Exporter that converts a [`PreTrainedModel`] to an ONNX `ONNXProgram`.

    Example:

    ```python
    >>> from transformers.exporters.exporter_onnx import OnnxExporter, OnnxConfig

    >>> exporter = OnnxExporter()
    >>> onnx_program = exporter.export(model, inputs, config=OnnxConfig(dynamic=True))
    >>> outputs = onnx_program(**inputs)  # run in-memory
    >>> exporter.export(model, inputs, config=OnnxConfig(output_path="model.onnx"))  # save to disk
    ```
    """

    export_format = ExportFormat.ONNX
    config_class = OnnxConfig
    artifact_suffix = ".onnx"

    required_packages = ["torch", "onnx", "onnxscript"]
    tested_versions = {"torch": "2.13.0", "onnx": "1.22.0", "onnxscript": "0.7.1"}

    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: OnnxConfig | dict[str, Any],
    ) -> ONNXProgram:
        config = self._as_config(config)

        with apply_patches("onnx"), patch_model_outputs(model) as (inputs_names, outputs_names):
            exported_program, metadata = super().export_artifact(model, sample_inputs, config=config)
            inputs_names, outputs_names = disambiguate_io_names(inputs_names, outputs_names)
            apply_fx_node_fixes("onnx", exported_program.graph_module)
            onnx_program: ONNXProgram = torch.onnx.export(
                exported_program,
                args=(),
                f=config.output_path,
                input_names=inputs_names,
                output_names=outputs_names,
                kwargs=copy.deepcopy(dict(sample_inputs)),
                custom_translation_table=_get_onnx_translation_table(),
                opset_version=config.opset_version,
                external_data=config.external_data,
                export_params=config.export_params,
                optimize=config.optimize,
            )

        apply_onnx_ir_fixes(onnx_program)
        # Read back via `session.get_modelmeta().custom_metadata_map`.
        onnx_program.model.metadata_props[EXPORT_METADATA_KEY] = json.dumps(metadata)
        return onnx_program, metadata

    @classmethod
    def save_artifact(cls, artifact, path) -> None:
        """The metadata is in `metadata_props`; ONNX decides whether initializers spill to a sidecar file."""
        artifact.save(path)


# ── ONNX helpers ────────────────────────────────────────────────────────────


@contextmanager
def patch_model_outputs(model):
    """Wrap `model.forward` to return a flat `dict[str, Tensor]`, capturing the I/O names it traces with."""

    inputs_names: list[str] = []
    outputs_names: list[str] = []
    original_forward = model.forward

    @functools.wraps(original_forward)
    def patched_forward(*args, **kwargs):
        # Before the forward: it mutates its kwargs in place, and cache states it creates would shift names.
        inputs = get_leaf_tensors(kwargs)
        inputs_names[:] = inputs.keys()
        outputs = get_leaf_tensors(
            duplicate_leaf_tensors(original_forward(*args, **kwargs), seen={id(tensor) for tensor in inputs.values()})
        )
        outputs_names[:] = outputs.keys()
        return outputs

    try:
        model.forward = patched_forward
        yield inputs_names, outputs_names
    finally:
        model.forward = original_forward


def disambiguate_io_names(inputs_names: list[str], outputs_names: list[str]) -> tuple[list[str], list[str]]:
    """Prefix any name that appears in both lists with `input.` / `output.`.

    An output named exactly `output` is prefixed too: it collides with FX's terminal node name (ORT:
    "Duplicate definition of name (output)").
    """
    collisions = set(inputs_names).intersection(outputs_names)
    return (
        [f"input.{name}" if name in collisions else name for name in inputs_names],
        [f"output.{name}" if name in collisions or name == "output" else name for name in outputs_names],
    )


# ── Stage 1: Torch patches ─────────────────────────────────────────────────────


@register_patch("onnx", "torch.unsqueeze", "torch.Tensor.unsqueeze")
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
def _patch_sdpa(original):
    """Zero rows that mask every key, the way torch's fused kernels do.

    ORT evaluates the softmax literally and returns `NaN` under a `-inf` mask (parakeet), which a later
    BatchNorm spreads over the batch.
    """

    def patch(query, key, value, attn_mask=None, *args, **kwargs):
        attn_output = original(query, key, value, attn_mask, *args, **kwargs)
        if attn_mask is None:
            return attn_output
        if attn_mask.dtype == torch.bool:
            unattended = ~attn_mask.any(dim=-1, keepdim=True)
        else:
            unattended = attn_mask.amax(dim=-1, keepdim=True) <= torch.finfo(attn_mask.dtype).min
        zero = torch.zeros((), dtype=attn_output.dtype, device=attn_output.device)
        return torch.where(unattended, zero, attn_output)

    return patch


@register_patch("onnx", "torch.split", "torch.Tensor.split")
def _patch_split(original):
    """Expand a symbolic split size into statically-counted `narrow`s.

    Otherwise it lowers to `SplitToSequence`, which onnxscript's constant folder crashes on
    (`'NoneType' object has no attribute 'ndim'`).
    """

    def patch(input, split_size_or_sections, dim=0):
        if not isinstance(split_size_or_sections, torch.SymInt):
            return original(input, split_size_or_sections, dim)
        split_size = split_size_or_sections
        total = input.size(dim)
        # Specializes the count, as `aten.split.Tensor`'s meta already guards on it.
        count = int((total + split_size - 1) // split_size)
        return tuple(
            input.narrow(dim, i * split_size, torch.sym_min(split_size, total - i * split_size)) for i in range(count)
        )

    return patch


@register_patch("onnx", "torch.randperm")
def _patch_randperm(original):
    """Implement randperm via argsort(rand(n)) — no ONNX decomposition for aten.randperm."""

    def patch(n, *, dtype=torch.int64, layout=torch.strided, device=None, pin_memory=False, generator=None):
        return torch.argsort(torch.rand(n, device=device)).to(dtype)

    return patch


@register_patch("onnx", "torch.nn.RMSNorm.forward")
def _patch_rms_norm_forward(original):
    """Unfused RMSNorm without an affine weight: `aten._fused_rms_norm` has no ONNX translation (diffllama)."""

    def patch(self, x):
        if not self.elementwise_affine:
            eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + eps)).to(x.dtype)
        return original(self, x)

    return patch


@register_patch("onnx", "onnxscript.onnx_opset._impl.opset13.Opset13.Constant")
def _patch_opset13_constant(original):
    """Substitute `op.Constant(value_ints=[])` with an explicit empty INT64 tensor.

    onnxscript's `aten_index_put` can pass an empty `value_ints`, which `onnx_ir` warns has an ambiguous type.
    """

    def patch(self, *args, **kwargs):
        if kwargs.get("value_ints") == []:
            kwargs.pop("value_ints")
            kwargs["value"] = onnx_ir.tensor(np.array([], dtype=np.int64))
        return original(self, *args, **kwargs)

    return patch


@register_patch("onnx", "onnxscript.optimizer.optimize_ir")
def _patch_optimize_ir(original):
    """Skip constant-folding `Resize` nodes during onnxscript optimization.

    The pure-Python reference `Resize` takes minutes per node (~4.5 min on the YOLOS test model).
    """

    def patch(model, *args, **kwargs):
        kwargs.setdefault("should_fold", lambda node: False if node.op_type == "Resize" else None)
        with _exact_identity_rewrites():
            return original(model, *args, **kwargs)

    return patch


@register_patch("onnx", "onnx_ir.passes.common.DeduplicateInitializersPass")
def _patch_deduplicate_initializers(original):
    """Keep distinct initializers distinct, even when they hold the same values.

    A merged initializer read on two devices (DPT's zero `cls_token` / `position_embeddings`) fails to open on
    onnxruntime-gpu 1.23.2 (`SaveInitializedTensors`: `!utils::HasExternalDataInMemory`).
    """
    from onnx_ir.passes import PassResult

    class KeepInitializers(original):
        def call(self, model):
            return PassResult(model, modified=False)

    return KeepInitializers


@functools.cache
def _identity_pattern_constants() -> tuple:
    """The `0` / `1` constants in onnxscript's default rewrite rules."""
    import onnxscript.rewriter as rewriter
    from onnxscript.rewriter import _pattern_ir

    def constants(node, seen):
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, _pattern_ir.Constant):
            yield node
            return
        for value in vars(node).values() if hasattr(node, "__dict__") else ():
            for item in value if isinstance(value, (list, tuple)) else (value,):
                if hasattr(item, "__dict__") and type(item).__module__.startswith("onnxscript"):
                    yield from constants(item, seen)

    seen = set()
    return tuple(
        constant
        for rule in rewriter._DEFAULT_REWRITE_RULES
        for constant in constants(rule, seen)
        if isinstance(constant._value, (int, float)) and constant._value in (0, 1)
    )


@contextmanager
def _exact_identity_rewrites():
    """Let onnxscript's identity rewrites (`x + 0`, `x * 1`, …) fire on an exact 0 or 1 only.

    Pattern constants match within `isclose`, so a `+ 1e-10` epsilon is deleted (patchtst: `0 / 0 = nan`).
    """
    tightened = [(constant, constant._rel_tol, constant._abs_tol) for constant in _identity_pattern_constants()]
    for constant, _, _ in tightened:
        constant._rel_tol = constant._abs_tol = 0.0
    try:
        yield
    finally:
        for constant, rel_tol, abs_tol in tightened:
            constant._rel_tol, constant._abs_tol = rel_tol, abs_tol


@register_patch("onnx", "torch.chunk", "torch.Tensor.chunk")
def _patch_chunk(original):
    """Lower `chunk` via `narrow` (→ ONNX `Slice`) under dynamic shapes.

    The `SplitToSequence` it lowers to has a symbolic split length that onnx_ir's `InlinePass` rejects
    (diffllama).
    """

    def patch(input, chunks, dim=0):
        total = input.size(dim)
        if not isinstance(total, torch.SymInt):
            return original(input, chunks, dim)
        # Assumes the axis divides evenly into `chunks`; otherwise the piece count would be symbolic.
        chunk_size = (total + chunks - 1) // chunks
        splits, start = [], 0
        for i in range(chunks):
            length = (total - start) if i == chunks - 1 else chunk_size
            splits.append(input.narrow(dim, start, length))
            start = start + chunk_size
        return tuple(splits)

    return patch


@register_patch("onnx", "torch.exp", "torch.Tensor.exp")
def _patch_exp(original):
    """Lower complex `exp` via Euler; onnxscript has no complex `aten.exp`."""

    def patch(input):
        if torch.is_complex(input):
            magnitude = original(input.real)
            return torch.complex(magnitude * input.imag.cos(), magnitude * input.imag.sin())
        return original(input)

    return patch


@register_patch("onnx", "torch.fft.irfft")
def _patch_irfft(original):
    """Replace `irfft` with `ifft` over the conjugate-mirrored input (assumes even `n`).

    ORT's `DFT` rejects the `is_onesided=1` + `inverse=1` combination `irfft` lowers to.
    """

    def patch(input, n=None, dim=-1, norm=None):
        if n is None:
            n = 2 * (input.shape[dim] - 1)
        slc = [slice(None)] * input.ndim
        slc[dim] = slice(1, -1)
        full = torch.cat([input, input[tuple(slc)].flip(dims=[dim]).conj()], dim=dim)
        return torch.fft.ifft(full, n=n, dim=dim, norm=norm).real

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


@register_patch("onnx", "torch.roll")
def _patch_roll(original):
    """Replace `torch.roll(input, shifts, dims)` with explicit `narrow + cat` shifts.

    The `roll` lowering can emit an empty `Shape(start, end)`, and ORT rejects the downstream `Slice` with
    `ShapeInferenceError` (Gemma4-Unified).
    """

    def patch(input, shifts, dims=None):
        if isinstance(shifts, int) and isinstance(dims, int):
            shifts = (shifts,)
            dims = (dims,)
        elif not (isinstance(shifts, (tuple, list)) and isinstance(dims, (tuple, list)) and len(shifts) == len(dims)):
            return original(input, shifts, dims)

        out = input
        for shift, dim in zip(shifts, dims):
            length = out.size(dim)
            shift = shift % length if length > 0 else 0
            if shift == 0:
                continue
            front = out.narrow(dim, length - shift, shift)
            back = out.narrow(dim, 0, length - shift)
            out = torch.cat([front, back], dim=dim)
        return out

    return patch


# ── Stage 2: ONNX patches ──────────────────────────────────────────────────────


@register_patch("onnx", "torch.onnx._internal.exporter._core._prepare_exported_program_for_export")
def _patch_prepare_for_export(original):
    """Re-run the FX node fixes after `torch.onnx`'s internal `run_decompositions`.

    The decomposition can introduce new guards (`operator.le(sym_size, int_oo)`) that overflow in
    translation. Hooks a private PyTorch API: if it moves, hook wherever `_core.py` calls
    `run_decompositions`.
    """

    def patch(ep, *, registry):
        result = original(ep, registry=registry)
        apply_fx_node_fixes("onnx", result.graph_module)
        return result

    return patch


# ── Stage 3: FX node fixes ───────────────────────────────────────────────────


_COMPARISON_OPS = frozenset({operator.le, operator.lt, operator.ge, operator.gt, operator.eq, operator.ne})


@register_fx_node_fix("onnx")
def _fix_dead_comparison(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Erase or constant-fold comparison nodes involving symbolic infinities.

    Guards like ``operator.le(sym_size, int_oo)`` overflow a C long in the ONNX translator. Unused ones are
    erased (DCE skips Python callables); ones with a constant arg are evaluated.
    """
    if node.target not in _COMPARISON_OPS:
        return False
    if len(node.users) == 0:
        gm.graph.erase_node(node)
        return True
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
def _fix_noop_squeeze(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Drop the axes of a `squeeze` that are not of size 1, which torch leaves in place.

    torchlib lowers it to `Squeeze` unconditionally and ORT rejects it (`Dimension of input 0 must be 1
    instead of 12`, mistral3). Symbolic sizes are left alone.
    """
    if node.target not in (torch.ops.aten.squeeze.dim, torch.ops.aten.squeeze.dims):
        return False
    source = node.args[0]
    value = source.meta.get("val") if isinstance(source, torch.fx.Node) else None
    if not isinstance(value, torch.Tensor) or value.dim() == 0:
        return False
    dims = [node.args[1]] if isinstance(node.args[1], int) else list(node.args[1])
    sizes = [value.shape[dim] for dim in dims]
    kept = [dim for dim, size in zip(dims, sizes) if not (isinstance(size, int) and size != 1)]
    if len(kept) == len(dims):
        return False
    if kept:
        node.target = torch.ops.aten.squeeze.dims
        node.args = (source, kept)
        return False
    node.replace_all_uses_with(source)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("onnx")
def _fix_slice_implicit_start(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Spell out a slice's implicit start as ``0``.

    ``start=None`` lowers to an `Unsqueeze` of nothing, which ORT rejects (``input 0 is marked single but has
    an empty string``, funnel) or the inliner raises a `PassError` on.
    """
    if node.target is not torch.ops.aten.slice.Tensor or len(node.args) < 3 or node.args[2] is not None:
        return False
    args = list(node.args)
    args[2] = 0
    node.args = tuple(args)
    return True


@register_fx_node_fix("onnx")
def _fix_index_put_last_dim_index(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite ``self[..., idx] = value`` as a mask + ``where`` when ``idx`` selects on the last dim.

    torchlib's `index_put` silently drops the write under dynamic shapes (chameleon's image-token mask).
    Only scalar values take this path.
    """
    if node.target not in (torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default):
        return False
    if len(node.args) < 3:
        return False
    self_arg, indices, values = node.args[0], node.args[1], node.args[2]
    accumulate = node.args[3] if len(node.args) > 3 else node.kwargs.get("accumulate", False)
    if accumulate or not isinstance(indices, (list, tuple)) or not indices or indices[-1] is None:
        return False
    if any(index is not None for index in indices[:-1]):
        return False

    index = indices[-1]
    index_val = getattr(index, "meta", {}).get("val")
    self_val = getattr(self_arg, "meta", {}).get("val")
    values_val = getattr(values, "meta", {}).get("val")
    if index_val is None or self_val is None or values_val is None:
        return False
    if index_val.dtype == torch.bool or values_val.numel() != 1 or len(indices) != self_val.ndim:
        return False

    # The write targets a view; walk back through full-`:` slices to the base later nodes read.
    base = self_arg
    while (
        base.op == "call_function"
        and base.target is torch.ops.aten.slice.Tensor
        and getattr(base.args[0], "meta", {}).get("val") is not None
        and base.meta["val"].shape == base.args[0].meta["val"].shape
    ):
        base = base.args[0]

    last_dim = self_val.ndim - 1
    with gm.graph.inserting_before(node):
        size = gm.graph.call_function(torch.ops.aten.sym_size.int, args=(self_arg, last_dim))
        arange = gm.graph.call_function(
            torch.ops.aten.arange.default, args=(size,), kwargs={"dtype": index_val.dtype, "device": index_val.device}
        )
        columns = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(arange, -1))
        selected = gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(index, 0))
        matches = gm.graph.call_function(torch.ops.aten.eq.Tensor, args=(columns, selected))
        mask = gm.graph.call_function(torch.ops.aten.any.dim, args=(matches, -1))
        result = gm.graph.call_function(torch.ops.aten.where.self, args=(mask, values, base))
        result.meta.update(node.meta)
    node.replace_all_uses_with(result)
    # Under dynamic shapes `index_put_` has no users and the mutation is lost; rewire later readers.
    ordering = {other: position for position, other in enumerate(gm.graph.nodes)}
    for user in list(base.users):
        if user is not node and ordering.get(user, -1) > ordering[node]:
            user.replace_input_with(base, result)
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
def _fix_sort_stable(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace aten.sort.stable with aten.sort.default (which has ONNX translation)."""
    if node.target is not torch.ops.aten.sort.stable:
        return False
    self_arg = node.args[0]
    # Keyword-only in the `sort.stable` schema.
    dim = node.kwargs.get("dim", -1)
    descending = node.kwargs.get("descending", False)
    with gm.graph.inserting_before(node):
        new = gm.graph.call_function(torch.ops.aten.sort.default, args=(self_arg, dim, descending))
    node.replace_all_uses_with(new)
    gm.graph.erase_node(node)
    return True


@functools.cache
def _integral_scalar_promotion_ops() -> frozenset:
    """Ops where a Python float meeting an integral tensor needs the tensor promoted first.

    Both overloads appear, since decomposition rewrites `.Tensor` to `.Scalar`. Resolved lazily so the module
    imports without torch.
    """
    return frozenset(
        {
            torch.ops.aten.rsub.Scalar,
            torch.ops.aten.sub.Scalar,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.mul.Tensor,
        }
    )


@register_fx_node_fix("onnx")
def _fix_integral_tensor_float_scalar(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Promote an integral tensor before it meets a Python float, which torch 2.13 mishandles.

    `1.0 - int_mask` crashes decomposition (pytorch/pytorch#194381) and `int_mask * 2.0` has no ONNX
    decomposition (pytorch/pytorch#194382); both export fine once the operand is already the result dtype.
    """
    if node.target not in _integral_scalar_promotion_ops():
        return False
    if len(node.args) < 2:
        return False
    tensor_arg, scalar_arg = node.args[0], node.args[1]
    scalar = scalar_arg.meta.get("val") if isinstance(scalar_arg, torch.fx.Node) else scalar_arg
    if not isinstance(tensor_arg, torch.fx.Node) or not isinstance(scalar, (float, torch.SymFloat)):
        return False
    operand, result = tensor_arg.meta.get("val"), node.meta.get("val")
    if operand is None or result is None:
        return False
    if operand.dtype.is_floating_point or not result.dtype.is_floating_point:
        return False
    with gm.graph.inserting_before(node):
        promoted = gm.graph.call_function(
            torch.ops.aten._to_copy.default, args=(tensor_arg,), kwargs={"dtype": result.dtype}
        )
    promoted.meta.update(node.meta)
    node.replace_input_with(tensor_arg, promoted)
    return True


_TENSOR_FACTORY_OPS = set()
if is_torch_available():
    _TENSOR_FACTORY_OPS.update(
        {
            torch.ops.aten.zeros.default,
            torch.ops.aten.ones.default,
            torch.ops.aten.empty.memory_format,
            torch.ops.aten.full.default,
            torch.ops.aten.new_zeros.default,
            torch.ops.aten.new_ones.default,
            torch.ops.aten.new_full.default,
        }
    )


def _sym_expr(value) -> str | None:
    """Return the sympy expression string of a SymInt-valued FX node/val, else None."""
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt):
        return str(value.node.expr)
    return None


@register_fx_node_fix("onnx")
def _fix_symbolic_factory_shape(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Size ``zeros``/``full``/… off a live tensor's runtime shape instead of a derived floor-div SymInt.

    ORT's CUDA EP mis-evaluates onnxscript's signed floor-div lowering (e.g. a conv output length) to a
    negative dim. Only factory size args are touched, never arithmetic `Div` nodes.
    """
    if node.target not in _TENSOR_FACTORY_OPS:
        return False
    size_pos = next(
        (i for i, a in enumerate(node.args) if isinstance(a, (list, tuple)) and any(_sym_expr(e) for e in a)),
        None,
    )
    if size_pos is None:
        return False
    size = list(node.args[size_pos])

    # Map each floor-div symbolic dim to a live tensor dim defined earlier in the graph.
    sources: dict[str, tuple[torch.fx.Node, int]] = {}
    for prior in gm.graph.nodes:
        if prior is node:
            break
        if prior.target in _TENSOR_FACTORY_OPS:
            continue
        val = prior.meta.get("val") if hasattr(prior, "meta") else None
        shape = getattr(val, "shape", None)
        if shape is None or isinstance(val, torch.SymInt):
            continue
        for dim, s in enumerate(shape):
            expr = _sym_expr(s)
            if expr and expr not in sources:
                sources[expr] = (prior, dim)

    changed = False
    for i, element in enumerate(size):
        expr = _sym_expr(element)
        if expr is None or "//" not in expr or expr not in sources:
            continue
        src_node, src_dim = sources[expr]
        with gm.graph.inserting_before(node):
            sym_size = gm.graph.call_function(torch.ops.aten.sym_size.int, args=(src_node, src_dim))
        sym_size.meta["val"] = element.meta["val"]
        size[i] = sym_size
        changed = True
    if not changed:
        return False
    node.update_arg(size_pos, size)
    return True


# ── Stage 4: ONNX translations ────────────────────────────────────────────────
# onnxscript lowerings overriding torchlib where its default is buggy or missing.

_ONNX_TRANSLATIONS: dict[Any, callable] = {}


def register_onnx_translation(*paths: str):
    """Append the decorated lowering `fn` to `_ONNX_TRANSLATIONS`, keyed by each op `path`.

    Each `path` is a dotted op overload or `operator` builtin, resolved at decoration time; unresolvable
    paths are skipped.
    """

    def decorator(fn):
        for path in paths:
            op = _resolve_dotted_path(path)
            if op is not None:
                _ONNX_TRANSLATIONS[op] = fn
        return fn

    return decorator


def _aten_grouped_mm(mat_a: TReal, mat_b: TReal, offs: INT64, bias=None, out_dtype=None) -> TReal:
    """ONNX implementation of `aten._grouped_mm.default`, unrolled per group as `Slice + MatMul` + `Concat`.

    `G` is static, and unrolling avoids the `(M, K, N)` materialisation of a `weight[group_idx]` gather.
    """
    G = mat_b.shape[0]
    if not isinstance(G, int):
        raise ValueError("_aten_grouped_mm: number of experts (mat_b.shape[0]) must be static at translation time")

    offs_i64 = op.Cast(offs, to=7)
    axes_0 = op.Constant(value_ints=[0])
    zero_1d = op.Constant(value_ints=[0])

    outputs = []
    prev_end = zero_1d
    for g in range(G):
        g_lo = op.Constant(value_ints=[g])
        g_hi = op.Constant(value_ints=[g + 1])
        end = op.Slice(offs_i64, g_lo, g_hi, axes_0)  # (1,) — offs[g]
        a_g = op.Slice(mat_a, prev_end, end, axes_0)  # (n_g, K)
        w_g = op.Squeeze(op.Slice(mat_b, g_lo, g_hi, axes_0), axes_0)  # (K, N)
        out_g = op.MatMul(a_g, w_g)  # (n_g, N)
        if bias is not None:
            out_g = op.Add(out_g, op.Squeeze(op.Slice(bias, g_lo, g_hi, axes_0), axes_0))
        outputs.append(out_g)
        prev_end = end

    result = op.Concat(*outputs, axis=0)  # (M, N)
    if out_dtype is not None:
        # torch.onnx hands a translation `torch.dtype` arguments already as `ir.DataType`
        result = op.Cast(result, to=out_dtype)
    return result


@register_onnx_translation("torch.ops.aten.repeat_interleave.self_int")
def _aten_repeat_interleave_self_int(self, repeats, dim=None, output_size=None):
    """ONNX implementation of `aten.repeat_interleave.self_int`.

    Torchlib raises on `dim is None` and tiles `[r, 1]` instead of `[1, r]` for 1-D symbolic repeats.
    """
    if dim is None:
        flat = op.Reshape(self, op.Constant(value_ints=[-1]))
        unsq = op.Unsqueeze(flat, op.Constant(value_ints=[1]))
        if isinstance(repeats, int):
            tile_shape = op.Constant(value_ints=[1, repeats])
        else:
            r = op.Reshape(op.Cast(repeats, to=7), op.Constant(value_ints=[-1]))
            tile_shape = op.Concat(op.Constant(value_ints=[1]), r, axis=0)
        return op.Reshape(op.Tile(unsq, tile_shape), op.Constant(value_ints=[-1]))

    self_rank = len(self.shape)
    pos_dim = (dim + self_rank) % self_rank
    unsq = op.Unsqueeze(self, op.Constant(value_ints=[pos_dim + 1]))
    if isinstance(repeats, int):
        tiles = [1] * (self_rank + 1)
        tiles[pos_dim + 1] = repeats
        tile_shape = op.Constant(value_ints=tiles)
    else:
        r = op.Reshape(op.Cast(repeats, to=7), op.Constant(value_ints=[-1]))
        tile_shape = op.Concat(
            op.Constant(value_ints=[1] * (pos_dim + 1)),
            r,
            op.Constant(value_ints=[1] * (self_rank - pos_dim - 1)),
            axis=0,
        )
    tiled = op.Tile(unsq, tile_shape)
    final_shape = op.Concat(
        op.Shape(self, start=0, end=pos_dim),
        op.Constant(value_ints=[-1]),
        op.Shape(self, start=pos_dim + 1),
        axis=0,
    )
    return op.Reshape(tiled, final_shape)


@register_onnx_translation("operator.floordiv")
def _operator_floordiv(self, other):
    """Correct floor division (toward -inf) for signed integer SymInts.

    Torchlib lowers it to a truncating `Div`, which breaks the ceil-div idiom `-(-x // y)`.
    """
    offset = op.And(
        op.Not(op.Equal(op.Sign(self), op.Sign(other))),
        op.Cast(op.Mod(self, other), to=onnx_ir.DataType.BOOL.value),
    )
    dtype = self.dtype.value if hasattr(self, "dtype") else other.dtype.value
    return op.Sub(op.Div(self, other), op.Cast(offset, to=dtype))


@register_onnx_translation("torch.ops.aten.masked_fill.Scalar", "torch.ops.aten.masked_fill.Tensor")
def _aten_masked_fill(self, mask, value):
    """ONNX implementation of `aten.masked_fill.{Scalar,Tensor}`.

    ORT's CPU EP has no BOOL `Where(16)` kernel, so a bool `self` uses `Or` / `And(Not)` instead.
    """
    if self.dtype == onnx_ir.DataType.BOOL:
        fill_true = bool(value) if isinstance(value, (bool, int, float)) else bool(value.const_value.numpy())
        if fill_true:
            return op.Or(self, mask)
        return op.And(self, op.Not(mask))
    value_cast = op.CastLike(value, self)
    return op.Where(mask, value_cast, self)


@functools.cache
def _compiled_varlen_translation():
    """Compile (once) the `@script` translation of `torch_attn::_varlen_attn`.

    Not compiled at import: it must run after `validate_environment` has confirmed a compatible onnxscript.
    """

    @script()
    def _aten_varlen_attn(
        query: FLOAT,
        key: FLOAT,
        value: FLOAT,
        cu_seq_q: INT64,
        cu_seq_k: INT64,
        max_q: INT64,
        max_k: INT64,
        is_causal: bool,
        scale: float,
        window_size: TypingSequence[int],
    ) -> tuple[FLOAT, FLOAT, FLOAT]:
        """`torch_attn::_varlen_attn` as a `Loop` of dense SDPAs over the `cu_seq_q` segments; aux outputs are stubs.

        `max_q`/`max_k` are tensor inputs since they arrive as symints."""
        one = op.Constant(value_ints=[1])
        two = op.Constant(value_ints=[2])
        three = op.Constant(value_ints=[3])
        axis0 = op.Constant(value_ints=[0])
        cu = op.Cast(cu_seq_q, to=7)  # INT64
        num_segments = op.Squeeze(op.Sub(op.Shape(cu), one))
        # GQA (e.g. Exaone4.5 vision): expand kv heads to the query's head count up front.
        q_heads = op.Slice(op.Shape(query), one, two, axis0)
        k_shape = op.Shape(key)
        k_len = op.Slice(k_shape, axis0, one, axis0)
        k_heads = op.Slice(k_shape, one, two, axis0)
        k_dim = op.Slice(k_shape, two, three, axis0)
        n_rep = op.Div(q_heads, k_heads)
        expand_shape = op.Concat(k_len, k_heads, n_rep, k_dim, axis=0)
        grouped_shape = op.Concat(k_len, q_heads, k_dim, axis=0)
        key = op.Reshape(op.Expand(op.Unsqueeze(key, two), expand_shape), grouped_shape)
        value = op.Reshape(op.Expand(op.Unsqueeze(value, two), expand_shape), grouped_shape)
        output = op.Slice(query, op.Constant(value_ints=[0]), op.Constant(value_ints=[0]), axis0)  # empty (0, H, D)
        for i in range(num_segments):
            index = op.Reshape(i, one)
            start = op.Slice(cu, index, op.Add(index, one), axis0)
            end = op.Slice(cu, op.Add(index, one), op.Add(index, two), axis0)
            query_heads = op.Transpose(op.Slice(query, start, end, axis0), perm=[1, 0, 2])
            key_heads = op.Transpose(op.Slice(key, start, end, axis0), perm=[1, 0, 2])
            value_heads = op.Transpose(op.Slice(value, start, end, axis0), perm=[1, 0, 2])
            scores = op.Mul(op.MatMul(query_heads, op.Transpose(key_heads, perm=[0, 2, 1])), scale)
            segment = op.Transpose(op.MatMul(op.Softmax(scores, axis=-1), value_heads), perm=[1, 0, 2])
            output = op.Concat(output, segment, axis=0)
        aux = op.CastLike(op.Constant(value_float=0.0), query)
        return output, aux, aux

    return _aten_varlen_attn


def _translate_associative_scan(combine, xs, additional_inputs):
    """Translate the SSM mixers' `higher_order.associative_scan`; only their first-order recurrence combine
    `(a_l * a_r, a_r * b_l + b_r)` is supported."""
    combine_ops = sorted(node.op_type for node in combine)
    if combine_ops != ["Add", "Mul", "Mul"] or len(xs) != 2 or len(additional_inputs) != 0:
        raise NotImplementedError(
            f"No ONNX translation for an associative_scan with combine {combine_ops} over {len(xs)} leaves "
            "— only the SSM mixers' first-order recurrence is supported."
        )
    return _compiled_ssm_scan_translation()(*xs)


@functools.cache
def _compiled_ssm_scan_translation():
    """Compile (once) the scan as an ONNX `Loop` with a dynamic trip count, so the step axis stays symbolic.

    Carried states start at the combine's identity; the mixer folds any cached state into `b`'s first step."""

    @script()
    def _transformers_ssm_scan(a: FLOAT, b: FLOAT) -> tuple[FLOAT, FLOAT]:
        one = op.Constant(value_ints=[1])
        axis0 = op.Constant(value_ints=[0])
        zero_f = op.CastLike(op.Constant(value_float=0.0), b)
        one_f = op.CastLike(op.Constant(value_float=1.0), a)
        seq_len = op.Squeeze(op.Slice(op.Shape(a), axis0, one, axis0))
        state = op.Mul(op.Slice(b, axis0, one, axis0), zero_f)
        a_acc = op.Add(op.Mul(op.Slice(a, axis0, one, axis0), zero_f), one_f)
        a_products = op.Slice(a, axis0, axis0, axis0)
        states = op.Slice(b, axis0, axis0, axis0)
        for i in range(seq_len):
            index = op.Reshape(i, one)
            a_step = op.Slice(a, index, op.Add(index, one), axis0)
            b_step = op.Slice(b, index, op.Add(index, one), axis0)
            a_acc = op.Mul(a_acc, a_step)
            state = op.Add(op.Mul(a_step, state), b_step)
            a_products = op.Concat(a_products, a_acc, axis=0)
            states = op.Concat(states, state, axis=0)
        return a_products, states

    return _transformers_ssm_scan


def _get_onnx_translation_table() -> dict[Any, Any]:
    """Assemble the `custom_translation_table` for `torch.onnx.export`.

    Not cached, so translations registered by late-imported modules are picked up. The ops probed below may
    not exist yet (older torch, or `grouped_mm_fallback` before `transformers.integrations.moe` is imported).
    """
    table = dict(_ONNX_TRANSLATIONS)
    if hasattr(torch.ops.aten, "_grouped_mm"):
        table[torch.ops.aten._grouped_mm.default] = _aten_grouped_mm
    if hasattr(torch.ops.transformers, "grouped_mm_fallback"):
        table[torch.ops.transformers.grouped_mm_fallback.default] = _aten_grouped_mm
    if hasattr(torch.ops.torch_attn, "_varlen_attn"):
        table[torch.ops.torch_attn._varlen_attn.default] = _compiled_varlen_translation()
    if hasattr(torch.ops.higher_order, "associative_scan"):
        table[torch.ops.higher_order.associative_scan] = _translate_associative_scan
    return table


# ── Stage 5: ONNX IR fixes ────────────────────────────────────────────────────
# Post-export `(graph_like) -> None` fixes for ORT, applied to the main graph and every function.


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
