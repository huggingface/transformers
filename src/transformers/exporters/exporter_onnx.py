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
4. **ONNX translations** (`_get_onnx_translation_table`): custom onnxscript functions
   passed as `custom_translation_table` that override the default torchlib
   lowering for specific aten ops where it's buggy or missing.
5. **ONNX IR fixes** (`_IR_FIXES` via `apply_onnx_ir_fixes`): post-export in-place
   fixes on the `ONNXProgram` IR for ORT compatibility.
"""

from __future__ import annotations

import copy
import functools
import json
import operator
from collections.abc import MutableMapping, Sequence
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
    from onnxscript.function_libs.torch_lib.ops.core import aten_index_put
    from onnxscript.onnx_opset import opset18 as op

    # torch.dtype -> onnx_ir.DataType, mirroring torch.onnx's private _TORCH_DTYPE_TO_ONNX so we
    # don't depend on that path. Only the dtypes a `Cast` target can realistically be; exotic
    # float8/float4 variants (never emitted as an ``out_dtype``) are omitted.
    _TORCH_DTYPE_TO_ONNX: dict[torch.dtype, onnx_ir.DataType] = {
        torch.float32: onnx_ir.DataType.FLOAT,
        torch.float64: onnx_ir.DataType.DOUBLE,
        torch.float16: onnx_ir.DataType.FLOAT16,
        torch.bfloat16: onnx_ir.DataType.BFLOAT16,
        torch.bool: onnx_ir.DataType.BOOL,
        torch.int8: onnx_ir.DataType.INT8,
        torch.int16: onnx_ir.DataType.INT16,
        torch.int32: onnx_ir.DataType.INT32,
        torch.int64: onnx_ir.DataType.INT64,
        torch.uint8: onnx_ir.DataType.UINT8,
        torch.uint16: onnx_ir.DataType.UINT16,
        torch.uint32: onnx_ir.DataType.UINT32,
        torch.uint64: onnx_ir.DataType.UINT64,
        torch.complex64: onnx_ir.DataType.COMPLEX64,
        torch.complex128: onnx_ir.DataType.COMPLEX128,
    }

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

    if is_onnxscript_available():
        from onnxscript.function_libs.torch_lib.ops.core import BOOL, TReal


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
    artifact_suffix = ".onnx"

    required_packages = ["torch", "onnx", "onnxscript"]
    tested_versions = {"torch": "2.13.0", "onnx": "1.22.0", "onnxscript": "0.7.1"}

    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: OnnxConfig | dict[str, Any],
    ) -> ONNXProgram:
        if isinstance(config, dict):
            config = OnnxConfig(**config)
        elif type(config) is not OnnxConfig:
            raise TypeError(f"Expected config to be an OnnxConfig or dict, got {type(config)}")

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
        # What the graph means, alongside what it declares: an ORT session reports names, shapes and types,
        # but nothing about precision or mask layout, so the runner would have to infer them
        # (`build_export_metadata`). `metadata_props` survives saving and comes back through
        # `session.get_modelmeta().custom_metadata_map`.
        # Also inside the file, so a lone `.onnx` handed to someone else still describes itself; the
        # authoritative copy for a saved directory is the one `export_artifact` hands back.
        onnx_program.model.metadata_props[EXPORT_METADATA_KEY] = json.dumps(metadata)
        return onnx_program, metadata

    @classmethod
    def save_artifact(cls, artifact, path) -> None:
        """`metadata_props` is part of the model proto, so the payload is already inside what gets written.
        Whether the initializers spill to a sidecar file is left to ONNX, which decides on the graph's size —
        the export-time `external_data` flag governs the trace, not this. Each component is saved under its
        own name because those sidecars are named after the file."""
        artifact.save(path)


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
        # Input names BEFORE the forward, and replacing rather than appending: the forward mutates its
        # pytree kwargs in place, so a cache whose states the model creates on this very call would
        # otherwise contribute leaf names the graph never took as inputs — shifting every later name onto
        # the wrong tensor (a recurrent model's `attention_mask` came out named
        # `cache_params.layers.0.conv_states.0`). Repeated traces then compounded it by appending again.
        inputs = get_leaf_tensors(kwargs)
        inputs_names[:] = inputs.keys()
        # The inputs count as already-seen identities: an output handed straight back is the graph's own
        # placeholder value, which would collapse the pair onto one name (see `duplicate_leaf_tensors`).
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

    Also prefix an output named exactly `output`: FX reserves that name for every graph's terminal node
    (top-level and HOP subgraphs alike), and `torch.onnx.export` can't disambiguate a requested output
    name from it — a component returning a bare unnamed tensor (the default leaf name `output`, see
    `_iter_leaf_tensors`) would define the name twice and produce an invalid model (ORT: "Duplicate
    definition of name (output)"). Consumers strip the `output.` prefix the same way they do for the
    input/output collision above.
    """
    collisions = set(inputs_names).intersection(outputs_names)
    return (
        [f"input.{name}" if name in collisions else name for name in inputs_names],
        [f"output.{name}" if name in collisions or name == "output" else name for name in outputs_names],
    )


# ── Stage 1: Torch patches ─────────────────────────────────────────────────────
# Each `_patch_*(original)` factory is registered via `@register_patch("onnx", path)`,
# where `path` is the dotted Python path of the attribute to swap (e.g. `"torch.where"`,
# `"torch.Tensor.unsqueeze"`). Installation and restoration go through `apply_patches`.
#
# To add a new patch: define a `_patch_*` factory and decorate it.


@register_patch(
    "onnx",
    "transformers.models.falcon_mamba.modeling_falcon_mamba.mamba_selective_scan",
    "transformers.models.jamba.modeling_jamba.mamba_selective_scan",
    "transformers.models.mamba.modeling_mamba.mamba_selective_scan",
    "transformers.models.zamba.modeling_zamba.mamba_selective_scan",
)
def _patch_mamba_selective_scan(original):
    """Keep an SSM scan sequential unless its `pointwise` combine mode is available. Only that mode (cuda /
    xpu) is ONNX-exportable: the `generic` mode a cpu tensor selects lowers through `vmap`, which
    `run_decompositions` cannot take apart ("tensor may have escaped from inside a function being
    vmapped"), and it pins the scan's step axis anyway. Read off the tensor that selects the mode, so the
    decision is per call rather than per export."""

    def patch(hidden_states, *args, **kwargs):
        if hidden_states.device.type not in ("cuda", "xpu"):
            kwargs["use_associative_scan"] = False
        return original(hidden_states, *args, **kwargs)

    return patch


@register_patch("onnx", "torch.where")
def _patch_where(original):
    """Normalize dtypes and scalars in torch.where."""

    def patch(condition, x=None, y=None):
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and x.dtype != y.dtype:
            y = y.to(x.dtype)
        elif isinstance(x, torch.Tensor) and isinstance(y, (int, float, bool)):
            # `full_like` (a traced op) rather than `torch.tensor(...)` (a fresh leaf constant): the
            # latter, if materialised during `run_decompositions`' retrace, becomes an unregistered
            # `_tensor_constant` → `alias` → `detach_` that trips aot's functional-graph assertion.
            y = torch.full_like(x, y)
        elif isinstance(y, torch.Tensor) and isinstance(x, (int, float, bool)):
            x = torch.full_like(y, x)
        if x is None and y is None:
            return original(condition)
        elif y is None:
            return original(condition, x)
        else:
            return original(condition, x, y)

    return patch


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

    A row masked at every key asks for a softmax over nothing. Torch's fused CUDA kernel answers with
    zeros; ONNX Runtime evaluates the softmax literally and, when the mask is `-inf` (parakeet's
    relative-position bias masks that way), returns `NaN` — which a later BatchNorm then spreads over
    the whole batch. Rows like these are routine: any padded frame under a padding mask has one.
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


@register_patch("onnx", "torch.nn.RMSNorm.forward")
def _patch_rms_norm_forward(original):
    """Use non-fused RMS normalization when elementwise_affine is False."""

    def patch(self, x):
        if not self.elementwise_affine:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(variance + self.eps)).to(x.dtype)
        return original(self, x)

    return patch


@register_patch("onnx", "torch.split", "torch.Tensor.split")
def _patch_split(original):
    """Expand a symbolic split size into statically-counted `narrow`s. A SymInt split size
    otherwise lowers to `SplitToSequence` with a symbolic scalar `split` input, which
    onnxscript's constant folder crashes on (`'NoneType' object has no attribute 'ndim'`).
    """

    def patch(input, split_size_or_sections, dim=0):
        if not isinstance(split_size_or_sections, torch.SymInt):
            return original(input, split_size_or_sections, dim)
        split_size = split_size_or_sections
        total = input.size(dim)
        # `int()` specializes the chunk count at trace time, exactly like enumerating the
        # list `aten.split.Tensor` returns (its meta guards on the same ceil division).
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


@register_patch("onnx", "onnxscript.onnx_opset._impl.opset13.Opset13.Constant")
def _patch_opset13_constant(original):
    """Substitute `op.Constant(value_ints=[])` with an explicit empty INT64 tensor.

    Upstream onnxscript's `aten_index_put` does `op.Constant(value_ints=none_indices)`
    where `none_indices` can be empty (when every input dim has an advanced index).
    `onnx_ir` then logs an ambiguous-type warning because an empty Python list has no
    derivable element type. Swap the empty-`value_ints` call for `value=ir.tensor([], INT64)`
    — semantically identical, no ambiguity. Drop once onnxscript fixes the call site.
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

    The optimizer's constant folder evaluates foldable nodes with onnx's pure-Python
    reference implementation. For `Resize` — e.g. the bicubic position-embedding
    interpolation in YOLOS/SegGPT-style vision models, whose inputs are constant
    initializers — that evaluation recurses per output element and takes minutes even
    on tiny graphs (~4.5 min per Resize node on the YOLOS test model, vs <1 s for the
    whole rest of the optimization). Keeping the Resize node in the graph costs one
    native ORT kernel launch at inference instead.
    """

    def patch(model, *args, **kwargs):
        kwargs.setdefault("should_fold", lambda node: False if node.op_type == "Resize" else None)
        with _exact_identity_rewrites():
            return original(model, *args, **kwargs)

    return patch


@contextmanager
def _exact_identity_rewrites():
    """Let onnxscript's identity rewrites (`x + 0`, `x * 1`, …) fire on an exact 0 or 1 only.

    A pattern constant matches within `math.isclose(rel_tol=1e-5, abs_tol=1e-8)`, so `x + 1e-10` counts as
    `x + 0` and the epsilon a guarded division adds is deleted: patchtst's `loss.sum() / (mask.sum() +
    1e-10)` turns into `0 / 0 = nan` whenever nothing is masked. Held only while the optimizer runs.
    """
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

    seen, tightened = set(), []
    for rule in rewriter._DEFAULT_REWRITE_RULES:
        for constant in constants(rule, seen):
            if isinstance(constant._value, (int, float)) and constant._value in (0, 1):
                tightened.append((constant, constant._rel_tol, constant._abs_tol))
                constant._rel_tol = constant._abs_tol = 0.0
    try:
        yield
    finally:
        for constant, rel_tol, abs_tol in tightened:
            constant._rel_tol, constant._abs_tol = rel_tol, abs_tol


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


@register_patch("onnx", "torch.cummax", "torch.Tensor.cummax")
def _patch_cummax(original):
    return _patch_cummax_or_cummin(original, mode="max")


@register_patch("onnx", "torch.cummin", "torch.Tensor.cummin")
def _patch_cummin(original):
    return _patch_cummax_or_cummin(original, mode="min")


@register_patch("onnx", "torch.chunk", "torch.Tensor.chunk")
def _patch_chunk(original):
    """Lower `chunk` via `narrow` (→ ONNX `Slice`) under dynamic shapes.

    `torch.chunk` lowers to an ONNX `SplitToSequence` whose split length is a symbolic floordiv of the
    (dynamic) axis size; onnx_ir's `InlinePass` rejects that graph (e.g. diffllama's differential
    attention splitting the head axis). Narrow-based slicing produces plain `Slice` ops instead, which
    lower and inline cleanly. Only rewrites when the split axis is dynamic — a static axis lets torch's
    own `chunk` lowering (fixed split sizes) through, which onnxscript handles fine.
    """

    def patch(input, chunks, dim=0):
        total = input.size(dim)
        if not isinstance(total, torch.SymInt):
            return original(input, chunks, dim)
        # `torch.chunk` splits into `chunks` pieces of `ceil(total / chunks)`, the last taking the
        # remainder. Emit that as narrows so nothing lowers to `SplitToSequence`. This assumes the
        # split axis divides evenly into `chunks` (true for the head-axis splits this targets); an
        # unevenly-divisible dynamic axis would need a symbolic piece count, which torch.export can't
        # express here.
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
    """Lower `exp` on complex tensors via Euler — onnxscript has no dispatch for `aten.exp` on
    complex inputs. Real inputs hit the original path."""

    def patch(input):
        if torch.is_complex(input):
            magnitude = original(input.real)
            return torch.complex(magnitude * input.imag.cos(), magnitude * input.imag.sin())
        return original(input)

    return patch


@register_patch("onnx", "torch.fft.irfft")
def _patch_irfft(original):
    """Replace `irfft` with `ifft` over the conjugate-mirrored input — ORT's `DFT` op rejects the
    `is_onesided=1`/`inverse=1` combination that torch's `irfft` lowers to. Mirroring restores the
    full spectrum so the inverse path uses two-sided DFT, which ORT accepts. Assumes even `n`
    (which is the common case for STFT-based audio codecs)."""

    def patch(input, n=None, dim=-1, norm=None):
        if n is None:
            n = 2 * (input.shape[dim] - 1)
        slc = [slice(None)] * input.ndim
        slc[dim] = slice(1, -1)
        full = torch.cat([input, input[tuple(slc)].flip(dims=[dim]).conj()], dim=dim)
        return torch.fft.ifft(full, n=n, dim=dim, norm=norm).real

    return patch


@register_patch("onnx", "torch.full")
def _patch_full(original):
    """Force dtype=torch.long when fill_value is int and no dtype specified (ONNX defaults to float32)."""

    def patch(*args, dtype=None, **kwargs):
        if dtype is None:
            # find fill_value: positional arg or kwarg
            fill_value = kwargs.get("fill_value", args[1] if len(args) > 1 else None)
            # `bool` is a subclass of `int` — exclude it so `torch.full(size, True)` stays bool.
            if isinstance(fill_value, int) and not isinstance(fill_value, bool):
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


@register_patch("onnx", "torch.roll")
def _patch_roll(original):
    """Replace `torch.roll(input, shifts, dims)` with explicit `narrow + cat` shifts.

    `torch.roll`'s torch.export lowering emits a `Shape(start, end)` op that can resolve to an
    empty INT64 result; the downstream `Slice` then has mismatched `axes` and `ends` lengths
    and ORT rejects the graph with `ShapeInferenceError` (seen in Gemma4-Unified Vision2Text,
    where roll is composed with an in-place scatter `[..., 0] = value`). The explicit form is
    bit-exact and traces to plain Slice + Concat nodes.
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
def _fix_noop_squeeze(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Drop the axes of a `squeeze` that are not of size 1, which torch leaves in place.

    `x.squeeze(0)` on a `[12, dim]` tensor is `x` in torch, but torchlib lowers `squeeze.dim` to ONNX `Squeeze`
    unconditionally, and ORT rejects the graph outright (`Dimension of input 0 must be 1 instead of 12`) —
    mistral3 squeezes its image features twice, once too often. Only a static size is decided here: a
    symbolic one may be 1 at runtime, and there the `Squeeze` is what torch would have done.
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


# Overloads torch.export emits that ONNX cannot take, each with a drop-in twin: the in-place ops
# (`aot_autograd` rejects them in a functional graph) and the `.Scalar` forms whose "scalar" arrives as a
# graph node after decomposition, where torchlib either has no translation or calls `int()` on it
# (pytorch/pytorch#194382). The rewrite is one shape -- insert the twin, forward the uses, erase -- so it
# is written once and the table says which op maps to which.
_FUNCTIONAL_TWINS = {}
_TENSOR_OVERLOAD_TWINS = {}
if is_torch_available():
    _FUNCTIONAL_TWINS.update(
        {
            torch.ops.aten.detach_.default: torch.ops.aten.detach.default,
            torch.ops.aten.index_put_.default: torch.ops.aten.index_put.default,
            torch.ops.aten.triu_.default: torch.ops.aten.triu.default,
        }
    )
    _TENSOR_OVERLOAD_TWINS.update(
        {
            torch.ops.aten.mul.Scalar: torch.ops.aten.mul.Tensor,
            torch.ops.aten.remainder.Scalar: torch.ops.aten.remainder.Tensor,
        }
    )


@register_fx_node_fix("onnx")
def _fix_overload_with_twin(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Swap an unexportable overload for its twin, per `_FUNCTIONAL_TWINS` / `_TENSOR_OVERLOAD_TWINS`.

    A `.Scalar` form is only swapped when its second argument really is a graph node holding a tensor or
    symbolic value -- with a Python literal there, the original overload is the right one.
    """
    if (replacement := _FUNCTIONAL_TWINS.get(node.target)) is None:
        replacement = _TENSOR_OVERLOAD_TWINS.get(node.target)
        if replacement is None or len(node.args) < 2 or not isinstance(node.args[1], torch.fx.Node):
            return False
        if not isinstance(node.args[1].meta.get("val"), (torch.Tensor, torch.SymFloat, torch.SymInt, torch.SymBool)):
            return False


@register_fx_node_fix("onnx")
def _fix_slice_implicit_start(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Spell out a slice's implicit start as ``0``.

    ``x[..., :end]`` traces as `aten.slice` with ``start=None``, which onnxscript lowers to a `Slice`
    whose `starts` input is an `Unsqueeze` of nothing. ORT then rejects the whole graph with
    ``input 0 is marked single but has an empty string`` (funnel's relative-shift gather), or, with
    optimisation on, the malformed node surfaces as an `onnx_ir` `PassError` from the inliner.
    ``None`` already means ``0`` here, so writing it out changes nothing but the emitted graph.
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

    torchlib's `index_put` lowering silently drops the write under dynamic shapes — the indexed
    columns come back unchanged (chameleon masks its image-token logits with `finfo.min` that way, and
    the sentinel never lands). Comparing an `arange` over the indexed dim against `idx` gives a mask
    that broadcasts against `self`, which ONNX handles identically in both shape modes. Only scalar
    (broadcastable) values take this path; anything else keeps the original lowering.
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
    # bool masks have their own translation, and only a broadcastable value can become a `where`
    if index_val.dtype == torch.bool or values_val.numel() != 1 or len(indices) != self_val.ndim:
        return False

    # `x[:, :, idx] = v` mutates a *view*: the graph slices, writes into the slice, and returns the
    # base it never re-reads. Walk back through slices that keep the shape (a full `:`) so the write
    # lands on the tensor later nodes actually read.
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
    # Under dynamic shapes `index_put_` is left with no users at all: the graph returns the tensor it
    # mutated and relies on the mutation, which a functional IR drops on the floor. Hand every later
    # reader of that tensor the value instead.
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
def _fix_sort_stable(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace aten.sort.stable with aten.sort.default (which has ONNX translation)."""
    if node.target is not torch.ops.aten.sort.stable:
        return False
    self_arg = node.args[0]
    # `dim`/`descending` are keyword-only in `sort.stable` (schema: `sort.stable(self, *, stable, dim=-1,
    # descending=False)`), so they arrive in `node.kwargs`, never `node.args`.
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

    Named for torch 2.13, where the mishandling appeared, but applied on every version: both rewrites are
    semantics-preserving — a cast to the dtype the op already produces, and an overload swap with the same
    meaning — so gating them on a version would add a branch that changes nothing except which torch the
    path is exercised on.

    `sub`/`rsub` and `mul` are what the affected models spell (`1.0 - attention_mask`, `mask * 2.0`); both
    overloads appear, since decomposition rewrites `.Tensor` to `.Scalar` when the operand is a constant.

    Resolved on first use rather than at import: this module is importable without torch, and naming an
    `OpOverload` at module scope breaks that.
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

    Two torch 2.13 regressions have the same shape — a float scalar against an *integral* tensor, whose
    promotion the export pipeline no longer gets right:

    - `1.0 - int_mask` (`aten.rsub.Scalar`) crashes the decomposition pass (pytorch/pytorch#194381),
    - `int_mask * 2.0` (`aten.mul.Tensor`, `aten.mul.Scalar` after decomposition) reaches translation with
      no ONNX decomposition registered for it (pytorch/pytorch#194382).

    Both go away once the tensor is already the dtype the op produces: `1.0 - float_tensor` and
    `float_tensor * 2.0` export fine on the same torch. So rather than rewriting the op — which would mean
    building the constant as a tensor and picking the right overload — cast its tensor operand up front and
    leave the op alone. The cast's value is the op's own output for these elementwise cases (same shape,
    the promoted dtype), so it carries `node.meta` unchanged.

    Self-limiting: once the operand is floating point the predicate no longer matches, so the walk cannot
    revisit it.
    """
    if node.target not in _integral_scalar_promotion_ops():
        return False
    if len(node.args) < 2:
        return False
    tensor_arg, scalar_arg = node.args[0], node.args[1]
    # A tensor on the left, a Python float on the right — a `Node` there is already a real tensor operand.
    if not isinstance(tensor_arg, torch.fx.Node) or not isinstance(scalar_arg, float):
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

    A conv/pool output length reaches a tensor factory (e.g. ``torch.zeros((batch, feat_len))`` for an
    audio feature mask) as a *derived* SymInt — a floor-div chain of the input length. onnxscript lowers
    that signed floor-div as ``Sub(Div, Cast(And(...Mod...Sign)))``, which ORT's CUDA EP mis-evaluates to a
    negative dim → ``Expand``/``Reshape`` failures. When another tensor already in the graph carries that
    exact symbolic dim in its shape, rewrite the factory's size element to read it as ``aten.sym_size``
    (a plain ``Shape`` gather at export) so no floor-div is recomputed. Only touches factory *size* args —
    never arithmetic ``Div`` nodes — so numeric floor-divs (relative-position lengths) are left intact.
    """
    if node.target not in _TENSOR_FACTORY_OPS:
        return False
    # The size list is the first list/tuple arg holding at least one SymInt-valued node.
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
        # Carry the original SymInt value so downstream shape reasoning / ONNX translation still sees it.
        sym_size.meta["val"] = element.meta["val"]
        size[i] = sym_size
        changed = True
    if not changed:
        return False
    node.update_arg(size_pos, size)
    return True


# ── Stage 4: ONNX translations ────────────────────────────────────────────────
# Custom onnxscript `_aten_*` functions that override `torchlib`'s default lowering for specific aten
# ops where the default is buggy or missing. Each is registered with `@register_onnx_translation` and
# assembled by `_get_onnx_translation_table` into `torch.onnx.export`'s `custom_translation_table`.
# ONNX-only (translation tables have no ExecuTorch / Dynamo equivalent), so the registry lives here
# rather than in the backend-generic `utils.py` alongside `_PATCHES` / `_FX_NODE_FIXES`.

_ONNX_TRANSLATIONS: dict[Any, callable] = {}


def register_onnx_translation(*paths: str):
    """Append the decorated lowering `fn` to `_ONNX_TRANSLATIONS`, keyed by each op `path`.

    Like `register_patch`, each `path` is a dotted string — an op overload
    (`"torch.ops.aten.index_put.default"`) or an `operator` builtin (`"operator.floordiv"`) — resolved
    to its object at decoration time. Unresolvable paths (torch not installed, op not yet registered)
    are skipped so the module still imports. Passing several paths registers the SAME `fn` for each
    (e.g. `masked_fill.Scalar` + `masked_fill.Tensor`). External code adds translations the same way;
    `_get_onnx_translation_table` reads them into the `custom_translation_table`.
    """

    def decorator(fn):
        for path in paths:
            op = _resolve_dotted_path(path)
            if op is not None:
                _ONNX_TRANSLATIONS[op] = fn
        return fn

    return decorator


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


@register_onnx_translation("torch.ops.aten.mul.Scalar")
def _aten_mul_scalar(self: TReal, other: float) -> TReal:
    """`aten.mul.Scalar` has no torchlib lowering, so any multiply the decompositions leave in that overload
    reaches translation and fails to dispatch. What gets there is the type-promoting case — an integer tensor
    times a float scalar (bros scales an int64 `bbox`) — which PyTorch computes in the promoted float type
    while ONNX has no promotion of its own, so make the cast explicit and multiply there."""
    if not isinstance(other, (bool, int, float)):
        # A symbolic scalar (a dynamic dim folded into the multiply) arrives as a graph value rather than a
        # python number, carrying its own dtype — line it up with the tensor's and multiply there.
        return op.Mul(self, op.CastLike(other, self))
    scalar = op.Constant(value_float=float(other))
    if isinstance(other, float) and not self.dtype.is_floating_point():
        return op.Mul(op.Cast(self, to=onnx_ir.DataType.FLOAT), scalar)
    return op.Mul(self, op.CastLike(scalar, self))


@register_onnx_translation("torch.ops.aten.rsub.Scalar")
def _aten_rsub_scalar(self: TReal, other: float, alpha: float = 1.0) -> TReal:
    """`aten.rsub.Scalar` (`scalar - tensor`, big_bird's `1.0 - to_mask`) has no torchlib lowering, and its
    decomposition emits `aten.sub(scalar, tensor)` — a scalar-first call no `sub` overload accepts, which
    the decomposition step's own type promotion then chokes on. Registering it here keeps the op out of the
    decomposition entirely and lowers it directly, promoting the same way `_aten_mul_scalar` does."""
    scalar = op.Constant(value_float=float(other))
    if isinstance(other, float) and not self.dtype.is_floating_point():
        self = op.Cast(self, to=onnx_ir.DataType.FLOAT)
    else:
        scalar = op.CastLike(scalar, self)
    if alpha != 1.0:
        self = op.Mul(self, op.CastLike(op.Constant(value_float=float(alpha)), self))
    return op.Sub(scalar, self)


@register_onnx_translation("torch.ops.aten.index_put.default")
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
    # The Where-based paths below overwrite; they can't express `self[mask] += values`. Delegate the
    # accumulate case (and any non-bool-mask index) to torchlib, which handles both correctly.
    if not is_bool or accumulate:
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


@register_onnx_translation("torch.ops.aten.bincount.default")
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


def _torch_dtype_to_onnx(dtype: torch.dtype) -> int:
    """Map a ``torch.dtype`` to the ONNX ``op.Cast(to=...)`` TensorProto int."""
    return _TORCH_DTYPE_TO_ONNX[dtype].value


def _aten_grouped_mm(mat_a: TReal, mat_b: TReal, offs: INT64, bias=None, out_dtype=None) -> TReal:
    """ONNX implementation of `aten._grouped_mm.default`.

    `_grouped_mm(mat_a: (M, K), mat_b: (G, K, N), offs: (G,))` computes `out[r] =
    mat_a[r] @ mat_b[group(r)]` where rows are sorted by group and `offs` holds the
    cumulative end index per group.

    Per-group `Slice + MatMul + Concat`. `G` (number of experts) is static for any
    concrete model, so unroll at translation time: emit one `Slice + MatMul` triple
    per group and a final `Concat`. Avoids the `(M, K, N)` materialisation a naive
    `weight[group_idx]` gather would emit — peak memory is `O(M·N + max(n_g)·K + K·N)`.
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
            # per-group bias ``(G, N)`` → ``(N,)`` broadcasts over the group's rows
            out_g = op.Add(out_g, op.Squeeze(op.Slice(bias, g_lo, g_hi, axes_0), axes_0))
        outputs.append(out_g)
        prev_end = end

    result = op.Concat(*outputs, axis=0)  # (M, N)
    if out_dtype is not None:
        result = op.Cast(result, to=_torch_dtype_to_onnx(out_dtype))
    return result


@register_onnx_translation("torch.ops.aten.repeat_interleave.self_int")
def _aten_repeat_interleave_self_int(self, repeats, dim=None, output_size=None):
    """ONNX implementation of `aten.repeat_interleave.self_int`.

    Torchlib's translation raises on `dim is None` and broadcasts incorrectly for the
    1-D + symbolic-repeats case (its tile shape is `[r, 1]` instead of `[1, r]`). We
    always rewrite: flatten when `dim is None`, then `Unsqueeze + Tile + Reshape` along
    the chosen axis. Handles both Python-int and 0-D-tensor `repeats`.
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

    Torchlib's `operator_floordiv` translation only handles positive operands (plain `Div`,
    which truncates). For signed ints — including SymInt shape arithmetic — apply the
    offset correction `floor(a/b) = trunc(a/b) - (sign(a) != sign(b) AND a mod b != 0)`,
    matching `aten_floor_divide`. Without this, the Python ceil-div idiom `-(-x // y)`
    produces `1 + trunc((y - 1 - x) / y)` instead of `ceil(x / y)`, which silently breaks
    any shape arithmetic that crosses zero.
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

    Upstream torchlib lowers this to `Where(mask, value, self)`. ORT's CPU EP has no
    `Where(16)` kernel for BOOL inputs, so when `self` is BOOL we rewrite the op using
    boolean primitives: `masked_fill(self, mask, True)` → `self | mask`,
    `masked_fill(self, mask, False)` → `self & ~mask`. Non-bool `self` keeps the default.
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

    Kept out of the module-level `@register_onnx_translation` decorators because the `@script` compile (and
    the onnxscript APIs it exercises) must run only inside `OnnxExporter.export`, after `validate_environment`
    has confirmed a compatible onnxscript — never on the bare import path. `@functools.cache` compiles at most
    once. The compile is self-contained (only `op.*` / `FLOAT` / `INT64`); importing `torch.nn.attention.varlen`
    to register the op key is the caller's concern, not a prerequisite for compiling.
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
        """ONNX translation of `torch_attn::_varlen_attn` — the varlen attention emitted by the chunked
        vision/audio attention export patch. Lowers to an ONNX `Loop` over the `cu_seq_q` segments (each
        iteration a dense SDPA on that segment's `n_i` tokens, concatenated), i.e. the true variable-length
        shape — O(sum n_i^2), no N×N block mask materialised. The op has three outputs `(out, softmax_lse,
        rng_state)`; only `out` is consumed downstream, so the two aux tensors are empty stubs.

        `max_q`/`max_k` are the (unused) flash-kernel bounds — kept as tensor inputs, not `int` attributes,
        because they arrive as symints under dynamic export. `window_size` must be typed via `typing.Sequence`
        (not `collections.abc`), which `onnxscript.script`'s annotation parser rejects."""
        one = op.Constant(value_ints=[1])
        two = op.Constant(value_ints=[2])
        three = op.Constant(value_ints=[3])
        axis0 = op.Constant(value_ints=[0])
        cu = op.Cast(cu_seq_q, to=7)  # INT64
        num_segments = op.Squeeze(op.Sub(op.Shape(cu), one))
        # Grouped-query attention: key/value carry fewer heads than query (e.g. Exaone4.5 vision). The
        # flash op broadcasts them internally; here we materialise the `repeat_kv` expansion once up
        # front — (L, Hkv, D) → (L, Hq, D) by repeating each kv head `Hq // Hkv` times — so the per-
        # segment MatMul sees matching head counts. `n_rep == 1` (multi-head attention) makes it a no-op.
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
    """Translate the `higher_order.associative_scan` the SSM mixers trace under export.

    The combine subgraph arrives as an `onnx_ir.Function`; the one combine the mixers use —
    `(a_l * a_r, a_r * b_l + b_r)`, a first-order recurrence over two leaves — is lowered by
    `_compiled_ssm_scan_translation` to an ONNX `Loop`. Anything else is rejected here, loudly."""
    combine_ops = sorted(node.op_type for node in combine)
    if combine_ops != ["Add", "Mul", "Mul"] or len(xs) != 2 or len(additional_inputs) != 0:
        raise NotImplementedError(
            f"No ONNX translation for an associative_scan with combine {combine_ops} over {len(xs)} leaves "
            "— only the SSM mixers' first-order recurrence is supported."
        )
    return _compiled_ssm_scan_translation()(*xs)


@functools.cache
def _compiled_ssm_scan_translation():
    """Compile (once) the `@script` body of `_translate_associative_scan` — see
    `_compiled_varlen_translation` for why the compile lives behind a cache instead of a module-level
    decorator. Lowers the first-order recurrence to an ONNX `Loop` with a *dynamic* trip count, so the
    step axis stays symbolic where torch's python scan loop would unroll it. Sequential like the mixers'
    own fallback — same numerics, O(seq) iterations. The carried states start at the combine monoid's
    identity (`a` products at 1, states at 0 — any cached initial state is already folded into `b`'s
    first step by the mixer)."""

    @script()
    def _transformers_ssm_scan(a: FLOAT, b: FLOAT) -> tuple[FLOAT, FLOAT]:
        one = op.Constant(value_ints=[1])
        axis0 = op.Constant(value_ints=[0])
        zero_f = op.CastLike(op.Constant(value_float=0.0), b)
        one_f = op.CastLike(op.Constant(value_float=1.0), a)
        seq_len = op.Squeeze(op.Slice(op.Shape(a), axis0, one, axis0))
        # carried states, kept `[1, ...]` so each step concatenates straight into the outputs
        state = op.Mul(op.Slice(b, axis0, one, axis0), zero_f)
        a_acc = op.Add(op.Mul(op.Slice(a, axis0, one, axis0), zero_f), one_f)
        # empty `(0, ...)` accumulators, the varlen translation's trick
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

    Merges the module-level `@register_onnx_translation(...)` entries (plus any registered by external
    code) with three ops whose availability is probed here rather than decorated at import: `_grouped_mm`,
    `grouped_mm_fallback`, and the `@script` varlen op (compiled once via `_compiled_varlen_translation`).
    Not cached, so a translation registered by a late-imported module is still picked up on the next export.
    """
    # None of these three op keys are guaranteed to exist as attributes here, so probe with `hasattr`
    # rather than referencing them unconditionally (a missing key raises AttributeError):
    #   - `aten._grouped_mm` is absent on older torch;
    #   - `transformers::grouped_mm_fallback` only exists once `transformers.integrations.moe` is imported
    #     (lazy — pulled in while tracing a model that uses it);
    #   - `torch_attn::_varlen_attn` is registered when `exporter_dynamo` imports `torch.nn.attention.varlen`.
    # Adding a translation whose op isn't in the graph is harmless — torch.onnx ignores unused entries.
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
