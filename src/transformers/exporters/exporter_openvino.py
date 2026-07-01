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
"""OpenVINO exporter.

Extends [`DynamoExporter`] with one extra stage that hands the ``ExportedProgram`` to
``openvino.convert_model``. The export pipeline runs:

1. **Torch patches** (``_PATCHES["openvino"]`` via ``apply_patches("openvino")``):
   reversibly swap ``torch`` ops the OV frontend can't lower (``torch.histc``,
   ``torch.empty_permuted``, …) with decomposed equivalents. Reverted on exit.
2. **Dynamo trace** (inherited from [`DynamoExporter`]): signature patch, model patches,
   pytree registration, dynamic shapes, state cleanup — same as for any other backend.
3. **FX program fixes** (``apply_fx_program_fixes("openvino", ep)``): repair the
   `ExportedProgram` in place where the fix needs program-level context — e.g. stripping
   SymInt graph outputs whose non-numeric FX names trip OV's ``is_number(name)`` check.
4. **FX node fixes** (``apply_fx_node_fixes("openvino", ep.graph_module)``): per-node
   in-place rewrites that work around quirks of OV's PyTorch frontend (e.g. dropping
   ``aten.cat`` of an empty operand the frontend can't broadcast).
5. **Conversion**: ``openvino.convert_model(exported_program, extensions=...)`` produces
   an ``openvino.Model``. Custom ``ConversionExtension``\\ s translate ops without a
   built-in OV frontend lowering (``aten._grouped_mm``,
   ``transformers.grouped_mm_fallback``). Optionally written to disk via
   ``openvino.save_model`` when ``OpenVINOConfig.output_path`` is set.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.import_utils import is_openvino_available, is_torch_available
from .configs import OpenVINOConfig
from .exporter_dynamo import DynamoExporter
from .exporter_onnx import disambiguate_io_names, patch_model_outputs
from .utils import (
    apply_fx_node_fixes,
    apply_fx_program_fixes,
    apply_patches,
    get_leaf_tensors,
    register_fx_node_fix,
    register_fx_program_fix,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram

    from .. import masking_utils


if is_openvino_available():
    import numpy as np
    import openvino
    import openvino.opset14 as ov_ops
    from openvino.frontend.pytorch import ConversionExtension


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


class OpenVINOExporter(DynamoExporter):
    """Exporter that converts a [`PreTrainedModel`] to an OpenVINO ``openvino.Model``.

    Example:

    ```python
    >>> from transformers.exporters.exporter_openvino import OpenVINOExporter, OpenVINOConfig

    >>> exporter = OpenVINOExporter()
    >>> ov_model = exporter.export(model, inputs, config=OpenVINOConfig(dynamic=True))
    >>> exporter.export(model, inputs, config=OpenVINOConfig(output_path="model.xml"))
    ```
    """

    required_packages = ["torch", "openvino"]
    tested_versions = {"torch": "2.12.0", "openvino": "2025.0.0"}

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: OpenVINOConfig | dict[str, Any],
    ) -> openvino.Model:
        if isinstance(config, dict):
            config = OpenVINOConfig(**config)
        elif type(config) is not OpenVINOConfig:
            raise TypeError(f"Expected config to be an OpenVINOConfig or dict, got {type(config)}")

        with patch_model_outputs(model) as (inputs_names, outputs_names), apply_patches("openvino"):
            exported_program: ExportedProgram = super().export(model, sample_inputs, config=config)

        apply_fx_program_fixes("openvino", exported_program)
        apply_fx_node_fixes("openvino", exported_program.graph_module)
        inputs_names, outputs_names = disambiguate_io_names(inputs_names, outputs_names)
        ov_model = openvino.convert_model(exported_program, extension=_OV_CONVERSION_EXTENSIONS)
        _rename_ports(ov_model.inputs, [n for n in inputs_names if n in get_leaf_tensors(sample_inputs)])
        _rename_ports(ov_model.outputs, outputs_names)

        if config.output_path is not None:
            openvino.save_model(ov_model, config.output_path, compress_to_fp16=config.compress_to_fp16)

        return ov_model


# ── Conversion helpers ──────────────────────────────────────────────────────
# Small helpers for ``OpenVINOExporter.export`` — extracted for readability and so each stage
# of the conversion has a single responsibility.


def _rename_ports(ports, names: list[str]) -> None:
    """Apply ``names`` to OV ports via ``set_names`` (works for inputs and outputs).

    OV's PyTorch frontend doesn't support an ``output=`` argument, and ``input=`` only accepts
    Python-identifier names (no dots) — so we restore the dotted ``get_leaf_tensors`` form
    post-conversion. Names line up positionally with the ports.
    """
    for port, name in zip(ports, names):
        port.get_tensor().set_names({name})


# ── FX program fixes ────────────────────────────────────────────────────────
# Program-level fixes applied to the `ExportedProgram` (not just the graph_module) before
# `openvino.convert_model`. Each `_fix_*(exported_program) -> None` is registered via
# `@register_fx_program_fix("openvino")` and runs once per export. Use this when the fix
# needs to update `graph_signature` / `range_constraints` alongside the FX graph itself.


_OV_NAME_OK = re.compile(r"_\d+$")


@register_fx_program_fix("openvino")
def _fix_rename_bare_node_names(exported_program):
    """Append a numeric suffix to FX node names that lack one.

    OV's PyTorch frontend strips a trailing ``_<digits>`` from each tensor name to recover the op
    kind, then validates the remainder — aborting with ``GeneralFailure: is_number(name)`` for
    bare names (``mul``, ``clone``, ``linear``). The first node of any kind in the FX graph has
    no ``_<digits>`` suffix, so the strip is a no-op and OV rejects it. We give every such node
    a ``_0`` suffix and mirror the rename into ``graph_signature.output_specs`` (which holds the
    user-output names OV reads to identify outputs). The post-conversion port rename restores
    user-facing output names; intermediates are internal to OV's translator.
    """
    graph = exported_program.graph_module.graph
    used = {n.name for n in graph.nodes}
    renames: dict[str, str] = {}
    for n in graph.nodes:
        if n.op in ("placeholder", "output"):
            continue
        if _OV_NAME_OK.search(n.name):
            continue
        candidate = f"{n.name}_0"
        i = 0
        while candidate in used:
            i += 1
            candidate = f"{n.name}_{i}"
        used.discard(n.name)
        used.add(candidate)
        renames[n.name] = candidate
        n._rename(candidate)
    for spec in exported_program.graph_signature.output_specs:
        if spec.arg.name in renames:
            spec.arg.name = renames[spec.arg.name]


# ── FX node fixes ───────────────────────────────────────────────────────────
# Per-node in-place rewrites applied to the `ExportedProgram` graph after the Dynamo trace
# but before `openvino.convert_model`. Each `_fix_*(gm, node) -> bool` factory is registered
# via `@register_fx_node_fix("openvino")` and returns `True` when it consumed the node
# (no further fixes run against it). Use this for OV-frontend quirks that are easier to
# repair at the FX level than to patch around at the torch op level.
#
# To add a new fix: define a `_fix_*` callable and decorate it.


@register_fx_node_fix("openvino")
def _fix_sym_float(gm, node):
    """``torch.sym_float`` is a no-op at the OV layer (it's a Python-level SymInt→SymFloat cast).
    Replace it with its input — affects deformable_detr, focalnet, mask2former, deepseek_ocr2.
    """
    if node.target is not torch.sym_float:
        return False
    node.replace_all_uses_with(node.args[0])
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_sym_min_max(gm, node):
    """Rewrite ``torch.sym_min``/``torch.sym_max`` to the built-in ``min``/``max``.

    OV's FX decoder keys translations on ``str(target)``. ``torch.sym_min`` reprs to
    ``<function sym_min at 0xADDRESS>`` — the address varies per process so no
    ``ConversionExtension`` string can match. ``min``/``max`` repr to stable
    ``<built-in function min>``/``<built-in function max>``, which we register translators
    for. The numeric behaviour is identical for SymInts.
    """
    if node.target is torch.sym_min:
        node.target = min
        return True
    if node.target is torch.sym_max:
        node.target = max
        return True
    return False


@register_fx_program_fix("openvino")
def _fix_to_dtype_layout_in_subgraphs(exported_program):
    """Rewrite ``aten.to.dtype_layout`` and ``aten.to.device`` in *submodule* FX graphs to
    ``aten._to_copy`` (with just the dtype kwarg).

    OV's PyTorch frontend registers ``ConversionExtension`` handlers only for the top-level
    graph; nested ``HigherOrderOp`` subgraphs (like the one wrapped by
    ``wrap_with_set_grad_enabled`` in Chameleon's rotary path) still see the raw
    ``aten.to.dtype_layout`` node, for which OV has no translator — resulting in a dangling
    ``torch::None`` constant. Rewriting the FX target here lets OV's built-in
    ``aten._to_copy.default`` handler take over (which we also override at the top level to
    swallow complex-dtype casts)."""
    # Walk the top-level graph AND every submodule's graph (higher-order-op subgraphs).
    graphs = [exported_program.graph_module]
    graphs.extend(m for _, m in exported_program.graph_module.named_children() if hasattr(m, "graph"))
    for gm_or_submod in graphs:
        for node in list(gm_or_submod.graph.nodes):
            if node.op != "call_function":
                continue
            target = node.target
            if target is torch.ops.aten.to.dtype:
                # ``aten.to.dtype(tensor, dtype)`` — dtype is positional arg[1].
                dtype = node.args[1] if len(node.args) > 1 else node.kwargs.get("dtype")
                node.target = torch.ops.aten._to_copy.default
                node.args = (node.args[0],)
                node.kwargs = {"dtype": dtype} if dtype is not None else {}
            elif target in (torch.ops.aten.to.dtype_layout, torch.ops.aten.to.device, torch.ops.aten.to.other):
                dtype = node.kwargs.get("dtype")
                node.target = torch.ops.aten._to_copy.default
                node.args = (node.args[0],)
                node.kwargs = {"dtype": dtype} if dtype is not None else {}
        gm_or_submod.recompile()


@register_fx_node_fix("openvino")
def _fix_drop_assert_ops(gm, node):
    """Erase ``aten._assert_tensor_metadata`` / ``aten._assert_scalar`` nodes.

    ``torch.export`` inserts these as dead-code (num_users=0) runtime assertions, but OV's
    frontend translates them into ``torch::None`` constants whose downstream consumers can't
    drop them — causing ``OpConversionFailure``. They have no semantic effect on the model.
    """
    if node.target not in (torch.ops.aten._assert_tensor_metadata.default, torch.ops.aten._assert_scalar.default):
        return False
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("openvino")
def _fix_scatter_reduce(gm, node):
    """Lower ``aten.scatter_reduce.two`` at the FX level — OV's frontend has no translation,
    and its ``ScatterElementsUpdate`` op can't accept the ``reduce`` string as a constant input.

    Handles two patterns the MoE/SSM models use:
      * ``reduce="sum", include_self=True`` → ``aten.scatter_add`` (BLT/JetMoe/NemotronH router).
      * ``reduce="amax", include_self=False`` → masked-max over a one-hot expansion of ``index``
        (BLT byte-pooling). Other combinations fall through to the generic OpConversionFailure.
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

    return False


@register_fx_node_fix("openvino")
def _fix_empty_cat(gm, node):
    """Drop ``aten.cat([empty, x], dim)`` constructed by ``DynamicLayer`` for prefill — the empty
    operand is a rank-1 ``f32[0]`` from ``aten.detach_(lift_fresh_copy(...))``, which OV's torch
    frontend can't broadcast against the non-empty 4D operand for a ``dim=-2`` cat (it rejects
    with ``Axis -2 out of the tensor rank range [-1, 0]``). Mathematically the cat is identity
    when one operand is 0-element, so replace its uses with the non-empty operand.
    """
    if node.target is not torch.ops.aten.cat.default:
        return False

    operands = node.args[0]
    if not isinstance(operands, (list, tuple)) or len(operands) != 2:
        return False

    from torch.fx.experimental.symbolic_shapes import guard_or_false

    def _is_empty(n):
        val = n.meta.get("val") if hasattr(n, "meta") else None
        if val is None:
            return False
        # ``numel() == 0`` on a compound SymInt expression trips ``GuardOnDataDependentSymNode``
        # (MinimaxM3VL — the concat operand has ``3*u0*u1*u2 + ...`` numel). Default to
        # ``False`` when we can't tell — treating the cat as non-empty keeps it in the graph,
        # which is always correct (the empty-cat optimisation just doesn't fire).
        return guard_or_false(val.numel() == 0)

    if _is_empty(operands[0]):
        keep = operands[1]
    elif _is_empty(operands[1]):
        keep = operands[0]
    else:
        return False

    node.replace_all_uses_with(keep)
    gm.graph.erase_node(node)
    return True


# ── Torch patches ───────────────────────────────────────────────────────────
# Each `_patch_*(original)` factory is registered via `@register_patch("openvino", path)`
# and reversibly swaps a `torch` op the OV frontend can't lower with a decomposed
# equivalent. Reverted on exit by `apply_patches("openvino")`.
#
# To add a new patch: define a `_patch_*` factory and decorate it.


@register_patch("openvino", "torch.nn.functional.layer_norm")
def _patch_layer_norm(original):
    """Substitute identity ``weight=ones``/``bias=zeros`` when either is ``None``.

    OV's frontend records a ``torch::None`` constant for any unwired optional, then refuses to
    convert it (``None constant cannot be converted to OpenVINO opset``). LayerNorm without
    affine still computes ``(x - mean) / sqrt(var + eps)``; passing identity tensors keeps the
    math unchanged and gives OV concrete operands. Affects Chameleon (no-affine RMSNorm path)
    and any model that calls ``F.layer_norm(..., weight=None, bias=None)``.
    """

    def patch(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        if weight is None:
            weight = torch.ones(normalized_shape, dtype=input.dtype, device=input.device)
        if bias is None:
            bias = torch.zeros(normalized_shape, dtype=input.dtype, device=input.device)
        return original(input, normalized_shape, weight, bias, eps)

    return patch


@register_patch("openvino", "torch.nn.functional.scaled_dot_product_attention")
def _patch_sdpa(original):
    """Pre-expand K/V to Q's head count before calling SDPA.

    OV's ``opset13::ScaledDotProductAttention`` op rejects GQA shapes (e.g. Q=[B,4,T,D],
    K/V=[B,2,T,D]) with ``Key input shape not compatible with other inputs``. Repeating K/V via
    ``repeat_interleave`` on the head axis keeps the math identical and gives OV matching shapes.
    """

    def patch(query, key, value, *args, **kwargs):
        q_heads, k_heads = query.shape[-3], key.shape[-3]
        if q_heads != k_heads and q_heads % k_heads == 0:
            reps = q_heads // k_heads
            key = key.repeat_interleave(reps, dim=-3)
            value = value.repeat_interleave(reps, dim=-3)
        return original(query, key, value, *args, **kwargs)

    return patch


@register_patch("openvino", "transformers.masking_utils._vmap_expansion_sdpa")
def _patch_broadcast_mask_expansion(_original):
    """Replace vmap-based mask expansion with broadcast expansion.

    OV's PyTorch frontend can't trace through ``torch.vmap`` — the input tensors look like
    they "escaped" the vmap context. Same shape of fix as the ONNX exporter's.
    """

    def patch(mask_function):
        def _expanded(batch_arange, head_arange, q_arange, kv_arange):
            broadcasted = masking_utils._non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange)
            return mask_function(*broadcasted).expand(
                batch_arange.shape[0], head_arange.shape[0], q_arange.shape[0], kv_arange.shape[0]
            )

        return _expanded

    return patch


@register_patch("openvino", "torch.histc")
def _patch_histc(original):
    """Replace ``torch.histc`` with a deterministic ``zeros + scatter_add_`` equivalent.

    OV's PyTorch frontend has no lowering for ``aten.histc``. The MoE token-counting path uses
    integer inputs (expert ids), which ``torch.histc`` doesn't support natively anyway. The
    decomposition pre-allocates a ``zeros(bins)`` (static shape) and accumulates via
    ``scatter_add_``, both OV-friendly primitives.
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


@register_patch("openvino", "torch.empty_permuted")
def _patch_empty_permuted(original):
    """Replace ``torch.empty_permuted(size, physical_layout, ...)`` with plain ``torch.empty(size, ...)``.

    OV's frontend has no ``aten.empty_permuted`` lowering. The op exists only to hint a memory
    layout (stride) — the values are uninitialised either way, and downstream reads see the same
    logical content. ``torch.empty`` is enough.
    """

    def patch(size, physical_layout, **kwargs):
        return torch.empty(size, **kwargs)

    return patch


@register_patch("openvino", "torch.polar")
def _patch_polar(original):
    """Build ``polar(abs, angle)`` as ``complex(abs*cos(angle), abs*sin(angle))``.

    OV has no ``aten.polar`` lowering. Euler's formula gives the same result through ops the
    frontend already supports.
    """

    def patch(abs, angle):
        return torch.complex(abs * angle.cos(), abs * angle.sin())

    return patch


@register_patch("openvino", "torch.bernoulli")
def _patch_bernoulli(original):
    """Strip randomness from ``torch.bernoulli`` — return ``zeros_like(p)`` during export.

    Stochastic ops have no place in an exported graph; the training-time sampling is
    deterministic-zero at inference (eval mode), so the export-time substitution is correct
    for the only modes that actually export.
    """

    def patch(input, *args, **kwargs):
        return torch.zeros_like(input)

    return patch


@register_patch("openvino", "torch.randn", "torch.randn_like")
def _patch_randn(original):
    """Strip randomness from ``torch.randn`` / ``torch.randn_like`` — return zeros.

    Same rationale as ``torch.bernoulli``: stochastic noise has no place in an exported graph;
    the inference-time path doesn't sample, so zero is what the model would see.
    """

    def patch(*args, **kwargs):
        if args and isinstance(args[0], torch.Tensor):
            return torch.zeros_like(args[0])
        return torch.zeros(*args, **kwargs)

    return patch


@register_patch("openvino", "torch.randint")
def _patch_randint(original):
    """Strip randomness from ``torch.randint`` — return zeros.

    Same rationale as ``torch.bernoulli`` / ``torch.randn``.
    """

    def patch(*args, **kwargs):
        # Signatures: ``randint(high, size, ...)`` or ``randint(low, high, size, ...)``.
        size = next((a for a in args if isinstance(a, (list, tuple, torch.Size))), kwargs.get("size"))
        return torch.zeros(size, dtype=kwargs.get("dtype", torch.int64), device=kwargs.get("device"))

    return patch


@register_patch("openvino", "torch.bincount", "torch.Tensor.bincount")
def _patch_bincount(original):
    """Replace ``torch.bincount`` with ``zeros + scatter_add_`` of size ``minlength`` (or input max+1
    when unknown).

    OV's PyTorch frontend has no ``aten.bincount`` lowering — same shape of fix as
    ``_patch_histc``. The static output shape ``minlength`` keeps shape inference happy.
    """

    def patch(input, weights=None, minlength=0):
        flat = input.reshape(-1)
        bins = max(int(minlength), int(flat.max().item()) + 1 if flat.numel() > 0 else 0)
        out_dtype = weights.dtype if weights is not None else torch.long
        counts = torch.zeros(bins, dtype=out_dtype, device=input.device)
        src = weights.reshape(-1).to(out_dtype) if weights is not None else torch.ones_like(flat, dtype=out_dtype)
        return counts.scatter_add_(0, flat.long(), src)

    return patch


@register_patch("openvino", "torch.nn.functional.interpolate")
def _patch_interpolate(original):
    """Disable antialias for ``F.interpolate(..., antialias=True)`` during OV export.

    OV's frontend has no ``aten._upsample_bilinear2d_aa`` lowering. Antialiasing is a
    pre-resample low-pass filter — turning it off costs a tiny amount of image-side quality but
    keeps the graph translatable. Affects siglip2 and lfm2_vl.
    """

    def patch(input, *args, **kwargs):
        kwargs.pop("antialias", None)
        return original(input, *args, **kwargs)

    return patch


@register_patch("openvino", "torch.fft.irfft")
def _patch_irfft(original):
    """Replace ``irfft`` with ``ifft`` over the conjugate-mirrored input — same shape of fix as
    the ONNX exporter's. OV's ``DFT`` op rejects ``is_onesided=1``/``inverse=1`` together; the
    mirrored ifft path sidesteps it.
    """

    def patch(input, n=None, dim=-1, norm=None):
        if n is None:
            n = 2 * (input.shape[dim] - 1)
        slc = [slice(None)] * input.ndim
        slc[dim] = slice(1, -1)
        full = torch.cat([input, input[tuple(slc)].flip(dims=[dim]).conj()], dim=dim)
        return torch.fft.ifft(full, n=n, dim=dim, norm=norm).real

    return patch


@register_patch("openvino", "torch.fft.rfft")
def _patch_rfft(original):
    """Replace ``rfft`` with ``fft`` + slice to the one-sided half. OV's ``DFT(is_onesided=1)``
    has no inverse-pair (see ``_patch_irfft``); using two-sided + slice gives the same result
    for the forward direction. Affects audio models (wav2vec*, seamless_m4t, pop2piano)."""

    def patch(input, n=None, dim=-1, norm=None):
        full = torch.fft.fft(input, n=n, dim=dim, norm=norm)
        n_full = full.shape[dim]
        slc = [slice(None)] * full.ndim
        slc[dim] = slice(0, n_full // 2 + 1)
        return full[tuple(slc)]

    return patch


@register_patch("openvino", "torch.fft.fft")
def _patch_fft(original):
    """``torch.fft.fft`` lowers to ``aten._fft_c2c.default`` which OV's frontend doesn't
    translate. Build the DFT manually from the twiddle matrix — quadratic but adequate for
    audio-encoder-sized FFTs that hit this path.
    """

    def patch(input, n=None, dim=-1, norm=None):
        if n is None:
            n = input.shape[dim]
        # Twiddle matrix W[k, j] = exp(-2j pi k j / n) — emit via complex(cos, -sin).
        k = torch.arange(n, device=input.device, dtype=torch.float32)
        j = k.view(-1, 1)
        angles = -2.0 * torch.pi * k * j / n
        twiddle = torch.complex(angles.cos(), angles.sin())
        # Move target dim to last, matmul against twiddle, move back.
        x = input.to(torch.complex64) if not torch.is_complex(input) else input
        x = x.movedim(dim, -1)
        out = x @ twiddle.T
        return out.movedim(-1, dim)

    return patch


@register_patch("openvino", "torch.fft.ifft")
def _patch_ifft(original):
    """Inverse of ``_patch_fft`` — uses conjugate twiddle and divides by ``n``."""

    def patch(input, n=None, dim=-1, norm=None):
        if n is None:
            n = input.shape[dim]
        k = torch.arange(n, device=input.device, dtype=torch.float32)
        j = k.view(-1, 1)
        angles = 2.0 * torch.pi * k * j / n
        twiddle = torch.complex(angles.cos(), angles.sin())
        x = input.to(torch.complex64) if not torch.is_complex(input) else input
        x = x.movedim(dim, -1)
        out = (x @ twiddle.T) / n
        return out.movedim(-1, dim)

    return patch


@register_patch("openvino", "torch.fft.fftn")
def _patch_fftn(original):
    """Multi-dim FFT decomposed as successive 1-D ``torch.fft.fft`` calls along each ``dim``.

    OV has no ``aten._fft_c2c`` lowering for N-D inputs; the iterative 1-D form composes with
    our ``_patch_fft`` so each axis is translated cleanly. Affects FNet.
    """

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = input
        for d, n in zip(dims, sizes):
            out = torch.fft.fft(out, n=n, dim=d, norm=norm)
        return out

    return patch


@register_patch("openvino", "torch.fft.ifftn")
def _patch_ifftn(original):
    """Multi-dim inverse FFT — same decomposition as ``_patch_fftn`` via ``torch.fft.ifft``."""

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = input
        for d, n in zip(dims, sizes):
            out = torch.fft.ifft(out, n=n, dim=d, norm=norm)
        return out

    return patch


@register_patch("openvino", "torch.fft.rfftn")
def _patch_rfftn(original):
    """Real N-D FFT — last dim uses ``rfft`` (one-sided), remaining dims use ``fft``."""

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = input
        for d, n in zip(dims[:-1], sizes[:-1]):
            out = torch.fft.fft(out, n=n, dim=d, norm=norm)
        return torch.fft.rfft(out, n=sizes[-1], dim=dims[-1], norm=norm)

    return patch


@register_patch("openvino", "torch.fft.irfftn")
def _patch_irfftn(original):
    """Real N-D inverse FFT — last dim uses ``irfft``, remaining dims use ``ifft``."""

    def patch(input, s=None, dim=None, norm=None):
        dims = list(range(input.ndim)) if dim is None else list(dim)
        sizes = [None] * len(dims) if s is None else list(s)
        out = torch.fft.irfft(input, n=sizes[-1], dim=dims[-1], norm=norm)
        for d, n in zip(dims[:-1], sizes[:-1]):
            out = torch.fft.ifft(out, n=n, dim=d, norm=norm)
        return out.real if torch.is_complex(out) else out

    return patch


@register_patch("openvino", "torch.Tensor.scatter_reduce_", "torch.Tensor.scatter_reduce")
def _patch_scatter_reduce(original):
    """Decompose ``scatter_reduce_(dim, index, src, reduce)`` into ``scatter_*`` variants OV
    can lower. ``sum``/``amax``/``amin`` map to ``scatter_add_``/``scatter_reduce(amax)`` /
    ``scatter_reduce(amin)`` already, but the ``two`` overload OV doesn't recognise has the
    same algorithmic content — replace with the plain ``scatter_add_`` for ``sum`` (the only
    reduce mode actually used in the failing model, BLT).
    """

    def patch(self, dim, index, src, *, reduce="sum", include_self=True):
        if reduce == "sum":
            if not include_self:
                self.zero_()
            return self.scatter_add_(dim, index, src)
        return original(self, dim, index, src, reduce=reduce, include_self=include_self)

    return patch


# ── OpenVINO conversion extensions ──────────────────────────────────────────
# Custom OV-side translations registered in ``_OV_CONVERSION_EXTENSIONS`` and passed to
# ``openvino.convert_model(extension=...)``. Mirrors the role of ONNX's
# ``_ONNX_TRANSLATION_TABLE``: use this when an op has no equivalent torch-level decomposition.
# Each ``_convert_*(context)`` receives a ``NodeContext`` (``context.get_input(i)`` for inputs)
# and returns a list of output ports built with ``openvino.opset14`` ops.
#
# To add a new translation: implement ``_convert_*`` and append a ``ConversionExtension`` to
# ``_OV_CONVERSION_EXTENSIONS``.


def _convert_grouped_mm(context):
    """Convert ``aten._grouped_mm`` / ``transformers.grouped_mm_fallback`` to OV ops.

    ``grouped_mm(mat_a: (M, K), mat_b: (G, K, N), offs: (G,)) -> (M, N)`` computes
    ``out[offs[g-1]:offs[g]] = mat_a[offs[g-1]:offs[g]] @ mat_b[g]`` per expert ``g``.
    ``G`` (number of experts) must be static at translation time, so we unroll the loop and
    emit ``G`` independent ``Slice + Gather + MatMul`` triples followed by a final ``Concat``.
    """
    mat_a = context.get_input(0)
    mat_b = context.get_input(1)
    offs = context.get_input(2)

    G = mat_b.get_partial_shape()[0].get_length()
    offs_i64 = ov_ops.convert(offs, "i64")
    axes_0 = ov_ops.constant(np.array([0], dtype=np.int64))
    step_1 = ov_ops.constant(np.array([1], dtype=np.int64))
    prev_end = ov_ops.constant(np.array([0], dtype=np.int64))

    outputs = []
    for g in range(G):
        g_lo = ov_ops.constant(np.array([g], dtype=np.int64))
        g_hi = ov_ops.constant(np.array([g + 1], dtype=np.int64))
        end = ov_ops.slice(offs_i64, g_lo, g_hi, step_1, axes_0)  # (1,) — offs[g]
        a_g = ov_ops.slice(mat_a, prev_end, end, step_1, axes_0)  # (n_g, K)
        w_g_3d = ov_ops.slice(mat_b, g_lo, g_hi, step_1, axes_0)  # (1, K, N)
        w_g = ov_ops.squeeze(w_g_3d, axes_0)  # (K, N)
        outputs.append(ov_ops.matmul(a_g, w_g, transpose_a=False, transpose_b=False).output(0))
        prev_end = end

    return [ov_ops.concat(outputs, axis=0).output(0)]


def _convert_empty_permuted(context):
    """Convert ``aten.empty_permuted`` to a zero-initialised constant of the requested shape.

    ``empty_permuted`` is uninitialised — only the shape matters for downstream ops. OV has no
    direct equivalent; emit a zero ``Broadcast`` of the right shape and dtype.
    """
    size = context.get_input(0)
    # Default to f32; in the MoE expert path the result feeds straight into integer index ops or
    # gets overwritten before any read, so dtype doesn't propagate to outputs.
    zero = ov_ops.constant(np.float32(0.0))
    return [ov_ops.broadcast(zero, size).output(0)]


def _convert_index_put(context):
    """Convert ``aten.index_put(self, indices, values, accumulate=False)`` via
    ``scatter_nd_update``. SAM2's mask-decoder mask-assignment pattern hits this — single index
    tensor over a slice, no accumulation. Mirrors ONNX's ``_aten_index_put`` bool-mask path."""
    self_tensor = context.get_input(0)
    indices = context.get_input(1)  # list of index tensors
    values = context.get_input(2)
    # Stack indices into the ``(K, N)`` shape OV's ScatterNDUpdate expects, then update.
    if isinstance(indices, list):
        stacked = ov_ops.concat([ov_ops.unsqueeze(i, ov_ops.constant(-1)) for i in indices], axis=-1)
    else:
        stacked = indices
    return [ov_ops.scatter_nd_update(self_tensor, stacked, values).output(0)]


def _convert_view_as_real(context):
    """``view_as_real(complex)`` reinterprets a complex tensor as ``[..., 2]`` real. Our
    ``_convert_complex`` already represents complex tensors that way, so this is identity."""
    return [context.get_input(0)]


def _convert_fft_c2c(context):
    """Convert ``aten._fft_c2c(self, dim, normalization, forward)`` to OV's ``DFT``/``IDFT``.

    OV's ``dft``/``idft`` expect a trailing ``[..., 2]`` real/imag pair. Our ``_convert_complex``
    produces that layout already. For models that call ``_fft_c2c`` on a real-valued tensor
    (FNet, where ``torch.fft.fftn(real)`` implicitly promotes to complex), we stack a zero
    imaginary component on the last dim first. We detect the input rank via partial shape and
    only inject the stack when there's no trailing ``[..., 2]`` already.
    """
    data = context.get_input(0)
    axes = context.get_input(1)
    forward = bool(context.get_values_from_const_input(3))
    # If the input doesn't already end in a 2-element axis, treat it as real and pad imag=0.
    pshape = data.get_partial_shape()
    needs_pair = pshape.rank.is_static and (
        not pshape[pshape.rank.get_length() - 1].is_static or pshape[pshape.rank.get_length() - 1].get_length() != 2
    )
    if needs_pair:
        zeros = ov_ops.broadcast(ov_ops.constant(np.float32(0.0)), ov_ops.shape_of(data))
        data = ov_ops.concat(
            [ov_ops.unsqueeze(data, ov_ops.constant(-1)), ov_ops.unsqueeze(zeros, ov_ops.constant(-1))],
            axis=-1,
        )
    op = ov_ops.dft if forward else ov_ops.idft
    return [op(data, ov_ops.convert(axes, "i64")).output(0)]


def _convert_conj(context):
    """Convert ``aten._conj(complex)`` — complex conjugate. With our ``[..., 2]`` real/imag
    representation, this negates the imaginary part. We split into real/imag, negate imag,
    and concat back. Used by manual FFT decompositions."""
    data = context.get_input(0)
    # last dim is 2 — split along axis -1 into real/imag, then concat [real, -imag]
    axes_neg1 = ov_ops.constant(np.array([-1], dtype=np.int64))
    real_part = ov_ops.gather(data, ov_ops.constant(np.int64(0)), axes_neg1)
    imag_part = ov_ops.gather(data, ov_ops.constant(np.int64(1)), axes_neg1)
    neg_imag = ov_ops.negative(imag_part)
    return [
        ov_ops.concat(
            [ov_ops.unsqueeze(real_part, axes_neg1), ov_ops.unsqueeze(neg_imag, axes_neg1)],
            axis=-1,
        ).output(0)
    ]


def _convert_bitwise_not(context):
    """Convert ``aten.bitwise_not`` — OV's default translator internally calls ``torch.sym_float``
    on the input's dynamic dims to compute output shape metadata, and that Python-level call
    remains as an unconverted node in the resulting graph. Emit ``LogicalNot`` on a boolean
    view of the input; ``bitwise_not`` on bool would reject with ``is_integral()`` check.
    Affects deformable_detr, mask2former."""
    data = context.get_input(0)
    return [ov_ops.logical_not(ov_ops.convert(data, "boolean")).output(0)]


def _convert_layer_norm(context):
    """Convert ``aten.layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable)`` to
    ``MVN + (weight * x + bias)``. OV's default translator decomposes to ``native_layer_norm``
    which returns a 3-tuple ``(out, mean, rstd)``; the unused ``mean`` / ``rstd`` outputs are
    emitted as ``torch::None`` constants that fail conversion (chameleon). Emitting MVN
    directly gives a single-output op with no dangling None."""
    data = context.get_input(0)
    normalized_shape = context.get_values_from_const_input(1)
    weight = context.get_input(2)
    bias = context.get_input(3)
    eps = float(context.get_values_from_const_input(4)) if context.get_input_size() > 4 else 1e-5
    ndim = data.get_partial_shape().rank.get_length()
    axes_len = len(normalized_shape) if hasattr(normalized_shape, "__len__") else 1
    axes = ov_ops.constant(np.array(list(range(ndim - axes_len, ndim)), dtype=np.int64))
    normalized = ov_ops.mvn(data, axes, normalize_variance=True, eps=eps, eps_mode="inside_sqrt")
    scaled = ov_ops.multiply(normalized, weight)
    shifted = ov_ops.add(scaled, bias)
    return [shifted.output(0)]


def _convert_aten_to(context):
    """Convert ``aten.to.{dtype,device,dtype_layout,other}`` — emit a real ``Convert`` when the
    target dtype is present, else identity.

    OV's frontend has no ``aten.to.*`` translations at all (only ``aten._to_copy.default``);
    every unhandled variant falls back to a ``torch::None`` constant that fails conversion
    (chameleon's rotary sub-module hits ``aten.to.dtype_layout``). ``layout`` / ``device``
    kwargs are silently dropped — OV exports are inherently device-neutral."""
    data = context.get_input(0)
    if not context.has_attribute("dtype"):
        # ``aten.to.device`` — just device move, no dtype. Emit identity.
        return [data]
    try:
        dtype = context.get_attribute("dtype")
    except Exception:
        # Complex dtypes throw. Skip (identity) — see ``_convert_to_copy``.
        return [data]
    if dtype is None:
        return [data]
    return [ov_ops.convert(data, dtype).output(0)]


def _convert_to_copy(context):
    """Convert ``aten._to_copy(self, dtype=..., ...)`` to an OV ``Convert``.

    OV's default translator throws (``Attribute dtype can't be converted to defined types``)
    when the target dtype is ``complex64`` — no native OV complex type. Our ``_convert_complex``
    uses a ``[..., 2]`` real representation, so the complex cast is a no-op we swallow. For all
    real dtypes we emit a real ``Convert`` — dropping the cast entirely regresses downstream
    ops like ``aten.bitwise_and.Tensor`` that need the mask to actually be ``bool`` (cpmant,
    chameleon)."""
    data = context.get_input(0)
    if not context.has_attribute("dtype"):
        return [data]
    try:
        dtype = context.get_attribute("dtype")
    except Exception:
        # Complex dtypes throw ``Attribute dtype can't be converted to defined types``. With
        # the ``[..., 2]`` real representation, the cast is a no-op.
        return [data]
    if dtype is None:
        return [data]
    return [ov_ops.convert(data, dtype).output(0)]


def _convert_sdpa(context):
    """Convert ``aten.scaled_dot_product_attention`` — wrapping OV's op with a mask-dtype fix.

    OV's ``opset13::ScaledDotProductAttention`` rejects int-typed masks. Under CUDA export
    ``aten.expand`` promotes bool masks to ``i64`` during OV translation, so we insert a
    ``Convert(→ boolean)`` on the mask input before instantiating the op. Q/K/V/scale pass
    through unchanged."""
    q, k, v = context.get_input(0), context.get_input(1), context.get_input(2)
    mask = context.get_input(3) if context.get_input_size() > 3 else None
    if mask is not None:
        mask = ov_ops.convert(mask, "boolean")
    is_causal = False
    if context.has_attribute("is_causal"):
        try:
            is_causal = bool(context.get_attribute("is_causal"))
        except Exception:
            is_causal = False
    return [ov_ops.scaled_dot_product_attention(q, k, v, mask, causal=is_causal).output(0)]


def _convert_complex(context):
    """Convert ``aten.complex(real, imag)`` by stacking as the last dim — OV represents complex
    tensors as ``[..., 2]`` real tensors via ``ComplexTypeMark``. Affects models that build
    complex tensors explicitly (RoPE polar form, manual FFT decompositions)."""
    real = context.get_input(0)
    imag = context.get_input(1)
    stacked = ov_ops.concat(
        [ov_ops.unsqueeze(real, ov_ops.constant(-1)), ov_ops.unsqueeze(imag, ov_ops.constant(-1))],
        axis=-1,
    )
    return [stacked.output(0)]


# ── SymInt builtin translations ─────────────────────────────────────────────
# torch.export records Python-level math on SymInts (``a % b``, ``a // b``, ``min(a, b)``)
# as ``call_function`` nodes whose target is the Python builtin or ``torch.sym_*`` callable.
# These survive into the EP because torch never lowers them — there's no aten op that
# produces a SymInt for ``mod``/``floordiv``/etc. OV's PyTorch frontend has no translation
# for them either, so we register one per builtin keyed on its ``str(target)`` literal.
# Each translator emits an OV opset17 elementwise op; the result is a 0-d integer tensor
# that downstream shape ops (view, reshape, expand) concat into shape lists natively.


def _convert_sym_binop(op):
    """Factory: build a 2-arg OV-op translator for SymInt binary builtins (add, mul, mod, …)."""

    def _convert(context):
        a, b = context.get_input(0), context.get_input(1)
        return [op(a, b).output(0)]

    return _convert


def _convert_sym_unop(op, *, cast_to_i64=False):
    """Factory: build a 1-arg OV-op translator for SymInt unary builtins (floor, ceil, sym_float).

    ``cast_to_i64`` casts the output back to ``i64`` — Python's ``floor(x)`` / ``ceil(x)`` on a
    SymFloat return an int, but OV's ``floor`` / ``ceiling`` are dtype-preserving, so a float
    input yields a float output. Downstream shape ops (SequenceMark → Concat) need i64;
    without the cast, mixed-dtype Concat fails ``element::Type::merge`` (focalnet)."""

    def _convert(context):
        out = op(context.get_input(0))
        if cast_to_i64:
            out = ov_ops.convert(out, "i64")
        return [out.output(0)]

    return _convert


def _convert_sym_floordiv(context):
    """``a // b`` over SymInts → ``floor(a / b)``, cast to i64. Used by patch/window-size
    computations (focalnet, donut_swin). The i64 cast keeps the result shape-op-friendly —
    downstream ``SequenceMark → Concat`` requires a uniform int dtype."""
    a, b = context.get_input(0), context.get_input(1)
    return [ov_ops.convert(ov_ops.floor(ov_ops.divide(a, b)), "i64").output(0)]


_OV_CONVERSION_EXTENSIONS: list[Any] = []
if is_openvino_available():
    _OV_CONVERSION_EXTENSIONS.extend(
        [
            ConversionExtension("aten._grouped_mm.default", _convert_grouped_mm),
            ConversionExtension("transformers.grouped_mm_fallback.default", _convert_grouped_mm),
            ConversionExtension("aten.empty_permuted.default", _convert_empty_permuted),
            ConversionExtension("aten.index_put.default", _convert_index_put),
            ConversionExtension("aten.complex.default", _convert_complex),
            ConversionExtension("aten.view_as_real.default", _convert_view_as_real),
            ConversionExtension("aten._fft_c2c.default", _convert_fft_c2c),
            ConversionExtension("aten._conj.default", _convert_conj),
            ConversionExtension("aten._to_copy.default", _convert_to_copy),
            ConversionExtension("aten.to.dtype", _convert_aten_to),
            ConversionExtension("aten.to.dtype_layout", _convert_aten_to),
            ConversionExtension("aten.to.device", _convert_aten_to),
            ConversionExtension("aten.to.other", _convert_aten_to),
            ConversionExtension("aten.layer_norm.default", _convert_layer_norm),
            ConversionExtension("aten.scaled_dot_product_attention.default", _convert_sdpa),
            ConversionExtension("aten.bitwise_not.default", _convert_bitwise_not),
            # SymInt builtins — see comment block above.
            ConversionExtension("<built-in function add>", _convert_sym_binop(ov_ops.add)),
            ConversionExtension("<built-in function sub>", _convert_sym_binop(ov_ops.subtract)),
            ConversionExtension("<built-in function mul>", _convert_sym_binop(ov_ops.multiply)),
            ConversionExtension("<built-in function truediv>", _convert_sym_binop(ov_ops.divide)),
            ConversionExtension("<built-in function floordiv>", _convert_sym_floordiv),
            ConversionExtension("<built-in function mod>", _convert_sym_binop(ov_ops.mod)),
            ConversionExtension("<built-in function pow>", _convert_sym_binop(ov_ops.power)),
            ConversionExtension("<built-in function floor>", _convert_sym_unop(ov_ops.floor, cast_to_i64=True)),
            ConversionExtension("<built-in function ceil>", _convert_sym_unop(ov_ops.ceiling, cast_to_i64=True)),
            ConversionExtension("<built-in function min>", _convert_sym_binop(ov_ops.minimum)),
            ConversionExtension("<built-in function max>", _convert_sym_binop(ov_ops.maximum)),
            # ``torch.sym_float`` has an address-based ``str()`` (not a stable ``<built-in ...>``
            # form), so we register by its runtime str. Emits a real→f32 Convert.
            ConversionExtension(str(torch.sym_float), _convert_sym_unop(lambda x: ov_ops.convert(x, "f32"))),
        ]
    )
