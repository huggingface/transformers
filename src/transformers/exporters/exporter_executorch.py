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
"""ExecuTorch exporter.

Extends `DynamoExporter` to produce an `ExecutorchProgramManager` for mobile and
edge deployment. The export pipeline runs:

1. **Backend preparation** (`_BACKEND_PREPARE`): `prepare_for_xnnpack` / `prepare_for_cuda`
   move the model to the target device/dtype and build the partitioner list.
2. **Torch patches** (`_PATCHES["executorch"]` via `apply_patches("executorch")`):
   reversibly swap `torch` ops the ExecuTorch backends can't accept (`split_copy`, `topk`,
   `avg_pool2d`, …) with decomposed equivalents. Reverted on exit.
3. **ExecuTorch patches** (`_PATCHES["executorch"]` via `apply_patches("executorch")`):
   reversibly swap ExecuTorch internals (`SpecPropPass`, `PruneEmptyTensorsPass`,
   `eval_upper_bound`, …) with versions that don't crash on legitimate dynamic-shape
   patterns. Same registry as stage 2, installed by the same `apply_patches` call.
4. **FX program fixes** (`apply_fx_program_fixes("executorch", ep)`): repair the
   `ExportedProgram` in place where the fix needs program-level context — widen
   `int_oo` upper bounds in `range_constraints`, fill missing placeholder `meta["val"]`.
5. **FX node fixes** (`apply_fx_node_fixes("executorch", ep.graph_module)`): per-node
   in-place rewrites — swap Python sym ops for their `executorch_prim.*` equivalents,
   rewrite `pow` as a `mul` chain, normalize amax/max negative dim, force contiguous clone.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.import_utils import is_executorch_available, is_torch_available
from .configs import ExecutorchConfig
from .exporter_dynamo import DynamoExporter
from .utils import (
    apply_fx_node_fixes,
    apply_fx_program_fixes,
    apply_patches,
    register_fx_node_fix,
    register_fx_program_fix,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.fx.experimental.symbolic_shapes import guard_or_true
    from torch.utils._sympy.numbers import IntInfinity
    from torch.utils._sympy.value_ranges import ValueRanges


if is_executorch_available():
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir.capture._config import EdgeCompileConfig
    from executorch.exir.passes.executorch_prim_ops_registry import _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS
    from executorch.exir.program import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower

if TYPE_CHECKING:
    if is_torch_available():
        from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)


class ExecutorchExporter(DynamoExporter):
    """Exporter that converts a [`PreTrainedModel`] to an ExecuTorch `ExecutorchProgramManager`.

    Example:

    ```python
    >>> from transformers.exporters.exporter_executorch import ExecutorchExporter, ExecutorchConfig

    >>> exporter = ExecutorchExporter()
    >>> et_program = exporter.export(model, inputs, config=ExecutorchConfig(backend="xnnpack"))
    >>> et_program.write_to_file("model.pte")
    ```
    """

    required_packages = ["torch", "executorch"]

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: dict[str, Any],
        config: ExecutorchConfig | dict[str, Any],
    ) -> ExecutorchProgramManager:
        """Export a model to ExecuTorch, applying backend preparation and torch op patches."""
        if isinstance(config, dict):
            config = ExecutorchConfig(**config)
        elif type(config) is not ExecutorchConfig:
            raise TypeError(f"Expected config to be an ExecutorchConfig or dict, got {type(config)}")

        prepare_for_backend = _BACKEND_PREPARE.get(config.backend)
        if prepare_for_backend is None:
            raise ValueError(f"Unsupported backend {config.backend} for ExecuTorch export")

        model, sample_inputs, partitioner = prepare_for_backend(model, sample_inputs)

        with apply_patches("executorch"):
            exported_program: ExportedProgram = super().export(model, sample_inputs, config=config)
            apply_fx_program_fixes("executorch", exported_program)
            apply_fx_node_fixes("executorch", exported_program.graph_module)
            edge_program_manager: EdgeProgramManager = to_edge_transform_and_lower(
                exported_program, partitioner=partitioner, compile_config=_get_edge_compile_config()
            )
            executorch_programs_manager: ExecutorchProgramManager = edge_program_manager.to_executorch()

        return executorch_programs_manager


def _get_edge_compile_config() -> EdgeCompileConfig:
    """Build the ``EdgeCompileConfig`` used for ``to_edge_transform_and_lower``.

    Adds non-core ATen ops to ``_core_aten_ops_exception_list`` so torch.export
    decompositions that produce these ops don't trip the edge-dialect verifier.
    These are ops that show up in transformers models (FFT in fnet, bucketize /
    is_all_true in T5 / mBart / Bart family, polar in seamless_m4t rotary, etc.)
    but aren't in the core ATen opset. The CPU portable kernels handle them at
    runtime; XNNPACK leaves them in the non-delegated CPU portion of the graph.
    """
    return EdgeCompileConfig(
        _core_aten_ops_exception_list=[
            torch.ops.aten._fft_c2c.default,
            torch.ops.aten._is_all_true.default,
            torch.ops.aten.bincount.default,
            torch.ops.aten.bucketize.Tensor,
            torch.ops.aten.cummax.default,
            torch.ops.aten.cummin.default,
            torch.ops.aten.polar.default,
            torch.ops.aten.rand_like.default,
            torch.ops.aten.randint.low,
            torch.ops.aten.randn_like.default,
            torch.ops.aten.searchsorted.Tensor,
            torch.ops.aten.unique_consecutive.default,
        ],
    )


# ── Stage 1: Backend preparation ──────────────────────────────────────────────
# Each prepare_for_* function receives the original model and sample inputs, applies backend-specific preparation,
# and returns the modified model, the list of partitioners to apply, and the modified sample inputs. Common patterns include:
# - Move the model to the target device.
# - Cast the model and inputs to the required dtype (e.g., bfloat16 for CUDA).
# - Build the backend-specific partitioner list passed to to_edge_transform_and_lower.
# To add a new backend: implement _prepare_for_new_backend and add it to the _BACKEND_PREPARE table.


def prepare_for_xnnpack(model: PreTrainedModel, sample_inputs: dict[str, Any]):
    """CPU inference via XNNPACK. Moves the model to CPU and uses the default XnnpackPartitioner."""

    model.requires_grad_(False)
    if model.device.type != "cpu":
        model = model.to(device="cpu")
    partitioner = [XnnpackPartitioner()]
    return model, sample_inputs, partitioner


def prepare_for_cuda(model: PreTrainedModel, sample_inputs: dict[str, Any]):
    """GPU inference via the ExecuTorch CUDA backend.

    Moves the model to CUDA and upcasts to bfloat16 — required by the CUDA backend.
    """
    model.requires_grad_(False)
    if model.device.type != "cuda":
        model = model.to(device="cuda")
    if model.dtype != torch.bfloat16:
        logger.warning(f"ExecuTorch CUDA backend requires bfloat16; upcasting model from {model.dtype}.")
        model = model.to(dtype=torch.bfloat16)
    partitioner = [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(model.__class__.__name__)])]
    return model, sample_inputs, partitioner


_BACKEND_PREPARE = {
    "xnnpack": prepare_for_xnnpack,
    "cuda": prepare_for_cuda,
}


# ── Stage 2: Torch patches ────────────────────────────────────────────────────
# Reversible swaps of `torch` ops the ExecuTorch backends can't lower (`split_copy`,
# `topk(k>dim)`, non-divisible `avg_pool2d`, `dropout`, in-place `view`, GQA-shaped
# SDPA …). Each `_patch_*(original)` factory is registered via
# `@register_patch("executorch", "dotted.path")` and installed through `apply_patches`.


@register_patch("executorch", "torch.split", "torch.Tensor.split")
def _patch_split(original):
    """Narrow-based split (split_copy not supported by CUDA backend)."""

    def patch(input, split_size_or_sections, dim=0):
        if isinstance(split_size_or_sections, int):
            splits = []
            total = input.size(dim)
            for i in range(0, total, split_size_or_sections):
                splits.append(input.narrow(dim, i, min(split_size_or_sections, total - i)))
            return tuple(splits)
        elif isinstance(split_size_or_sections, torch.SymInt):
            # Dynamic split size: `range(0, total, sym_int)` needs a concrete step, so
            # the narrow-based loop above doesn't apply. Defer to the original torch.split.
            return original(input, split_size_or_sections, dim)
        else:
            splits = []
            start = 0
            for size in split_size_or_sections:
                splits.append(input.narrow(dim, start, size))
                start += size
            return tuple(splits)

    return patch


@register_patch("executorch", "torch.chunk", "torch.Tensor.chunk")
def _patch_chunk(original):
    """Narrow-based chunk (delegates to split patch)."""

    def patch(input, chunks, dim=0):
        total = input.size(dim)
        chunk_size = (total + chunks - 1) // chunks
        # Call through torch.split which is already patched
        return torch.split(input, chunk_size, dim)

    return patch


@register_patch("executorch", "torch.topk", "torch.Tensor.topk")
def _patch_topk(original):
    """Argsort-based topk fallback."""

    def patch(input, k, dim=None, largest=True, sorted=True):
        if dim is None:
            dim = -1
        indices = torch.argsort(input, dim=dim, descending=largest)
        topk_indices = indices.narrow(dim, 0, k)
        topk_values = torch.gather(input, dim, topk_indices)
        return torch.return_types.topk((topk_values, topk_indices))

    return patch


@register_patch("executorch", "torch.detach", "torch.Tensor.detach")
def _patch_detach(_original):
    """No-op detach."""

    def patch(input):
        return input

    return patch


@register_patch("executorch", "torch.nn.functional.avg_pool2d")
def _patch_avg_pool2d(original):
    """Decompose avg_pool2d as depthwise conv2d (no CUDA ExecuTorch kernel)."""

    def patch(
        input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        kh, kw = kernel_size
        h, w = input.shape[-2:]
        channels = input.shape[1]
        actual_kh = min(kh, h + padding[0] * 2)
        actual_kw = min(kw, w + padding[1] * 2)
        divisor = divisor_override if divisor_override is not None else actual_kh * actual_kw
        weight = input.new_ones(channels, 1, actual_kh, actual_kw) / divisor
        return torch.nn.functional.conv2d(input, weight, bias=None, stride=stride, padding=padding, groups=channels)

    return patch


@register_patch("executorch", "torch.nn.functional.scaled_dot_product_attention")
def _patch_scaled_dot_product_attention(original):
    """Manual matmul+softmax fallback for cases unsupported by the ExecuTorch CUDA backend.

    Falls back to eager attention when:
    - enable_gqa=True
    - D_q != D_v (asymmetric head dims, e.g. MLA attention)
    - attn_mask is float (ExecuTorch CUDA SDPA only accepts bool masks)
    - any input shape contains unbacked SymInts (CPU path) — SDPA's internal
      dispatch branches on shapes (e.g. ``Eq(query_len, 1)`` for the decode
      fast-path) and trips ``GuardOnDataDependentSymNode`` on unbacked dims
      (idefics2, sam3).
    """

    def has_unbacked_shape(t):
        if t is None:
            return False
        return any(isinstance(s, torch.SymInt) and not s.node.expr.is_number for s in t.shape)

    def patch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        needs_eager_attention = (
            query.device.type == "cuda"
            and (
                kwargs.get("enable_gqa", False)
                or query.shape[-1] != value.shape[-1]
                or (attn_mask is not None and attn_mask.is_floating_point())
            )
        ) or any(has_unbacked_shape(t) for t in (query, key, value, attn_mask))
        if needs_eager_attention:
            scale_factor = scale if scale is not None else math.sqrt(query.shape[-1]) ** -1
            if key.shape[1] != query.shape[1]:
                n_rep = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(n_rep, dim=1)
                value = value.repeat_interleave(n_rep, dim=1)
            attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
            if is_causal:
                L, S = query.shape[-2], key.shape[-2]
                causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
                attn_weight = attn_weight.masked_fill(~causal_mask, float("-inf"))
            if attn_mask is not None:
                attn_weight = attn_weight + attn_mask
            attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
            return torch.matmul(attn_weight, value)
        return original(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
        )

    return patch


@register_patch("executorch", "torch.nn.functional.dropout")
def _patch_dropout(_original):
    """No-op dropout for inference export."""

    def patch(input, p=0.5, training=True, inplace=False):
        return input

    return patch


@register_patch("executorch", "torch.bernoulli", "torch.Tensor.bernoulli")
def _patch_bernoulli(_original):
    """Sample Bernoulli via ``rand_like`` + comparison.

    ExecuTorch ships no out-variant kernel for ``aten::bernoulli`` (used by
    SpeechT5's consistent dropout), so ``to_executorch`` fails with
    ``Missing out variants: {'aten::bernoulli'}``. Rewrite the call into
    ``(rand_like(input) < probs).to(input.dtype)`` — both ops have out variants.
    """

    def patch(input, *args, p=None, generator=None, out=None):
        # Two API shapes: bernoulli(input) (elementwise probabilities) and
        # bernoulli(input, p=...) (scalar probability, shape from input).
        if p is None and len(args) == 1:
            p = args[0]
        probs = input if p is None else p
        return (torch.rand_like(input) < probs).to(input.dtype)

    return patch


@register_patch("executorch", "torch.Tensor.expand")
def _patch_expand(original):
    """Force a contiguous copy after ``expand``.

    ``Tensor.expand`` produces a view with stride ``0`` along broadcast dims.
    ExecuTorch's memory planner rejects ``stride == 0`` (raises "0 in strides is not
    supported for ExecuTorch"): see
    https://github.com/pytorch/executorch/blob/main/exir/tensor.py
    Materialise the broadcast so the captured tensor has standard strides downstream.
    """

    def patch(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple, torch.Size)):
            sizes = tuple(sizes[0])
        return original(self, *sizes).clone(memory_format=torch.contiguous_format)

    return patch


# ── Stage 3: ExecuTorch patches ───────────────────────────────────────────────
# Reversible swaps of ExecuTorch internals (passes, verifiers, op dicts) that crash
# on legitimate dynamic-shape patterns: `SpecPropPass.update_placeholder_tensor_specs`,
# `eval_upper_bound`, `dim_order_from_stride`, XNNPACK squeeze/unsqueeze, complex-dtype
# validator, edge-dialect sym-op allowlist. Same registry as Stage 2 — each
# `_patch_*(original)` factory is registered via `@register_patch("executorch", path)`
# and installed by the single `apply_patches("executorch")` wrapping the export.


@register_patch(
    "executorch",
    "executorch.exir.sym_util.eval_upper_bound",
    "executorch.exir.passes.sym_shape_eval_pass.eval_upper_bound",
)
def _patch_eval_upper_bound(original):
    """Constraint-based bound, then trace hint, then ``_MAX_DIM_FLOOR``.

    Constraint propagation returns ``int_oo`` for compound expressions whose
    constraints don't compose (e.g. ``((s43*s53)//s70)``) or for sums of
    unbacked symbols (e.g. MoE per-expert cats ``u320+u321+...``); the
    fallbacks guarantee an ``int`` so ``ConstraintBasedSymShapeEvalPass``
    doesn't raise.
    """
    from executorch.exir.sym_util import eval_expr

    def patch(maybe_symint):
        result = original(maybe_symint)
        if isinstance(result, int):
            return result
        hint = eval_expr(maybe_symint)
        return hint if isinstance(hint, int) else _MAX_DIM_FLOOR

    return patch


@register_patch(
    "executorch", "executorch.exir.passes.prune_empty_tensors_pass.PruneEmptyTensorsPass.remove_empty_tensors_from_cat"
)
def _patch_remove_empty_tensors_from_cat(_original):
    """Replacement for ``PruneEmptyTensorsPass.remove_empty_tensors_from_cat``.

    The original checks ``input.numel() != 0`` directly; for tensors with
    unbacked dynamic shapes (e.g. ``74 * u176``) that raises
    ``GuardOnDataDependentSymNode`` because ``Ne(74*u176, 0)`` can't be proved
    either way at trace time. Using ``guard_or_true`` keeps unbacked-shape
    inputs conservatively (the pass is purely an optimisation).
    """
    from executorch.exir.dialects._ops import ops as exir_ops

    def patch(self, graph_module, cat_node):
        pruned = [arg for arg in cat_node.args[0] if guard_or_true(arg.meta["val"].numel() != 0)]
        cat_node.args = (pruned,) + cat_node.args[1:]
        if not pruned:
            cat_tensor = cat_node.meta["val"]
            with graph_module.graph.inserting_after(cat_node):
                full_like = graph_module.graph.create_node(
                    "call_function",
                    target=exir_ops.edge.aten.full.default,
                    args=(tuple(cat_tensor.shape), 0),
                    kwargs={"dtype": cat_tensor.dtype},
                )
                full_like.meta = cat_node.meta
                cat_node.replace_all_uses_with(full_like)

    return patch


@register_patch("executorch", "executorch.exir.verification.verifier._check_tensor_args_matching_op_allowed_dtype")
def _patch_check_tensor_args_dtype(original):
    """Suppress complex-dtype violations in
    ``_check_tensor_args_matching_op_allowed_dtype``.

    The validator's per-op allowed-dtype tables don't include ``complex64`` /
    ``complex128``, so models using complex tensors (FFT in fnet, complex-valued
    rotary embeddings in deepseek_v2) trip the check on ops like
    ``aten.unsqueeze_copy`` / ``aten.view_as_real_copy``. Those ops handle
    complex tensors correctly at runtime; the violation is purely cosmetic.
    """

    def patch(gm):
        try:
            original(gm)
        except Exception as exc:
            msg = str(exc)
            if "mismatched dtypes" in msg and ("complex64" in msg or "complex128" in msg):
                return
            raise

    return patch


@register_patch(
    "executorch",
    "executorch.exir.tensor.dim_order_from_stride",
    "executorch.exir.tensor_layout.dim_order_from_stride",
    "executorch.exir.emit._emitter.dim_order_from_stride",
    "executorch.exir.passes.replace_view_copy_with_view_pass.dim_order_from_stride",
)
def _patch_dim_order_from_stride(_original):
    """Replacement for ``executorch.exir.tensor.dim_order_from_stride``.

    The upstream version compares strides with ``guard_size_oblivious`` to sort
    them. When the strides are unbacked SymInts (e.g. ``splinter`` slicing on a
    data-dependent index), the comparison raises ``GuardOnDataDependentSymNode``
    deep inside ``spec_prop_pass``. Use ``guard_or_true`` / ``guard_or_false``
    so the sort still produces *a* dim order when the comparison is unbacked —
    the exact order on unbacked dims doesn't affect correctness, just memory layout.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

    def patch(stride):
        for s in stride:
            if guard_or_false(s == 0):
                raise ValueError("0 in strides is not supported for ExecuTorch.")

        class K:
            __slots__ = ("stride",)

            def __init__(self, stride):
                self.stride = stride

            def __lt__(self, other):
                return guard_or_true(self.stride < other.stride)

        sorted_dims = [i[0] for i in sorted(enumerate(stride), key=lambda x: K(x[1]), reverse=True)]
        return tuple(sorted_dims)

    return patch


@register_patch("executorch", "executorch.exir.passes.spec_prop_pass.SpecPropPass.update_placeholder_tensor_specs")
def _patch_update_placeholder_tensor_specs(_original):
    """Replacement for ``SpecPropPass.update_placeholder_tensor_specs``.

    The original unconditionally sets ``spec.const = True`` for placeholders in
    ``inputs_to_parameters``/``inputs_to_buffers``/``inputs_to_lifted_tensor_constants``.
    ``insert_write_back_for_buffers_pass`` can leave ``inputs_to_buffers``
    shifted by one slot, so a user input placeholder (e.g. ``input_ids``) is
    keyed as a buffer with a stale FQN; ``SpecPropPass`` builds no spec for it
    (``val`` is ``None``) and the assignment raises ``AttributeError``. Skip
    ``None`` specs so user inputs aren't mis-marked const.
    """
    from executorch.exir.passes.spec_prop_pass import _is_mutable_buffer

    def patch(self, exported_program, graph_module):
        sig = exported_program.graph_signature
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            if "spec" not in node.meta:
                raise RuntimeError(f"Placeholder node {node} missing meta['spec']")
            spec = node.meta["spec"]
            # make_spec returns the raw int/bool/float for scalar placeholders and
            # None for unsupported types — neither has a ``const`` attribute.
            if not hasattr(spec, "const"):
                continue
            if isinstance(node.target, str) and (
                node.target in sig.inputs_to_parameters
                or (node.target in sig.inputs_to_buffers and not _is_mutable_buffer(node, sig))
                or node.target in sig.inputs_to_lifted_tensor_constants
            ):
                spec.const = True

    return patch


@register_patch(
    "executorch",
    "executorch.exir.passes.executorch_prim_ops_registry._EXECUTORCH_SYM_OPS",
    "executorch.exir.verification.verifier._EXECUTORCH_SYM_OPS",
)
def _extend_sym_ops_allowlist(original):
    """Return the edge-dialect sym-op allowlist extended with sym ops that have no `executorch_prim.*`
    equivalent (`sym_ite`, `sym_not`, `sym_int`, `sym_sum`, `sym_float`).

    Trace-time-only ops don't need a runtime kernel; without this they still trip the verifier.
    """
    return original | {torch.sym_ite, torch.sym_not, torch.sym_int, torch.sym_sum, torch.sym_float}


def _make_squeeze_define_node(original):
    """Allow XNNPACK's squeeze/unsqueeze to serialize when output has multiple dynamic dims.

    The original ``define_node`` rejects any reshape with >1 dynamic output dim, but
    squeeze (removes a size-1 dim) and unsqueeze (adds a size-1 dim) don't change the
    number of dynamic dimensions — they're not really reshapes. The check is triggered
    when XNNPACK's ``conv1d_unsqueeze_pass`` wraps a conv1d in unsqueeze/conv2d/squeeze
    and the surrounding tensor has multiple dynamic dims (typical of audio/speech models
    where both batch and time are dynamic). Replace the strict check with a no-op when
    the dynamic-dim count is preserved across the squeeze.
    """
    from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (  # type: ignore[import-not-found]
        XNNStaticReshape,
        XNode,
    )
    from executorch.backends.xnnpack.utils.utils import get_input_node
    from torch.fx.experimental.symbolic_shapes import free_symbols

    def patch(self, node, xnn_graph, vals_to_ids, debug_handle):
        self.define_nodes_tensor_inputs_outputs(node, xnn_graph, vals_to_ids)
        input_id = vals_to_ids[get_input_node(node, 0)]
        output_id = vals_to_ids[node]
        new_shape = [0 if free_symbols(dim) else dim for dim in node.meta["val"].shape]
        xnn_graph.xnodes.append(
            XNode(
                xnode_union=XNNStaticReshape(
                    num_dims=len(new_shape),
                    new_shape=new_shape,
                    input_id=input_id,
                    output_id=output_id,
                    flags=0,
                ),
                debug_handle=debug_handle,
            )
        )

    return patch


@register_patch("executorch", "executorch.backends.xnnpack.operators.node_visitor._node_visitor_dict")
def _patch_squeeze_node_visitors(original):
    """Swap the squeeze/unsqueeze visitor entries in ``_node_visitor_dict`` with subclasses
    whose ``define_node`` skips the strict reshape check.

    XNNPACK's ``conv1d_unsqueeze_pass`` wraps conv1d in unsqueeze/conv2d/squeeze; the squeeze
    then trips the "reshape only supports 1 dynamic dimension" check when the surrounding
    tensor has multiple dynamic dims (audio / speech models). Squeeze/unsqueeze of a size-1
    dim doesn't actually change dynamism, so skip the check.

    The visitor classes live behind a dict-key lookup because ``@register_node_visitor``
    rebinds the decorated class name to ``None`` — there's no dotted path to them. Instead,
    swap the whole dict for a copy where the two affected keys point at subclasses with the
    patched method, so the production classes stay untouched.
    """
    new = dict(original)
    for key in ("aten.squeeze_copy.dim", "aten.unsqueeze_copy.default"):
        cls = original[key]
        new[key] = type(cls.__name__, (cls,), {"define_node": _make_squeeze_define_node(cls.define_node)})
    return new


# ── Stage 4: FX program fixes ─────────────────────────────────────────────────
# `@register_fx_program_fix("executorch")` on `(exported_program) -> None` callables
# applied in place between ``torch.export.export`` and ``to_edge_transform_and_lower``.
# Program-level fixes need context the per-node walk doesn't have: `range_constraints`,
# `graph_signature`, `state_dict`.

_MAX_DIM_MULTIPLIER = 4  # upper bound = max(lower * multiplier, floor)
_MAX_DIM_FLOOR = 1024  # minimum upper bound for any dynamic dim


def _as_int(x, default: int = 0) -> int:
    """Best-effort ``int(x)`` for sympy values, with a fallback for infinities.

    ``int(sympy.oo / -oo / IntInfinity)`` raises ``OverflowError`` → falls through
    to ``AttributeError`` on ``'Infinity'._mpf_``. Catch both so unbounded ends
    fall back to ``default`` instead of propagating sympy's internals.
    """
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return default


@register_fx_program_fix("executorch")
def _fix_range_constraints(exported_program: ExportedProgram) -> None:
    """Cap ``int_oo`` upper bounds for ExecuTorch compatibility.

    Uses ``max(lower * 4, trace_value * 4, 1024)`` per dim — keeps bounds
    proportional to actual sample sizes so XNNPACK memory planning doesn't
    overflow, while still covering trace-time values (e.g. VLM image tokens).
    """
    # Collect all range dicts that need patching: range_constraints (torch.export
    # verifiers) + shape_env.var_to_range (ExecuTorch sym_shape_eval_pass).
    range_dicts = [exported_program._range_constraints]
    var_to_val = {}
    for node in exported_program.graph_module.graph.nodes:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor) and hasattr(val, "fake_mode"):
            shape_env = val.fake_mode.shape_env
            range_dicts.append(shape_env.var_to_range)
            var_to_val = getattr(shape_env, "backed_var_to_val", None) or shape_env.var_to_val
            break  # all nodes share the same shape_env, so we only need one

    for rd in range_dicts:
        for sym, vr in rd.items():
            if isinstance(vr.upper, IntInfinity):
                lower = _as_int(vr.lower, 2)
                trace_val = _as_int(var_to_val.get(sym), 0)
                upper = max(lower * _MAX_DIM_MULTIPLIER, trace_val * _MAX_DIM_MULTIPLIER, _MAX_DIM_FLOOR)
                rd[sym] = ValueRanges(vr.lower, upper)


@register_fx_program_fix("executorch")
def _fix_missing_placeholder_vals(exported_program: ExportedProgram) -> None:
    """Ensure parameter/buffer/lifted-constant placeholders have a tensor ``meta["val"]``.

    ExecuTorch's ``SpecPropPass`` builds ``node.meta["spec"]`` from ``meta["val"]``
    via ``TensorSpec.from_tensor``. If ``val`` is ``None`` (or a non-tensor) for a
    placeholder that the graph signature marks as a parameter/buffer/lifted
    constant, ``spec`` stays ``None`` and a later ``spec.const = True`` crashes
    with ``AttributeError``. Fill in the missing val from the actual state-dict
    tensor so the spec round-trips correctly.
    """
    sig = exported_program.graph_signature
    state_dict = exported_program.state_dict
    constants = getattr(exported_program, "constants", {}) or {}

    sources = (
        (sig.inputs_to_parameters, state_dict),
        (sig.inputs_to_buffers, state_dict),
        (sig.inputs_to_lifted_tensor_constants, constants),
    )

    for node in exported_program.graph_module.graph.nodes:
        if node.op != "placeholder" or not isinstance(node.target, str):
            continue
        if isinstance(node.meta.get("val"), torch.Tensor):
            continue
        for input_map, store in sources:
            fqn = input_map.get(node.target)
            if fqn is None:
                continue
            tensor = store.get(fqn)
            if isinstance(tensor, torch.Tensor):
                node.meta["val"] = tensor
            break


# ── Stage 5: FX node fixes ────────────────────────────────────────────────────
# `@register_fx_node_fix("executorch")` on `(gm, node) -> bool` per-node fixers,
# applied in place by ``apply_fx_node_fixes("executorch", gm)`` right after the
# program fixes. Return ``True`` to consume the node; DCE runs at the end of the walk.


@register_fx_node_fix("executorch")
def _fix_amax_dim(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite negative ``dim`` indices on max/amax ops to positive ones.

    XNNPACK's ``op_max_dim`` visitor compares ``node.args[1]`` directly against
    2 and 3 without normalizing, so a ``dim=-1`` call on a 4-D tensor fails with
    ``amax.default only supports dim == 2 or dim == 3`` even though dim 3 is
    what was meant. Done for both ``aten.amax.default`` and ``aten.max.dim``
    (which gets folded into amax later during lowering).
    """
    if node.target not in (torch.ops.aten.amax.default, torch.ops.aten.max.dim) or len(node.args) < 2:
        return False
    input_node = node.args[0]
    input_val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
    rank = input_val.dim() if isinstance(input_val, torch.Tensor) else None
    if rank is None:
        return False
    dim_arg = node.args[1]
    if isinstance(dim_arg, int) and dim_arg < 0:
        new_args = list(node.args)
        new_args[1] = rank + dim_arg
        node.args = tuple(new_args)
        return True
    if isinstance(dim_arg, (list, tuple)) and any(isinstance(d, int) and d < 0 for d in dim_arg):
        new_dims = [d + rank if isinstance(d, int) and d < 0 else d for d in dim_arg]
        new_args = list(node.args)
        new_args[1] = type(dim_arg)(new_dims)
        node.args = tuple(new_args)
        return True
    return False


@register_fx_node_fix("executorch")
def _fix_python_sym_op(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Swap Python sym ops (``torch.sym_min``, ``math.ceil``, ...) for their
    ``executorch_prim.*`` equivalents.

    The edge-dialect verifier rejects Python ``FunctionType`` ops other than
    ``alloc`` (``verifier.py:317``). ExecuTorch has its own pass
    (``EdgeToBackendOpsPass``) that swaps these, but it only runs during
    ``to_executorch``, after the edge verifier already runs in
    ``to_edge_transform_and_lower``. Apply the same swap here.

    Only ``torch.sym_*`` and ``math.*`` targets are swapped — ``operator.add`` /
    ``mul`` / etc. are also used for tensor-tensor ops, where the ``Scalar``
    overload fails at runtime with ``Cannot cast NotImplemented to number``.
    """
    if node.target not in (torch.sym_float, torch.sym_max, torch.sym_min, math.ceil, math.trunc, round):
        return False
    replacement = _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS.get(node.target)
    if replacement is None:
        return False
    node.target = replacement
    return True


@register_fx_node_fix("executorch")
def _fix_clone_memory_format(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Force ``contiguous_format`` on ``aten.clone`` whose input has a non-standard dim order.

    ``Tensor.clone()`` defaults to ``preserve_format`` and inherits the source's stride
    layout. When a cache tensor has been transposed earlier (dim order e.g. ``[1, 0, 2, 3]``),
    the clone inherits it and ExecuTorch's ``dim_order_from_stride`` fails to map it to a
    ``torch.memory_format``. Only rewrite clones whose input ``meta["val"]`` is non-contiguous
    so we don't disturb the (much more common) clones of already-contiguous tensors — those
    can otherwise get optimised into pass-through nodes that XNNPACK rejects.
    """
    if node.target is not torch.ops.aten.clone.default:
        return False
    if node.kwargs.get("memory_format") is not None:
        return False
    input_val = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
    if not (isinstance(input_val, torch.Tensor) and not input_val.is_contiguous()):
        return False
    node.kwargs = {**node.kwargs, "memory_format": torch.contiguous_format}
    return True


@register_fx_node_fix("executorch")
def _fix_sym_pow_as_mul(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace ``operator.pow(sym_int, n)`` with a chain of ``executorch_prim.mul.Scalar``.

    The emitter has no entry for ``operator.pow`` (no ``executorch_prim.pow``), so a
    ``sym_size ** 2`` in model code (e.g. seamless_m4t's relative positional bias)
    raises ``invalid target for call_function <built-in function pow>`` at to_executorch.
    Rewrite small-integer exponents (n >= 1) as a multiplication chain — the
    ``executorch_prim.mul.Scalar`` op accepts SymInt operands.
    """
    if node.target is not operator.pow:
        return False
    base, exp = node.args
    if not isinstance(exp, int) or exp < 1:
        return False
    # `operator.mul` is kept out of `_fix_python_sym_op`'s allowlist (it crashes on
    # tensor-tensor calls), but the `executorch_prim.mul.Scalar` overload is fine when
    # we construct the chain ourselves with SymInt operands.
    mul_scalar = _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS.get(operator.mul)
    if mul_scalar is None:
        return False
    with gm.graph.inserting_before(node):
        running = base
        for _ in range(exp - 1):
            running = gm.graph.call_function(mul_scalar, (running, base))
    node.replace_all_uses_with(running)
    gm.graph.erase_node(node)
    return True
