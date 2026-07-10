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
import re
from collections.abc import MutableMapping
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_executorch_available, is_torch_available
from .configs import ExecutorchConfig
from .exporter_dynamo import DynamoExporter
from .utils import (
    apply_fx_node_fixes,
    apply_fx_program_fixes,
    apply_patches,
    module_dtype,
    register_fx_node_fix,
    register_fx_program_fix,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.fx.experimental.symbolic_shapes import (
        free_symbols,
        free_unbacked_symbols,
        guard_or_false,
        guard_or_true,
        statically_known_true,
    )
    from torch.fx.passes.infra.pass_base import PassResult
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils._sympy.numbers import IntInfinity
    from torch.utils._sympy.value_ranges import ValueRanges

    from .. import masking_utils
    from ..modeling_utils import PreTrainedModel


if is_executorch_available():
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (  # type: ignore[import-not-found]
        XNNStaticReshape,
        XNode,
    )
    from executorch.backends.xnnpack.utils.utils import get_input_node
    from executorch.exir.capture._config import EdgeCompileConfig
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.passes.executorch_prim_ops_registry import _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS
    from executorch.exir.passes.replace_view_copy_with_view_pass import _VIEW_OP, _is_view_copy, _ViewSpec
    from executorch.exir.passes.spec_prop_pass import _is_mutable_buffer
    from executorch.exir.program import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower
    from executorch.exir.sym_util import eval_expr
    from executorch.exir.tensor import determine_tensor_dynanism

    # The ExecuTorch CUDA backend pulls in `triton`, which CPU-only torch builds don't ship. Guard the
    # import on CUDA availability so the module still imports (and the xnnpack CPU path still works) on
    # CPU-only builds; `prepare_for_cuda` raises a clear error if the `cuda` backend is requested when
    # it isn't available.
    if torch.cuda.is_available():
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner


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
    tested_versions = {"torch": "2.11.0", "executorch": "1.2.0"}

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
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
    """CPU inference via XNNPACK.

    Moves the model to CPU: XNNPACK's partitioner/serializer and the edge-lowering passes all
    require a CPU-typed graph, and tracing on CPU also sidesteps per-model device bugs — models
    create in-``forward`` tensors (``arange``/``zeros``/sinusoids) without ``device=``, which
    default to CPU and would mismatch a CUDA model (``FakeTensor Device Propagation ... cuda, cpu``).
    ``prepare_for_export`` then casts the inputs to CPU during the trace."""

    model.requires_grad_(False)
    model = model.to(device="cpu")
    # XNNPACK has no `_grouped_mm.out` kernel — force MoE experts to `batched_mm`.
    if isinstance(model, PreTrainedModel) and model._can_set_experts_implementation():
        model.set_experts_implementation("batched_mm")
    partitioner = [XnnpackPartitioner()]
    return model, sample_inputs, partitioner


def prepare_for_cuda(model: PreTrainedModel, sample_inputs: dict[str, Any]):
    """GPU inference via the ExecuTorch CUDA backend, decoupled from the model's device.

    The backend requires bfloat16 (upcast here) and a visible GPU — it delegates ops to Triton
    kernels compiled by AOTInductor, which needs a GPU to compile/autotune. The model itself can
    stay on any device (e.g. CPU): AOTInductor targets the machine's GPU regardless of where the
    traced tensors live, so no `.to("cuda")` is needed."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment; cannot export to the ExecuTorch CUDA backend.")

    model.requires_grad_(False)
    dtype = module_dtype(model)
    if dtype is not None and dtype != torch.bfloat16:
        logger.warning(f"ExecuTorch CUDA backend requires bfloat16; upcasting model from {dtype}.")
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
    """`torch.chunk` decomposes through `aten.split_copy.Tensor`, which AOT inductor for the
    ExecuTorch CUDA backend can't lower (`split_copy.Tensor is missing a c-shim implementation`).
    Same root cause as `_patch_split`; route `chunk` through the already-patched `torch.split`
    so it ends up as a sequence of `narrow`s instead. XNNPACK lowers `chunk` natively, so we
    only swap when the input lives on CUDA.
    """

    def patch(input, chunks, dim=0):
        if input.device.type != "cuda":
            return original(input, chunks, dim)
        total = input.size(dim)
        chunk_size = (total + chunks - 1) // chunks
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


@register_patch("executorch", "transformers.masking_utils._vmap_expansion_sdpa")
def _patch_broadcast_mask_expansion(_original):
    """Replace vmap-based mask expansion with broadcast expansion. `aot_autograd` and
    `gen_vmap_plumbing` reject vmap-built masks under ExecuTorch's lowering passes."""

    def patch(mask_function):
        def _expanded(batch_arange, head_arange, q_arange, kv_arange):
            broadcasted = masking_utils._non_vmap_expansion_sdpa(batch_arange, head_arange, q_arange, kv_arange)
            return mask_function(*broadcasted).expand(
                batch_arange.shape[0], head_arange.shape[0], q_arange.shape[0], kv_arange.shape[0]
            )

        return _expanded

    return patch


@register_patch("executorch", "torch.nn.functional.scaled_dot_product_attention")
def _patch_scaled_dot_product_attention(original):
    """Route SDPA through the MATH backend, plus a manual matmul+softmax fallback for cases
    unsupported by the ExecuTorch CUDA backend.

    ``sdpa_kernel(MATH)`` forces the decomposable SDPA variant on any device — without it,
    CUDA traces pick ``_scaled_dot_product_efficient_attention``, which XNNPACK's edge-dialect
    verifier rejects as non-core-ATen. Same shape of fix as the Dynamo-path ``_patch_sdpa``,
    but unconditional here since the CUDA fused kernel is never lowerable by ExecuTorch's
    xnnpack backend. No-op on CPU (MATH is already the default), so this is safe everywhere.

    The eager-fallback path matches PyTorch's SDPA math kernel exactly
    (``_scaled_dot_product_attention_math`` in ``aten/src/ATen/native/transformers/attention.cpp``)
    — notably, the softmax stays in the input dtype rather than promoting to fp32. Falls back to
    eager (CUDA-backend only) when:
    - enable_gqa=True
    - D_q != D_v (asymmetric head dims, e.g. MLA attention)
    - attn_mask is float (ExecuTorch CUDA SDPA only accepts bool masks)

    The MATH-path output gets an explicit ``clone(memory_format=contiguous_format)`` so
    downstream strides don't depend on which SDPA layout torch picks: the pre-dispatch trace
    sees a contiguous ``(N, H, L, E)`` fake output (so ``.contiguous()`` would trace to
    nothing) and records downstream ``reshape``s as bare ``view`` nodes, but decomposition
    re-traces SDPA via ``scaled_dot_product_flash_attention_for_cpu``, which materializes an
    ``(L, N, H, E)`` buffer — invalidating those recorded views (``Cannot view a tensor with
    shape/strides``). ``clone`` records unconditionally and re-executes correctly under either
    layout, normalizing the strides the rest of the graph was recorded against.

    The eager fallback also fires on **any** device when ``attn_mask`` has a data-dependent
    (unbacked) batch dim — the Idefics2/3 / SmolVLM vision tower drops padding images via
    ``pixel_values[real_images_inds]`` (a boolean index → unbacked ``u0`` image count), so the
    vision attention mask carries batch ``u0``. ``to_edge_transform_and_lower`` decomposes the
    surviving ``aten.scaled_dot_product_attention`` node through the SDPA math CIA kernel, which
    guards ``Eq(u0, 1)`` on the mask's batch (broadcast-vs-not) and raises
    ``GuardOnDataDependentSymNode``. The manual matmul+softmax path masks against ``attn_weight``
    (both batch ``u0``) with plain broadcasting, so no ``Eq(u0, 1)`` guard is needed and no SDPA
    node survives to be re-decomposed.
    """

    def _has_unbacked_batch(t):
        # True when ``t``'s batch dim is a data-dependent (unbacked, ``u*``) SymInt.
        if t is None or t.ndim == 0:
            return False
        batch = t.shape[0]
        return isinstance(batch, torch.SymInt) and bool(free_unbacked_symbols(batch.node.expr))

    def patch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        needs_eager_attention = (
            query.device.type == "cuda"
            and (
                kwargs.get("enable_gqa", False)
                or query.shape[-1] != value.shape[-1]
                or (attn_mask is not None and attn_mask.is_floating_point())
            )
        ) or (attn_mask is not None and _has_unbacked_batch(attn_mask))
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
                if attn_mask.dtype == torch.bool:
                    attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn_weight = attn_weight + attn_mask
            attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1)
            return torch.matmul(attn_weight, value)
        with sdpa_kernel(SDPBackend.MATH):
            return original(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
            ).clone(memory_format=torch.contiguous_format)

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
    ExecuTorch's memory planner rejects ``stride == 0`` and raises "0 in strides is not
    supported for ExecuTorch" — see ``TensorSpec.__init__`` in
    https://github.com/pytorch/executorch/blob/v1.0.0/exir/tensor.py#L72. Materialise
    the broadcast so the captured tensor has standard strides downstream.
    """

    def patch(self, *sizes, **kwargs):
        # Forward whatever form the caller used — positional ``expand(*sizes)``, a single
        # list/tuple, or the keyword form ``expand(size=...)`` — straight to the original.
        result = original(self, *sizes, **kwargs)
        # Only materialise when ``expand`` actually introduced a stride-0 (broadcast) dim; a
        # no-broadcast expand is a plain view ExecuTorch's memory planner accepts as-is.
        if 0 in result.stride():
            return result.clone(memory_format=torch.contiguous_format)
        return result

    return patch


@register_patch("executorch", "torch.reshape", "torch.Tensor.reshape")
def _patch_reshape(original):
    """Materialise a non-contiguous input before ``reshape``.

    ExecuTorch's edge-lowering reshape reference refuses a non-contiguous input (e.g. the
    ``transpose(1, 2).reshape(...)`` in the packed vision-attention forward). A plain
    ``.contiguous()`` gets folded away by functionalization, but a ``.clone()`` survives. Eager
    ``reshape`` already copies a non-contiguous tensor, so this adds no extra work — it just moves
    the copy where ExecuTorch's lowering needs it.

    The clone must force ``contiguous_format``: a bare ``.clone()`` defaults to
    ``preserve_format``, keeping a transposed dim-order (e.g. ``[0, 2, 1]``) that ExecuTorch's
    clone lowering can't map to a ``torch.memory_format`` (``Failed to map a given dim_order`` —
    hit by xcodec2's ISTFT head).
    """

    def patch(input, *shape, **kwargs):
        if not input.is_contiguous():
            input = input.clone(memory_format=torch.contiguous_format)
        return original(input, *shape, **kwargs)

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
    """Constraint-based bound, clamped to a trace-hint-proportional cap.

    Constraint propagation misbehaves on compound expressions in two ways, and
    ``ConstraintBasedSymShapeEvalPass`` needs an ``int`` in both cases:

    - It returns ``int_oo`` when constraints don't compose (e.g. ``((s43*s53)//s70)``)
      or for sums of unbacked symbols (e.g. MoE per-expert cats ``u320+u321+...``).
    - It returns absurdly large *finite* bounds for floordiv ratios: interval
      arithmetic evaluates ``x // (x // 2)`` (window-count ratios in the Swin family,
      true value 2) as ``upper(x) // lower(x // 2)``, e.g. ``513 // 1``. These
      ratios appear squared in window-partition reshapes and compound across
      stages, so worst-case tensor sizes reach ~2^63 bytes and ExecuTorch's memory
      planner overflows (``mem_offset does not fit in 64 bits``).

    Clamp every symbolic bound to ``max(hint * _MAX_DIM_MULTIPLIER, _MAX_DIM_FLOOR)``
    — the same trace-proportional heuristic ``_fix_range_constraints`` applies to the
    per-symbol ranges — so planned buffers stay proportional to the sampled inputs.
    """

    def patch(maybe_symint):
        if isinstance(maybe_symint, int):
            return maybe_symint
        result = original(maybe_symint)
        hint = eval_expr(maybe_symint)
        cap = max(hint * _MAX_DIM_MULTIPLIER, _MAX_DIM_FLOOR) if isinstance(hint, int) else _MAX_DIM_FLOOR
        return min(result, cap) if isinstance(result, int) else cap

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


@register_patch("executorch", "executorch.exir.program._program.lift_constant_tensor_pass")
def _patch_lift_constant_tensor_pass(original):
    """Realign ``input_specs`` with the graph placeholder order after constant lifting.

    The upstream pass picks the graph insertion point for newly lifted constant
    placeholders by matching node names against ``graph_signature.user_inputs`` —
    but for user inputs exported as ``ConstantArgument`` (e.g. ``input_ids=None``
    in a prefill component that runs from ``inputs_embeds``), ``user_inputs``
    holds the argument's *value* (``None``), not its name, so the match fails and
    the new placeholders land *after* that input while their signature specs land
    *before* it. Later positional signature rebuilds then shift every buffer arg
    name by one slot, and the emitter serializes the wrong tensor for each lifted
    constant (``Tensor spec has buffer of size 4, but expected nbytes of 8``).
    Reordering ``input_specs`` to match the placeholders restores the invariant
    the rest of the pipeline assumes.
    """

    def patch(exported_program):
        exported_program = original(exported_program)
        signature = exported_program.graph_signature
        placeholder_names = [node.name for node in exported_program.graph.nodes if node.op == "placeholder"]
        specs_by_name = {getattr(spec.arg, "name", None): spec for spec in signature.input_specs}
        if (
            None not in specs_by_name
            and len(specs_by_name) == len(signature.input_specs)
            and sorted(specs_by_name) == sorted(placeholder_names)
        ):
            signature.input_specs = [specs_by_name[name] for name in placeholder_names]
        return exported_program

    return patch


def _view_replaceable_nodes(graph_module):
    """Yield ``(node, shape)`` for non-output ``view_copy`` nodes whose view shape has the same
    shape dynamism as their base — the nodes ``ReplaceViewCopyWithViewPass`` may safely replace.

    ``view`` nodes share storage with their base during memory planning, so ``_ViewSpec``
    requires both to have the same ``shape_dynamism``. Models that reshape a static parameter
    with input-derived dynamic dims (the ``pos_embed.reshape(1, height, width, -1)`` position-
    embedding interpolation in Pvt / DepthPro / VitDet) produce a dynamic-shaped view of a
    static const base, and ``_ViewSpec.__init__`` raises ``_ViewSpec is incompatible with its
    base``. Those nodes must stay ``view_copy`` (an out-variant copy op, always correct) —
    only the storage-sharing optimisation is skipped for them.
    """

    for node in graph_module.graph.nodes:
        if _is_view_copy(node) and all(user.op != "output" for user in node.users):
            # The view shape is node.meta["val"].shape, not node.args[1], which can contain
            # an inferred -1 (same as the original pass).
            shape = node.meta["val"].shape
            base = node.args[0]
            if determine_tensor_dynanism(shape) == base.meta["spec"].shape_dynamism:
                yield node, shape


@register_patch(
    "executorch", "executorch.exir.passes.replace_view_copy_with_view_pass.ReplaceViewCopyWithViewPass.call"
)
def _patch_replace_view_copy_with_view_call(_original):
    """Replacement for ``ReplaceViewCopyWithViewPass.call`` that only replaces ``view_copy``
    nodes whose shape dynamism matches their base's — see ``_view_replaceable_nodes``."""

    def patch(self, graph_module):
        n_replaced = 0
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node, shape in _view_replaceable_nodes(module):
                node.target = _VIEW_OP
                node.meta["spec"] = _ViewSpec(node.args[0].meta["spec"], shape)
                n_replaced += 1
            module.recompile()
        return PassResult(graph_module, n_replaced > 0)

    return patch


@register_patch(
    "executorch", "executorch.exir.passes.replace_view_copy_with_view_pass.ReplaceViewCopyWithViewPass.ensures"
)
def _patch_replace_view_copy_with_view_ensures(_original):
    """Companion to ``_patch_replace_view_copy_with_view_call``: the original ``ensures`` asserts
    that no non-output ``view_copy`` node remains, but the patched ``call`` deliberately keeps
    the ones whose shape dynamism differs from their base's."""

    def patch(self, graph_module):
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            remaining = [node for node, _ in _view_replaceable_nodes(module)]
            assert not remaining, f"view_copy nodes were not replaced with views: {remaining}"

    return patch


@register_patch("executorch", "torch.export.exported_program._convert_guards_to_code")
def _patch_convert_guards_to_code(_original):
    """Skip stringifying ShapeEnv guards on every ``ExportedProgram`` construction.

    ``ExportedProgram.__init__`` unconditionally pretty-prints every ShapeEnv guard
    into ``_guards_code``. ExecuTorch lowering constructs hundreds of intermediate
    ``ExportedProgram``s (every ``_transform`` / decomposition / partition), each
    re-printing the full guard set. For dynamic-shape guards with deeply nested
    ``FloorDiv``/``Add`` expressions (Swin/Hiera window partitioning, BigBird
    block-sparse indexing) the printer walks the *unshared* expression tree —
    minutes of sympy printing per export, and on Mask2Former/Sam2 a recursion deep
    enough to overflow the C stack (segfault). The strings are only consumed when
    ``ExportedProgram.module()`` builds a guards fn, which torch itself force-disables
    for ExecuTorch callers (``torch.export._unlift._ok_to_generate_guards_fn``), so
    they are pure waste during lowering.
    """

    def patch(graph_module):
        return []

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


@register_patch(
    "executorch", "executorch.backends.transforms.remove_clone_ops.RemoveCloneOpsTransform._is_non_identity_clone"
)
def _patch_is_non_identity_clone(original):
    """Keep identity clones that feed the graph output.

    XNNPACK's delegate preprocess runs ``RemoveCloneOpsTransform``, which folds identity
    clones (same dim order — including ``_clone_dim_order`` of a ``permute_copy``, identity
    only *after* the view-to-copy pass) onto their input. When both the clone and its input
    are outputs of the delegated submodule (a value and its ``.contiguous()`` copy both
    crossing the partition boundary — Clvp's mel-attention residual, PerceptionLM's
    eval-mode dropout of a returned hidden state), the fold leaves
    the same node twice in the output list and ``generate_node_to_external_map`` rejects the
    submodule with ``Output node ... is already in the inputs``. Report output-feeding clones
    as non-identity so they are kept — the partitioner only admits dim-order-preserving
    clones, which XNNPACK serializes as ``XNNCopy``.
    """

    def patch(self, node):
        if any(user.op == "output" for user in node.users):
            return True
        return original(self, node)

    return patch


@register_patch(
    "executorch", "executorch.backends.xnnpack.partition.config.node_configs.PreluConfig.check_constraints"
)
def _patch_prelu_check_constraints(original):
    """Only delegate ``prelu`` to XNNPACK when its input is 4-D.

    ``PreluConfig.check_constraints`` only verifies the weight is a parameter, but
    XNNPACK's ``ChannelsLastTaggedReshapePass`` lists ``prelu`` among the ops that
    require NHWC input and asserts the input can be converted (i.e. is 4-D) —
    ``Attempting to convert non-NHWC compatible node to NHWC`` otherwise. Models
    that apply ``nn.PReLU`` to 3-D transformer activations (dab_detr) crash there.
    Rejecting the node keeps it on the portable CPU ops instead.
    """

    def patch(self, node, ep):
        input_node = node.all_input_nodes[0]
        val = input_node.meta.get("val")
        if not (isinstance(val, torch.Tensor) and val.dim() == 4):
            return False
        return original(self, node, ep)

    return patch


@register_patch(
    "executorch",
    "executorch.exir.backend.canonical_partitioners.group_partitioner.GroupBasedPartitioner.propose_partitions",
)
def _patch_group_partitioner_break_quotient_cycles(original):
    """Split delegated partitions that would form a dependency cycle once fused.

    ``to_backend`` fuses each partition into a single ``call_module`` node, one at a time. A
    fused node conservatively depends on *all* of the partition's inputs and feeds *all* of its
    outputs — even when the partition internally holds independent sub-computations. The XNNPACK
    config partitioner does create such partitions: its disjoint-set grouping
    (``get_matched_nodes_from_configs``) unions nodes that merely share a constant, so a single
    tag can cover two independent chains. When one such partition ``A`` collapses to one node, it
    introduces an edge from every ``A``-input to every ``A``-output, which can make a *previously
    convex* partition ``B`` non-convex if ``B`` has nodes on both sides of ``A`` — i.e. the
    quotient graph (each partition contracted to a node) has a cycle ``B -> A -> B``.
    ``create_submodule_from_nodes`` then raises ``Invalid partition, found dependency cycles``
    (seen on FLAVA, where the text/image encoder streams and the multimodal encoder interleave).

    ``GroupBasedPartitioner`` only checks *pairwise merges* (``_can_merge_partitions``) against
    the original graph, so it misses this fusion-induced cycle. Enforce the real fuseability
    condition here: the quotient graph over the proposed partitions must be a DAG. While it isn't,
    pick a multi-node partition on a detected cycle (only a multi-node partition can create the
    false all-inputs-to-all-outputs edge) and split it around the other cycle members: nodes
    upstream of that barrier form one partition, the rest form another, ordering the halves as
    ``upstream -> barrier -> downstream``. Both halves stay delegated to XNNPACK, so nodes the
    partitioner marked "do not decompose" (e.g. ``linear``) aren't left orphaned and un-lowered.
    Prefer the straddling partition (whose split lands nodes on both sides); each split strictly
    increases the partition count while shrinking the straddler, so this converges. Cycle-free
    partitionings (every currently-passing model) are returned untouched.
    """
    from torch.fx.passes.infra.partitioner import Partition

    def find_cycle(adjacency):
        # Iterative DFS returning the node ids on one cycle, or None if the graph is a DAG.
        color = dict.fromkeys(adjacency, 0)  # 0 = unvisited, 1 = on stack, 2 = done
        for root in adjacency:
            if color[root] != 0:
                continue
            path = [root]
            stack = [(root, iter(adjacency[root]))]
            color[root] = 1
            while stack:
                _, neighbors = stack[-1]
                advanced = False
                for nxt in neighbors:
                    if color[nxt] == 1:  # back-edge into the current DFS path
                        return path[path.index(nxt) :]
                    if color[nxt] == 0:
                        color[nxt] = 1
                        path.append(nxt)
                        stack.append((nxt, iter(adjacency[nxt])))
                        advanced = True
                        break
                if not advanced:
                    color[stack.pop()[0]] = 2
                    path.pop()
        return None

    def ancestors_of(barrier):
        # All nodes with a path *into* the barrier set (barrier included), via reverse BFS.
        seen, worklist = set(barrier), list(barrier)
        while worklist:
            for inp in worklist.pop().all_input_nodes:
                if inp not in seen:
                    seen.add(inp)
                    worklist.append(inp)
        return seen

    def split_around(cycle, victim_id, by_id):
        # Split victim's nodes into (before, after) around the other cycle members: `before` =
        # victim nodes upstream of the barrier, `after` = the rest. Ordered before -> barrier ->
        # after, this breaks the victim's participation in the cycle. Returns None if one side
        # is empty (this victim doesn't straddle the barrier).
        barrier = set()
        for member in cycle:
            if member == victim_id:
                continue
            barrier.update(by_id[member].nodes if isinstance(member, int) else {member})
        ancestors = ancestors_of(barrier)
        victim_nodes = list(by_id[victim_id].nodes)
        before = [n for n in victim_nodes if n in ancestors]
        after = [n for n in victim_nodes if n not in ancestors]
        return (before, after) if before and after else None

    def patch(self):
        partitions = original(self)
        by_id = {p.id: p for p in partitions}
        while by_id:
            # Build the quotient graph: every partition contracts to its (integer) id, every
            # un-delegated node stays as itself. A cycle here is exactly a fusion cycle. The
            # `tag66 -> tag95` return edge can run through an un-delegated node (e.g. FLAVA's
            # multimodal layer_norm), so those nodes must be vertices too — a partition-only
            # quotient misses the cycle.
            node_to_pid = {node: pid for pid, p in by_id.items() for node in p.nodes}
            adjacency: dict = {}
            for node in self.graph_module.graph.nodes:
                src = node_to_pid.get(node, node)
                adjacency.setdefault(src, set())
                for user in node.users:
                    dst = node_to_pid.get(user, user)
                    adjacency.setdefault(dst, set())
                    if dst != src:
                        adjacency[src].add(dst)
            cycle = find_cycle(adjacency)
            if cycle is None:
                break

            # Only a multi-node partition can introduce the false fusion edge (a one-node
            # partition contracts to a single graph vertex, which the DAG can't put on a cycle),
            # so the victim must be one of these. Prefer the straddler — the partition whose split
            # lands nodes on both sides of the barrier — smallest first to minimise churn.
            candidates = sorted(
                (v for v in cycle if isinstance(v, int) and v in by_id and len(by_id[v].nodes) > 1),
                key=lambda pid: len(by_id[pid].nodes),
            )
            if not candidates:
                break
            chosen_id, halves = None, None
            for victim_id in candidates:
                halves = split_around(cycle, victim_id, by_id)
                if halves is not None:
                    chosen_id = victim_id
                    break

            next_id = max(by_id) + 1
            if chosen_id is not None:
                del by_id[chosen_id]
                for nodes in halves:
                    by_id[next_id] = Partition(id=next_id, nodes=set(nodes))
                    next_id += 1
            else:
                # No straddler splits cleanly — dissolve the smallest candidate into single-node
                # partitions, which cannot sit on a fusion cycle. Rare; keeps the loop converging.
                victim = by_id.pop(candidates[0])
                for node in victim.nodes:
                    by_id[next_id] = Partition(id=next_id, nodes={node})
                    next_id += 1
        return list(by_id.values())

    return patch


@register_patch(
    "executorch",
    "executorch.exir.lowered_backend_module._unsafe_adjust_original_program",
    "executorch.exir.backend.backend_api._unsafe_adjust_original_program",
)
def _patch_unsafe_adjust_original_program(original):
    """Delete each consumed parameter/constant target at most once when adjusting the
    original program after delegation.

    After a partition is lowered, ``_unsafe_adjust_original_program`` strips the params/buffers
    the delegate absorbed from the top-level graph signature and state dict. Its dedup guard
    (``currently_used_targets``) only skips targets still referenced by a *remaining* input spec —
    it does not dedup *within* the batch being deleted. When one delegate consumes several
    duplicated copies of the same shared parameter (the constant-dedup pass emits ``..._copy_1``,
    ``..._copy_2`` … all keeping the original FQN as their ``target``), the loop deletes that
    target on the first copy and then raises ``KeyError`` on the next. Transformers detection
    models hit this because a single head is applied at every decoder layer and tied to the
    encoder head — e.g. PPDocLayoutV3's ``model.decoder.class_embed.weight`` / ``bbox_embed``.

    Track the targets already removed and delete each only once (and tolerate an already-absent
    key). Removing a target once is correct: its data is baked into the delegate blob, so the
    repeated deletes are no-ops. The rest of the routine (graph-node erasure, output-spec and
    getitem-index fixups) is unchanged, so all duplicate placeholders are still erased.
    """
    from torch.export.graph_signature import InputKind

    def patch(original_program, call_delegate_node, input_specs_to_delete, output_specs_to_delete):
        original_program._graph_signature.input_specs = [
            input_spec
            for input_spec in original_program.graph_signature.input_specs
            if input_spec.arg.name not in input_specs_to_delete
        ]

        currently_used_targets = {
            input_spec.target
            for input_spec in original_program._graph_signature.input_specs
            if input_spec.target is not None
        }

        original_program._graph_signature.output_specs = [
            output_spec
            for output_spec in original_program.graph_signature.output_specs
            if output_spec.arg.name not in output_specs_to_delete
        ]

        for node in original_program.graph.nodes:
            if node.op == "placeholder":
                if node.name in input_specs_to_delete:
                    assert len(node.users) == 0
                    original_program.graph.erase_node(node)
            else:
                break

        deleted_targets: set = set()
        for input_spec in input_specs_to_delete.values():
            input_target = input_spec.target
            assert input_target is not None
            # Skip targets still referenced elsewhere, and targets already removed by an earlier
            # duplicate copy in this same batch (the fix: the stock routine omits this second case).
            if input_target in currently_used_targets or input_target in deleted_targets:
                continue
            deleted_targets.add(input_target)

            if input_spec.kind == InputKind.PARAMETER:
                original_program._state_dict.pop(input_target, None)
            elif input_spec.kind == InputKind.BUFFER:
                if input_spec.persistent:
                    original_program._state_dict.pop(input_target, None)
                else:
                    original_program._constants.pop(input_target, None)
            elif input_spec.kind == InputKind.CONSTANT_TENSOR:
                original_program._constants.pop(input_target, None)
            else:
                raise RuntimeError(f"Invalid input spec {input_spec} received")

        toplevel_output_node = original_program.graph.output_node()
        assert toplevel_output_node is not None
        assert len(toplevel_output_node.args) == 1, (
            f"Invalid output node: {toplevel_output_node} with args {toplevel_output_node.args}"
        )

        new_output_args = [
            arg
            for arg in toplevel_output_node.args[0]
            if not isinstance(arg, torch.fx.Node) or arg.name not in output_specs_to_delete
        ]
        toplevel_output_node.args = (tuple(new_output_args),)

        getitem_idxs: list = []
        user_nodes = list(call_delegate_node.users.keys())
        for user in user_nodes:
            if user.name in output_specs_to_delete:
                assert user.op == "call_function" and user.target == operator.getitem
                user_idx = user.args[1]
                assert isinstance(user_idx, int), f"Invalid getitem type: {type(user_idx)}"
                getitem_idxs.append(user_idx)
                original_program.graph.erase_node(user)

        getitem_idxs.sort(reverse=True)

        user_nodes = list(call_delegate_node.users.keys())
        for user in user_nodes:
            assert user.op == "call_function" and user.target == operator.getitem
            user_idx = user.args[1]
            assert isinstance(user_idx, int)
            for i, idx in enumerate(getitem_idxs):
                if user_idx > idx:
                    user.args = (user.args[0], user_idx - (len(getitem_idxs) - i))
                    break

    return patch


@register_patch(
    "executorch",
    "executorch.backends.xnnpack.serialization.xnnpack_graph_serialize._flatc_compile",
)
def _patch_flatc_compile_nonfinite(original):
    """Rewrite non-finite float literals in the XNNPACK delegate JSON before ``flatc``.

    XNNPACK serializes its delegate graph via ``json.dumps``, which emits non-finite floats as the
    bare tokens ``-Infinity`` / ``Infinity`` / ``NaN`` — not part of the flatbuffers JSON grammar, so
    ``flatc`` fails with ``cannot parse value starting with: -``. MiniMaxM3's lightning-indexer block
    padding (``F.pad(scores, ..., value=float("-inf"))``) lowers to a ``constant_pad_nd`` whose
    ``-inf`` ``padding_value`` hits this. Swap the tokens for flatbuffers' own ``-inf`` / ``inf`` /
    ``nan`` (parsed to the identical IEEE value) so the exact ``-inf`` semantics are preserved.
    """

    def patch(output_dir, schema_path, json_path):
        with open(json_path) as f:
            data = f.read()
        # Lookbehind/lookahead on JSON delimiters so only bare numeric literals match (quoted
        # strings are bounded by `"` and never touched).
        fixed = re.sub(r"(?<=[:\[,\s])-Infinity(?=[,\]}\s])", "-inf", data)
        fixed = re.sub(r"(?<=[:\[,\s])Infinity(?=[,\]}\s])", "inf", fixed)
        fixed = re.sub(r"(?<=[:\[,\s])NaN(?=[,\]}\s])", "nan", fixed)
        if fixed != data:
            with open(json_path, "w") as f:
                f.write(fixed)
        return original(output_dir, schema_path, json_path)

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

# Heuristic caps for `int_oo` dynamic-dim upper bounds, used by `_fix_range_constraints`.
# ExecuTorch's XNNPACK memory planner pre-allocates buffers from the upper bound, so leaving
# `int_oo` blows up memory; capping too tight rejects legitimate trace-time shapes (e.g. VLM
# image-token counts). The pair below — 4x the observed lower/trace value, with a 1024 floor —
# was tuned empirically on the export test suite to keep every passing trace under realistic
# planner memory while still covering the largest sampled inputs. Bump together if a new model
# hits a "bound too tight" error during ExecuTorch lowering.
_MAX_DIM_MULTIPLIER = 4
_MAX_DIM_FLOOR = 1024


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

    unbounded = []
    for rd in range_dicts:
        for sym, vr in rd.items():
            if isinstance(vr.upper, IntInfinity):
                lower = _as_int(vr.lower, 2)
                trace_val = _as_int(var_to_val.get(sym), 0)
                upper = max(lower * _MAX_DIM_MULTIPLIER, trace_val * _MAX_DIM_MULTIPLIER, _MAX_DIM_FLOOR)
                rd[sym] = ValueRanges(vr.lower, upper)
                unbounded.append((str(sym), lower, upper))

    if unbounded:
        # dedupe across the two range_dicts since they share symbols
        seen = {name: (lower, upper) for name, lower, upper in unbounded}
        details = ", ".join(f"{name} → [{lower}, {upper}]" for name, (lower, upper) in seen.items())
        logger.warning(
            "ExecuTorch export: %d dynamic dim(s) had no upper bound (int_oo) and were capped "
            "heuristically (%s). The XNNPACK memory planner pre-allocates from these bounds, so "
            "loose caps mean wasted device memory. For best memory planning, pass explicit "
            "`dynamic_shapes` with fine-grained `torch.export.Dim(name, min=..., max=...)` "
            "covering the smallest and largest shapes you expect at runtime.",
            len(seen),
            details,
        )


@register_fx_program_fix("executorch")
def _drop_runtime_asserts(exported_program: ExportedProgram) -> None:
    """Drop ``_assert_scalar`` / ``_assert_tensor_metadata`` runtime asserts before lowering.

    ``_assert_scalar`` lowers a ``torch._check`` on an unbacked symint (e.g. the image-token
    count in ``get_placeholder_mask``) into a ``cast_symbool_to_symint`` + ``eq`` chain whose
    ``Piecewise`` result the ``_ModuleStackTracer`` used by ``to_edge_transform_and_lower``'s
    decomposition pass cannot proxy (``... is not tracked with proxy``). The range facts these
    asserts encode survive on ``exported_program.range_constraints`` (further capped by
    ``_fix_range_constraints``), so dropping the nodes (and the now-dead symint feeders) is safe.
    """
    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in list(module.graph.nodes):
            if node.op == "call_function" and node.target in (
                torch.ops.aten._assert_tensor_metadata.default,
                torch.ops.aten._assert_scalar.default,
            ):
                module.graph.erase_node(node)
        module.graph.eliminate_dead_code()
        module.recompile()


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
    base_val = base.meta.get("val") if isinstance(base, torch.fx.Node) else base
    with gm.graph.inserting_before(node):
        running = base
        running_val = base_val
        for _ in range(exp - 1):
            running = gm.graph.call_function(mul_scalar, (running, base))
            # Propagate the symbolic value so downstream passes / the emitter see a ``meta["val"]``
            # on the synthesised products (matches the original ``pow`` node's value).
            if base_val is not None and running_val is not None:
                running_val = running_val * base_val
                running.meta["val"] = running_val
    node.replace_all_uses_with(running)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("executorch")
def _fix_negative_slice_start(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite a data-dependent negative slice start into its positive ``dim_size + start`` form.

    A negative slice on an unbacked length (VideoMAE's decoder keeps only the masked tokens via
    ``hidden_states[:, -return_token_num:]``, ``return_token_num`` being the symbolic masked-patch
    count) records ``start = -(u // 2)``. ``to_edge_transform_and_lower`` re-runs ``slice_forward``'s
    meta, whose ``if start_val < 0`` guard can't be decided on a size-like symbol
    (``GuardOnDataDependentSymNode``). Replace the start with ``sym_size(input, dim) + start`` — for a
    tail slice this is the (size-like, hence provably ``>= 0``) number of leading elements, so the
    guard is statically false. Drop the stale ``unbacked_bindings``: the output length is now a
    computable expression, not the fresh unbacked symbol ``run_decompositions`` recorded.
    """

    if node.target not in (torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor) or len(node.args) < 3:
        return False
    start = node.args[2]
    if not isinstance(start, torch.fx.Node):
        return False
    start_val = start.meta.get("val")
    if not isinstance(start_val, torch.SymInt) or statically_known_true(start_val >= 0):
        return False
    input_node, dim = node.args[0], node.args[1]
    input_val = input_node.meta.get("val")
    if not isinstance(input_val, torch.Tensor):
        return False
    with gm.graph.inserting_before(node):
        size_node = gm.graph.call_function(torch.ops.aten.sym_size.int, (input_node, dim))
        size_node.meta["val"] = input_val.shape[dim]
        add_node = gm.graph.call_function(operator.add, (size_node, start))
        add_node.meta["val"] = input_val.shape[dim] + start_val
    node.args = (*node.args[:2], add_node, *node.args[3:])
    node.meta.pop("unbacked_bindings", None)
    return True
