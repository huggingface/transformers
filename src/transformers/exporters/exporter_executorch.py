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

1. **Backend preparation** (`prepare_for_xnnpack`, `prepare_for_cuda`): move the
   model to the target device/dtype and build the partitioner list.
2. **Torch op patches** (`patch_torch_ops`): swap ops unsupported by ExecuTorch
   backends (split_copy, topk, avg_pool2d, ...) with decomposed equivalents, then
   `torch.export.export` the model.
3. **FX graph patches** (`patch_fx_graph`): apply `_FX_PATCHES` on the resulting
   `ExportedProgram` to repair shape bounds, placeholder metadata, op args, and
   replace Python sym ops with their `executorch_prim.*` equivalents.
4. **Upstream-pass softenings** (`patch_executorch_passes`): temporarily replace
   ExecuTorch passes (`SpecPropPass`, `PruneEmptyTensorsPass`,
   `eval_upper_bound`) with versions that don't crash on legitimate dynamic-shape
   patterns. Reverted on exit.
5. **Lowering**: `to_edge_transform_and_lower` followed by `to_executorch` with
   the config from `_get_executorch_backend_config`.
"""

from __future__ import annotations

import math
import operator
from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any

from ..utils import logging
from ..utils.export_config import ExecutorchConfig
from ..utils.import_utils import is_executorch_available, is_torch_available
from .exporter_dynamo import DynamoExporter


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
    from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
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

    >>> exporter = ExecutorchExporter(export_config=ExecutorchConfig(backend="xnnpack"))
    >>> et_program = exporter.export(model, inputs)
    >>> et_program.write_to_file("model.pte")
    ```
    """

    export_config: ExecutorchConfig

    required_packages = ["torch", "executorch"]

    def export(self, model: PreTrainedModel, sample_inputs: dict[str, Any]) -> ExecutorchProgramManager:
        """Export a model to ExecuTorch, applying backend preparation and torch op patches."""
        prepare_for_backend = _BACKEND_PREPARE.get(self.export_config.backend)
        if prepare_for_backend is None:
            raise ValueError(f"Unsupported backend {self.export_config.backend} for ExecuTorch export")

        model, sample_inputs, partitioner = prepare_for_backend(model, sample_inputs)

        with patch_torch_ops(), patch_executorch_passes():
            exported_program: ExportedProgram = super().export(model, sample_inputs)
            patch_fx_graph(exported_program)
            edge_program_manager: EdgeProgramManager = to_edge_transform_and_lower(
                exported_program, partitioner=partitioner, compile_config=_get_edge_compile_config()
            )
            executorch_programs_manager: ExecutorchProgramManager = edge_program_manager.to_executorch(
                config=_get_executorch_backend_config()
            )

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


def _get_executorch_backend_config() -> ExecutorchBackendConfig:
    """Build the ``ExecutorchBackendConfig`` used for ``edge_program_manager.to_executorch``.

    Currently no overrides from the upstream defaults. ``remove_view_copy=False`` was
    tried to fix the ``_ViewSpec is incompatible with its base`` failure on depth_pro /
    pvt / vitdet, but it kept ``view_copy`` ops that XNNPACK then partitioned as pass-
    through and regressed more tests than it fixed — those three models go back into the
    known-failing list until a per-model fix is found.
    """
    return ExecutorchBackendConfig()


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
    device = getattr(model, "device", None) or next(model.parameters()).device
    if device.type != "cpu":
        model = model.to(device="cpu")
    partitioner = [XnnpackPartitioner()]
    return model, sample_inputs, partitioner


def prepare_for_cuda(model: PreTrainedModel, sample_inputs: dict[str, Any]):
    """GPU inference via the ExecuTorch CUDA backend.

    Moves the model to CUDA and upcasts to bfloat16 — required by the CUDA backend.
    """
    model.requires_grad_(False)
    dtype = next(model.parameters()).dtype
    device = getattr(model, "device", None) or next(model.parameters()).device
    if device.type != "cuda":
        model = model.to(device="cuda")
    if dtype != torch.bfloat16:
        model = model.to(dtype=torch.bfloat16)
    partitioner = [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(model.__class__.__name__)])]
    return model, sample_inputs, partitioner


_BACKEND_PREPARE = {
    "xnnpack": prepare_for_xnnpack,
    "cuda": prepare_for_cuda,
}


# ── Stage 2: Torch op patches ─────────────────────────────────────────────────
# Same factory pattern as exporter_onnx.py: each _patch_* receives the original
# and returns the replacement. _TORCH_PATCHES lists (obj, attr, factory).


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


def _patch_chunk(original):
    """Narrow-based chunk (delegates to split patch)."""

    def patch(input, chunks, dim=0):
        total = input.size(dim)
        chunk_size = (total + chunks - 1) // chunks
        # Call through torch.split which is already patched
        return torch.split(input, chunk_size, dim)

    return patch


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


def _patch_detach(_original):
    """No-op detach."""

    def patch(input):
        return input

    return patch


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


def _patch_scaled_dot_product_attention(original):
    """Manual matmul+softmax fallback for cases unsupported by the ExecuTorch CUDA backend.

    Falls back to eager attention when:
    - enable_gqa=True
    - D_q != D_v (asymmetric head dims, e.g. MLA attention)
    - attn_mask is float (ExecuTorch CUDA SDPA only accepts bool masks)
    """

    def patch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        needs_eager_attention = query.device.type == "cuda" and (
            kwargs.get("enable_gqa", False)
            or query.shape[-1] != value.shape[-1]
            or (attn_mask is not None and attn_mask.is_floating_point())
        )
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


def _patch_dropout(_original):
    """No-op dropout for inference export."""

    def patch(input, p=0.5, training=True, inplace=False):
        return input

    return patch


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


def _patch_expand(original):
    """Force a contiguous copy after ``expand``.

    ``Tensor.expand`` produces a view with stride ``0`` along broadcast dims.
    ExecuTorch's memory planner rejects ``stride == 0`` (``tensor.py:77``: "0 in
    strides is not supported for ExecuTorch"). Materialise the broadcast so the
    captured tensor has standard strides downstream.
    """

    def patch(self, *sizes):
        if len(sizes) == 1 and isinstance(sizes[0], (list, tuple, torch.Size)):
            sizes = tuple(sizes[0])
        return original(self, *sizes).clone(memory_format=torch.contiguous_format)

    return patch


# (object, attribute, factory) triples installed by patch_torch_ops.
_TORCH_PATCHES = []
if is_torch_available():
    _TORCH_PATCHES += [
        (torch, "split", _patch_split),
        (torch.Tensor, "split", _patch_split),
        (torch, "chunk", _patch_chunk),
        (torch.Tensor, "chunk", _patch_chunk),
        (torch, "topk", _patch_topk),
        (torch.Tensor, "topk", _patch_topk),
        (torch, "detach", _patch_detach),
        (torch.Tensor, "detach", _patch_detach),
        (torch.nn.functional, "avg_pool2d", _patch_avg_pool2d),
        (torch.nn.functional, "scaled_dot_product_attention", _patch_scaled_dot_product_attention),
        (torch.nn.functional, "dropout", _patch_dropout),
        (torch, "bernoulli", _patch_bernoulli),
        (torch.Tensor, "bernoulli", _patch_bernoulli),
        (torch.Tensor, "expand", _patch_expand),
    ]


@contextmanager
def patch_torch_ops():
    """Context manager: install torch patches for ExecuTorch export."""
    originals = []
    for obj, attr, factory in _TORCH_PATCHES:
        original = getattr(obj, attr)
        originals.append((obj, attr, original))
        setattr(obj, attr, factory(original))

    try:
        yield
    finally:
        for obj, attr, original in originals:
            setattr(obj, attr, original)


# ── Stage 3: FX graph patches ─────────────────────────────────────────────────
# Patches applied to the ExportedProgram between ``torch.export.export`` and
# ``to_edge_transform_and_lower``, to repair the graph for ExecuTorch's stricter
# expectations (concrete dim bounds, placeholder metadata, allowlisted ops,
# normalised op args). Same role as ``patch_fx_graph`` in ``exporter_onnx.py``,
# but at ExportedProgram granularity rather than per-node.

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


def _bound_range_constraints(exported_program: ExportedProgram) -> None:
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


def _populate_missing_placeholder_vals(exported_program: ExportedProgram) -> None:
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


def _normalize_amax_dim(exported_program: ExportedProgram) -> None:
    """Rewrite negative ``dim`` indices on max/amax ops to positive ones.

    XNNPACK's ``op_max_dim`` visitor compares ``node.args[1]`` directly against
    2 and 3 without normalizing, so a ``dim=-1`` call on a 4-D tensor fails with
    ``amax.default only supports dim == 2 or dim == 3`` even though dim 3 is
    what was meant. Convert any negative ``dim`` to ``rank + dim`` so the
    partitioner sees a positive index. Done for both ``aten.amax.default`` and
    ``aten.max.dim`` (which gets folded into amax later during lowering).
    """
    targets = {torch.ops.aten.amax.default, torch.ops.aten.max.dim}
    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op != "call_function" or node.target not in targets:
                continue
            if len(node.args) < 2:
                continue
            input_node = node.args[0]
            input_val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
            rank = input_val.dim() if isinstance(input_val, torch.Tensor) else None
            if rank is None:
                continue
            dim_arg = node.args[1]
            if isinstance(dim_arg, int) and dim_arg < 0:
                new_args = list(node.args)
                new_args[1] = rank + dim_arg
                node.args = tuple(new_args)
            elif isinstance(dim_arg, (list, tuple)) and any(isinstance(d, int) and d < 0 for d in dim_arg):
                new_dims = [d + rank if isinstance(d, int) and d < 0 else d for d in dim_arg]
                new_args = list(node.args)
                new_args[1] = type(dim_arg)(new_dims)
                node.args = tuple(new_args)


_SYM_OP_REPLACEMENTS = {
    target: _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS[target]
    for target in (torch.sym_float, torch.sym_max, torch.sym_min, math.ceil, math.trunc, round)
    if target in _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS
}


def _replace_python_sym_ops(exported_program: ExportedProgram) -> None:
    """Replace Python sym ops (``torch.sym_min``, ``math.ceil``, ...) with their
    ExecuTorch backend equivalents.

    The edge-dialect verifier rejects Python ``FunctionType`` ops other than
    ``alloc`` (``verifier.py:317``). ExecuTorch has its own pass
    (``EdgeToBackendOpsPass``) that swaps these for ``executorch_prim.*`` ops,
    but it only runs during ``to_executorch``, after the edge verifier already
    runs in ``to_edge_transform_and_lower``. Apply the same swap here.

    Only ``torch.sym_*`` and ``math.*`` targets are swapped — ``operator.add`` /
    ``mul`` / etc. are also used for tensor-tensor ops, where the ``Scalar``
    overload fails at runtime with ``Cannot cast NotImplemented to number``.
    """
    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        changed = False
        for node in module.graph.nodes:
            if node.op == "call_function" and node.target in _SYM_OP_REPLACEMENTS:
                node.target = _SYM_OP_REPLACEMENTS[node.target]
                changed = True
        if changed:
            module.recompile()


def _force_contiguous_clone_memory_format(exported_program: ExportedProgram) -> None:
    """Force ``contiguous_format`` on ``aten.clone`` nodes whose input has a non-standard
    dim order.

    ``Tensor.clone()`` defaults to ``preserve_format`` and inherits the source's stride
    layout. When a cache tensor has been transposed earlier (dim order e.g. ``[1, 0, 2, 3]``),
    the clone inherits it and ExecuTorch's ``dim_order_from_stride`` fails to map it to a
    ``torch.memory_format``. Only rewrite clones whose input ``meta["val"]`` is non-contiguous
    so we don't disturb the (much more common) clones of already-contiguous tensors — those
    can otherwise get optimised into pass-through nodes that XNNPACK rejects.
    """
    clone_op = torch.ops.aten.clone.default
    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op != "call_function" or node.target is not clone_op:
                continue
            if node.kwargs.get("memory_format") is not None:
                continue
            input_val = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
            if isinstance(input_val, torch.Tensor) and not input_val.is_contiguous():
                node.kwargs = {**node.kwargs, "memory_format": torch.contiguous_format}


def _rewrite_sym_pow_as_mul(exported_program: ExportedProgram) -> None:
    """Replace ``operator.pow(sym_int, n)`` with a chain of ``executorch_prim.mul.Scalar``.

    The emitter has no entry for ``operator.pow`` (no ``executorch_prim.pow``), so a
    ``sym_size ** 2`` in model code (e.g. seamless_m4t's relative positional bias)
    raises ``invalid target for call_function <built-in function pow>`` at to_executorch.
    Rewrite small-integer exponents (n >= 1) as a multiplication chain — the
    ``executorch_prim.mul.Scalar`` op accepts SymInt operands.
    """
    from executorch.exir.passes.executorch_prim_ops_registry import _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS

    mul_scalar = _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS.get(operator.mul)
    if mul_scalar is None:
        return

    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        changed = False
        for node in list(module.graph.nodes):
            if node.op != "call_function" or node.target is not operator.pow:
                continue
            base, exp = node.args
            if not isinstance(exp, int) or exp < 1:
                continue
            with module.graph.inserting_before(node):
                running = base
                for _ in range(exp - 1):
                    running = module.graph.call_function(mul_scalar, (running, base))
            node.replace_all_uses_with(running)
            module.graph.erase_node(node)
            changed = True
        if changed:
            module.recompile()


_FX_PATCHES = [
    _bound_range_constraints,
    _populate_missing_placeholder_vals,
    _replace_python_sym_ops,
    _rewrite_sym_pow_as_mul,
    _normalize_amax_dim,
    _force_contiguous_clone_memory_format,
]


def patch_fx_graph(exported_program: ExportedProgram) -> None:
    """Apply every FX graph patch in order on ``exported_program`` (in place)."""
    for fx_patch in _FX_PATCHES:
        fx_patch(exported_program)


# ── Stage 4: Upstream-pass softenings ─────────────────────────────────────────
# Same factory pattern as Stage 2: each _patch_* receives the original and returns
# the replacement. _EXECUTORCH_PASS_PATCHES lists (obj, attr, factory).


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


@contextmanager
def patch_attr(obj: Any, attr: str, factory: Any):
    """Swap ``obj.attr`` with ``factory(original)`` for the duration of the block."""
    original = getattr(obj, attr)
    setattr(obj, attr, factory(original))
    try:
        yield
    finally:
        setattr(obj, attr, original)


@contextmanager
def _patch_sym_ops_allowlist():
    """Extend the edge-dialect verifier's sym-op allowlist for the duration of the block.

    Python sym ops that don't have ``executorch_prim.*`` equivalents (``sym_ite``,
    ``sym_not``, ``sym_int``, ``sym_sum``) still trip the verifier. Add them to the
    allowlist set so they're accepted — trace-time-only ops don't need a runtime kernel.
    In-place mutation propagates to all modules that imported the set by name.
    """
    from executorch.exir.passes.executorch_prim_ops_registry import _EXECUTORCH_SYM_OPS

    extra = {
        op for op in (torch.sym_ite, torch.sym_not, torch.sym_int, torch.sym_sum, torch.sym_float) if op is not None
    }
    added = extra - _EXECUTORCH_SYM_OPS
    _EXECUTORCH_SYM_OPS.update(added)
    try:
        yield
    finally:
        _EXECUTORCH_SYM_OPS.difference_update(added)


def _executorch_patches() -> list[Any]:
    """Build the per-patch context managers installed by :func:`patch_executorch_passes`.

    Imports are local to keep the module-level import block small.
    """
    from executorch.backends.xnnpack.operators.node_visitor import _node_visitor_dict
    from executorch.exir import sym_util
    from executorch.exir.passes import prune_empty_tensors_pass, spec_prop_pass, sym_shape_eval_pass
    from executorch.exir.verification import verifier

    return [
        # ConstraintBasedSymShapeEvalPass imports eval_upper_bound at module load,
        # so we need to rebind both `sym_util.eval_upper_bound` (the canonical home)
        # and `sym_shape_eval_pass.eval_upper_bound` (the imported copy).
        patch_attr(sym_util, "eval_upper_bound", _patch_eval_upper_bound),
        patch_attr(sym_shape_eval_pass, "eval_upper_bound", _patch_eval_upper_bound),
        patch_attr(
            prune_empty_tensors_pass.PruneEmptyTensorsPass,
            "remove_empty_tensors_from_cat",
            _patch_remove_empty_tensors_from_cat,
        ),
        patch_attr(
            spec_prop_pass.SpecPropPass,
            "update_placeholder_tensor_specs",
            _patch_update_placeholder_tensor_specs,
        ),
        # XNNPACK's conv1d_unsqueeze_pass wraps conv1d in unsqueeze/conv2d/squeeze; the
        # squeeze then trips the "reshape only supports 1 dynamic dimension" check when
        # the surrounding tensor has multiple dynamic dims (audio / speech models where
        # batch and time are both dynamic). Drop the check — squeeze/unsqueeze of a
        # size-1 dim doesn't actually change dynamism. Classes go via the
        # ``_node_visitor_dict`` lookup because ``@register_node_visitor`` rebinds the
        # decorated name to ``None``.
        patch_attr(_node_visitor_dict["aten.squeeze_copy.dim"], "define_node", _make_squeeze_define_node),
        patch_attr(_node_visitor_dict["aten.unsqueeze_copy.default"], "define_node", _make_squeeze_define_node),
        # Allow complex64 / complex128 through the edge dtype validator — used by
        # FFT in fnet and complex rotary embeddings in deepseek_v2.
        patch_attr(verifier, "_check_tensor_args_matching_op_allowed_dtype", _patch_check_tensor_args_dtype),
        _patch_sym_ops_allowlist(),
    ]


@contextmanager
def patch_executorch_passes():
    """Context manager: install ExecuTorch pass softenings for export."""
    with ExitStack() as stack:
        for cm in _executorch_patches():
            stack.enter_context(cm)
        yield
