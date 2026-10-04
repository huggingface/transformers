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
"""ExecuTorch exporter: `DynamoExporter` followed by edge lowering to a `.pte` program.

1. **Backend preparation** (`_BACKEND_PREPARE`): target device / dtype and partitioners per backend (XNNPACK,
   CUDA, MLX, ...).
2. **Patches** (`apply_patches("executorch")`, plus `executorch.<backend>`): reversible swaps of `torch` ops the
   backends reject (`split_copy`, `avg_pool2d`, varlen attention, ...) and of ExecuTorch internals
   (`SpecPropPass`, `eval_upper_bound`, ...) that crash on valid dynamic shapes.
3. **FX program fixes** (`apply_fx_program_fixes`): program-level repairs, e.g. widening `int_oo` bounds in
   `range_constraints`.
4. **FX node fixes** (`apply_fx_node_fixes`): per-node rewrites, e.g. sym ops to `executorch_prim.*`, `pow` as a
   `mul` chain, floored modulo.
5. **Lowering** (`to_edge_transform_and_lower`), with the export metadata carried in the program as a constant
   method.
"""

from __future__ import annotations

import contextlib
import functools
import math
import operator
import re
from collections.abc import MutableMapping
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_executorch_available, is_torch_available
from .configs import ExecutorchConfig, ExportFormat
from .decompose import _MODALITY_SPECS
from .exporter_dynamo import DynamoExporter, varlen_attn_masked_sdpa
from .utils import (
    apply_fx_node_fixes,
    apply_fx_program_fixes,
    apply_patches,
    drop_runtime_asserts,
    module_dtype,
    register_fx_node_fix,
    register_fx_program_fix,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.fx.passes.infra.pass_base import PassResult
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils._sympy.numbers import IntInfinity
    from torch.utils._sympy.value_ranges import ValueRanges

    from ..cache_utils import EncoderDecoderCache, StaticCache
    from ..modeling_utils import PreTrainedModel


if is_executorch_available():
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (  # type: ignore[import-not-found]
        XNNStaticReshape,
        XNode,
    )
    from executorch.backends.xnnpack.utils.utils import get_input_node
    from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.passes.executorch_prim_ops_registry import _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from executorch.exir.passes.replace_view_copy_with_view_pass import _VIEW_OP, _is_view_copy, _ViewSpec
    from executorch.exir.passes.spec_prop_pass import _is_mutable_buffer
    from executorch.exir.program import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower
    from executorch.exir.sym_util import eval_expr
    from executorch.exir.tensor import determine_tensor_dynanism, get_scalar_type, num_bytes_from_shape_and_dtype

    # The CUDA backend imports `triton`, which CPU-only torch builds don't ship.
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

    export_format = ExportFormat.EXECUTORCH
    config_class = ExecutorchConfig
    artifact_suffix = ".pte"

    required_packages = ["torch", "executorch"]
    tested_versions = {"torch": "2.13.0", "executorch": "1.4.1"}

    def export_artifact(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: ExecutorchConfig | dict[str, Any],
    ) -> ExecutorchProgramManager:
        """Export a model to ExecuTorch, applying backend preparation and torch op patches."""
        config = self._as_config(config)

        prepare_for_backend = _BACKEND_PREPARE.get(config.backend)
        if prepare_for_backend is None:
            raise ValueError(f"Unsupported backend {config.backend} for ExecuTorch export")

        model.requires_grad_(False)
        model, partitioner = prepare_for_backend(model, sample_inputs, exclude=tuple(config.partition_exclude))
        partitioner = partitioner if config.partition else []
        # `spec_prop` rejects stride-0 broadcast inputs ("0 in strides").
        sample_inputs = torch.utils._pytree.tree_map_only(torch.Tensor, lambda t: t.contiguous(), sample_inputs)

        with (
            contiguous_nonzero_fake(),
            apply_patches("executorch"),
            apply_patches(f"executorch.{config.backend}"),
        ):
            exported_program, metadata = super().export_artifact(model, sample_inputs, config=config)
            apply_fx_program_fixes("executorch", exported_program)

            with keep_backed_symbols_symbolic(exported_program):
                apply_fx_node_fixes("executorch", exported_program.graph_module)
                edge_program_manager: EdgeProgramManager = to_edge_transform_and_lower(
                    exported_program,
                    partitioner=partitioner,
                    compile_config=_get_edge_compile_config(exported_program, config.backend),
                    transform_passes=_get_transform_passes(config.backend),
                )
                executorch_programs_manager = edge_program_manager.to_executorch(config=_get_backend_config(config))

        return executorch_programs_manager, metadata

    @classmethod
    def save_artifact(cls, artifact, path) -> None:
        """Stream the `.pte` to `path` (`.buffer` would materialize the whole program first)."""
        with open(path, "wb") as file:
            artifact.write_to_file(file)


@register_patch("executorch", "torch.nn.attention.varlen.varlen_attn")
def _patch_varlen_attn(original):
    """Lower `varlen_attn` to masked SDPA: the edge verifier trips on the flash op's aux outputs."""

    def varlen_attn(*args, **kwargs):
        return varlen_attn_masked_sdpa(*args, **kwargs)

    return varlen_attn


@register_patch("executorch", "executorch.exir.program._program.serialize_for_executorch")
def _patch_serialize_for_executorch(original):
    """Repair the emitted program right before serialization, the last point it is still editable."""

    def serialize_for_executorch(emitter_output, *args, **kwargs):
        canonicalize_size_one_dim_orders(getattr(emitter_output, "program", None))
        widen_underplanned_arenas(getattr(emitter_output, "program", None))
        return original(emitter_output, *args, **kwargs)

    return serialize_for_executorch


def canonicalize_size_one_dim_orders(executorch_program) -> None:
    """Give canonical dim order to tensors whose only non-canonical axes are size-1.

    Size-1 axes make dim order ambiguous, but portable kernels require identical dim orders across operands
    (`tensors_have_same_dim_order`, `0x12`). Genuinely permuted layouts are left alone.
    """
    for plan in getattr(executorch_program, "execution_plan", []):
        for value in getattr(plan, "values", []):
            tensor = getattr(value, "val", None)
            sizes = getattr(tensor, "sizes", None)
            dim_order = getattr(tensor, "dim_order", None)
            if not sizes or not dim_order or 1 not in list(sizes):
                continue
            with_extent = [axis for axis in dim_order if sizes[axis] != 1]
            if with_extent == sorted(with_extent):
                tensor.dim_order = type(dim_order)(range(len(sizes)))


def _backed_var_to_val(shape_env) -> dict:
    """`shape_env.backed_var_to_val`, falling back to `var_to_val` on older torch."""
    values = getattr(shape_env, "backed_var_to_val", None)
    return shape_env.var_to_val if values is None else values


def _contiguous_nonzero(input):
    count = torch.library.get_ctx().new_dynamic_size(max=input.numel() if isinstance(input.numel(), int) else None)
    return input.new_empty((count, input.dim()), dtype=torch.long)


@contextlib.contextmanager
def contiguous_nonzero_fake():
    """Report `nonzero`'s result as contiguous, the layout ExecuTorch's kernel actually writes.

    torch's fake rule returns a transposed layout, so the program re-reads row-major bytes with the wrong strides
    (scrambled values, or `0x12` in consumers). Fixed at the fake rule since both lowering stages re-run it, and
    registered for the export only.
    """
    library = torch.library.Library("aten", "FRAGMENT")
    torch.library.register_fake("aten::nonzero", _contiguous_nonzero, lib=library)
    try:
        yield
    finally:
        library._destroy()


@contextlib.contextmanager
def keep_backed_symbols_symbolic(exported_program: ExportedProgram):
    """Block replacing a *backed* symbol with a constant during lowering.

    Lowering passes re-run ops on fake tensors and refine dynamic axes to their hints (`s70 = 2`), which the
    memory plan bakes in ("Attempted to resize a static tensor", 0x12). Unbacked resolution and
    symbol-to-symbol unification still go through (blocking them breaks data-dependent sizes).
    """
    shape_env = next(
        (
            val.fake_mode.shape_env
            for node in exported_program.graph_module.graph.nodes
            if isinstance(val := node.meta.get("val"), torch.Tensor) and hasattr(val, "fake_mode")
        ),
        None,
    )
    if shape_env is None:
        yield
        return
    original = shape_env._set_replacement

    backed_values = _backed_var_to_val(shape_env)

    def selective(symbol, replacement, *args, **kwargs):
        if symbol in backed_values and getattr(replacement, "is_number", False):
            return None
        return original(symbol, replacement, *args, **kwargs)

    shape_env._set_replacement = selective
    try:
        yield
    finally:
        del shape_env._set_replacement


def _has_layout_copies(exported_program: ExportedProgram) -> bool:
    """Whether the graph holds a dim-order copy, which only the `dim_order_ops` variants can express."""
    target = getattr(getattr(torch.ops, "dim_order_ops", None), "_to_dim_order_copy", None)
    return target is not None and any(
        node.target is target.default for node in exported_program.graph.nodes if node.op == "call_function"
    )


def _uses_channels_last(exported_program: ExportedProgram) -> bool:
    """Whether any tensor is channels-last, the one layout that cannot be lowered without `dim_order_ops`.

    Transposed views are deliberately not counted: keeping dim-order ops for them re-freezes symbolic buffers.
    """
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    for node in exported_program.graph_module.graph.nodes:
        val = node.meta.get("val")
        tensors = val if isinstance(val, (tuple, list)) else [val]
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.dim() not in (4, 5):
                continue
            layout = torch.channels_last if tensor.dim() == 4 else torch.channels_last_3d
            try:
                if tensor.is_contiguous(memory_format=layout) and not tensor.is_contiguous():
                    return True
            except GuardOnDataDependentSymNode:
                # Data-dependent sizes can't answer `is_contiguous`; treat as not channels-last.
                continue
    return False


def _get_transform_passes(backend: str):
    """Return backend-specific graph transforms, or ``None`` for defaults."""
    if backend == "mlx":
        from executorch.backends.mlx.passes import get_default_passes

        return get_default_passes()
    return None


def _get_edge_compile_config(exported_program: ExportedProgram, backend: str) -> EdgeCompileConfig:
    """Build the ``EdgeCompileConfig`` for ``to_edge_transform_and_lower``.

    Non-core ATen ops that transformers models produce (FFT in fnet, bucketize, polar, ...) are exempted
    from the edge-dialect verifier; the portable kernels run them.
    """
    if backend == "mlx":
        return EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
    return EdgeCompileConfig(
        # `_empty_dim_order` takes `int[]` sizes (not `SymInt[]`), freezing symbolic buffers at their hints
        # ("Attempted to resize a static tensor"); keep dim-order ops only where channels-last needs them.
        _skip_dim_order=not (_uses_channels_last(exported_program) or _has_layout_copies(exported_program)),
        _core_aten_ops_exception_list=[
            torch.ops.aten._embedding_bag_forward_only.default,
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


def widen_underplanned_arenas(executorch_program) -> None:
    """Grow a planned memory arena whose declared size is smaller than its own offsets reach.

    The greedy planner can place a buffer past the arena size it declares (SmolVLM vision), and the loader
    refuses the method (`MemoryAllocationFailed`, `0x21`). Only the size is raised; offsets stay put.
    """
    if executorch_program is None:
        return
    for plan in executorch_program.execution_plan:
        needed = dict.fromkeys(range(len(plan.non_const_buffer_sizes)), 0)
        for item in plan.values:
            value = item.val
            info = getattr(value, "allocation_info", None)
            sizes = getattr(value, "sizes", None)
            if info is None or not sizes or info.memory_id >= len(plan.non_const_buffer_sizes):
                continue
            end = info.memory_offset + num_bytes_from_shape_and_dtype(sizes, get_scalar_type(value.scalar_type))
            needed[info.memory_id] = max(needed[info.memory_id], end)
        for memory_id, end in needed.items():
            if end > plan.non_const_buffer_sizes[memory_id]:
                logger.warning_once(
                    f"The memory plan plots {end - plan.non_const_buffer_sizes[memory_id]} bytes past the "
                    f"arena it asks for, which the runtime refuses; asking for {end} instead."
                )
                plan.non_const_buffer_sizes[memory_id] = end


def _get_backend_config(config):
    """Build the ``ExecutorchBackendConfig`` for ``to_executorch``, or ``None`` when no ``alloc_*`` flag changed."""
    if config.alloc_graph_input and config.alloc_graph_output and config.alloc_mutable_buffers:
        return None
    return ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=config.alloc_graph_input,
            alloc_graph_output=config.alloc_graph_output,
            alloc_mutable_buffers=config.alloc_mutable_buffers,
        )
    )


# ── Stage 1: Backend preparation ──────────────────────────────────────────────
# Each `prepare_for_<backend>` returns `(model, partitioners)`.


def prepare_for_xnnpack(model: PreTrainedModel, sample_inputs: dict[str, Any], exclude: tuple[str, ...] = ()):
    """CPU inference via XNNPACK. The model is moved to CPU, which the partitioner and edge passes require."""

    model = model.to(device="cpu")
    # XNNPACK has no `_grouped_mm.out` kernel.
    if isinstance(model, PreTrainedModel):
        model.set_experts_implementation("batched_mm")
    # XNNPACK can claim a partition its compiler then refuses (`ViewCopyConfig` for qwen3_next).
    if exclude:
        from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS

        unknown = set(exclude) - {config.__name__ for config in ALL_PARTITIONER_CONFIGS}
        if unknown:
            raise ValueError(f"Unknown XNNPACK partitioner config(s): {sorted(unknown)}")
        configs = [config for config in ALL_PARTITIONER_CONFIGS if config.__name__ not in exclude]
        if not configs:
            # `XnnpackPartitioner` treats an empty list as every config.
            raise ValueError("partition_exclude removes every partitioner config; pass partition=False instead")
        partitioner = [XnnpackPartitioner(configs=configs)]
    else:
        partitioner = [XnnpackPartitioner()]
    return model, partitioner


def prepare_for_openvino(model: PreTrainedModel, sample_inputs: dict[str, Any], exclude: tuple[str, ...] = ()):
    """CPU inference through the OpenVINO delegate. `partition_exclude` does not apply."""
    from executorch.backends.openvino.partitioner import OpenvinoPartitioner
    from executorch.exir.backend.backend_details import CompileSpec

    model = model.to(device="cpu")
    partitioner = [OpenvinoPartitioner([CompileSpec("device", b"CPU")])]
    return model, partitioner


def prepare_for_cuda(model: PreTrainedModel, sample_inputs: dict[str, Any], exclude: tuple[str, ...] = ()):
    """GPU inference via the ExecuTorch CUDA backend.

    Requires bfloat16 (upcast here) and a visible GPU; the model can stay on any device.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment; cannot export to the ExecuTorch CUDA backend.")

    dtype = module_dtype(model)
    if dtype is not None and dtype != torch.bfloat16:
        logger.warning(f"ExecuTorch CUDA backend requires bfloat16; upcasting model from {dtype}.")
        model = model.to(dtype=torch.bfloat16)
    partitioner = [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(model.__class__.__name__)])]
    return model, partitioner


def prepare_for_mlx(model: PreTrainedModel, sample_inputs: dict[str, Any], exclude: tuple[str, ...] = ()):
    """Apple Silicon GPU inference via the ExecuTorch MLX backend."""
    for value in sample_inputs.values():
        caches = [value]
        if isinstance(value, EncoderDecoderCache):
            caches = [value.self_attention_cache, value.cross_attention_cache]
        if any(isinstance(cache, StaticCache) for cache in caches):
            raise ValueError(
                "StaticCache is not supported by the ExecuTorch MLX backend. "
                "Use DynamicCache or set cache_implementation='dynamic' in GenerationConfig."
            )

    from executorch.backends.mlx import MLXPartitioner

    model = model.to(device="cpu")
    # MLX does not support grouped MoE kernels.
    if isinstance(model, PreTrainedModel):
        model.set_experts_implementation("batched_mm")
    partitioner = [MLXPartitioner()]
    return model, partitioner


_BACKEND_PREPARE = {
    "openvino": prepare_for_openvino,
    "xnnpack": prepare_for_xnnpack,
    "cuda": prepare_for_cuda,
    "mlx": prepare_for_mlx,
}


# ── Stage 2: Torch patches ────────────────────────────────────────────────────
# Reversible swaps of `torch` ops the ExecuTorch backends can't lower.


@register_patch("executorch", "torch._higher_order_ops.associative_scan.associative_scan")
def _patch_associative_scan(original):
    """Run `associative_scan` as a sequential `scan`, which ExecuTorch lowers at a symbolic length.

    Without it, the mamba family falls back to a Python loop that unrolls over the traced length.
    """
    from torch._higher_order_ops.scan import scan
    from torch.utils._pytree import tree_flatten, tree_unflatten

    def patch(combine_fn, xs, dim, reverse=False, combine_mode="pointwise"):
        leaves, spec = tree_flatten(xs)
        leaves = [leaf.movedim(dim, 0) for leaf in leaves]
        if reverse:
            leaves = [leaf.flip(0) for leaf in leaves]
        # `scan` checks the carry it returns against the one it was given, strides included: both contiguous.
        first = [leaf[0].contiguous() for leaf in leaves]

        def step(carry, element):
            combined = tree_flatten(combine_fn(tree_unflatten(carry, spec), tree_unflatten(element, spec)))[0]
            combined = [value.contiguous() for value in combined]
            return combined, [value.clone() for value in combined]

        _, rest = scan(step, first, [leaf[1:] for leaf in leaves])
        outputs = [torch.cat([head.unsqueeze(0), tail], dim=0) for head, tail in zip(first, rest)]
        if reverse:
            outputs = [output.flip(0) for output in outputs]
        return tree_unflatten([output.movedim(0, dim) for output in outputs], spec)

    return patch


def _has_unbacked_sizes(split_size_or_sections) -> bool:
    """Whether a `split` sections argument carries a data-dependent size."""
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    return isinstance(split_size_or_sections, (list, tuple)) and any(
        isinstance(size, torch.SymInt) and bool(free_unbacked_symbols(size.node.expr))
        for size in split_size_or_sections
    )


@register_patch(
    "executorch",
    "executorch.exir.tensor.dim_order_from_stride",
    "executorch.exir.tensor_layout.dim_order_from_stride",
    "executorch.exir.emit._emitter.dim_order_from_stride",
    "executorch.exir.passes.replace_view_copy_with_view_pass.dim_order_from_stride",
)
def _patch_dim_order_from_stride(original):
    """Order dims when a stride carries a data-dependent size.

    ExecuTorch's stride sort falls back to a plain `<` that raises `GuardOnDataDependentSymNode` on
    `64*u21 < 64`; sort size-obliviously instead. Registered at every call site that imported the name.
    """
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode, guard_or_false

    def compare(left, right) -> int:
        if guard_or_false(left == right):
            return 0
        if guard_or_false(left < right):
            return -1
        if guard_or_false(right < left):
            return 1
        return _compare_assuming_nonempty(left, right)

    def dim_order_from_stride(stride):
        try:
            return original(stride)
        except GuardOnDataDependentSymNode:
            order = sorted(range(len(stride)), key=functools.cmp_to_key(lambda a, b: compare(stride[a], stride[b])))
            return tuple(reversed(order))

    return dim_order_from_stride


# An unbacked size with no finite bound: large, so a size derived from it by floor division stays nonzero.
_UNBOUNDED_STAND_IN = 2**20


def _compare_assuming_nonempty(left, right) -> int:
    """Compare `left` and `right` (-1, 0, 1), backed symbols at their hints and unbacked at their upper bound.

    The upper bound rather than 2, since a derived size like `u42 // 104` would floor to zero. Substituting
    rather than guarding leaves the shape environment untouched.
    """

    def concrete(side):
        node = getattr(side, "node", None)
        if node is None:
            return int(side)
        expression = node.expr
        shape_env = getattr(node, "shape_env", None)
        hints = _backed_var_to_val(shape_env) if shape_env is not None else {}
        ranges = getattr(shape_env, "var_to_range", {})

        def stand_in(symbol):
            if symbol in hints:
                return hints[symbol]
            bound = ranges.get(symbol)
            return max(2, _as_int(bound.upper, _UNBOUNDED_STAND_IN)) if bound is not None else _UNBOUNDED_STAND_IN

        return int(expression.xreplace({symbol: stand_in(symbol) for symbol in expression.free_symbols}))

    try:
        left_value, right_value = concrete(left), concrete(right)
    except (TypeError, ValueError):
        # Keep the incoming order.
        return 0
    # Equal strides must stay equal, or operands of one op end up in different dim orders (`0x12`).
    return (left_value > right_value) - (left_value < right_value)


@register_patch("executorch", "torch.split", "torch.Tensor.split")
def _patch_unbacked_split(original):
    """Split with data-dependent sizes as a chain of `narrow`s.

    `split_with_sizes_copy` with unbacked sizes fails to lower ("Could not extract specialized integer"), e.g.
    grid VLMs splitting vision output per image. Gated on *unbacked* sizes: `aten.split.Tensor`'s
    decomposition re-enters `torch.split` with backed-symbolic sizes, which stay with the original.
    """

    def patch(input, split_size_or_sections, dim=0):
        if _has_unbacked_sizes(split_size_or_sections):
            pieces, start = [], 0
            for size in split_size_or_sections:
                pieces.append(input.narrow(dim, start, size))
                start = start + size
            return tuple(pieces)
        return original(input, split_size_or_sections, dim)

    return patch


@register_patch("executorch.cuda", "torch.split", "torch.Tensor.split")
def _patch_split(original):
    """Narrow-based split for the CUDA backend, which can't lower `split_copy`."""

    def patch(input, split_size_or_sections, dim=0):
        if isinstance(split_size_or_sections, int):
            splits = []
            total = input.size(dim)
            for i in range(0, total, split_size_or_sections):
                splits.append(input.narrow(dim, i, min(split_size_or_sections, total - i)))
            return tuple(splits)
        elif isinstance(split_size_or_sections, torch.SymInt):
            # `range` needs a concrete step.
            return original(input, split_size_or_sections, dim)
        elif _has_unbacked_sizes(split_size_or_sections):
            # Handled by `_patch_unbacked_split`, which this patch is layered over.
            return original(input, split_size_or_sections, dim)
        else:
            splits = []
            start = 0
            for size in split_size_or_sections:
                splits.append(input.narrow(dim, start, size))
                start += size
            return tuple(splits)

    return patch


@register_patch("executorch.cuda", "torch.chunk", "torch.Tensor.chunk")
def _patch_chunk(original):
    """Route `chunk` through the patched `torch.split`: CUDA can't lower `split_copy.Tensor` (no c-shim)."""

    def patch(input, chunks, dim=0):
        total = input.size(dim)
        chunk_size = (total + chunks - 1) // chunks
        return torch.split(input, chunk_size, dim)

    return patch


@register_patch("executorch.cuda", "torch.topk", "torch.Tensor.topk")
def _patch_topk(original):
    """Argsort-based topk for the CUDA backend, which has no topk kernel.

    Not for XNNPACK: the portable runtime has no `sort` (`Missing operator: aten::sort.values`).
    """

    def patch(input, k, dim=None, largest=True, sorted=True):
        if dim is None:
            dim = -1
        indices = torch.argsort(input, dim=dim, descending=largest)
        topk_indices = indices.narrow(dim, 0, k)
        topk_values = torch.gather(input, dim, topk_indices)
        return torch.return_types.topk((topk_values, topk_indices))

    return patch


@register_patch("executorch.xnnpack", "torch.argsort", "torch.Tensor.argsort")
def _patch_argsort(original):
    """Topk-based argsort for XNNPACK, whose portable runtime has no `sort` (`Missing operator: aten::sort.values`)."""

    def patch(input, dim=-1, descending=False, stable=False):
        return torch.topk(input, input.shape[dim], dim=dim, largest=descending, sorted=True).indices

    return patch


@register_patch("executorch", "transformers.models.deepseek_v2.modeling_deepseek_v2.DeepseekV2RotaryEmbedding.forward")
def _patch_deepseek_v2_rotary_forward(original):
    """Carry deepseek_v2's rotary frequencies as a real `[cos, sin]` pair rather than a complex tensor.

    The portable registry has no `polar`/`view_as_complex_copy` (load fails, `0x14`). Paired with
    `_patch_deepseek_v2_apply_rotary_emb`.
    """

    def patch(self, x, position_ids):
        from ..utils.generic import maybe_autocast

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            freqs_cis = torch.stack((torch.cos(freqs), torch.sin(freqs)), dim=-1)
            freqs_cis = freqs_cis * self.attention_scaling
        return freqs_cis

    return patch


@register_patch("executorch", "transformers.models.deepseek_v2.modeling_deepseek_v2.apply_rotary_emb")
def _patch_deepseek_v2_apply_rotary_emb(original):
    """deepseek_v2's rotary as real arithmetic on the `[cos, sin]` pair from the patch above."""

    def patch(xq, xk, freqs_cis):
        cos = freqs_cis[..., 0].unsqueeze(1).to(xq.device)
        sin = freqs_cis[..., 1].unsqueeze(1).to(xq.device)

        def rotate(x):
            paired = x.float().reshape(*x.shape[:-1], -1, 2)
            real, imaginary = paired[..., 0], paired[..., 1]
            rotated = torch.stack((real * cos - imaginary * sin, real * sin + imaginary * cos), dim=-1)
            return rotated.flatten(3).type_as(x)

        return rotate(xq), rotate(xk)

    return patch


@register_patch("executorch", "torch.unsqueeze", "torch.Tensor.unsqueeze")
def _patch_unsqueeze(original):
    """Unsqueeze via reshape: XNNPACK claims `unsqueeze_copy` then rejects it on dynamic graphs (`0x1`)."""

    def patch(input, dim):
        shape = list(input.shape)
        shape.insert(dim if dim >= 0 else dim + len(shape) + 1, 1)
        return input.reshape(shape)

    return patch


@register_patch("executorch.xnnpack", "torch.detach", "torch.Tensor.detach")
@register_patch("executorch.cuda", "torch.detach", "torch.Tensor.detach")
def _patch_detach(_original):
    """No-op detach."""

    def patch(input):
        return input

    return patch


@register_patch("executorch.cuda", "torch.nn.functional.avg_pool2d")
def _patch_avg_pool2d(original):
    """Decompose avg_pool2d as depthwise conv2d for the CUDA backend, which has no avg_pool2d kernel."""

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


@register_patch(
    "executorch", "executorch.exir.passes.prune_empty_tensors_pass.PruneEmptyTensorsPass.remove_empty_tensors_from_cat"
)
def _patch_remove_empty_tensors_from_cat(_original):
    """``PruneEmptyTensorsPass.remove_empty_tensors_from_cat`` that keeps unbacked-size inputs.

    The original's ``numel() != 0`` raises ``GuardOnDataDependentSymNode`` on sizes like ``74 * u176``.
    """
    from torch.fx.experimental.symbolic_shapes import guard_or_true

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


@register_patch("executorch", "torch.nn.functional.pad")
def _patch_pad(original):
    """Split a negative pad into a crop plus a non-negative pad ("Padding values must be non-negative", 0x12)."""

    def patch(input, pad, mode="constant", value=None):
        # SymInt amounts have unknown sign and take the general form.
        if all(isinstance(amount, int) and amount >= 0 for amount in pad):
            return original(input, pad, mode, value)
        output = original(input, [torch.sym_max(amount, 0) for amount in pad], mode, value)
        for i in range(len(pad) // 2):
            left, right, dim = pad[2 * i], pad[2 * i + 1], input.dim() - 1 - i
            start = torch.sym_max(-left, 0)
            stop = output.size(dim) - torch.sym_max(-right, 0)
            output = output.narrow(dim, start, stop - start)
        return output

    return patch


@register_patch("executorch", "torch.nn.functional.conv3d")
def _patch_conv3d(original):
    """Decompose conv3d into a sum of conv2d's over the kernel depth.

    The portable convolution takes 3-D/4-D inputs only ("Expect input tensor to be 3-D or 4-D", 0x12).
    """

    def patch(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        def triple(value):
            return (value, value, value) if isinstance(value, int) else tuple(value)

        (stride_t, stride_h, stride_w) = triple(stride)
        (pad_t, pad_h, pad_w) = triple(padding)
        (dil_t, dil_h, dil_w) = triple(dilation)
        batch, _, frames, height, width = input.shape
        out_channels, _, kernel_t, _, _ = weight.shape
        if pad_t:
            input = torch.nn.functional.pad(input, (0, 0, 0, 0, pad_t, pad_t))
            frames = frames + 2 * pad_t
        out_frames = (frames - dil_t * (kernel_t - 1) - 1) // stride_t + 1
        output = None
        for kt in range(kernel_t):
            start = kt * dil_t
            frame_slice = input[:, :, start : start + (out_frames - 1) * stride_t + 1 : stride_t]
            folded = frame_slice.transpose(1, 2).reshape(batch * out_frames, input.shape[1], height, width)
            planes = torch.nn.functional.conv2d(
                folded, weight[:, :, kt], None, (stride_h, stride_w), (pad_h, pad_w), (dil_h, dil_w), groups
            )
            contribution = planes.reshape(batch, out_frames, out_channels, *planes.shape[2:]).transpose(1, 2)
            output = contribution if output is None else output + contribution
        if bias is not None:
            output = output + bias.reshape(1, -1, 1, 1, 1)
        return output

    return patch


@register_patch("executorch", "torch.nn.functional.adaptive_avg_pool2d")
def _patch_adaptive_avg_pool2d(original):
    """Decompose adaptive_avg_pool2d (no portable adaptive-pool kernel) for static spatial dims."""

    def patch(input, output_size):
        oh, ow = (output_size, output_size) if isinstance(output_size, int) else output_size
        h, w = input.shape[-2], input.shape[-1]
        oh, ow = (h if oh is None else oh), (w if ow is None else ow)
        if not (isinstance(h, int) and isinstance(w, int)):
            return original(input, output_size)
        if h % oh == 0 and w % ow == 0:
            return torch.nn.functional.avg_pool2d(input, kernel_size=(h // oh, w // ow))

        def bounds(size, out):
            return [((i * size) // out, -(-(i + 1) * size // out)) for i in range(out)]

        rows = [
            torch.cat([input[..., hs:he, ws:we].mean(dim=(-2, -1), keepdim=True) for ws, we in bounds(w, ow)], dim=-1)
            for hs, he in bounds(h, oh)
        ]
        return torch.cat(rows, dim=-2)

    return patch


@register_patch("executorch", "torch.bernoulli", "torch.Tensor.bernoulli")
def _patch_bernoulli(_original):
    """Rewrite ``bernoulli`` as ``rand_like < p`` (``Missing out variants: {'aten::bernoulli'}``).

    ``generator`` is ignored.
    """

    def patch(input, *args, p=None, generator=None, out=None):
        if p is None and len(args) == 1:
            p = args[0]
        probs = input if p is None else p
        result = (torch.rand_like(input) < probs).to(input.dtype)
        return out.copy_(result) if out is not None else result

    return patch


@register_patch("executorch", "torch.nn.functional.scaled_dot_product_attention")
def _patch_scaled_dot_product_attention(original):
    """Route SDPA through the MATH backend, with a manual matmul+softmax fallback.

    MATH keeps CUDA traces off `_scaled_dot_product_efficient_attention`, which the edge verifier rejects.
    The fallback runs on CUDA for GQA, D_q != D_v, or float masks (CUDA SDPA takes bool masks only), and on
    any device when the mask has an unbacked batch (the SDPA math kernel guards `Eq(u0, 1)`).
    The MATH output is cloned contiguous: decomposition re-traces SDPA with a different output layout,
    invalidating recorded views ("Cannot view a tensor with shape/strides").
    """
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

    def _has_unbacked_batch(t):
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


@register_patch("executorch", "torch.Tensor.expand")
def _patch_expand(original):
    """Materialize broadcasting ``expand``s: ExecuTorch rejects stride 0 ("0 in strides is not supported")."""

    def patch(self, *sizes, **kwargs):
        result = original(self, *sizes, **kwargs)
        if 0 in result.stride():
            return result.clone(memory_format=torch.contiguous_format)
        return result

    return patch


# ── Stage 3: ExecuTorch patches ───────────────────────────────────────────────
# Reversible swaps of ExecuTorch internals that crash on legitimate dynamic-shape patterns.


@register_patch(
    "executorch",
    "executorch.exir.sym_util.eval_upper_bound",
    "executorch.exir.passes.sym_shape_eval_pass.eval_upper_bound",
)
def _patch_eval_upper_bound(original):
    """Constraint-based bound, clamped to ``max(hint * _MAX_DIM_MULTIPLIER, _MAX_DIM_FLOOR)``.

    The original returns ``int_oo`` for compound or unbacked-sum expressions, and huge finite bounds for
    floordiv ratios (Swin windows), overflowing the planner (``mem_offset does not fit in 64 bits``).
    """

    def patch(maybe_symint):
        if isinstance(maybe_symint, int):
            return maybe_symint
        result = original(maybe_symint)
        hint = eval_expr(maybe_symint)
        # An unbacked size's finite bound comes from torch's constraints; capping it at the floor under-plans it.
        if not isinstance(hint, int) and isinstance(result, int) and result <= _MAX_UNBOUNDED_PRODUCT:
            return result
        cap = max(hint * _MAX_DIM_MULTIPLIER, _MAX_DIM_FLOOR) if isinstance(hint, int) else _MAX_DIM_FLOOR
        return min(result, cap) if isinstance(result, int) else cap

    return patch


@register_patch("executorch", "executorch.exir.verification.verifier._check_tensor_args_matching_op_allowed_dtype")
def _patch_check_tensor_args_dtype(original):
    """Suppress complex-dtype violations, which the op dtype tables lack but the kernels handle (fnet FFT)."""

    def patch(gm):
        try:
            original(gm)
        except Exception as exc:
            msg = str(exc)
            if "mismatched dtypes" in msg and ("complex64" in msg or "complex128" in msg):
                return
            raise

    return patch


@register_patch("executorch", "executorch.exir.passes.spec_prop_pass.SpecPropPass.update_placeholder_tensor_specs")
def _patch_update_placeholder_tensor_specs(_original):
    """``SpecPropPass.update_placeholder_tensor_specs`` that skips placeholders without a tensor spec.

    ``inputs_to_buffers`` can be shifted by one slot, keying a user input as a buffer with no spec
    (``AttributeError`` on ``spec.const``).
    """

    def patch(self, exported_program, graph_module):
        sig = exported_program.graph_signature
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            if "spec" not in node.meta:
                raise RuntimeError(f"Placeholder node {node} missing meta['spec']")
            spec = node.meta["spec"]
            # Scalar or unsupported placeholders have no ``const``.
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
    """Realign ``input_specs`` with the placeholder order after constant lifting.

    With a ``ConstantArgument`` user input (``input_ids=None``) the upstream pass misplaces lifted
    placeholders, shifting buffer names by one (``Tensor spec has buffer of size 4, but expected nbytes of 8``).
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
    """Yield ``(node, shape)`` for non-output ``view_copy`` nodes with the same shape dynamism as their base.

    A dynamic view of a static base (Pvt's ``pos_embed.reshape``) raises ``_ViewSpec is incompatible with its
    base``; those stay ``view_copy``.
    """

    for node in graph_module.graph.nodes:
        if _is_view_copy(node) and all(user.op != "output" for user in node.users):
            # `node.args[1]` can hold an inferred -1.
            shape = node.meta["val"].shape
            base = node.args[0]
            if determine_tensor_dynanism(shape) == base.meta["spec"].shape_dynamism:
                yield node, shape


@register_patch(
    "executorch", "executorch.exir.passes.replace_view_copy_with_view_pass.ReplaceViewCopyWithViewPass.call"
)
def _patch_replace_view_copy_with_view_call(_original):
    """``ReplaceViewCopyWithViewPass.call`` restricted to ``_view_replaceable_nodes``."""

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
    """``ensures`` matching ``_patch_replace_view_copy_with_view_call``, which keeps some ``view_copy``s."""

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

    Lowering builds hundreds of programs; printing nested guards takes minutes and can overflow the C stack
    (Mask2Former). Torch never uses them for ExecuTorch callers (``_ok_to_generate_guards_fn``).
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
    """Let the edge verifier accept the Python sym ops the trace emits (`torch.sym_min`, `math.ceil`, ...).

    `to_executorch` maps them to `executorch_prim.*` itself; swapping them earlier pins dynamic axes to their hint.
    `operator.*` is left out: it also covers tensor ops. `sym_sum` has no mapping and stays as is.
    """
    python_sym_ops = {op for op in _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS if op not in vars(operator).values()}
    return original | python_sym_ops | {torch.sym_sum}


def _make_squeeze_define_node(original):
    """``define_node`` for XNNPACK squeeze/unsqueeze without the ">1 dynamic dim" reshape check."""
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


@register_patch(
    "executorch", "executorch.backends.transforms.remove_clone_ops.RemoveCloneOpsTransform._is_non_identity_clone"
)
def _patch_is_non_identity_clone(original):
    """Keep identity clones that feed the graph output.

    Folding a clone whose input is also an output duplicates it in the output list
    (``Output node ... is already in the inputs``).
    """

    def patch(self, node):
        if any(user.op == "output" for user in node.users):
            return True
        return original(self, node)

    return patch


@register_patch(
    "executorch.xnnpack", "executorch.backends.xnnpack.partition.config.node_configs.PreluConfig.check_constraints"
)
def _patch_prelu_check_constraints(original):
    """Only delegate ``prelu`` to XNNPACK for 4-D input (``Attempting to convert non-NHWC compatible node``)."""

    def patch(self, node, ep):
        input_node = node.all_input_nodes[0]
        val = input_node.meta.get("val")
        if not (isinstance(val, torch.Tensor) and val.dim() == 4):
            return False
        return original(self, node, ep)

    return patch


# Delimiter lookarounds match bare literals only, never quoted strings.
_JSON_NONFINITE_SUBS = (
    (re.compile(r"(?<=[:\[,\s])-Infinity(?=[,\]}\s])"), "-inf"),
    (re.compile(r"(?<=[:\[,\s])Infinity(?=[,\]}\s])"), "inf"),
    (re.compile(r"(?<=[:\[,\s])NaN(?=[,\]}\s])"), "nan"),
)


@register_patch(
    "executorch",
    "executorch.backends.xnnpack.serialization.xnnpack_graph_serialize._flatc_compile",
)
def _patch_flatc_compile_nonfinite(original):
    """Rewrite JSON ``-Infinity``/``Infinity``/``NaN`` to flatbuffers' ``-inf``/``inf``/``nan`` before ``flatc``.

    ``flatc`` otherwise fails with ``cannot parse value starting with: -`` (``-inf`` padding values).
    """

    def patch(output_dir, schema_path, json_path):
        with open(json_path, encoding="utf-8") as f:
            data = f.read()
        fixed = data
        for pattern, repl in _JSON_NONFINITE_SUBS:
            fixed = pattern.sub(repl, fixed)
        if fixed != data:
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(fixed)
        return original(output_dir, schema_path, json_path)

    return patch


@register_patch("executorch.xnnpack", "executorch.backends.xnnpack.operators.node_visitor._node_visitor_dict")
def _patch_squeeze_node_visitors(original):
    """Swap the squeeze/unsqueeze visitors for subclasses that skip the strict reshape check.

    ``conv1d_unsqueeze_pass`` squeezes trip "reshape only supports 1 dynamic dimension" on audio models.
    The dict is swapped because ``@register_node_visitor`` leaves no dotted path to the classes.
    """
    new = dict(original)
    for key in ("aten.squeeze_copy.dim", "aten.unsqueeze_copy.default"):
        cls = original[key]
        new[key] = type(cls.__name__, (cls,), {"define_node": _make_squeeze_define_node(cls.define_node)})
    return new


# ── Stage 4: FX program fixes ─────────────────────────────────────────────────
# In-place `(exported_program) -> None` fixes needing program-level context.

# The memory planner allocates from upper bounds, so `int_oo` dims get a finite cap:
# `max(lower, trace) * multiplier`, floored so small-traced dims stay usable.
_MAX_DIM_MULTIPLIER = 4
# Floor range for a single unbounded dim; `_dim_floor` shrinks it as more dims are unbounded.
_MAX_DIM_FLOOR = 1024
_MIN_DIM_FLOOR = 64
# Budget for the product of all unbounded dims (~16M elements); the arena grows with that product.
_MAX_UNBOUNDED_PRODUCT = 2**24


def _dim_floor(num_unbounded: int) -> int:
    """Per-dim cap floor so `num_unbounded` dims multiply to about `_MAX_UNBOUNDED_PRODUCT`."""
    per_dim = round(_MAX_UNBOUNDED_PRODUCT ** (1.0 / max(num_unbounded, 1)))
    return max(_MIN_DIM_FLOOR, min(_MAX_DIM_FLOOR, per_dim))


def _as_int(x, default: int = 0) -> int:
    """Best-effort ``int(x)`` for sympy values, ``default`` for infinities."""
    try:
        return int(x)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return default


@register_fx_program_fix("executorch")
def _fix_constant_dim_orders(exported_program: ExportedProgram) -> None:
    """Make constant tensors contiguous: channels-last weights trip "2 input tensors have different dim orders"."""
    for holder in (exported_program.state_dict, exported_program.constants):
        for name, tensor in holder.items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
                holder[name] = tensor.contiguous()


def _query_axis_symbols(exported_program: ExportedProgram, var_to_val: dict) -> tuple[set, int]:
    """Token-axis symbols of the text inputs, and the larger of their hint and the cache-length hint."""

    def hint(dim) -> int:
        return dim if isinstance(dim, int) else _as_int(var_to_val.get(dim.node.expr), 0)

    placeholders = {
        node.name: node.meta.get("val")
        for node in exported_program.graph_module.graph.nodes
        if node.op == "placeholder"
    }
    symbols, sequence_hint = set(), 0
    for name, axis in (("input_ids", 1), ("inputs_embeds", 1), ("decoder_input_ids", 1), ("position_ids", -1)):
        value = placeholders.get(name)
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and not isinstance(value.shape[axis], int):
            symbols.add(value.shape[axis].node.expr)
            sequence_hint = max(sequence_hint, hint(value.shape[axis]))
    for name, value in placeholders.items():
        if (
            name.startswith(("past_key_values", "cache_params"))
            and isinstance(value, torch.Tensor)
            and value.dim() == 4
        ):
            sequence_hint = max(sequence_hint, hint(value.shape[2]))
    return symbols, sequence_hint


def _modality_axis_symbols(exported_program: ExportedProgram) -> set:
    """Symbols on modality inputs' axes (patches, frames), bounded by their trace values with no floor.

    The text-axis floor multiplied through the planner's buffers (SmolVLM arena 13.5 GB instead of 0.9 GB).
    """
    modality_inputs = tuple(
        {key for spec in _MODALITY_SPECS for key in spec.input_keys}
        | {spec.grid_key for spec in _MODALITY_SPECS if spec.grid_key}
    )
    symbols = set()
    for node in exported_program.graph_module.graph.nodes:
        if node.op != "placeholder" or not node.name.startswith(modality_inputs):
            continue
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            continue
        for dim in value.shape:
            if not isinstance(dim, int):
                symbols.add(dim.node.expr)
    return symbols


@register_fx_program_fix("executorch")
def _fix_range_constraints(exported_program: ExportedProgram) -> None:
    """Cap ``int_oo`` upper bounds at ``max(lower, trace) * _MAX_DIM_MULTIPLIER`` or `_dim_floor`."""
    # `range_constraints` feeds torch.export verifiers, `var_to_range` ExecuTorch's sym_shape_eval_pass.
    range_dicts = [exported_program._range_constraints]
    var_to_val = {}
    for node in exported_program.graph_module.graph.nodes:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor) and hasattr(val, "fake_mode"):
            shape_env = val.fake_mode.shape_env
            range_dicts.append(shape_env.var_to_range)
            var_to_val = _backed_var_to_val(shape_env)
            break

    floor = _dim_floor(len({sym for rd in range_dicts for sym, vr in rd.items() if isinstance(vr.upper, IntInfinity)}))

    # A merged decode is captured at two tokens but serves the whole prompt, so the query axis is bounded
    # from the cache-length hint too ("Attempted to resize a bounded tensor").
    query_symbols, sequence_hint = _query_axis_symbols(exported_program, var_to_val)
    # A symbol that is both keeps the text treatment, the safe direction.
    modality_symbols = _modality_axis_symbols(exported_program) - query_symbols

    unbounded = []
    for rd in range_dicts:
        for sym, vr in rd.items():
            if isinstance(vr.upper, IntInfinity):
                lower = _as_int(vr.lower, 2)
                trace_val = _as_int(var_to_val.get(sym), 0)
                if sym in query_symbols:
                    trace_val = max(trace_val, sequence_hint)
                if sym in modality_symbols:
                    upper = max(trace_val, lower, _MIN_DIM_FLOOR)
                else:
                    upper = max(lower * _MAX_DIM_MULTIPLIER, trace_val * _MAX_DIM_MULTIPLIER, floor)
                rd[sym] = ValueRanges(vr.lower, upper)
                unbounded.append((str(sym), lower, upper))

    if unbounded:
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
    drop_runtime_asserts(exported_program.graph_module)


@register_fx_program_fix("executorch")
def _fix_missing_placeholder_vals(exported_program: ExportedProgram) -> None:
    """Fill missing ``meta["val"]`` on parameter/buffer/constant placeholders, which ``SpecPropPass`` needs."""
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
# `(gm, node) -> bool` per-node fixers; return ``True`` to consume the node.


# `a % b` and the ops that rebuild it, for symbolic ints and for tensors.
_FLOORED_MOD_OPS = {operator.mod: (operator.add, operator.mod)}
if is_torch_available():
    _FLOORED_MOD_OPS[torch.ops.aten.remainder.Scalar] = (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.remainder.Scalar,
    )


@register_fx_node_fix("executorch")
def _fix_max_values_keepdim(gm, node):
    """Rewrite a values-only ``aten.max.dim`` as squeezed ``amax(keepdim=True)``.

    XNNPACK claims ``amax`` with ``keepdim=False`` then refuses it (``amax.default only supports keep_dim == True``).
    """
    if node.target is not torch.ops.aten.max.dim or len(node.args) < 2:
        return False
    keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
    users = list(node.users)
    if keepdim or any(user.target is not operator.getitem for user in users):
        return False
    # An unread indices output is not a use.
    if any(user.args[1] != 0 and user.users for user in users):
        return False
    values_users = [user for user in users if user.args[1] == 0]
    if not values_users:
        return False
    source, dim = node.args[0], node.args[1]
    with gm.graph.inserting_before(node):
        kept = gm.graph.call_function(torch.ops.aten.amax.default, args=(source, [dim], True))
        values = gm.graph.call_function(torch.ops.aten.squeeze.dims, args=(kept, [dim]))
    val = node.meta.get("val")
    if val is not None:
        values.meta["val"] = val[0]
        kept.meta["val"] = val[0].unsqueeze(dim)
    for user in users:
        if user.args[1] == 0:
            user.replace_all_uses_with(values)
        gm.graph.erase_node(user)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("executorch")
def _fix_convolution_input_layout(gm, node):
    """Copy a transposed convolution input to the default layout.

    The portable kernel accepts only default/channels-last dim orders
    (``tensor_is_default_or_channels_last_dim_order``, 0x12).
    """
    convolutions = (
        torch.ops.aten.convolution.default,
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv3d.default,
    )
    if node.target not in convolutions or getattr(node, "_layout_fixed", False):
        return False
    from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

    source = node.args[0]
    val = getattr(source, "meta", {}).get("val")
    if not isinstance(val, torch.Tensor) or val.dim() not in (3, 4, 5):
        return False
    try:
        if val.is_contiguous() or val.is_contiguous(
            memory_format=torch.channels_last if val.dim() == 4 else torch.contiguous_format
        ):
            return False
    except GuardOnDataDependentSymNode:
        return False
    # A dim-order copy, not a contiguous `clone`: the portable `clone` keeps its input's dim order.
    import executorch.exir.passes.dim_order_ops_registry  # noqa: F401  (registers `dim_order_ops`)

    with gm.graph.inserting_before(node):
        contiguous = gm.graph.call_function(
            torch.ops.dim_order_ops._to_dim_order_copy.default,
            args=(source,),
            kwargs={"dim_order": list(range(val.dim()))},
        )
        contiguous.meta["val"] = val.contiguous()
    node.replace_input_with(source, contiguous)
    node._layout_fixed = True
    return True


@register_fx_node_fix("executorch")
def _fix_empty_like(gm, node):
    """Rewrite ``aten.empty_like`` as ``aten.empty``.

    Non-contiguous ``empty_like`` decomposes to ``empty_permuted``, which has no kernel.
    """
    if node.target is not torch.ops.aten.empty_like.default:
        return False
    source = node.args[0]
    val = getattr(source, "meta", {}).get("val")
    if val is None:
        return False
    with gm.graph.inserting_before(node):
        sizes = [
            size if isinstance(size, int) else gm.graph.call_function(torch.ops.aten.sym_size.int, args=(source, axis))
            for axis, size in enumerate(val.shape)
        ]
        empty = gm.graph.call_function(
            torch.ops.aten.empty.memory_format,
            args=(sizes,),
            kwargs={"dtype": node.kwargs.get("dtype") or val.dtype, "device": node.kwargs.get("device") or val.device},
        )
        empty.meta.update(node.meta)
    node.replace_all_uses_with(empty)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("executorch")
def _fix_floored_mod(gm, node) -> bool:
    """Rewrite `a % b` (positive literal `b`) as `((a % b) + b) % b`, equal under floored and truncated modulo.

    ExecuTorch computes C's truncated modulo, both symbolically and in `remainder`, giving negative results
    (negative pads, out-of-bounds `gather` indices).
    """
    add_op, mod_op = _FLOORED_MOD_OPS.get(node.target, (None, None))
    if node.op != "call_function" or add_op is None:
        return False
    divisor = node.args[1]
    if not isinstance(divisor, int) or divisor <= 0:
        return False
    # Already wrapped; the walk reaches inserted nodes, so this avoids wrapping forever.
    operand = node.args[0]
    if (
        getattr(operand, "op", None) == "call_function"
        and operand.target is add_op
        and len(operand.args) == 2
        and operand.args[1] == divisor
    ):
        return False
    graph = gm.graph
    with graph.inserting_after(node):
        shifted = graph.call_function(add_op, (node, divisor))
    with graph.inserting_after(shifted):
        rewrapped = graph.call_function(mod_op, (shifted, divisor))
    rewrapped.meta = dict(node.meta)
    node.replace_all_uses_with(rewrapped)
    # `replace_all_uses_with` also rewired the chain itself.
    shifted.update_arg(0, node)
    return True


@register_fx_node_fix("executorch")
def _fix_amax_dim(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Normalize negative ``dim`` on max/amax (``amax.default only supports dim == 2 or dim == 3``)."""
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
def _fix_clone_memory_format(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Force ``contiguous_format`` on ``aten.clone`` of a non-contiguous input.

    ``dim_order_from_stride`` can't map inherited transposed layouts to a ``torch.memory_format``.
    """
    if node.target is not torch.ops.aten.clone.default:
        return False
    if node.kwargs.get("memory_format") is not None:
        return False
    from torch._prims_common import is_contiguous_or_false

    input_val = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
    # Sizes may be unbacked.
    if not (isinstance(input_val, torch.Tensor) and not is_contiguous_or_false(input_val)):
        return False
    node.kwargs = {**node.kwargs, "memory_format": torch.contiguous_format}
    return True


@register_fx_node_fix("executorch")
def _fix_sym_pow_as_mul(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Replace ``operator.pow(sym_int, n)`` with ``executorch_prim.mul.Scalar`` (no ``executorch_prim.pow``)."""
    if node.target is not operator.pow:
        return False
    base, exp = node.args
    if not isinstance(exp, int) or exp < 1:
        return False
    # Safe here: the operands are SymInts.
    mul_scalar = _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS.get(operator.mul)
    if mul_scalar is None:
        return False
    base_val = base.meta.get("val") if isinstance(base, torch.fx.Node) else base
    with gm.graph.inserting_before(node):
        running = base
        running_val = base_val
        for _ in range(exp - 1):
            running = gm.graph.call_function(mul_scalar, (running, base))
            if base_val is not None and running_val is not None:
                running_val = running_val * base_val
                running.meta["val"] = running_val
    node.replace_all_uses_with(running)
    gm.graph.erase_node(node)
    return True


@register_fx_node_fix("executorch")
def _fix_negative_slice_start(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    """Rewrite a data-dependent negative slice start as ``dim_size + start``.

    ``slice_forward``'s ``start < 0`` guard can't be decided on an unbacked start (VideoMAE's
    ``[:, -return_token_num:]``). The stale ``unbacked_bindings`` are dropped.
    """
    from torch.fx.experimental.symbolic_shapes import statically_known_true

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


@register_patch("executorch", "torch.nn.functional.one_hot")
def _patch_one_hot(original):
    """One-hot via `arange` comparison: `aten.one_hot` raises `DynamicOutputShapeException` on symbolic counts."""

    def patch(input, num_classes=-1):
        if isinstance(num_classes, int) and num_classes < 0:
            return original(input, num_classes)
        return (input.unsqueeze(-1) == torch.arange(num_classes, device=input.device)).to(torch.long)

    return patch
