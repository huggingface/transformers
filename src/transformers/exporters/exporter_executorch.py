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
2. **Torch patches** (`_PATCHES["executorch"]` via `apply_patches("executorch")`, plus the
   backend-specific `_PATCHES[f"executorch.{backend}"]`): reversibly swap `torch` ops the
   ExecuTorch backends can't accept (`split_copy`, `avg_pool2d`, …) with decomposed equivalents.
   Backend-specific ones (e.g. the CUDA-only `topk` fallback) apply to just one backend.
   Reverted on exit.
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

import contextlib
import functools
import json
import math
import operator
import re
from collections.abc import MutableMapping
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_executorch_available, is_torch_available
from .configs import ExecutorchConfig, ExportFormat
from .exporter_dynamo import DynamoExporter, varlen_attn_masked_sdpa
from .metadata import (
    EXPORT_METADATA_KEY,
)
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
    from torch.fx.passes.infra.pass_base import PassResult
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils._sympy.numbers import IntInfinity
    from torch.utils._sympy.value_ranges import ValueRanges

    from ..modeling_utils import PreTrainedModel

    # Runtime-assert ops dropped before lowering (see `_drop_runtime_asserts`).
    _RUNTIME_ASSERT_TARGETS = (
        torch.ops.aten._assert_tensor_metadata.default,
        torch.ops.aten._assert_scalar.default,
    )


if is_executorch_available():
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import (  # type: ignore[import-not-found]
        XNNStaticReshape,
        XNode,
    )
    from executorch.backends.xnnpack.utils.utils import get_input_node
    from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
    from executorch.exir.passes.executorch_prim_ops_registry import _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
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

    export_format = ExportFormat.EXECUTORCH
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
        if isinstance(config, dict):
            config = ExecutorchConfig(**config)
        elif type(config) is not ExecutorchConfig:
            raise TypeError(f"Expected config to be an ExecutorchConfig or dict, got {type(config)}")

        prepare_for_backend = _BACKEND_PREPARE.get(config.backend)
        if prepare_for_backend is None:
            raise ValueError(f"Unsupported backend {config.backend} for ExecuTorch export")

        model, sample_inputs, partitioner = prepare_for_backend(
            model, sample_inputs, exclude=tuple(config.partition_exclude)
        )
        partitioner = partitioner if config.partition else []

        with (
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
                    compile_config=_get_edge_compile_config(exported_program),
                    # A `.pte` binds its inputs positionally and reports only counts and shapes, so what
                    # the graph *is* rides along as one constant method — the same schema the ONNX
                    # exporter writes into `metadata_props` (`build_export_metadata`).
                    constant_methods={EXPORT_METADATA_KEY: [json.dumps(metadata)]},
                )
                executorch_programs_manager = edge_program_manager.to_executorch(config=_get_backend_config(config))

        return executorch_programs_manager, metadata

    @classmethod
    def save_artifact(cls, artifact, path) -> None:
        """The metadata is a constant method inside the program (`_patch_metadata_method`), so it is already
        part of the serialized `.pte`. Streamed rather than taken through `.buffer`, which materializes the
        whole program as bytes first."""
        with open(path, "wb") as file:
            artifact.write_to_file(file)


@register_patch("executorch", "executorch.exir.program._program.serialize_for_executorch")
def _patch_serialize_for_executorch(original):
    """Settle the size-1 dim orders on the way into the binary.

    `ExecutorchProgramManager` serializes in its constructor, so there is no moment afterwards at which the
    program can still be edited — the bytes already exist. This is that moment.
    """

    def serialize_for_executorch(emitter_output, *args, **kwargs):
        canonicalize_size_one_dim_orders(getattr(emitter_output, "program", None))
        return original(emitter_output, *args, **kwargs)

    return serialize_for_executorch


def canonicalize_size_one_dim_orders(executorch_program) -> None:
    """Order a tensor's size-1 axes the way every other tensor orders them.

    A dim order is the axes sorted by stride, and a size-1 axis has no extent to sort by — it shares its
    neighbour's stride, so whichever side of the tie it lands on is arbitrary. ExecuTorch's portable kernels
    do not treat it as arbitrary: they require every operand of an op to carry the *same* dim order, and
    refuse the call otherwise (`tensors_have_same_dim_order`, `0x12`) — which splinter hits on `aten::repeat`
    with `[64, 64, 1]` ordered `[0, 2, 1]` against `[64, 64, 32]` ordered `[0, 1, 2]`.

    So where the ambiguity is the only difference, it is settled one way: a tensor whose axes are in
    canonical order once the size-1 ones are ignored gets the canonical order outright. A tensor that is
    really laid out differently (channels-last, a transpose of two axes that both have extent) keeps what it
    has — its order describes something.
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


@contextlib.contextmanager
def keep_backed_symbols_symbolic(exported_program: ExportedProgram):
    """Selective: keep *backed* symbols symbolic through the lowering, let everything else proceed.

    The lowering's passes re-execute ops on the program's live fake tensors, and each shape relation those
    re-executions evaluate against the trace hints ends in `ShapeEnv.set_replacement` — refining a
    declared-dynamic axis down to its hint (`s70 = 2`), which the memory plan then bakes in ("Attempted to
    resize a static tensor", 0x12). But wholesale guard suppression breaks the models that carry *unbacked*
    symbols (bart's data-dependent sizes): those are legitimately resolved during lowering by the very same
    replacement machinery, and blocking it leaves a later pass guarding on an unresolvable expression
    (`GuardOnDataDependentSymNode`). So filter exactly the harmful case: a symbol the trace *hinted* (backed,
    present in `var_to_val`) being replaced by a *constant*. Unbacked resolution and symbol-to-symbol
    unification pass through untouched.
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

    # `var_to_val` was renamed `backed_var_to_val` (the old name warns); both hold exactly the backed
    # symbols, which is the distinction this wrapper turns on.
    backed_values = getattr(shape_env, "backed_var_to_val", None)
    if backed_values is None:
        backed_values = shape_env.var_to_val

    def selective(symbol, replacement, *args, **kwargs):
        if symbol in backed_values and getattr(replacement, "is_number", False):
            return None
        return original(symbol, replacement, *args, **kwargs)

    shape_env._set_replacement = selective
    try:
        yield
    finally:
        del shape_env._set_replacement


def _uses_channels_last(exported_program: ExportedProgram) -> bool:
    """Whether any tensor in the graph carries a channels-last layout — the case the `dim_order_ops` variants
    exist to represent, and the one graph shape that cannot be lowered without them.

    Deliberately *only* channels-last, not every non-contiguous tensor: a transpose-derived view (the rotary
    pattern, dozens per audio tower) is also a non-standard dim order, and keeping the dim-order ops for
    those graphs was measured to fix nothing while re-freezing the symbolic-size buffers the plain schemas
    keep dynamic (musicflamingo/qwen3_asr stayed red, deepseek_v3's merged decode broke)."""
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
                # Contiguity is a question about strides, and a tensor whose size is data-dependent cannot
                # answer it — `is_contiguous` raises rather than returning. Count it as not channels-last,
                # which is this function's default answer and the one that keeps the plain ATen schemas.
                continue
    return False


def _get_edge_compile_config(exported_program: ExportedProgram) -> EdgeCompileConfig:
    """Build the ``EdgeCompileConfig`` used for ``to_edge_transform_and_lower``.

    Adds non-core ATen ops to ``_core_aten_ops_exception_list`` so torch.export
    decompositions that produce these ops don't trip the edge-dialect verifier.
    These are ops that show up in transformers models (FFT in fnet, bucketize /
    is_all_true in T5 / mBart / Bart family, polar in seamless_m4t rotary, etc.)
    but aren't in the core ATen opset. The CPU portable kernels handle them at
    runtime; XNNPACK leaves them in the non-delegated CPU portion of the graph.
    """
    return EdgeCompileConfig(
        # Keep the plain ATen ops instead of the `dim_order_ops` variants wherever the graph allows it:
        # `_empty_dim_order` declares its size as `int[]` where `aten.empty`'s is `SymInt[]`, so dispatch
        # coerces every symbolic size to its trace hint and the fake kernel cannot produce a symbolic val —
        # freezing the buffer (and everything downstream) at the traced shape, which the memory plan then
        # enforces at runtime ("Attempted to resize a static tensor", 0x12). With the plain schema the
        # symbols survive to the spec pass and the buffers plan at their bounds. The one graph shape that
        # *needs* the dim-order variants is a channels-last tensor — the emitter refuses the layout without
        # them ("Tensor has a memory_format that is unsupported") — so those graphs keep them.
        # ET-version-sensitive, like every internals patch here: revisit when the dim-order schemas learn
        # `SymInt[]`.
        _skip_dim_order=not _uses_channels_last(exported_program),
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


def _get_backend_config(config):
    """Build the ``ExecutorchBackendConfig`` for ``to_executorch``, or ``None`` for defaults.

    Only overrides the memory-planning pass when the caller changed an ``alloc_*`` flag. Turning off
    ``alloc_graph_input``/``alloc_graph_output`` hands input/output memory ownership to the caller
    (see [`ExecutorchConfig`]) — the prerequisite for zero-copy in-place ``USER_INPUT_MUTATION``.
    """
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
# Each prepare_for_* function receives the original model and sample inputs, applies backend-specific preparation,
# and returns the modified model, the list of partitioners to apply, and the modified sample inputs. Common patterns include:
# - Move the model to the target device.
# - Cast the model and inputs to the required dtype (e.g., bfloat16 for CUDA).
# - Build the backend-specific partitioner list passed to to_edge_transform_and_lower.
# To add a new backend: implement _prepare_for_new_backend and add it to the _BACKEND_PREPARE table.


def _make_contiguous(sample_inputs: dict[str, Any]) -> dict[str, Any]:
    """Materialise input tensors to contiguous for ExecuTorch.

    ExecuTorch rejects a 0 (broadcast) stride on any tensor, including graph inputs — a sample input
    can be a non-contiguous broadcast view (e.g. a batch dim expanded from 1), whose stride-0 is
    captured on the input placeholder and later rejected by `spec_prop` ("0 in strides is not
    supported"). `.contiguous()` is a no-op for already-contiguous tensors, so cache buffers keep
    their identity for in-place static-cache writes.
    """
    return torch.utils._pytree.tree_map_only(torch.Tensor, lambda t: t.contiguous(), sample_inputs)


def prepare_for_xnnpack(model: PreTrainedModel, sample_inputs: dict[str, Any], exclude: tuple[str, ...] = ()):
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
    # Withholding a config leaves its ops to the portable kernels and keeps the rest delegated: XNNPACK
    # can claim a partition its compiler then refuses over a single op pattern (`ViewCopyConfig` for
    # qwen3_next), where dropping the whole partition would cost every other op its acceleration.
    if exclude:
        from executorch.backends.xnnpack.partition.config import ALL_PARTITIONER_CONFIGS

        unknown = set(exclude) - {config.__name__ for config in ALL_PARTITIONER_CONFIGS}
        if unknown:
            raise ValueError(f"Unknown XNNPACK partitioner config(s): {sorted(unknown)}")
        configs = [config for config in ALL_PARTITIONER_CONFIGS if config.__name__ not in exclude]
        if not configs:
            # `XnnpackPartitioner` reads `configs or ALL_PARTITIONER_CONFIGS`, so an empty list means
            # *every* config, the opposite of what excluding all of them asks for. Say so instead.
            raise ValueError("partition_exclude removes every partitioner config; pass partition=False instead")
        partitioner = [XnnpackPartitioner(configs=configs)]
    else:
        partitioner = [XnnpackPartitioner()]
    return model, _make_contiguous(sample_inputs), partitioner


def prepare_for_cuda(model: PreTrainedModel, sample_inputs: dict[str, Any], exclude: tuple[str, ...] = ()):
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
    return model, _make_contiguous(sample_inputs), partitioner


_BACKEND_PREPARE = {
    "xnnpack": prepare_for_xnnpack,
    "cuda": prepare_for_cuda,
}


# ── Stage 2: Torch patches ────────────────────────────────────────────────────
# Reversible swaps of `torch` ops the ExecuTorch backends can't lower (`split_copy`,
# `topk(k>dim)`, non-divisible `avg_pool2d`, `dropout`, in-place `view`, GQA-shaped
# SDPA …). Each `_patch_*(original)` factory is registered via
# `@register_patch("executorch", "dotted.path")` and installed through `apply_patches`.


@register_patch(
    "executorch",
    "transformers.models.falcon_mamba.modeling_falcon_mamba.mamba_selective_scan",
    "transformers.models.jamba.modeling_jamba.mamba_selective_scan",
    "transformers.models.mamba.modeling_mamba.mamba_selective_scan",
    "transformers.models.zamba.modeling_zamba.mamba_selective_scan",
)
def _patch_mamba_selective_scan(original):
    """Keep the SSM scan sequential: the associative scan has no ExecuTorch lowering, and the runtime has no
    loop primitive to run it as. Forced at the scan's own call site rather than through a mixer's
    `use_associative_scan` attribute — every family passes it here as a keyword, whereas the attribute is a
    per-instance copy of the config knob and would take the live model to reach. The sequential path
    unrolls, pinning the traced step length — why the multi-token decode variants are skipped here."""

    def patch(*args, **kwargs):
        return original(*args, **{**kwargs, "use_associative_scan": False})

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
    """Order a tensor's dims when one of its strides carries a data-dependent size.

    ExecuTorch reads dim order by sorting strides, and its comparator already answers what it can without a
    hint (`guard_or_false`), falling back to a plain `<` — which on `64*u21 < 64` has nothing to decide with
    and raises `GuardOnDataDependentSymNode` from inside the lowering, after a graph exported cleanly.

    The fallback is sorted *size-obliviously* instead. That is sound for this question: the symbol is
    size-like, so it is at least one, so a stride of `64*u21` is at least the `64` it multiplies — and the
    answer being sought is only which axis sits outermost, never the extent itself. Registered against
    every call site, because three of the four imported the name rather than the module.
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


def _compare_assuming_nonempty(left, right) -> int:
    """`left` against `right` (-1, 0, 1) with every data-dependent size in it taken to be 2.

    What `guard_size_oblivious` used to answer, written out because it is deprecated in favour of explicit
    unbacked handling. Two is the size-oblivious convention: the symbol is a size, and the cases that make
    an ordering question unanswerable are the degenerate 0 and 1. Substituting rather than guarding also
    keeps the lowering's shape environment untouched, which is the point of `keep_backed_symbols_symbolic`.
    """

    def concrete(side):
        node = getattr(side, "node", None)
        if node is None:
            return int(side)
        expression = node.expr
        # A stride mixes both kinds of symbol: a backed one stands for a real traced size and has a hint to
        # put in its place, an unbacked one has none and takes the 2. Leaving either symbolic would make the
        # comparison unanswerable again, and an unanswerable comparison is what puts two operands of one op
        # in different dim orders — which ExecuTorch's kernels reject at run time (`0x12`).
        shape_env = getattr(node, "shape_env", None)
        hints = getattr(shape_env, "backed_var_to_val", None) or getattr(shape_env, "var_to_val", None) or {}
        return int(expression.xreplace({symbol: hints.get(symbol, 2) for symbol in expression.free_symbols}))

    try:
        left_value, right_value = concrete(left), concrete(right)
    except (TypeError, ValueError):
        # Still not a number: treat the two as indistinguishable, which keeps the order they came in.
        return 0
    # Equal reads as equal, so the sort leaves them in place. Forcing an order on strides that are really
    # the same (a size-1 axis has its neighbour's stride) is what puts two tensors of one op in different
    # dim orders, and ExecuTorch's kernels reject that at run time.
    return (left_value > right_value) - (left_value < right_value)


@register_patch("executorch", "torch.split", "torch.Tensor.split")
def _patch_unbacked_split(original):
    """Keep a split whose *sizes are data-dependent* out of the graph.

    A grid VLM cuts its flat vision output back into per-image runs with
    `(grid_thw.prod(-1) // merge**2).tolist()`, i.e. sizes read off a tensor. Under export those are
    unbacked symbols, and `split_with_sizes_copy` with unbacked sizes is the one thing ExecuTorch's
    verifier cannot lower ("Could not extract specialized integer") — it fails every variant of every
    such model, ~17 of them. Both consumers in this position immediately concatenate the pieces again
    (the model's own `torch.cat(image_features, dim=0)`, and `ModalityEncoder.forward`), so handing back
    the tensor whole is the same value with nothing for the verifier to choke on. A consumer that really
    wanted the pieces indexes past the end of a 1-tuple, which fails loudly rather than quietly.

    The test is for an *unbacked* size, not merely "not a Python int": `aten.split.Tensor`'s own
    decomposition rewrites a constant chunk size into a size list whose last element is a backed-symbolic
    expression of the split dim, then re-enters this same public `torch.split`. Short-circuiting there
    hands the base tensor back from inside a view op's decomposition, which autograd rejects with "View
    operation returned a tensor that is the same as the input base tensor" — that would fail every
    `.split(int)` / `.chunk()` taken over a dynamic dim (qwen3_omni_moe chunks its audio conv that way).
    """

    def patch(input, split_size_or_sections, dim=0):
        if _has_unbacked_sizes(split_size_or_sections):
            return (input,)
        return original(input, split_size_or_sections, dim)

    return patch


@register_patch("executorch.cuda", "torch.split", "torch.Tensor.split")
def _patch_split(original):
    """Narrow-based split for the CUDA backend, which can't lower `split_copy`.

    Not registered for XNNPACK: the portable runtime has native `split_copy`/`slice_copy` kernels,
    so native `torch.split` lowers there and delegates better than a chain of narrows.
    """

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
        elif _has_unbacked_sizes(split_size_or_sections):
            # Data-dependent section sizes: narrowing to them puts the unbacked symbol in the graph, which
            # is what `_patch_unbacked_split` -- the patch this one is layered over on CUDA -- exists to
            # prevent. Defer to it rather than reimplementing the half of the decision this branch can see.
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
    """`torch.chunk` decomposes through `aten.split_copy.Tensor`, which AOT inductor for the
    ExecuTorch CUDA backend can't lower (`split_copy.Tensor is missing a c-shim implementation`).
    Same root cause as `_patch_split`; route `chunk` through the (also CUDA-scoped) `torch.split`
    so it ends up as a sequence of `narrow`s instead. Not registered for XNNPACK, which lowers
    `chunk` natively via the portable `split_copy` kernel.
    """

    def patch(input, chunks, dim=0):
        total = input.size(dim)
        chunk_size = (total + chunks - 1) // chunks
        return torch.split(input, chunk_size, dim)

    return patch


@register_patch("executorch.cuda", "torch.topk", "torch.Tensor.topk")
def _patch_topk(original):
    """Argsort-based topk fallback for the CUDA backend, which has no topk kernel.

    Not registered for XNNPACK: the portable runtime ships a `topk.values` kernel but no `sort`,
    so rewriting `topk` to `argsort` (which lowers to `aten.sort.values`) would make the program
    fail to load (`Missing operator: aten::sort.values`). Native `topk` lowers to the supported
    `aten.topk.values` there instead.
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
    """Topk-based argsort for XNNPACK, whose portable runtime ships no `sort` kernel.

    The mirror image of `_patch_topk`: that one rewrites `topk` as `argsort` for CUDA, which has no topk
    kernel, and is deliberately not registered here because the portable registry is the other way round —
    it has `aten.topk.values` but no `aten.sort.values`, so a graph reaching lowering with an `argsort` in it
    produces a `.pte` that fails to *load* (`Missing operator: aten::sort.values`). A model that sorts on its
    own hits that wall the same way (muse_glimmer's vision tower inverts its window permutation with
    `torch.argsort`), so the rewrite is applied here rather than left to each model.

    `topk` over the whole axis is a full sort, and `largest=descending` matches `argsort`'s ordering.
    """

    def patch(input, dim=-1, descending=False, stable=False):
        return torch.topk(input, input.shape[dim], dim=dim, largest=descending, sorted=True).indices

    return patch


@register_patch("executorch", "transformers.models.deepseek_v2.modeling_deepseek_v2.DeepseekV2RotaryEmbedding.forward")
def _patch_deepseek_v2_rotary_forward(original):
    """Carry deepseek_v2's rotary frequencies as a real `[cos, sin]` pair rather than a complex tensor.

    ExecuTorch's portable registry ships neither `aten::polar.out` nor `aten::view_as_complex_copy.out`, so
    a graph building `torch.polar(ones, freqs)` produces a `.pte` that cannot load (`0x14`). The pair is the
    same numbers — `polar(1, θ)` is `cos θ + i sin θ` — and `_patch_deepseek_v2_apply_rotary_emb` consumes
    it, so no complex tensor is created. The two are a set: neither works without the other.

    `@dynamic_rope_update` on the original updates `inv_freq` for the dynamic RoPE types before the body
    runs; it is kept by wrapping the original rather than reimplementing that, and only the complex tail
    differs.
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
    """deepseek_v2's rotary as real arithmetic, against the pair the patch above produces.

    A complex multiply is two real multiplies: `(a + bi)(c + di)` is `(ac - bd) + (ad + bc)i`. Writing it
    out keeps the same float operations in the same order, so the result matches the complex path rather
    than approximating it."""

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
    """Insert the axis with a reshape, so XNNPACK never sees an `unsqueeze_copy` to refuse.

    XNNPACK's partitioner claims `aten.unsqueeze_copy` and its compiler then rejects the partition on a
    dynamic graph (`0x1`, `Propagating input shapes failed with code: xnn_status_invalid_parameter`). It is
    the only one of the 52 partitioner configs that does: excluding `UnsqueezeCopyConfig` alone lets
    deepseek_v2's dynamic export run delegated, and excluding any of the other 51 changes nothing. A reshape
    to the same shape is the same operation and is claimed by a config XNNPACK honours.
    """

    def patch(input, dim):
        shape = list(input.shape)
        shape.insert(dim if dim >= 0 else dim + len(shape) + 1, 1)
        return input.reshape(shape)

    return patch


@register_patch("executorch", "torch.detach", "torch.Tensor.detach")
def _patch_detach(_original):
    """No-op detach."""

    def patch(input):
        return input

    return patch


@register_patch("executorch.cuda", "torch.nn.functional.avg_pool2d")
def _patch_avg_pool2d(original):
    """Decompose avg_pool2d as depthwise conv2d for the CUDA backend, which has no avg_pool2d kernel.

    Not registered for XNNPACK: the portable runtime ships a native `avg_pool2d.out` kernel, so the
    decomposition is unnecessary there.
    """

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


@register_patch("executorch", "torch.nn.functional.pad")
def _patch_pad(original):
    """Split a negative pad into the crop it means plus the non-negative remainder — torch treats a negative
    amount as trimming that edge, and ExecuTorch's portable kernel refuses it outright ("Padding values must
    be non-negative", execute 0x12; longt5's local-attention blocking and rwkv's shifted state both pad
    negatively)."""

    def patch(input, pad, mode="constant", value=None):
        # Only a provably non-negative pad goes through untouched: under a dynamic export the amounts are
        # SymInts (`block_len - seq`) whose sign is not knowable here, so those take the general form too.
        if all(isinstance(amount, int) and amount >= 0 for amount in pad):
            return original(input, pad, mode, value)
        output = original(input, [torch.sym_max(amount, 0) for amount in pad], mode, value)
        # Pad and crop act on opposite edges independently, so clamping first and trimming after is the
        # same tensor as torch's crop-then-pad — and it needs no branch on a symbolic sign.
        for i in range(len(pad) // 2):
            left, right, dim = pad[2 * i], pad[2 * i + 1], input.dim() - 1 - i
            start = torch.sym_max(-left, 0)
            stop = output.size(dim) - torch.sym_max(-right, 0)
            output = output.narrow(dim, start, stop - start)
        return output

    return patch


@register_patch("executorch", "torch.nn.functional.conv3d")
def _patch_conv3d(original):
    """Decompose conv3d into a sum of conv2d's over the kernel's depth — ExecuTorch's portable convolution
    kernel takes 3-D/4-D inputs only ("Expect input tensor to be 3-D or 4-D, but got, 5", execute 0x12), and
    a video/temporal vision tower's patch embedding is a Conv3d (glm4v and family, cosmos3_omni,
    cohere_compass).

    For each depth offset `kt`, the matching strided temporal slice contributes one conv2d over `(H, W)`:
    the output frames fold into the batch axis, the 2-D geometry (stride/padding/dilation/groups) applies
    unchanged, and the contributions sum. Bias is added once at the end.
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
    """Decompose adaptive_avg_pool2d (no portable adaptive-pool kernel).

    When the input spatial dims divide the output dims evenly it's a plain ``avg_pool2d``; otherwise
    (e.g. pyramid pooling with output 2/3/6) each output cell averages a variable, overlapping window
    per the adaptive formula — ``start = i*S // O``, ``end = ceil((i+1)*S / O)`` — which we build from
    concrete slices + ``mean``. Falls back to the original for symbolic (dynamic) spatial dims, where
    the window bounds aren't computable at trace time.
    """

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
    """Rewrite ``bernoulli`` as ``rand_like`` + comparison (no portable ``bernoulli`` out-variant).

    ExecuTorch ships no out-variant kernel for ``aten::bernoulli`` (SpeechT5's speech-decoder prenet
    consistent-dropout), so ``to_executorch`` fails with ``Missing out variants: {'aten::bernoulli'}``.
    ``rand_like`` *does* have a portable kernel (it's in the core-aten exception list), so
    ``(rand_like(input) < probs)`` is a faithful Bernoulli sample — the real randomness is preserved.
    ``generator`` is ignored (``rand_like`` seeds from the default generator; export doesn't carry a
    per-call generator anyway).
    """

    def patch(input, *args, p=None, generator=None, out=None):
        # Two API shapes: bernoulli(input) (elementwise probabilities) and
        # bernoulli(input, p=...) (scalar probability, shape from input).
        if p is None and len(args) == 1:
            p = args[0]
        probs = input if p is None else p
        result = (torch.rand_like(input) < probs).to(input.dtype)
        return out.copy_(result) if out is not None else result

    return patch


@register_patch("executorch", "torch.nn.attention.varlen.varlen_attn")
def _patch_varlen_attn(original):
    """The chunked vision/audio attention patch calls `varlen_attn`, whose CUDA flash op stays opaque
    through edge lowering (its aux outputs trip the edge-dialect verifier). Swap it for the block-diagonal
    masked SDPA — core-aten ops ExecuTorch can lower — returning just the output tensor (`varlen_attn`'s
    contract, vs the underlying op's `(output, *aux)` tuple)."""

    def varlen_attn(*args, **kwargs):
        return varlen_attn_masked_sdpa(*args, **kwargs)

    return varlen_attn


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
    from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols

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
    "executorch.xnnpack", "executorch.backends.xnnpack.partition.config.node_configs.PreluConfig.check_constraints"
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


# Lookbehind/lookahead on JSON delimiters so only bare numeric literals match (quoted strings are
# bounded by `"` and never touched). Compiled once at import rather than per delegate-serialize call.
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
        fixed = data
        for pattern, repl in _JSON_NONFINITE_SUBS:
            fixed = pattern.sub(repl, fixed)
        if fixed != data:
            with open(json_path, "w") as f:
                f.write(fixed)
        return original(output_dir, schema_path, json_path)

    return patch


@register_patch("executorch.xnnpack", "executorch.backends.xnnpack.operators.node_visitor._node_visitor_dict")
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

# Caps for `int_oo` dynamic-dim upper bounds. ExecuTorch's XNNPACK memory planner pre-allocates
# buffers from the upper bound, so an unbounded dim must get a finite cap; capping too tight rejects
# legitimate trace-time shapes (e.g. VLM image-token counts). Each dim's cap is `max(lower, trace) *
# multiplier`, floored so a dim traced small still gets a usable range.
_MAX_DIM_MULTIPLIER = 4
# 1024 covers the largest single unbounded dim we see in practice (VLM image-token counts, seq lens)
# without over-allocating; 64 keeps a dim usable even when several are unbounded (see `_dim_floor`).
_MAX_DIM_FLOOR = 1024  # cap floor for a single unbounded dim
_MIN_DIM_FLOOR = 64  # cap floor never drops below this, so dims stay usable at runtime
# The planner's arena grows with the *product* of the unbounded dims, so a fixed floor lets several
# small-traced dims multiply into a huge arena. `_dim_floor` instead splits this product budget
# across the N unbounded dims, keeping that product bounded. 2**24 (~16M elements) is the largest
# element-count product we allow across all unbounded dims — a few hundred MB at fp32, the ceiling
# before XNNPACK's memory planner starts overflowing/thrashing on the CI runners.
_MAX_UNBOUNDED_PRODUCT = 2**24


def _dim_floor(num_unbounded: int) -> int:
    """Cap floor for each of `num_unbounded` simultaneously-unbounded dims, sized so their product
    stays near `_MAX_UNBOUNDED_PRODUCT` and clamped to ``[_MIN_DIM_FLOOR, _MAX_DIM_FLOOR]``."""
    per_dim = round(_MAX_UNBOUNDED_PRODUCT ** (1.0 / max(num_unbounded, 1)))
    return max(_MIN_DIM_FLOOR, min(_MAX_DIM_FLOOR, per_dim))


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
def _fix_constant_dim_orders(exported_program: ExportedProgram) -> None:
    """Make every constant tensor contiguous — a weight stored channels-last (an audio tower's conv kernels)
    keeps those strides through serialization, invisible to any scan of the graph's activation vals, and at
    runtime feeds a portable kernel whose planned output is contiguous ("2 input tensors have different dim
    orders", `slice_copy`/`squeeze_copy` refusing 0x12). The values are unchanged; only the layout is."""
    for holder in (exported_program.state_dict, exported_program.constants):
        for name, tensor in holder.items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
                holder[name] = tensor.contiguous()


def _query_axis_symbols(exported_program: ExportedProgram, var_to_val: dict) -> tuple[set, int]:
    """The symbols on the token axis of the graph's text inputs, and the sequence scale to bound them by:
    the larger of their own hint and the cache's length hint (the prompt a merged decode was captured after).
    """

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


@register_fx_program_fix("executorch")
def _fix_range_constraints(exported_program: ExportedProgram) -> None:
    """Cap ``int_oo`` upper bounds for ExecuTorch compatibility.

    Caps each unbounded dim at ``max(lower, trace) * _MAX_DIM_MULTIPLIER`` or a floor that shrinks
    with the number of unbounded dims (see `_dim_floor`), so bounds cover the sampled shapes without
    the unbounded dims' product overflowing XNNPACK memory planning.
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
            # `is None`, not `or`: an empty backed mapping is falsy and would fall through to the
            # deprecated name.
            var_to_val = getattr(shape_env, "backed_var_to_val", None)
            if var_to_val is None:
                var_to_val = shape_env.var_to_val
            break  # all nodes share the same shape_env, so we only need one

    floor = _dim_floor(len({sym for rd in range_dicts for sym, vr in rd.items() if isinstance(vr.upper, IntInfinity)}))

    # The query axis is bounded from the *sequence* scale the graph was traced at, not from its own hint: a
    # merged multi-token decode is captured at two tokens yet serves the whole prompt (a multi-modal export
    # has no separate prefill graph), and its cache already carries that prompt's length — so the query
    # symbols take `max(query hint, cache-length hint)`. Traced at two, a 64-token bound refused a 71-token
    # prompt ("Attempted to resize a bounded tensor with a maximum capacity of 512 elements to 568").
    query_symbols, sequence_hint = _query_axis_symbols(exported_program, var_to_val)

    unbounded = []
    for rd in range_dicts:
        for sym, vr in rd.items():
            if isinstance(vr.upper, IntInfinity):
                lower = _as_int(vr.lower, 2)
                trace_val = _as_int(var_to_val.get(sym), 0)
                if sym in query_symbols:
                    trace_val = max(trace_val, sequence_hint)
                upper = max(lower * _MAX_DIM_MULTIPLIER, trace_val * _MAX_DIM_MULTIPLIER, floor)
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
    ``_assert_tensor_metadata`` (input dtype/device/layout) has no ``range_constraints`` equivalent,
    but ExecuTorch re-validates every input's spec against the method signature at load / ``set_inputs``,
    so those checks are re-established at runtime rather than lost.
    """
    for module in exported_program.graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        asserts = [
            node
            for node in module.graph.nodes
            if node.op == "call_function" and node.target in _RUNTIME_ASSERT_TARGETS
        ]
        # Erase the asserts and only the symint feeders they leave dead, walking back from each assert's
        # inputs. We avoid a global `eliminate_dead_code` on purpose: it also visits unrelated dead nodes
        # and on some graphs (e.g. `minimax_m3_vl` under static shapes) trips an fx `SystemError`/`KeyError`
        # inside `_update_args_kwargs` erasing an unrelated `expand_as`. Targeted removal only touches the
        # assert chain, so those unrelated nodes are left for the downstream lowering passes to clean up.
        stack = [feeder for node in asserts for feeder in node.all_input_nodes]
        for node in asserts:
            module.graph.erase_node(node)
        # A feeder can be reached more than once (shared input / diamond); track erased nodes so we
        # never call `erase_node` twice on the same one (its `users` is empty after the first erase,
        # which would otherwise pass the guard below and corrupt the graph).
        erased = set()
        while stack:
            feeder = stack.pop()
            if feeder in erased or feeder.op in ("placeholder", "output") or feeder.users or feeder.is_impure():
                continue
            # Best effort: on some graphs (glmasr / musicflamingo audio towers) erasing a dead `sym_size`
            # feeder trips a C-level fx bug (`SystemError` in `_update_args_kwargs`, an arg already nulled).
            # A leftover dead `sym_size` is harmless — the tracer only chokes on the `Piecewise` cast/eq
            # chain, which erases fine — so skip the node and keep the rest of the cleanup.
            try:
                module.graph.erase_node(feeder)
            except SystemError:
                continue
            stack.extend(feeder.all_input_nodes)
            erased.add(feeder)
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


# `a % b` and the ops that rebuild it, per level: symbolic ints go through `operator`, tensors through the
# aten schemas `torch.export` records for `tensor % int`.
_FLOORED_MOD_OPS = {operator.mod: (operator.add, operator.mod)}
if is_torch_available():
    _FLOORED_MOD_OPS[torch.ops.aten.remainder.Scalar] = (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.remainder.Scalar,
    )


@register_fx_node_fix("executorch")
def _fix_floored_mod(gm, node) -> bool:
    """Rewrite `a % b` (positive literal `b`) as `((a % b) + b) % b` — the same value under floored *and*
    truncated modulo.

    torch computes Python's floored modulo (`-9 % 4 = 3`), ExecuTorch C's truncated one (`-9 % 4 = -1`), and
    both its symbolic evaluator and its `remainder` kernel take the C answer. Symbolically, a pad amount like
    longt5's `-seq % block` arrives negative and the kernel refuses it ("Padding values must be non-negative").
    On tensors the disagreement is silent and worse: `(-5) % 16` returns `-5`, so timesfm's
    `(idx_range - indices) % num_seq` produces negative indices and the `gather` that consumes them fails
    bounds-checking (`0x12` at `aten::gather.out`) — for a positive `indices` the values were simply wrong.

    The double-mod form is a graph-level identity no simplifier removes, and evaluates identically either way:
    under flooring the inner result is already in `[0, b)` so the outer mod is a no-op, and under truncation
    `(trunc + b)` lands in `(0, 2b)` and the outer mod brings it back.
    """
    add_op, mod_op = _FLOORED_MOD_OPS.get(node.target, (None, None))
    if node.op != "call_function" or add_op is None:
        return False
    divisor = node.args[1]
    if not isinstance(divisor, int) or divisor <= 0:
        return False
    # Already the wrapped form (its operand is the `+ divisor` this fix inserts) — the node list iteration
    # reaches freshly inserted nodes, so without this the rewrap would wrap itself forever.
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
    # `replace_all_uses_with` also rewired the chain itself — point it back at the original.
    shifted.update_arg(0, node)
    return True


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
    from torch._prims_common import is_contiguous_or_false

    input_val = node.args[0].meta.get("val") if hasattr(node.args[0], "meta") else None
    # Guard-free contiguity, for the same reason as the reshape patch above: this runs on the traced
    # `FakeTensor`, whose sizes may be unbacked.
    if not (isinstance(input_val, torch.Tensor) and not is_contiguous_or_false(input_val)):
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


@register_patch("executorch", "torch._subclasses.fake_impls.op_implementations_dict")
def _patch_nonzero_fake_layout(original):
    """Report `nonzero`'s result as contiguous, matching the layout ExecuTorch's kernel actually writes.

    ATen's CPU `nonzero` fills a `(ndim, nnz)` buffer and returns its transpose, and the fake kernel is
    faithful to it: `new_empty_strided((nnz, ndim), (1, nnz))`. ExecuTorch's planner turns those strides into
    a dim order of `(1, 0)`, but its `nonzero.out` kernel writes the tensor row-major — so the label
    contradicts the bytes, and reading it back scrambles the values. For `[[1, -1, 1], [-1, 1, -1]]` the
    program returns `[[0, 2], [0, 1], [0, 1]]` where eager gives `[[0, 0], [0, 2], [1, 1]]` (row-major bytes
    `[0, 0, 0, 2, 1, 1]` re-read with stride `(1, 3)`). Consumers that assert
    `tensors_have_same_dim_order(in, out)` against their contiguous, planner-allocated output refuse the call
    instead — `0x12` at `aten::slice_copy.Tensor_out` for musicflamingo's audio-token positions
    (`torch.where(diff == 1)`) and `aten::squeeze_copy.dims_out` for qwen3_asr's (`.nonzero().squeeze(-1)`).

    Fixed at the fake kernel because that is the only place it holds: `to_edge_transform_and_lower` and
    `to_executorch` each re-run it, so a rewritten stride on either graph is recomputed, and the emitted
    program is already serialized by the time it can be edited. Swapping the registry dict for a copy leaves
    the key set intact, so the membership check that `register_op_impl` bound to the original dict still
    agrees with this lookup (`dispatch_to_op_implementations_dict` reads the module attribute).
    """
    inner = original.get(torch.ops.aten.nonzero.default)
    if inner is None:
        return original

    def contiguous_nonzero(fake_mode, func, arg):
        result = inner(fake_mode, func, arg)
        return result.new_empty(result.shape) if isinstance(result, torch.Tensor) else result

    return {**original, torch.ops.aten.nonzero.default: contiguous_nonzero}


@register_patch("executorch", "torch.nn.functional.one_hot")
def _patch_one_hot(original):
    """Build the one-hot matrix by comparison against `arange` instead of calling `aten.one_hot`.

    `one_hot`'s fake kernel has to know `num_classes` to give the result a shape, and raises
    `DynamicOutputShapeException` when it cannot — which includes a symbolic count, as in longt5's
    transient-global attention (`one_hot(block_ids, global_seq_len + 1)`, where the global length comes from
    the input's block count). `arange` takes a `SymInt` happily, and broadcasting the comparison gives the
    same matrix with a shape expressed in that symbol.
    """

    def patch(input, num_classes=-1):
        if isinstance(num_classes, int) and num_classes < 0:
            return original(input, num_classes)
        return (input.unsqueeze(-1) == torch.arange(num_classes, device=input.device)).to(torch.long)

    return patch
