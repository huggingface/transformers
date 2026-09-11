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

1. **Backend preparation** (the recipe's `prepare`): the built-in `prepare_for_xnnpack` /
   `prepare_for_cuda` move the model to the target device/dtype and build the partitioner list.
   Built-in and externally registered backends share the same `ExecutorchBackendRecipe` contract.
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
import copy
import math
import operator
import re
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_executorch_available, is_torch_available
from .configs import ExecutorchConfig
from .exporter_dynamo import DynamoExporter
from .utils import (
    apply_fx_node_fixes,
    apply_fx_program_fixes,
    apply_patches,
    export_patch_scope,
    module_dtype,
    patch_attributes,
    prepare_for_export,
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

    from .. import masking_utils
    from ..modeling_utils import PreTrainedModel

    # Runtime-assert ops dropped before lowering (see `_drop_runtime_asserts`).
    _RUNTIME_ASSERT_TARGETS = (
        torch.ops.aten._assert_tensor_metadata.default,
        torch.ops.aten._assert_scalar.default,
    )


if is_executorch_available():
    from executorch.exir.capture._config import EdgeCompileConfig, ExecutorchBackendConfig
    from executorch.exir.dialects._ops import ops as exir_ops
    from executorch.exir.passes.executorch_prim_ops_registry import _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass
    from executorch.exir.passes.replace_view_copy_with_view_pass import _VIEW_OP, _is_view_copy, _ViewSpec
    from executorch.exir.passes.spec_prop_pass import _is_mutable_buffer
    from executorch.exir.program import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower
    from executorch.exir.sym_util import eval_expr
    from executorch.exir.tensor import determine_tensor_dynanism


logger = logging.get_logger(__name__)


@dataclass(frozen=True)
class ExecutorchExportPatch:
    """A backend-owned patch description applied by the ExecuTorch exporter."""

    targets: tuple[str, ...]
    factory: Callable[[Any], Any]


@dataclass(frozen=True)
class ExecutorchAttention:
    """Backend attention and an explicit mask policy.

    A callable builds masks; None removes any stale mask registration and leaves masking
    to the backend. Already supplied 4-D masks can still pass through Transformers helpers.
    """

    implementation: str
    attention_function: Callable[..., Any]
    mask_function: Callable[..., Any] | None


@dataclass(frozen=True)
class ExecutorchCompatibilityPolicy:
    """ExecuTorch compatibility bundles, enabled by default for existing backends.

    Disabling these does not disable Dynamo capture support, explicit recipe patches,
    or patches registered in the selected backend's namespace. ``excluded_patch_targets``
    suppresses only matching common/backend/Dynamo registry installations, including aliases.
    Graph fixes, recipe patches, capture contexts and signature/config helpers still run;
    exclusions are not inherited by nested exports.
    """

    common_patches: bool = True
    common_graph_fixes: bool = True
    excluded_patch_targets: tuple[str, ...] = ()


@dataclass
class ExecutorchBackendPreparation:
    """Backend-owned model, inputs, and state retained through lowering, without model copies.

    Preparation may mutate the caller's model and inputs; the backend owns those changes
    and their cleanup. Only exporter-managed patch/attention scopes are restored automatically.
    ``dynamic_shapes`` overrides the config when non-None (including an empty dict), then
    config shapes and finally ``dynamic=True`` automatic shapes apply. ``capture_contexts``
    are entered once for capture only, never for transforms or deferred lowering.

    With ``normalize_inputs=True`` (the default), shared normalization handles input keys,
    dtypes and devices. False bypasses it entirely, leaving the input ABI to the backend.
    ``attention_target`` explicitly selects recipe attention after preparation and before
    normalization. None only installs registries; wrapper attributes are never guessed.
    Identity-sensitive resources belong in ``state``, not in the copied export config.
    ``output_flags`` are reversible capture-time model-config overrides; explicit backend
    flags take precedence over flags extracted by shared normalization.
    """

    model: torch.nn.Module
    sample_inputs: MutableMapping[str, Any]
    dynamic_shapes: dict[str, Any] | None = None
    capture_contexts: tuple[contextlib.AbstractContextManager[None], ...] = ()
    state: Any = None
    normalize_inputs: bool = True
    output_flags: dict[str, Any] = field(default_factory=dict)
    attention_target: torch.nn.Module | None = None


@dataclass(frozen=True)
class ExecutorchBackendRecipe:
    """Explicit prepare/transform/lower contract for an out-of-tree ExecuTorch backend.

    Patch factories create fresh run-scoped patches for capture and deferred lowering.
    Preparation, normalization, transforms and common graph fixes run only once.
    The frozen recipe is retained by captures, independent of subsequent registrations.
    """

    prepare: Callable[[Any, MutableMapping[str, Any], ExecutorchConfig], ExecutorchBackendPreparation]
    lower: Callable[[ExportedProgram, ExecutorchBackendPreparation, ExecutorchConfig], Any]
    patches: tuple[ExecutorchExportPatch, ...] = ()
    attention: ExecutorchAttention | None = None
    transform_exported_program: Callable[[ExportedProgram, ExecutorchBackendPreparation], ExportedProgram] | None = (
        None
    )
    compatibility: ExecutorchCompatibilityPolicy = field(default_factory=ExecutorchCompatibilityPolicy)


@dataclass(frozen=True)
class ExecutorchCapture:
    """In-process capture result with backend-owned preparation/state and a single lowering attempt.

    Obtain this from ``ExecutorchExporter.capture``. The program and preparation are shared,
    not cloned; callers must not mutate them while exporting/lowering or reuse model state
    incompatibly before lowering. This handle is not a serialization format. ``config``
    returns a defensive copy of the retained configuration, while ``recipe`` remains the
    resolved frozen recipe. Even failed lowering consumes the attempt because lowerers may
    mutate the graph. Capture-only contexts are not replayed.
    """

    exported_program: ExportedProgram
    preparation: ExecutorchBackendPreparation
    recipe: ExecutorchBackendRecipe
    _config: ExecutorchConfig = field(repr=False)
    _attention_state: list[tuple[Any, dict[str, Any]]] = field(default_factory=list, repr=False)
    _lower_attempted: bool = field(default=False, init=False, repr=False)

    @property
    def config(self) -> ExecutorchConfig:
        """An independent copy; changing it cannot change deferred lowering."""
        return _snapshot_config(self._config, "capture.config access")


ExecutorchBackendRecipeFactory = Callable[[Mapping[str, Any]], ExecutorchBackendRecipe]
# External backends registered via ``register_executorch_backend``.
_EXECUTORCH_BACKEND_RECIPES: dict[str, ExecutorchBackendRecipeFactory] = {}
# Built-in backends (xnnpack, cuda), populated at import next to their ``prepare_for_*`` helpers.
# Kept in a separate table so external registrations can neither shadow nor clear them.
_BUILTIN_EXECUTORCH_BACKEND_RECIPES: dict[str, ExecutorchBackendRecipeFactory] = {}


def register_executorch_backend(
    name: str, recipe_factory: ExecutorchBackendRecipeFactory, *, overwrite: bool = False
) -> None:
    """Register a factory explicitly; no discovery or implicit backend imports are performed.

    Registering the same factory is idempotent. Replacing another external factory requires
    ``overwrite=True``; builtin names remain protected even with overwrite enabled.
    """
    if not isinstance(name, str):
        raise TypeError("ExecuTorch backend name must be a string")
    if not name.strip():
        raise ValueError("ExecuTorch backend name must not be empty")
    if not callable(recipe_factory):
        raise TypeError("ExecuTorch backend recipe factory must be callable")
    # Registration can happen during module import; do not take the export lock here.
    if name in _BUILTIN_EXECUTORCH_BACKEND_RECIPES:
        raise ValueError(f"ExecuTorch backend {name!r} is built in and cannot be replaced")
    previous = _EXECUTORCH_BACKEND_RECIPES.get(name)
    if previous is recipe_factory:
        return
    if previous is not None and not overwrite:
        raise ValueError(f"ExecuTorch backend {name!r} is already registered; use overwrite=True to replace it")
    _EXECUTORCH_BACKEND_RECIPES[name] = recipe_factory


def _resolve_patch_target(target: str) -> tuple[Any, str]:
    """Resolve a dotted patch target to its owning object and attribute."""
    from .utils import _resolve_dotted_path

    owner_path, separator, attribute = target.rpartition(".")
    if not separator or (owner := _resolve_dotted_path(owner_path)) is None:
        raise ImportError(f"Could not resolve ExecuTorch backend patch target {target!r}")
    return owner, attribute


@contextlib.contextmanager
def _select_attention(attention, model):
    """Select only an explicit target, restoring managed fields even if its setter fails."""
    if model is None:
        yield
        return
    if attention is None:
        raise ValueError("attention_target requires recipe attention")
    if not callable(getattr(model, "set_attn_implementation", None)):
        raise TypeError("ExecuTorch attention target requires set_attn_implementation()")
    snapshot = _snapshot_attention_state(model)
    try:
        model.set_attn_implementation(attention.implementation)
        yield
    finally:
        _restore_attention_state(snapshot)


@contextlib.contextmanager
def _apply_external_patches(patches: tuple[ExecutorchExportPatch, ...]):
    with export_patch_scope():
        resolved = []
        seen = {}
        for patch in patches:
            for target in patch.targets:
                owner, attribute = _resolve_patch_target(target)
                slot = (id(owner), attribute)
                if slot in seen:
                    raise ValueError(
                        f"Duplicate recipe patch targets {seen[slot]!r} and {target!r} resolve to one slot"
                    )
                seen[slot] = target
                resolved.append((owner, attribute, patch.factory))
        with patch_attributes(resolved, exclusive=True, source="ExecuTorch recipe"):
            yield


_MISSING = object()
_ATTENTION_CONFIG_FIELDS = ("_attn_implementation_internal", "_attn_was_changed")


def _snapshot_attention_state(model) -> list[tuple[Any, dict[str, Any]]]:
    snapshot = []
    seen = set()
    pending = [getattr(module, "config", None) for module in chain((model,), model.modules())]
    while pending:
        config = pending.pop()
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        snapshot.append(
            (
                config,
                {name: config.__dict__.get(name, _MISSING) for name in _ATTENTION_CONFIG_FIELDS},
            )
        )
        pending.extend(getattr(config, name, None) for name in getattr(config, "sub_configs", ()))
    return snapshot


def _restore_attention_state(snapshot: list[tuple[Any, dict[str, Any]]]) -> None:
    for config, fields in reversed(snapshot):
        for name, value in fields.items():
            if value is _MISSING:
                config.__dict__.pop(name, None)
            else:
                config.__dict__[name] = value


def _validate_attention(attention: ExecutorchAttention) -> None:
    if not isinstance(attention, ExecutorchAttention):
        raise TypeError("attention must be an ExecutorchAttention")
    if not isinstance(attention.implementation, str) or not attention.implementation.strip():
        raise ValueError("attention implementation must be a nonempty string")
    if not callable(attention.attention_function):
        raise TypeError("attention_function must be callable")
    if attention.mask_function is not None and not callable(attention.mask_function):
        raise TypeError("mask_function must be callable or None")


@contextlib.contextmanager
def scoped_executorch_attention(attention: ExecutorchAttention | None, model=None):
    """Temporarily register backend attention, optionally selecting it on a model.

    With ``model=None`` this only changes registries, supporting ordinary modules without
    a Transformers setter. Otherwise the model's setter validates/selects the implementation
    and its attention config fields are restored on exit, including on setter failure.
    Cooperating exports share a reentrant lock, including when attention is None. Unrelated
    eager code is not isolated; do not await worker-thread exports while holding this scope.
    """
    with export_patch_scope():
        if attention is None:
            with _select_attention(attention, model):
                yield None
            return
        _validate_attention(attention)

        from ..masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        from ..modeling_utils import ALL_ATTENTION_FUNCTIONS

        sentinel = object()
        registries = [
            (ALL_ATTENTION_FUNCTIONS, attention.attention_function),
            (ALL_MASK_ATTENTION_FUNCTIONS, attention.mask_function),
        ]
        previous = [registry._global_mapping.get(attention.implementation, sentinel) for registry, _ in registries]
        try:
            for registry, implementation in registries:
                if implementation is None:
                    registry._global_mapping.pop(attention.implementation, None)
                else:
                    registry.register(attention.implementation, implementation)
            with _select_attention(attention, model):
                yield attention.implementation
        finally:
            for (registry, _), prior in zip(registries, previous):
                mapping = type(registry)._global_mapping
                if prior is sentinel:
                    mapping.pop(attention.implementation, None)
                else:
                    mapping[attention.implementation] = prior


@contextlib.contextmanager
def _reenter_attention_state(snapshot):
    """Reenter saved fields without replaying the potentially mutating model setter."""
    with export_patch_scope():
        previous = [
            (config, {name: config.__dict__.get(name, _MISSING) for name in fields}) for config, fields in snapshot
        ]
        try:
            _restore_attention_state(snapshot)
            yield
        finally:
            _restore_attention_state(previous)


def _snapshot_config(value, context):
    try:
        return copy.deepcopy(value)
    except Exception as error:
        raise TypeError(
            f"ExecuTorch {context} snapshot failed: config and backend_options must be deepcopy-compatible "
            "configuration data; keep identity-sensitive resources in preparation.state"
        ) from error


def _valid_patch_target(target):
    return isinstance(target, str) and "." in target and all(part.isidentifier() for part in target.split("."))


def _validate_recipe(recipe) -> None:
    if not isinstance(recipe, ExecutorchBackendRecipe):
        raise TypeError("ExecuTorch backend recipe factories must return an ExecutorchBackendRecipe")
    for name in ("prepare", "lower"):
        if not callable(getattr(recipe, name)):
            raise TypeError(f"ExecuTorch recipe {name} must be callable")
    if recipe.transform_exported_program is not None and not callable(recipe.transform_exported_program):
        raise TypeError("transform_exported_program must be callable or None")
    if not isinstance(recipe.compatibility, ExecutorchCompatibilityPolicy):
        raise TypeError("compatibility must be an ExecutorchCompatibilityPolicy")
    for name in ("common_patches", "common_graph_fixes"):
        if not isinstance(getattr(recipe.compatibility, name), bool):
            raise TypeError(f"compatibility {name} must be a bool")
    exclusions = recipe.compatibility.excluded_patch_targets
    if not isinstance(exclusions, tuple):
        raise TypeError("excluded_patch_targets must be a tuple of dotted attribute paths")
    if any(not _valid_patch_target(target) for target in exclusions):
        raise ValueError("excluded_patch_targets must contain valid dotted attribute paths")
    if not isinstance(recipe.patches, tuple):
        raise TypeError("recipe patches must be a tuple of ExecutorchExportPatch objects")
    for patch in recipe.patches:
        if not isinstance(patch, ExecutorchExportPatch) or not callable(patch.factory):
            raise TypeError("recipe patches must contain ExecutorchExportPatch objects with callable factories")
        if not isinstance(patch.targets, tuple) or not patch.targets:
            raise TypeError("patch targets must be a nonempty tuple of dotted strings")
        if any(not _valid_patch_target(target) for target in patch.targets):
            raise ValueError("patch targets must be nonempty dotted strings")
    if recipe.attention is not None:
        _validate_attention(recipe.attention)


def _validate_preparation(prepared, attention) -> None:
    if not isinstance(prepared, ExecutorchBackendPreparation):
        raise TypeError("ExecuTorch backend recipes must prepare an ExecutorchBackendPreparation")
    if not isinstance(prepared.model, torch.nn.Module):
        raise TypeError("ExecuTorch prepared model must be a torch.nn.Module")
    if not isinstance(prepared.sample_inputs, MutableMapping) or any(
        not isinstance(key, str) for key in prepared.sample_inputs
    ):
        raise TypeError("ExecuTorch prepared sample_inputs must be a string-keyed mutable mapping")
    if not isinstance(prepared.normalize_inputs, bool):
        raise TypeError("normalize_inputs must be a bool")
    if prepared.attention_target is not None:
        if attention is None:
            raise ValueError("attention_target requires recipe attention")
        if not isinstance(prepared.attention_target, torch.nn.Module) or not callable(
            getattr(prepared.attention_target, "set_attn_implementation", None)
        ):
            raise TypeError("attention_target must be a torch.nn.Module with set_attn_implementation()")
    if not isinstance(prepared.output_flags, Mapping) or any(
        not isinstance(key, str) for key in prepared.output_flags
    ):
        raise TypeError("output_flags must be a string-keyed mapping")
    if prepared.dynamic_shapes is not None and not isinstance(prepared.dynamic_shapes, dict):
        raise TypeError("prepared dynamic_shapes must be a dict or None")
    if not isinstance(prepared.capture_contexts, tuple) or any(
        not callable(getattr(scope, "__enter__", None)) or not callable(getattr(scope, "__exit__", None))
        for scope in prepared.capture_contexts
    ):
        raise TypeError("capture_contexts must be a tuple of context managers")


@contextlib.contextmanager
def _compatibility_patches(recipe, backend):
    with contextlib.ExitStack() as stack:
        if recipe.compatibility.common_patches:
            stack.enter_context(apply_patches("executorch", exclude=recipe.compatibility.excluded_patch_targets))
        stack.enter_context(
            apply_patches(f"executorch.{backend}", exclude=recipe.compatibility.excluded_patch_targets)
        )
        yield


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
    tested_versions = {"torch": "2.12.0", "executorch": "1.3.1"}

    def export(
        self,
        model: PreTrainedModel | torch.nn.Module,
        sample_inputs: MutableMapping[str, Any],
        config: ExecutorchConfig | dict[str, Any],
    ) -> Any:
        """Prepare, capture and lower with uninterrupted recipe/attention/compatibility scopes.

        Returns the backend's lowering result (an ExecutorchProgramManager for builtins).
        Model/input mutation belongs to preparation; models are never deep-copied.
        """
        return self._export_registered_backend(model, sample_inputs, config, lower=True)

    def capture(
        self,
        model: PreTrainedModel | torch.nn.Module,
        sample_inputs: MutableMapping[str, Any],
        config: ExecutorchConfig | dict[str, Any],
    ) -> ExecutorchCapture:
        """Capture once and retain preparation, resolved recipe and config for deferred lowering.

        All temporary scopes have exited when this returns. Backend-owned preparation
        mutations remain. Cooperating exports are serialized for the entire lifecycle;
        unrelated eager execution is not protected by the shared lock.
        """
        return self._export_registered_backend(model, sample_inputs, config, lower=False)

    def lower(self, capture: ExecutorchCapture) -> Any:
        """Lower a capture once using fresh scopes, without replaying any capture-time work.

        Uses the saved recipe/config, not the registry or caller's mutable config. Reenters
        attention registry and saved fields without calling the model setter again. A failed
        attempt is consumed too, since backend lowering may already have mutated the graph.
        """
        if not isinstance(capture, ExecutorchCapture):
            raise TypeError("Expected an ExecutorchCapture from capture()")
        with export_patch_scope():
            if capture._lower_attempted:
                raise RuntimeError("An ExecutorchCapture permits only one lowering attempt")
            object.__setattr__(capture, "_lower_attempted", True)
            recipe = capture.recipe
            config = capture.config
            with (
                _apply_external_patches(recipe.patches),
                scoped_executorch_attention(recipe.attention),
                _reenter_attention_state(capture._attention_state),
                _compatibility_patches(recipe, config.backend),
            ):
                return recipe.lower(capture.exported_program, capture.preparation, config)

    def _export_registered_backend(self, model, sample_inputs, config, *, lower):
        with export_patch_scope():
            if isinstance(config, dict):
                config = ExecutorchConfig(**config)
            elif not isinstance(config, ExecutorchConfig):
                raise TypeError(f"Expected config to be an ExecutorchConfig or dict, got {type(config)}")
            config._validate_backend()
            config = _snapshot_config(config, "initial config")
            recipe_factory = _BUILTIN_EXECUTORCH_BACKEND_RECIPES.get(config.backend)
            if recipe_factory is None:
                recipe_factory = _EXECUTORCH_BACKEND_RECIPES.get(config.backend)
            if recipe_factory is None:
                available = sorted(set(_BUILTIN_EXECUTORCH_BACKEND_RECIPES) | set(_EXECUTORCH_BACKEND_RECIPES))
                raise ValueError(
                    f"Unsupported backend {config.backend} for ExecuTorch export; available backends: {available}"
                )
            recipe = recipe_factory(_snapshot_config(dict(config.backend_options), "recipe factory backend_options"))
            _validate_recipe(recipe)

            with (
                _apply_external_patches(recipe.patches),
                scoped_executorch_attention(recipe.attention),
                contextlib.ExitStack() as attention_stack,
            ):
                prepared = recipe.prepare(model, sample_inputs, config)
                _validate_preparation(prepared, recipe.attention)
                attention_target = prepared.attention_target
                attention_stack.enter_context(_select_attention(recipe.attention, attention_target))
                if prepared.normalize_inputs:
                    prepared.model, prepared.sample_inputs, output_flags = prepare_for_export(
                        prepared.model, prepared.sample_inputs
                    )
                    prepared.output_flags = {**output_flags, **prepared.output_flags}

                with _compatibility_patches(recipe, config.backend):
                    with contextlib.ExitStack() as stack:
                        for capture_context in prepared.capture_contexts:
                            stack.enter_context(capture_context)
                        exported_program = self._export_prepared(
                            prepared.model,
                            prepared.sample_inputs,
                            config,
                            output_flags=prepared.output_flags,
                            dynamic_shapes=prepared.dynamic_shapes,
                            patch_exclusions=recipe.compatibility.excluded_patch_targets,
                        )
                    if not isinstance(exported_program, ExportedProgram):
                        raise TypeError("ExecuTorch capture must return an ExportedProgram")
                    if recipe.transform_exported_program is not None:
                        exported_program = recipe.transform_exported_program(exported_program, prepared)
                        if not isinstance(exported_program, ExportedProgram):
                            raise TypeError("transform_exported_program must return an ExportedProgram")
                    if recipe.compatibility.common_graph_fixes:
                        apply_fx_program_fixes("executorch", exported_program)
                        apply_fx_node_fixes("executorch", exported_program.graph_module)
                    if lower:
                        return recipe.lower(exported_program, prepared, config)
                    attention_state = []
                    if recipe.attention is not None:
                        seen = set()
                        for root in (model, prepared.model, attention_target):
                            if root is None:
                                continue
                            for attention_config, fields in _snapshot_attention_state(root):
                                if id(attention_config) not in seen:
                                    seen.add(id(attention_config))
                                    attention_state.append((attention_config, fields))
                    return ExecutorchCapture(
                        exported_program=exported_program,
                        preparation=prepared,
                        recipe=recipe,
                        _config=_snapshot_config(config, "retained capture config"),
                        _attention_state=attention_state,
                    )


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
# To add a built-in backend: implement prepare_for_<name>, wrap it with _builtin_lowering_recipe, and
# register it in _BUILTIN_EXECUTORCH_BACKEND_RECIPES. Out-of-tree backends use register_executorch_backend().


def _make_contiguous(sample_inputs: dict[str, Any]) -> dict[str, Any]:
    """Materialise input tensors to contiguous for ExecuTorch.

    ExecuTorch rejects a 0 (broadcast) stride on any tensor, including graph inputs — a sample input
    can be a non-contiguous broadcast view (e.g. a batch dim expanded from 1), whose stride-0 is
    captured on the input placeholder and later rejected by `spec_prop` ("0 in strides is not
    supported"). `.contiguous()` is a no-op for already-contiguous tensors, so cache buffers keep
    their identity for in-place static-cache writes.
    """
    return torch.utils._pytree.tree_map_only(torch.Tensor, lambda t: t.contiguous(), sample_inputs)


def prepare_for_xnnpack(model: PreTrainedModel, sample_inputs: dict[str, Any]):
    """CPU inference via XNNPACK.

    Moves the model to CPU: XNNPACK's partitioner/serializer and the edge-lowering passes all
    require a CPU-typed graph, and tracing on CPU also sidesteps per-model device bugs — models
    create in-``forward`` tensors (``arange``/``zeros``/sinusoids) without ``device=``, which
    default to CPU and would mismatch a CUDA model (``FakeTensor Device Propagation ... cuda, cpu``).
    ``prepare_for_export`` then casts the inputs to CPU during the trace."""

    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

    model.requires_grad_(False)
    model = model.to(device="cpu")
    # XNNPACK has no `_grouped_mm.out` kernel — force MoE experts to `batched_mm`.
    if isinstance(model, PreTrainedModel) and model._can_set_experts_implementation():
        model.set_experts_implementation("batched_mm")
    partitioner = [XnnpackPartitioner()]
    return model, _make_contiguous(sample_inputs), partitioner


def prepare_for_cuda(model: PreTrainedModel, sample_inputs: dict[str, Any]):
    """GPU inference via the ExecuTorch CUDA backend, decoupled from the model's device.

    The backend requires bfloat16 (upcast here) and a visible GPU — it delegates ops to Triton
    kernels compiled by AOTInductor, which needs a GPU to compile/autotune. The model itself can
    stay on any device (e.g. CPU): AOTInductor targets the machine's GPU regardless of where the
    traced tensors live, so no `.to("cuda")` is needed."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment; cannot export to the ExecuTorch CUDA backend.")

    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    model.requires_grad_(False)
    dtype = module_dtype(model)
    if dtype is not None and dtype != torch.bfloat16:
        logger.warning(f"ExecuTorch CUDA backend requires bfloat16; upcasting model from {dtype}.")
        model = model.to(dtype=torch.bfloat16)
    partitioner = [CudaPartitioner([CudaBackend.generate_method_name_compile_spec(model.__class__.__name__)])]
    return model, _make_contiguous(sample_inputs), partitioner


def _builtin_lowering_recipe(prepare_for_backend) -> ExecutorchBackendRecipe:
    """Wrap a built-in ``prepare_for_*`` in the recipe SPI with the standard edge lowering.

    Built-in backends dogfood the same ``ExecutorchBackendRecipe`` contract as externally
    registered ones: ``prepare`` runs the backend's device/dtype setup and hands its partitioner
    to ``lower`` via ``ExecutorchBackendPreparation.state``; ``lower`` runs the shared
    edge-lowering + ``to_executorch`` pipeline.
    """

    def prepare(
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: ExecutorchConfig,
    ) -> ExecutorchBackendPreparation:
        model, sample_inputs, partitioner = prepare_for_backend(model, sample_inputs)
        # dynamic_shapes stays None so capture falls back to config.dynamic_shapes; the partitioner
        # rides ``state`` from prepare to lower.
        return ExecutorchBackendPreparation(model=model, sample_inputs=sample_inputs, state=partitioner)

    def lower(
        exported_program: ExportedProgram,
        prepared: ExecutorchBackendPreparation,
        config: ExecutorchConfig,
    ) -> ExecutorchProgramManager:
        edge_program_manager: EdgeProgramManager = to_edge_transform_and_lower(
            exported_program, partitioner=prepared.state, compile_config=_get_edge_compile_config()
        )
        return edge_program_manager.to_executorch(config=_get_backend_config(config))

    return ExecutorchBackendRecipe(prepare=prepare, lower=lower)


def _xnnpack_backend_recipe(backend_options: Mapping[str, Any]) -> ExecutorchBackendRecipe:
    """Built-in XNNPACK recipe (CPU inference); see ``prepare_for_xnnpack``."""
    if backend_options:
        raise ValueError("The builtin xnnpack backend does not support backend_options")
    return _builtin_lowering_recipe(prepare_for_xnnpack)


def _cuda_backend_recipe(backend_options: Mapping[str, Any]) -> ExecutorchBackendRecipe:
    """Built-in CUDA recipe (GPU inference); see ``prepare_for_cuda``."""
    if backend_options:
        raise ValueError("The builtin cuda backend does not support backend_options")
    return _builtin_lowering_recipe(prepare_for_cuda)


_BUILTIN_EXECUTORCH_BACKEND_RECIPES.update(
    {
        "xnnpack": _xnnpack_backend_recipe,
        "cuda": _cuda_backend_recipe,
    }
)


# ── Stage 2: Torch patches ────────────────────────────────────────────────────
# Reversible swaps of `torch` ops the ExecuTorch backends can't lower (`split_copy`,
# `topk(k>dim)`, non-divisible `avg_pool2d`, `dropout`, in-place `view`, GQA-shaped
# SDPA …). Each `_patch_*(original)` factory is registered via
# `@register_patch("executorch", "dotted.path")` and installed through `apply_patches`.


@register_patch("executorch.cuda", "torch.split", "torch.Tensor.split", lazy=True)
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
        else:
            splits = []
            start = 0
            for size in split_size_or_sections:
                splits.append(input.narrow(dim, start, size))
                start += size
            return tuple(splits)

    return patch


@register_patch("executorch.cuda", "torch.chunk", "torch.Tensor.chunk", lazy=True)
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


@register_patch("executorch.cuda", "torch.topk", "torch.Tensor.topk", lazy=True)
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


@register_patch("executorch", "torch.detach", "torch.Tensor.detach")
def _patch_detach(_original):
    """No-op detach."""

    def patch(input):
        return input

    return patch


@register_patch("executorch.cuda", "torch.nn.functional.avg_pool2d", lazy=True)
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


@register_patch("executorch", "torch.bucketize")
def _patch_bucketize(original):
    """Decompose bucketize into a broadcasted comparison + sum.

    The portable runtime ships no `bucketize.Tensor_out` kernel (used by VLM vision position ids —
    idefics2/3, smolvlm, phi4_multimodal). `boundaries` is 1-D and sorted, so the bucket index is
    just the count of boundaries below each value — comparison and sum, both portable ops.
    """

    def patch(input, boundaries, *, out_int32=False, right=False, out=None):
        below = (boundaries <= input.unsqueeze(-1)) if right else (boundaries < input.unsqueeze(-1))
        result = below.sum(dim=-1)
        result = result.to(torch.int32) if out_int32 else result
        return out.copy_(result) if out is not None else result

    return patch


@register_patch("executorch", "torch.searchsorted")
def _patch_searchsorted(original):
    """Decompose searchsorted into a broadcasted comparison + sum (no portable kernel; same idea as
    bucketize). ``sorted_sequence`` is sorted, so the insertion index is the count of entries below.
    """

    def patch(sorted_sequence, input, *, out_int32=False, right=False, side=None, out=None, sorter=None):
        if side is not None:
            right = side == "right"
        seq, val = sorted_sequence.unsqueeze(-2), input.unsqueeze(-1)
        below = (seq <= val) if right else (seq < val)
        result = below.sum(dim=-1)
        result = result.to(torch.int32) if out_int32 else result
        return out.copy_(result) if out is not None else result

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


def _cumulative_reduce(input: torch.Tensor, dim: int, maximum: bool) -> torch.Tensor:
    """``cummax``/``cummin`` values via a triangular-masked ``amax``/``amin`` (no portable scan
    kernel). Output ``[..., i]`` reduces over ``j <= i``: broadcast the sequence against a
    lower-triangular keep-mask, fill the rest with the dtype's min/max, then reduce."""
    seq = input.transpose(dim, -1)
    length = seq.shape[-1]
    positions = torch.arange(length, device=input.device)
    keep = positions.unsqueeze(0) <= positions.unsqueeze(1)  # [i, j] = j <= i
    info = torch.finfo if input.is_floating_point() else torch.iinfo
    fill = info(input.dtype).min if maximum else info(input.dtype).max
    windows = torch.where(keep, seq.unsqueeze(-2), fill)
    reduced = windows.amax(dim=-1) if maximum else windows.amin(dim=-1)
    return reduced.transpose(dim, -1)


@register_patch("executorch", "torch.cummax", "torch.Tensor.cummax")
def _patch_cummax(_original):
    """Decompose ``cummax`` (no portable cumulative-scan kernel) — see ``_cumulative_reduce``.
    Returns ``(values, indices)`` like ``torch.cummax``; indices are zeros (callers use the values)."""

    def patch(input, dim):
        values = _cumulative_reduce(input, dim, maximum=True)
        return torch.return_types.cummax((values, torch.zeros_like(values, dtype=torch.long)))

    return patch


@register_patch("executorch", "torch.cummin", "torch.Tensor.cummin")
def _patch_cummin(_original):
    """Decompose ``cummin`` (no portable cumulative-scan kernel) — see ``_cumulative_reduce``.
    Returns ``(values, indices)`` like ``torch.cummin``; indices are zeros (callers use the values)."""

    def patch(input, dim):
        values = _cumulative_reduce(input, dim, maximum=False)
        return torch.return_types.cummin((values, torch.zeros_like(values, dtype=torch.long)))

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


def _normalize_tensor_shape_args(args, kwargs, keyword):
    """Normalize Tensor varargs or a single shape sequence without concretizing SymInts."""
    if kwargs:
        if args or set(kwargs) != {keyword}:
            raise TypeError(f"Expected positional dimensions or a single {keyword}= argument")
        args = (kwargs[keyword],)
    if not args:
        raise TypeError(f"Missing required {keyword} argument")
    return args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args


@register_patch("executorch", "torch.Tensor.expand")
def _patch_expand(_original):
    """Force a contiguous copy after ``expand``.

    ``Tensor.expand`` produces a view with stride ``0`` along broadcast dims.
    ExecuTorch's memory planner rejects ``stride == 0`` and raises "0 in strides is not
    supported for ExecuTorch" — see ``TensorSpec.__init__`` in
    https://github.com/pytorch/executorch/blob/v1.0.0/exir/tensor.py#L72. Materialise
    the broadcast so the captured tensor has standard strides downstream.
    """

    def patch(self, *sizes, **kwargs):
        # Saved TensorBase descriptors are not traceable under strict torch.export.
        size = _normalize_tensor_shape_args(sizes, kwargs, "size")
        result = torch.ops.aten.expand.default(self, size)
        # Only materialise when ``expand`` actually introduced a stride-0 (broadcast) dim; a
        # no-broadcast expand is a plain view ExecuTorch's memory planner accepts as-is.
        if 0 in result.stride():
            return result.clone(memory_format=torch.contiguous_format)
        return result

    return patch


@register_patch("executorch", "torch.reshape")
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

    def patch(input, shape):
        if not input.is_contiguous():
            input = input.clone(memory_format=torch.contiguous_format)
        return original(input, shape)

    return patch


@register_patch("executorch", "torch.Tensor.reshape")
def _patch_tensor_reshape(_original):
    """Use traceable ATen instead of calling a saved TensorBase reshape descriptor."""

    def patch(self, *shape, **kwargs):
        shape = _normalize_tensor_shape_args(shape, kwargs, "shape")
        if not self.is_contiguous():
            self = self.clone(memory_format=torch.contiguous_format)
        return torch.ops.aten.reshape.default(self, shape)

    return patch


@register_patch("executorch", "torch.Tensor.view")
def _patch_tensor_view(_original):
    """Preserve both view overloads while materialising non-contiguous inputs."""

    def patch(self, *size, **kwargs):
        size = _normalize_tensor_shape_args(size, kwargs, "dtype" if "dtype" in kwargs else "size")
        if not self.is_contiguous():
            self = self.clone(memory_format=torch.contiguous_format)
        if len(size) == 1 and isinstance(size[0], torch.dtype):
            return torch.ops.aten.view.dtype(self, size[0])
        return torch.ops.aten.view.default(self, size)

    return patch


# ── Stage 3: ExecuTorch patches

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
    from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import XNNStaticReshape, XNode
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
    "executorch.xnnpack",
    "executorch.backends.xnnpack.partition.config.node_configs.PreluConfig.check_constraints",
    lazy=True,
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
    "executorch.xnnpack",
    "executorch.backends.xnnpack.serialization.xnnpack_graph_serialize._flatc_compile",
    lazy=True,
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


@register_patch(
    "executorch.xnnpack", "executorch.backends.xnnpack.operators.node_visitor._node_visitor_dict", lazy=True
)
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
            var_to_val = getattr(shape_env, "backed_var_to_val", None) or shape_env.var_to_val
            break  # all nodes share the same shape_env, so we only need one

    floor = _dim_floor(len({sym for rd in range_dicts for sym, vr in rd.items() if isinstance(vr.upper, IntInfinity)}))

    unbounded = []
    for rd in range_dicts:
        for sym, vr in rd.items():
            if isinstance(vr.upper, IntInfinity):
                lower = _as_int(vr.lower, 2)
                trace_val = _as_int(var_to_val.get(sym), 0)
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
            stack.extend(feeder.all_input_nodes)
            module.graph.erase_node(feeder)
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
