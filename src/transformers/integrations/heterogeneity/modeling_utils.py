# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import contextvars
import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any

from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import (
    SkipDescriptor,
    get_heterogeneous_modeling_spec,
)
from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByAttributeValue


if TYPE_CHECKING:
    from torch import nn

    from transformers import PreTrainedModel


@dataclass(frozen=True)
class _LayerInitContext:
    model: PreTrainedModel
    layer_cls: type[nn.Module]
    layer_idx_variable_name: str
    skip_descriptors: dict[str, SkipDescriptor]


_layer_init_context: contextvars.ContextVar[tuple[_LayerInitContext, ...]] = contextvars.ContextVar(
    "_layer_init_context", default=()
)
_model_init_stack: contextvars.ContextVar[tuple[PreTrainedModel, ...]] = contextvars.ContextVar(
    "_model_init_stack", default=()
)
_layer_patching_lock = threading.Lock()


def apply_heterogeneous_modeling(model: PreTrainedModel) -> None:
    """Apply heterogeneous per-layer modeling during model construction.

    This function resolves the model's ``HeterogeneousModelingSpec``, validates its
    configured skips, records which layers do not update the KV cache, and registers
    the layer-construction context. The patched layer class uses this context to
    initialize each layer with its resolved config, apply skip replacements, and
    select layer-specific attention masks.

    Args:
        model: The model being initialized.
    """
    if not model.config.is_heterogeneous:
        return

    heterogeneous_modeling_spec = get_heterogeneous_modeling_spec(model)
    if heterogeneous_modeling_spec is None:
        return

    per_layer_skip_types = [layer_config.skip for layer_config in model.config.per_layer_config]
    skip_descriptors = heterogeneous_modeling_spec.skip_descriptors or {}
    _validate_skip_descriptors(per_layer_skip_types, skip_descriptors)

    # Record which layers have their KV-cache update disabled on the config's heterogeneity spec,
    # where cache construction (e.g. `StaticCache`) can read it from the config alone.
    model.config._heterogeneity_spec.disabled_kv_layer_indices = tuple(
        layer_idx
        for layer_idx, skip_types in enumerate(per_layer_skip_types)
        if any(skip_descriptors[skip_type].replaces_kv_cache_updater for skip_type in skip_types)
    )

    context = _LayerInitContext(
        model=model,
        layer_cls=heterogeneous_modeling_spec.layer_cls,
        layer_idx_variable_name=heterogeneous_modeling_spec.layer_idx_variable_name,
        skip_descriptors=skip_descriptors,
    )
    _layer_init_context.set((*_layer_init_context.get(), context))
    _patch_layer_init(heterogeneous_modeling_spec.layer_cls)


def wrap_model_init_with_heterogeneous_context(orig_init: Callable[..., None]) -> Callable[..., None]:
    if getattr(orig_init, "_wrapped_with_heterogeneous_context", False):
        return orig_init

    @wraps(orig_init)
    def _patched_init(self, *args, **kwargs):
        model_init_stack = _model_init_stack.get()
        if any(model is self for model in model_init_stack):
            return orig_init(self, *args, **kwargs)

        model_init_stack_token = _model_init_stack.set((*model_init_stack, self))
        layer_init_context_token = _layer_init_context.set(_layer_init_context.get())
        try:
            return orig_init(self, *args, **kwargs)
        finally:
            _layer_init_context.reset(layer_init_context_token)
            _model_init_stack.reset(model_init_stack_token)

    _patched_init._wrapped_with_heterogeneous_context = True
    return _patched_init


def _patch_layer_init(layer_cls: type[nn.Module]) -> None:
    """Patch ``layer_cls.__init__`` to resolve each layer's index and pass its matching per-layer config to the original init function."""
    if getattr(layer_cls.__init__, "_patched_by_heterogeneity", False):
        return

    with _layer_patching_lock:
        if getattr(layer_cls.__init__, "_patched_by_heterogeneity", False):
            return

        orig_layer_init = layer_cls.__init__

        @wraps(orig_layer_init)
        def _patched_layer_init(self, config, *args, **kwargs):
            context = next(
                (
                    context
                    for context in reversed(_layer_init_context.get())
                    if context.layer_cls is layer_cls and context.model.config is config
                ),
                None,
            )
            if context is None or not getattr(config, "is_heterogeneous", False):
                return orig_layer_init(self, config, *args, **kwargs)

            # --- Resolve layer index ---
            layer_idx_source = "constructor arguments"
            layer_idx = _get_variable_from_passed_arguments(
                func=orig_layer_init,
                args=(self, config, *args),
                kwargs=kwargs,
                names=[context.layer_idx_variable_name],
            )
            if layer_idx is None:
                layer_idx_source = "model construction stack"
                layer_idx = _get_variable_from_model_construction_stack(
                    model=context.model,
                    names=[context.layer_idx_variable_name],
                )
            if layer_idx is None:
                raise RuntimeError(
                    f"Could not determine layer index `{context.layer_idx_variable_name}` for heterogeneous model "
                    f"initialization of `{context.model.__class__.__name__}`. Make sure it is a layer constructor "
                    "argument or a local variable in the model's layer construction path."
                )
            _validate_layer_idx(
                layer_idx,
                variable_name=context.layer_idx_variable_name,
                source=layer_idx_source,
                num_layers=config.num_hidden_layers,
            )

            # --- Apply per-layer config ---
            layer_config = config.per_layer_config[layer_idx]
            orig_layer_init(self, layer_config, *args, **kwargs)

            # --- Replace skipped sublayers ---
            for skip_type in layer_config.skip:
                _apply_skip_descriptor(
                    layer=self,
                    skip_descriptor=context.skip_descriptors[skip_type],
                    layer_idx=layer_idx,
                )

            # --- Patch forward for attention mask selection ---
            if {"sliding_window", "attention_chunk_size"} & context.model.config.per_layer_attributes:
                mask_key = getattr(layer_config, "sliding_window", None) or getattr(
                    layer_config, "attention_chunk_size", None
                )  # Relies on having exclusivity validation in the heterogeneous configuration_utils
                if mask_key is not None:
                    _patch_layer_forward_for_attention_mask_selection(layer=self, mask_key=mask_key)

        _patched_layer_init._patched_by_heterogeneity = True
        layer_cls.__init__ = _patched_layer_init


def _patch_layer_forward_for_attention_mask_selection(
    *,
    layer: nn.Module,
    mask_key: int,
) -> None:
    orig_forward = layer.forward

    @wraps(orig_forward)
    def _patched_forward(self, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, AttentionMasksByAttributeValue):
            kwargs["attention_mask"] = attention_mask[mask_key]
        return orig_forward(*args, **kwargs)

    layer.forward = MethodType(_patched_forward, layer)


def _validate_skip_descriptors(
    per_layer_skip_types: list[list[str]], skip_descriptors: dict[str, SkipDescriptor]
) -> None:
    skip_types = {skip_type for layer_skip_types in per_layer_skip_types for skip_type in layer_skip_types}
    missing_descriptors = skip_types - skip_descriptors.keys()
    if missing_descriptors:
        raise ValueError(f"No-op descriptors are missing for the following types: {missing_descriptors}")


def _apply_skip_descriptor(
    *,
    layer: nn.Module,
    skip_descriptor: SkipDescriptor,
    layer_idx: int,
) -> None:
    generic_replacements = {}
    class_specific_replacements = {}

    for key, replacement_module in skip_descriptor.replacements.items():
        if isinstance(key, tuple):
            member_name, cls = key
        else:
            member_name = key
            cls = None

        if not _hasattr_by_path(layer, member_name):
            raise AttributeError(
                f"Layer {layer_idx} in class {layer.__class__.__name__} has no attribute {member_name}"
            )

        if cls is None:
            generic_replacements[member_name] = replacement_module
            continue

        if not isinstance(_getattr_by_path(layer, member_name), cls):
            continue

        if member_name in class_specific_replacements:
            raise ValueError(
                f"Multiple class-specific skip replacements match layer {layer_idx} "
                f"attribute {member_name} in class {layer.__class__.__name__}"
            )
        class_specific_replacements[member_name] = replacement_module

    for member_name, replacement_module in class_specific_replacements.items():
        _setattr_by_path(layer, member_name, replacement_module())

    for member_name, replacement_module in generic_replacements.items():
        if member_name not in class_specific_replacements:
            _setattr_by_path(layer, member_name, replacement_module())


def _getattr_by_path(obj: Any, attribute_path: str) -> Any:
    for attribute_name in attribute_path.split("."):
        obj = getattr(obj, attribute_name)
    return obj


def _hasattr_by_path(obj: Any, attribute_path: str) -> bool:
    try:
        _getattr_by_path(obj, attribute_path)
    except AttributeError:
        return False
    return True


def _setattr_by_path(obj: Any, attribute_path: str, value: Any) -> None:
    parent_path, _, attribute_name = attribute_path.rpartition(".")
    parent = _getattr_by_path(obj, parent_path) if parent_path else obj
    setattr(parent, attribute_name, value)


def _get_variable_from_passed_arguments(
    *,
    func: Callable,
    args: tuple,
    kwargs: dict,
    names: list[str],
) -> Any | None:
    signature = inspect.signature(func)
    try:
        bound_arguments = signature.bind(*args, **kwargs)
    except TypeError as e:
        raise TypeError(f"{func.__qualname__}() {e}") from None
    bound_arguments.apply_defaults()
    for name in names:
        if name in bound_arguments.arguments:
            return bound_arguments.arguments[name]
    return None


def _get_variable_from_model_construction_stack(*, model: PreTrainedModel, names: list[str]) -> Any | None:
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return None

    # Skip this helper and the patched layer initializer so its own `layer_idx` local cannot shadow the model's.
    frame = frame.f_back.f_back

    while frame is not None:
        for name in names:
            if name in frame.f_locals:
                return frame.f_locals[name]
        if frame.f_code.co_name == "__init__" and frame.f_locals.get("self") is model:
            return None
        frame = frame.f_back
    return None


def _validate_layer_idx(layer_idx: Any, *, variable_name: str, source: str, num_layers: int) -> None:
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
        raise TypeError(
            f"Layer index `{variable_name}` resolved from the {source} must be an integer, "
            f"but got {layer_idx!r} ({type(layer_idx).__name__})."
        )
    if not 0 <= layer_idx < num_layers:
        raise IndexError(
            f"Layer index `{variable_name}` resolved from the {source} is out of range for "
            f"a model with {num_layers} layers: {layer_idx}."
        )
