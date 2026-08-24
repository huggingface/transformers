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
from transformers.integrations.heterogeneity.layer_idx_resolvers import LayerIdxResolver
from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByLayerIdx


if TYPE_CHECKING:
    from torch import nn

    from transformers import PreTrainedModel


@dataclass(frozen=True)
class _LayerInitContext:
    model: PreTrainedModel
    layer_cls: type[nn.Module]
    layer_idx_resolver: LayerIdxResolver
    skip_descriptors: dict[str, SkipDescriptor]


_layer_init_contexts: contextvars.ContextVar[tuple[_LayerInitContext, ...]] = contextvars.ContextVar(
    "_layer_init_contexts", default=()
)
_model_init_contexts: contextvars.ContextVar[tuple[PreTrainedModel, ...]] = contextvars.ContextVar(
    "_model_init_contexts", default=()
)
_layer_patching_lock = threading.Lock()


def apply_generic_heterogeneous_modeling_if_applicable(model: PreTrainedModel) -> None:
    """Apply heterogeneous per-layer modeling during model initialization.

    This function resolves the model's ``HeterogeneousModelingSpec``, validates its
    configured skips, records which layers do not update the KV cache, and registers
    the layer-initialization context. The patched layer class uses this context to
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
        layer_idx_resolver=heterogeneous_modeling_spec.layer_idx_resolver,
        skip_descriptors=skip_descriptors,
    )
    _layer_init_contexts.set((*_layer_init_contexts.get(), context))
    _patch_layer_init(heterogeneous_modeling_spec.layer_cls)


def support_generic_heterogeneous_modeling(orig_init: Callable[..., None]) -> Callable[..., None]:
    """Create the model-initialization scope required by ``apply_generic_heterogeneous_modeling_if_applicable``.

    That function runs inside ``PreTrainedModel.__init__`` and registers temporary state that is used later, when the
    model subclass creates its layers. This wrapper keeps that state available across the model's ``super().__init__()``
    chain and restores the previous state when initialization finishes. Nested models receive their own nested scope.
    If generic heterogeneous modeling is not applied, the wrapper does not change model initialization.
    """
    if getattr(orig_init, "_scoped_for_heterogeneous_modeling", False):
        return orig_init

    @wraps(orig_init)
    def _scoped_init(self, *args, **kwargs):
        model_init_contexts = _model_init_contexts.get()
        if any(model is self for model in model_init_contexts):
            return orig_init(self, *args, **kwargs)

        model_init_contexts_token = _model_init_contexts.set((*model_init_contexts, self))
        layer_init_contexts_token = _layer_init_contexts.set(_layer_init_contexts.get())
        try:
            return orig_init(self, *args, **kwargs)
        finally:
            _layer_init_contexts.reset(layer_init_contexts_token)
            _model_init_contexts.reset(model_init_contexts_token)

    _scoped_init._scoped_for_heterogeneous_modeling = True
    return _scoped_init


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
                    for context in reversed(_layer_init_contexts.get())
                    if context.layer_cls is layer_cls and context.model.config is config
                ),
                None,
            )
            if context is None or not getattr(config, "is_heterogeneous", False):
                return orig_layer_init(self, config, *args, **kwargs)

            # --- Resolve layer index ---
            layer_idx = context.layer_idx_resolver.resolve(
                layer_init=orig_layer_init,
                args=(self, config, *args),
                kwargs=kwargs,
                model=context.model,
            )
            _validate_layer_idx(
                layer_idx,
                resolver=context.layer_idx_resolver,
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
            _patch_layer_forward_for_attention_mask_layer_selection(layer=self, layer_idx=layer_idx)

        _patched_layer_init._patched_by_heterogeneity = True
        layer_cls.__init__ = _patched_layer_init


def _patch_layer_forward_for_attention_mask_layer_selection(
    *,
    layer: nn.Module,
    layer_idx: int,
) -> None:
    orig_forward = layer.forward

    @wraps(orig_forward)
    def _patched_forward(self, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, AttentionMasksByLayerIdx):
            kwargs["attention_mask"] = attention_mask[layer_idx]
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


def _validate_layer_idx(layer_idx: Any, *, resolver: LayerIdxResolver, num_layers: int) -> None:
    resolver_description = f"{type(resolver).__name__}({resolver.variable_name!r})"
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
        raise TypeError(
            f"Layer index `{resolver.variable_name}` must be an integer, but `{resolver_description}` got "
            f"{layer_idx!r} ({type(layer_idx).__name__})."
        )
    if not 0 <= layer_idx < num_layers:
        raise IndexError(
            f"Layer index `{resolver.variable_name}` is out of range for a model with {num_layers} layers: "
            f"`{resolver_description}` got {layer_idx}."
        )
