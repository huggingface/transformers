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
from functools import partial, wraps
from typing import TYPE_CHECKING, Any

from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import (
    SkipDescriptor,
    get_heterogeneous_modeling_spec,
)
from transformers.integrations.heterogeneity.layer_idx_resolvers import LayerIdxResolver
from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByLayerIdx


if TYPE_CHECKING:
    from torch import nn

    from transformers import PreTrainedConfig, PreTrainedModel


@dataclass(frozen=True)
class _LayerInitContext:
    model: PreTrainedModel
    layer_cls: type[nn.Module]
    layer_idx_resolver: LayerIdxResolver
    skip_descriptors: dict[str, SkipDescriptor]
    model_layer_configs: dict[int, PreTrainedConfig]


_layer_init_contexts: contextvars.ContextVar[tuple[_LayerInitContext, ...]] = contextvars.ContextVar(
    "_layer_init_contexts", default=()
)
_initializing_models: contextvars.ContextVar[tuple[PreTrainedModel, ...]] = contextvars.ContextVar(
    "_initializing_models", default=()
)
_layer_patching_lock = threading.Lock()


def apply_generic_heterogeneous_modeling_if_applicable(model: PreTrainedModel) -> None:
    """Apply heterogeneous per-layer modeling during model initialization.

    This function resolves the model's ``HeterogeneousModelingSpec``, validates its
    configured skips, and registers the layer-initialization context. The patched layer class uses this context to
    initialize each layer with its resolved config, apply skip replacements, and
    select layer-specific attention masks.

    Args:
        model: The model being initialized.
    """
    if not model.config.is_heterogeneous:
        return

    if model.config.generic_modeling_applied:
        raise ValueError(
            f"This {type(model.config).__name__}, or the config it was copied from, was already used to construct a "
            "model with generic heterogeneous modeling. Its `per_layer_config` returns the layer configs resolved "
            "when that model was built, so later changes to the global config would not reach a new model. Create a "
            "new config with `type(config).from_dict(config.to_dict())`."
        )

    heterogeneous_modeling_spec = get_heterogeneous_modeling_spec(model)
    if heterogeneous_modeling_spec is None:
        return

    per_layer_skip_types = [layer_config.skip for layer_config in model.config.per_layer_config]
    skip_descriptors = heterogeneous_modeling_spec.skip_descriptors or {}
    _validate_skip_descriptors(per_layer_skip_types, skip_descriptors)

    layer_init_contexts = _layer_init_contexts.get()
    model_layer_configs = next(
        (context.model_layer_configs for context in layer_init_contexts if context.model.config is model.config),
        {},
    )
    context = _LayerInitContext(
        model=model,
        layer_cls=heterogeneous_modeling_spec.layer_cls,
        layer_idx_resolver=heterogeneous_modeling_spec.layer_idx_resolver,
        skip_descriptors=skip_descriptors,
        model_layer_configs=model_layer_configs,
    )
    _layer_init_contexts.set((*layer_init_contexts, context))
    _patch_layer_init(heterogeneous_modeling_spec.layer_cls)


def support_generic_heterogeneous_modeling(orig_init: Callable[..., None]) -> Callable[..., None]:
    """Create the model-initialization scope required by ``apply_generic_heterogeneous_modeling_if_applicable``.

    That function runs inside ``PreTrainedModel.__init__`` and registers temporary state that is used later, when the
    model subclass creates its layers. This wrapper keeps that state available across the model's
    ``super().__init__()`` chain and restores the previous state when initialization finishes. Generic heterogeneous
    modeling is marked as applied only after the root model's construction succeeds. If generic heterogeneous modeling
    is not applied, the wrapper does not change model initialization.
    """
    if getattr(orig_init, "_scoped_for_heterogeneous_modeling", False):
        return orig_init

    @wraps(orig_init)
    def _scoped_init(self, *args, **kwargs):
        initializing_models = _initializing_models.get()
        # The model's `super().__init__()` chain calls the wrapper of each class in its MRO; the first call owns
        # the scope.
        if any(model is self for model in initializing_models):
            return orig_init(self, *args, **kwargs)

        # Mark this model as initializing
        initializing_models_token = _initializing_models.set((*initializing_models, self))
        # Setting the current value gives us a token to restore it after initialization.
        layer_init_contexts_token = _layer_init_contexts.set(_layer_init_contexts.get())
        try:
            result = orig_init(self, *args, **kwargs)
        except BaseException:
            # Initialization failed, so discard the layer init contexts registered during it
            _layer_init_contexts.reset(layer_init_contexts_token)
            raise
        finally:
            _initializing_models.reset(initializing_models_token)

        if initializing_models:
            # Models containing this one are still initializing
            return result
        try:
            layer_init_contexts = _layer_init_contexts.get()
            # Validate all contexts before publishing any, so that a failed initialization publishes nothing
            for context in layer_init_contexts:
                _validate_layer_configs_collected(context)
            # The root model completed initialization, so now we can set the models' layer configs on their configs
            for context in layer_init_contexts:
                context.model.config._heterogeneity_spec.model_layer_configs = context.model_layer_configs
        finally:
            _layer_init_contexts.reset(layer_init_contexts_token)
        return result

    _scoped_init._scoped_for_heterogeneous_modeling = True
    return _scoped_init


def _patch_layer_init(layer_cls: type[nn.Module]) -> None:
    """Patch ``layer_cls.__init__`` to resolve each layer's index and pass its matching per-layer config to the
    original init function."""
    if getattr(layer_cls.__init__, "_heterogeneity_layer_cls", None) is layer_cls:
        return

    with _layer_patching_lock:
        if getattr(layer_cls.__init__, "_heterogeneity_layer_cls", None) is layer_cls:
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
            layer_config = context.model_layer_configs.get(layer_idx)
            if layer_config is None:
                layer_config = config.per_layer_config[layer_idx]
            orig_layer_init(self, layer_config, *args, **kwargs)

            # --- Replace skipped sublayers ---
            for skip_type in layer_config.skip:
                _apply_skip_descriptor(
                    layer=self,
                    skip_type=skip_type,
                    skip_descriptor=context.skip_descriptors[skip_type],
                    layer_idx=layer_idx,
                )

            # --- Register attention mask selection forward pre-hook ---
            _register_layer_attention_mask_selection_hook(layer=self, layer_idx=layer_idx)

            context.model_layer_configs[layer_idx] = layer_config

        _patched_layer_init._heterogeneity_layer_cls = layer_cls
        layer_cls.__init__ = _patched_layer_init


def _register_layer_attention_mask_selection_hook(
    *,
    layer: nn.Module,
    layer_idx: int,
) -> None:
    positional_names = [
        name
        for name, parameter in inspect.signature(layer.forward).parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    mask_position = positional_names.index("attention_mask") if "attention_mask" in positional_names else None
    layer.register_forward_pre_hook(
        partial(_select_attention_mask_by_layer_idx, layer_idx=layer_idx, mask_position=mask_position),
        with_kwargs=True,
    )


def _select_attention_mask_by_layer_idx(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    layer_idx: int,
    mask_position: int | None,
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    if mask_position is not None and mask_position < len(args):
        # Mask is positionally passed
        attention_mask = args[mask_position]
        if isinstance(attention_mask, AttentionMasksByLayerIdx):
            return (*args[:mask_position], attention_mask[layer_idx], *args[mask_position + 1 :]), kwargs
    else:
        # Mask is keyword-passed
        attention_mask = kwargs.get("attention_mask")
        if isinstance(attention_mask, AttentionMasksByLayerIdx):
            kwargs["attention_mask"] = attention_mask[layer_idx]
            return args, kwargs


def _validate_layer_configs_collected(context: _LayerInitContext) -> None:
    if not context.model_layer_configs and context.model.config.num_hidden_layers > 0:
        layer_cls_name = context.layer_cls.__name__
        raise ValueError(
            f"The heterogeneous modeling spec of {type(context.model).__name__} targets `{layer_cls_name}`, but no "
            f"`{layer_cls_name}` layer was constructed with the model's config. Check that the spec's `layer_cls` is "
            "the layer class the model constructs, and that the layers receive the model's config object itself, not a "
            "copy."
        )


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
    skip_type: str,
    skip_descriptor: SkipDescriptor,
    layer_idx: int,
) -> None:
    """Apply the selected skip replacements."""
    generic_targets = {}
    class_specific_targets = {}
    targeted_members = set()

    for key, replacement_factory in skip_descriptor.items():
        if isinstance(key, tuple):
            member_name, cls = key
        else:
            member_name = key
            cls = None

        if not _hasattr_by_path(layer, member_name):
            raise AttributeError(
                f"Layer {layer_idx} skips '{skip_type}', but class {layer.__class__.__name__} "
                f"has no attribute '{member_name}'."
            )

        targeted_members.add(member_name)
        if cls is None:
            generic_targets[member_name] = replacement_factory
            continue

        if not isinstance(_getattr_by_path(layer, member_name), cls):
            continue

        if member_name in class_specific_targets:
            raise ValueError(
                f"Layer {layer_idx} skips '{skip_type}', but multiple class-specific replacements "
                f"match member '{member_name}' in class {layer.__class__.__name__}."
            )
        class_specific_targets[member_name] = replacement_factory

    selected_targets = generic_targets | class_specific_targets
    for member_name in targeted_members:
        if member_name not in selected_targets:
            member_cls = type(_getattr_by_path(layer, member_name))
            raise ValueError(
                f"Layer {layer_idx} skips '{skip_type}', but that descriptor has no replacement for member "
                f"'{member_name}' with class {member_cls.__name__}. Add a class-specific or generic replacement."
            )

    for member_name, replacement_factory in selected_targets.items():
        original = _getattr_by_path(layer, member_name)
        replacement = replacement_factory()
        replacement._heterogeneity_skipped_class = type(original)
        _setattr_by_path(layer, member_name, replacement)


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
