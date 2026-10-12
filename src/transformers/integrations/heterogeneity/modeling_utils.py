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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial, wraps
from typing import TYPE_CHECKING, Any

from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import (
    HeterogeneousModelingSpec,
    MemberKey,
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
    heterogeneous_modeling_spec: HeterogeneousModelingSpec
    model_layer_configs: dict[int, PreTrainedConfig]
    unconsumed_attention_mask_layers: set[int]
    per_layer_unwritten_caches: dict[int, set[str]]


_layer_init_contexts: contextvars.ContextVar[tuple[_LayerInitContext, ...]] = contextvars.ContextVar(
    "_layer_init_contexts", default=()
)
_initializing_models: contextvars.ContextVar[tuple[PreTrainedModel, ...]] = contextvars.ContextVar(
    "_initializing_models", default=()
)
_layer_patching_lock = threading.Lock()


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


def _get_member_path(member_key: MemberKey) -> str:
    return member_key[0] if isinstance(member_key, tuple) else member_key


def _get_key_rank(member_key: MemberKey, member_mro: tuple[type, ...]) -> int:
    # More specific keys rank lower: a (path, class) key ranks by the position of its class in the member's MRO, and a
    # plain path ranks after every class
    return member_mro.index(member_key[1]) if isinstance(member_key, tuple) else len(member_mro)


def _get_matching_member_keys(
    layer: nn.Module, member_keys: Iterable[MemberKey], *, error_prefix: str
) -> list[MemberKey]:
    """Return the keys in `member_keys` that match members of `layer`, at most one per member path. A (path, class)
    key matches only when the member is an instance of that class, and takes precedence over the plain path. Among
    matching (path, class) keys, the one with the most specific class takes precedence."""
    member_path_to_key = {}
    for member_key in member_keys:
        member_path = _get_member_path(member_key)
        if not _hasattr_by_path(layer, member_path):
            raise AttributeError(
                f"{error_prefix}, but class {layer.__class__.__name__} has no attribute '{member_path}'."
            )

        member_mro = type(_getattr_by_path(layer, member_path)).__mro__
        if isinstance(member_key, tuple) and member_key[1] not in member_mro:
            continue
        matched_key = member_path_to_key.get(member_path)
        if matched_key is None or _get_key_rank(member_key, member_mro) < _get_key_rank(matched_key, member_mro):
            member_path_to_key[member_path] = member_key
    return list(member_path_to_key.values())


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
    """Replace the layer's members with the replacements of the matching keys in `skip_descriptor`."""
    error_prefix = f"Layer {layer_idx} skips '{skip_type}'"
    matching_member_keys = _get_matching_member_keys(layer, skip_descriptor, error_prefix=error_prefix)
    matched_member_paths = {_get_member_path(member_key) for member_key in matching_member_keys}

    # Before replacing anything, check that every member path in the descriptor has a matching key. Otherwise the skip
    # would only partly apply, and the layer would silently compute an unintended output.
    for member_key in skip_descriptor:
        member_path = _get_member_path(member_key)
        if member_path not in matched_member_paths:
            member_cls = type(_getattr_by_path(layer, member_path))
            raise ValueError(
                f"{error_prefix}, but that descriptor has no replacement for member '{member_path}' with class "
                f"{member_cls.__name__}. Add a class-specific or generic replacement."
            )

    for member_key in matching_member_keys:
        member_path = _get_member_path(member_key)
        original = _getattr_by_path(layer, member_path)
        replacement_factory = skip_descriptor[member_key]
        replacement = replacement_factory()
        replacement._heterogeneity_skipped_class = type(original)
        _setattr_by_path(layer, member_path, replacement)


def _apply_skips(*, layer: nn.Module, layer_idx: int, skip_types: list[str], context: _LayerInitContext) -> None:
    """Apply the layer's skips, and keep track in `context` of whether they leave its attention mask unconsumed and
    which of its caches they leave unwritten."""
    heterogeneous_modeling_spec = context.heterogeneous_modeling_spec
    # Find the declared members before the skips replace them
    attention_mask_consumers = [
        _getattr_by_path(layer, _get_member_path(member_key))
        for member_key in _get_matching_member_keys(
            layer,
            heterogeneous_modeling_spec.attention_mask_consumers,
            error_prefix=f"The spec's `attention_mask_consumers` lists a member of layer {layer_idx}",
        )
    ]
    cache_writers = {
        _getattr_by_path(layer, _get_member_path(member_key)): heterogeneous_modeling_spec.cache_writers[member_key]
        for member_key in _get_matching_member_keys(
            layer,
            heterogeneous_modeling_spec.cache_writers,
            error_prefix=f"The spec's `cache_writers` lists a member of layer {layer_idx}",
        )
    }

    for skip_type in skip_types:
        _apply_skip_descriptor(
            layer=layer,
            skip_type=skip_type,
            skip_descriptor=heterogeneous_modeling_spec.skip_descriptors[skip_type],
            layer_idx=layer_idx,
        )

    remaining_modules = set(layer.modules())
    if attention_mask_consumers and remaining_modules.isdisjoint(attention_mask_consumers):
        context.unconsumed_attention_mask_layers.add(layer_idx)

    declared_caches = set()
    written_caches = set()
    for writer, caches in cache_writers.items():
        declared_caches |= caches
        if writer in remaining_modules:
            written_caches |= caches
    if unwritten_caches := declared_caches - written_caches:
        context.per_layer_unwritten_caches[layer_idx] = unwritten_caches


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
                    if context.heterogeneous_modeling_spec.layer_cls is layer_cls and context.model.config is config
                ),
                None,
            )
            if context is None or not getattr(config, "is_heterogeneous", False):
                return orig_layer_init(self, config, *args, **kwargs)
            heterogeneous_modeling_spec = context.heterogeneous_modeling_spec

            # --- Resolve layer index ---
            layer_idx = heterogeneous_modeling_spec.layer_idx_resolver.resolve(
                layer_init=orig_layer_init,
                args=(self, config, *args),
                kwargs=kwargs,
                model=context.model,
            )
            _validate_layer_idx(
                layer_idx,
                resolver=heterogeneous_modeling_spec.layer_idx_resolver,
                num_layers=config.num_hidden_layers,
            )

            # --- Apply per-layer config ---
            layer_config = context.model_layer_configs.get(layer_idx)
            if layer_config is None:
                layer_config = config.per_layer_config[layer_idx]
            orig_layer_init(self, layer_config, *args, **kwargs)

            # --- Replace skipped sublayers ---
            if layer_config.skip:
                _apply_skips(layer=self, layer_idx=layer_idx, skip_types=layer_config.skip, context=context)

            # --- Register attention mask selection forward pre-hook ---
            _register_layer_attention_mask_selection_hook(layer=self, layer_idx=layer_idx)

            context.model_layer_configs[layer_idx] = layer_config

        _patched_layer_init._heterogeneity_layer_cls = layer_cls
        layer_cls.__init__ = _patched_layer_init


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

    if model.config.generic_heterogeneous_modeling_applied:
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
    _validate_skip_descriptors(per_layer_skip_types, heterogeneous_modeling_spec.skip_descriptors)

    layer_init_contexts = _layer_init_contexts.get()
    same_config_context = next(
        (context for context in layer_init_contexts if context.model.config is model.config),
        None,
    )
    context = _LayerInitContext(
        model=model,
        heterogeneous_modeling_spec=heterogeneous_modeling_spec,
        model_layer_configs=same_config_context.model_layer_configs if same_config_context else {},
        unconsumed_attention_mask_layers=(
            same_config_context.unconsumed_attention_mask_layers if same_config_context else set()
        ),
        per_layer_unwritten_caches=(same_config_context.per_layer_unwritten_caches if same_config_context else {}),
    )
    _layer_init_contexts.set((*layer_init_contexts, context))
    _patch_layer_init(heterogeneous_modeling_spec.layer_cls)


def _validate_layer_configs_collected(context: _LayerInitContext) -> None:
    if not context.model_layer_configs and context.model.config.num_hidden_layers > 0:
        layer_cls_name = context.heterogeneous_modeling_spec.layer_cls.__name__
        raise ValueError(
            f"The heterogeneous modeling spec of {type(context.model).__name__} targets `{layer_cls_name}`, but no "
            f"`{layer_cls_name}` layer was constructed with the model's config. Check that the spec's `layer_cls` is "
            "the layer class the model constructs, and that the layers receive the model's config object itself, not a "
            "copy."
        )


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
            # The root model completed initialization, so now we can set what was collected for the models on their
            # configs
            for context in layer_init_contexts:
                heterogeneity_spec = context.model.config._heterogeneity_spec
                heterogeneity_spec.model_layer_configs = context.model_layer_configs
                heterogeneity_spec.unconsumed_attention_mask_layers = context.unconsumed_attention_mask_layers
                heterogeneity_spec.per_layer_unwritten_caches = context.per_layer_unwritten_caches
        finally:
            _layer_init_contexts.reset(layer_init_contexts_token)
        return result

    _scoped_init._scoped_for_heterogeneous_modeling = True
    return _scoped_init
