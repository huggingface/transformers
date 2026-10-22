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

from transformers.integrations.heterogeneity.cache_utils import validate_no_unwritten_caches
from transformers.integrations.heterogeneity.configuration_utils import _GenericModelingState
from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import (
    HeterogeneousModelingSpec,
    get_heterogeneous_modeling_specs,
)
from transformers.integrations.heterogeneity.layer_idx_resolvers import LayerIdxResolver
from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByLayerIdx
from transformers.integrations.heterogeneity.skip_utils import apply_skips, validate_skip_descriptors


if TYPE_CHECKING:
    from torch import nn

    from transformers import PreTrainedModel


@dataclass(frozen=True)
class _LayerInitContext:
    model: PreTrainedModel
    heterogeneous_modeling_specs: dict[type[nn.Module], HeterogeneousModelingSpec]

    # Shared by the contexts of models built from the same config:
    generic_modeling_state: _GenericModelingState
    built_layer_indices: set[int]


_layer_init_contexts: contextvars.ContextVar[tuple[_LayerInitContext, ...]] = contextvars.ContextVar(
    "_layer_init_contexts", default=()
)
_initializing_models: contextvars.ContextVar[tuple[PreTrainedModel, ...]] = contextvars.ContextVar(
    "_initializing_models", default=()
)
_layer_patching_lock = threading.Lock()


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


def _validate_layer_cache_entry(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    layer_idx: int,
    unwritten_caches: set[str],
    past_key_values_position: int | None,
) -> None:
    if past_key_values_position is not None and past_key_values_position < len(args):
        past_key_values = args[past_key_values_position]
    else:
        past_key_values = kwargs.get("past_key_values")
    if past_key_values is not None:
        validate_no_unwritten_caches(past_key_values, layer_idx=layer_idx, unwritten_caches=unwritten_caches)


def _get_argument_position(function: Callable, argument_name: str) -> int | None:
    """Return the position of `argument_name` in the positional arguments of `function`, or `None` if it can't be
    passed positionally. For a method, pass the bound method, so that `self` isn't counted."""
    positional_names = [
        name
        for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return positional_names.index(argument_name) if argument_name in positional_names else None


def _register_layer_attention_mask_selection_hook(
    *,
    layer: nn.Module,
    layer_idx: int,
) -> None:
    layer.register_forward_pre_hook(
        partial(
            _select_attention_mask_by_layer_idx,
            layer_idx=layer_idx,
            mask_position=_get_argument_position(layer.forward, "attention_mask"),
        ),
        with_kwargs=True,
    )


def _register_layer_cache_validation_hook(*, layer: nn.Module, layer_idx: int, unwritten_caches: set[str]) -> None:
    layer.register_forward_pre_hook(
        partial(
            _validate_layer_cache_entry,
            layer_idx=layer_idx,
            unwritten_caches=unwritten_caches,
            past_key_values_position=_get_argument_position(layer.forward, "past_key_values"),
        ),
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
                    if layer_cls in context.heterogeneous_modeling_specs and context.model.config is config
                ),
                None,
            )
            if context is None or not getattr(config, "is_heterogeneous", False):
                return orig_layer_init(self, config, *args, **kwargs)
            heterogeneous_modeling_spec = context.heterogeneous_modeling_specs[layer_cls]

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
            layer_config = config.per_layer_config[layer_idx]
            orig_layer_init(self, layer_config, *args, **kwargs)

            # --- Replace skipped sublayers ---
            if layer_config.skip:
                apply_skips(
                    layer=self,
                    layer_idx=layer_idx,
                    skip_types=layer_config.skip,
                    heterogeneous_modeling_spec=heterogeneous_modeling_spec,
                    generic_modeling_state=context.generic_modeling_state,
                )

            # --- Register forward pre-hooks ---
            _register_layer_attention_mask_selection_hook(layer=self, layer_idx=layer_idx)
            if unwritten_caches := context.generic_modeling_state.per_layer_unwritten_caches.get(layer_idx):
                _register_layer_cache_validation_hook(
                    layer=self, layer_idx=layer_idx, unwritten_caches=unwritten_caches
                )

            context.built_layer_indices.add(layer_idx)

        _patched_layer_init._heterogeneity_layer_cls = layer_cls
        layer_cls.__init__ = _patched_layer_init


def apply_generic_heterogeneous_modeling_if_applicable(model: PreTrainedModel) -> None:
    """Apply heterogeneous per-layer modeling during model initialization.

    This function resolves the model's ``HeterogeneousModelingSpec``s, validates its
    configured skips, and registers the layer-initialization context. The patched layer classes use this context to
    initialize each layer with its resolved config, apply skip replacements, and
    select layer-specific attention masks.

    Args:
        model: The model being initialized.
    """
    if not model.config.is_heterogeneous:
        return

    heterogeneous_modeling_specs = get_heterogeneous_modeling_specs(model)
    if not heterogeneous_modeling_specs:
        return

    per_layer_skip_types = [layer_config.skip for layer_config in model.config.per_layer_config]
    validate_skip_descriptors(per_layer_skip_types, heterogeneous_modeling_specs)

    layer_init_contexts = _layer_init_contexts.get()
    same_config_context = next(
        (context for context in layer_init_contexts if context.model.config is model.config),
        None,
    )
    context = _LayerInitContext(
        model=model,
        heterogeneous_modeling_specs={spec.layer_cls: spec for spec in heterogeneous_modeling_specs},
        generic_modeling_state=(
            same_config_context.generic_modeling_state if same_config_context else _GenericModelingState()
        ),
        built_layer_indices=same_config_context.built_layer_indices if same_config_context else set(),
    )
    _layer_init_contexts.set((*layer_init_contexts, context))
    for heterogeneous_modeling_spec in heterogeneous_modeling_specs:
        _patch_layer_init(heterogeneous_modeling_spec.layer_cls)


def _validate_all_layers_built(context: _LayerInitContext) -> None:
    missing_layer_indices = sorted(set(range(context.model.config.num_hidden_layers)) - context.built_layer_indices)
    if missing_layer_indices:
        layer_cls_names = " or ".join(f"`{layer_cls.__name__}`" for layer_cls in context.heterogeneous_modeling_specs)
        raise ValueError(
            f"The heterogeneous modeling specs of {type(context.model).__name__} target {layer_cls_names}, but no "
            f"such layer was constructed with the model's config for layers {missing_layer_indices}. Check that the "
            "model has a spec for every layer class it constructs, and that the layers receive the model's config "
            "object itself, not a copy."
        )


def _validate_matches_existing_modeling_state(context: _LayerInitContext) -> None:
    """Raise if the config already has a generic modeling state, from building another model, that differs from
    this build's.

    Mask and cache creation read the state from the config, not from the model, so all models built with the config
    must have the same state."""
    existing_state = context.model.config._heterogeneity_spec.generic_modeling_state
    # Layer configs belong to the config, so the states can only differ in what the skips left behind
    if existing_state is not None and context.generic_modeling_state != existing_state:
        raise ValueError(
            f"The config of {type(context.model).__name__} was already used to build a model with generic "
            "heterogeneous modeling, and the two models differ in which layers are left without attention mask "
            "consumers or cache writers. Models that share a config must agree on these, because masks and caches are "
            "built from the config. Use a separate config, for example `type(config).from_dict(config.to_dict())`."
        )


def support_generic_heterogeneous_modeling(orig_init: Callable[..., None]) -> Callable[..., None]:
    """Create the model-initialization scope required by ``apply_generic_heterogeneous_modeling_if_applicable``.

    That function runs inside ``PreTrainedModel.__init__`` and registers temporary state that is used later, when the
    model subclass creates its layers. This wrapper keeps that state available across the model's
    ``super().__init__()`` chain and restores the previous state when initialization finishes. Generic heterogeneous
    modeling is marked as applied only after the root model's construction succeeds. If generic heterogeneous modeling
    is not applied, the wrapper does not change model initialization.

    The wrapper is installed on every subclass through ``__init_subclass__`` because any of them, including a user's,
    can be the root model, and no other hook runs when the root model's ``__init__`` returns or raises. ``post_init``
    runs at the end of each nested model's ``__init__``, such as ``LlamaModel`` inside ``LlamaForCausalLM``, and not
    at all on failure.
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
                _validate_all_layers_built(context)
                _validate_matches_existing_modeling_state(context)
            # The root model completed initialization, so now we can set what was collected for the models on their
            # configs
            for context in layer_init_contexts:
                context.model.config._heterogeneity_spec.generic_modeling_state = context.generic_modeling_state
        finally:
            _layer_init_contexts.reset(layer_init_contexts_token)
        return result

    _scoped_init._scoped_for_heterogeneous_modeling = True
    return _scoped_init
