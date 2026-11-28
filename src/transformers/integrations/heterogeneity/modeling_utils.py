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

import inspect
import threading
from functools import partial, wraps
from typing import TYPE_CHECKING, Any

from transformers.integrations.heterogeneity.cache_utils import validate_no_unwritten_caches
from transformers.integrations.heterogeneity.configuration_utils import _GenericModelingContext
from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import get_heterogeneous_modeling_specs
from transformers.integrations.heterogeneity.layer_idx_resolvers import LayerIdxResolver
from transformers.integrations.heterogeneity.masking_utils import AttentionMasksByLayerIdx
from transformers.integrations.heterogeneity.skip_utils import apply_skips, validate_skip_descriptors


if TYPE_CHECKING:
    from torch import nn

    from transformers import PreTrainedConfig, PreTrainedModel


_layer_patching_lock = threading.Lock()


def _get_layer_attention_mask(
    attention_masks: AttentionMasksByLayerIdx, *, config: PreTrainedConfig, layer_idx: int
) -> Any:
    if layer_idx not in attention_masks:
        raise ValueError(
            f"Layer {layer_idx} has no attention mask, because the model doesn't create attention masks for layers "
            f"of type '{config.layer_types[layer_idx]}'."
        )
    return attention_masks[layer_idx]


def _select_attention_masks_by_layer_idx(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    config: PreTrainedConfig,
    layer_idx: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    # A layer can get several masks, so every argument is checked
    get_layer_mask = partial(_get_layer_attention_mask, config=config, layer_idx=layer_idx)
    args = tuple(get_layer_mask(arg) if isinstance(arg, AttentionMasksByLayerIdx) else arg for arg in args)
    kwargs = {
        name: get_layer_mask(value) if isinstance(value, AttentionMasksByLayerIdx) else value
        for name, value in kwargs.items()
    }
    return args, kwargs


def _validate_layer_cache_entry(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    layer_idx: int,
    unwritten_caches: set[str],
) -> None:
    # Imported here because the core `cache_utils` module imports this package
    from transformers.cache_utils import Cache

    cache = next((argument for argument in (*args, *kwargs.values()) if isinstance(argument, Cache)), None)
    if cache is not None:
        validate_no_unwritten_caches(cache, layer_idx=layer_idx, unwritten_caches=unwritten_caches)


def _register_layer_attention_mask_selection_hook(
    *,
    layer: nn.Module,
    config: PreTrainedConfig,
    layer_idx: int,
) -> None:
    layer.register_forward_pre_hook(
        partial(_select_attention_masks_by_layer_idx, config=config, layer_idx=layer_idx),
        with_kwargs=True,
    )


def _register_layer_cache_validation_hook(*, layer: nn.Module, layer_idx: int, unwritten_caches: set[str]) -> None:
    layer.register_forward_pre_hook(
        partial(_validate_layer_cache_entry, layer_idx=layer_idx, unwritten_caches=unwritten_caches),
        with_kwargs=True,
    )


def _validate_layer_idx(layer_idx: Any, *, resolver: LayerIdxResolver, num_layers: int) -> None:
    if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
        raise TypeError(
            f"The layer index must be an integer, but `{resolver!r}` returned {layer_idx!r} "
            f"({type(layer_idx).__name__})."
        )
    if not 0 <= layer_idx < num_layers:
        raise IndexError(
            f"The layer index is out of range for a model with {num_layers} layers: `{resolver!r}` returned "
            f"{layer_idx}."
        )


def _patch_layer_init(layer_cls: type[nn.Module]) -> None:
    """Patch ``layer_cls.__init__`` to resolve each layer's index and pass its matching per-layer config to the
    original init function."""
    if getattr(layer_cls.__init__, "_heterogeneity_patched_layer_cls", None) is layer_cls:
        return

    with _layer_patching_lock:
        if getattr(layer_cls.__init__, "_heterogeneity_patched_layer_cls", None) is layer_cls:
            return

        orig_layer_init = layer_cls.__init__
        orig_layer_init_signature = inspect.signature(orig_layer_init)

        @wraps(orig_layer_init)
        def _patched_layer_init(self, *args, **kwargs):
            try:
                bound_arguments = orig_layer_init_signature.bind(self, *args, **kwargs)
            except TypeError:
                # The arguments don't fit `__init__`, so the original `__init__` raises its own error
                return orig_layer_init(self, *args, **kwargs)
            config = bound_arguments.arguments.get("config")

            heterogeneous_modeling_spec = None
            if getattr(config, "used_for_generic_heterogeneous_modeling", False):
                generic_modeling_context = config._heterogeneity_spec.generic_modeling_context
                heterogeneous_modeling_spec = generic_modeling_context.heterogeneous_modeling_specs.get(layer_cls)
            if heterogeneous_modeling_spec is None:
                return orig_layer_init(self, *args, **kwargs)

            # --- Resolve layer index ---
            layer_idx = heterogeneous_modeling_spec.layer_idx_resolver.resolve(
                layer_init=orig_layer_init,
                args=(self, *args),
                kwargs=kwargs,
            )
            _validate_layer_idx(
                layer_idx,
                resolver=heterogeneous_modeling_spec.layer_idx_resolver,
                num_layers=config.num_hidden_layers,
            )

            # --- Apply per-layer config ---
            layer_config = config.per_layer_config[layer_idx]
            bound_arguments.arguments["config"] = layer_config
            orig_layer_init(*bound_arguments.args, **bound_arguments.kwargs)

            # --- Replace skipped members ---
            if layer_config.skip:
                apply_skips(
                    layer=self,
                    layer_idx=layer_idx,
                    skip_types=layer_config.skip,
                    heterogeneous_modeling_spec=heterogeneous_modeling_spec,
                    generic_modeling_context=generic_modeling_context,
                )

            # --- Register forward pre-hooks ---
            _register_layer_attention_mask_selection_hook(layer=self, config=config, layer_idx=layer_idx)
            if unwritten_caches := generic_modeling_context.per_layer_unwritten_caches.get(layer_idx):
                _register_layer_cache_validation_hook(
                    layer=self, layer_idx=layer_idx, unwritten_caches=unwritten_caches
                )

        _patched_layer_init._heterogeneity_patched_layer_cls = layer_cls
        layer_cls.__init__ = _patched_layer_init


def apply_generic_heterogeneous_modeling_if_applicable(model: PreTrainedModel) -> None:
    """Apply heterogeneous per-layer modeling during model initialization.

    This function resolves the model's ``HeterogeneousModelingSpec``s, validates its
    configured skips, and sets the specs on the model's config. The patched layer classes read them from the config to
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

    # The patched layer `__init__` gets only the config, so it reads the specs there, and records there what the skips
    # of each layer removed. A model and its inner model, such as `LlamaForCausalLM` and `LlamaModel`, get the same
    # config, so they share one context.
    heterogeneity_spec = model.config._heterogeneity_spec
    if heterogeneity_spec.generic_modeling_context is None:
        heterogeneity_spec.generic_modeling_context = _GenericModelingContext()
    heterogeneity_spec.generic_modeling_context.heterogeneous_modeling_specs.update(
        {spec.layer_cls: spec for spec in heterogeneous_modeling_specs}
    )
    for heterogeneous_modeling_spec in heterogeneous_modeling_specs:
        _patch_layer_init(heterogeneous_modeling_spec.layer_cls)
