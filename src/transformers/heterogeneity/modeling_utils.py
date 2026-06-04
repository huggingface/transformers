from __future__ import annotations

import contextvars
import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any

from transformers.heterogeneity.heterogeneous_modeling_spec import SkipDescriptor, get_heterogeneous_modeling_spec


if TYPE_CHECKING:
    from torch import nn

    from transformers import PreTrainedModel

_LAYER_IDX_POSSIBLE_NAMES = ("layer_idx", "idx", "layer_id", "layer_number", "i", "_")


@dataclass
class _LayerInitContext:
    per_layer_skip_types: list[list[str]]
    skip_descriptors: dict[str, SkipDescriptor]
    per_layer_attributes: set[str]
    layer_idx_variable_name: str | None


_layer_init_context: contextvars.ContextVar[_LayerInitContext | None] = contextvars.ContextVar(
    "_layer_init_context", default=None
)
_layer_patching_lock = threading.Lock()


def apply_heterogeneous_modeling(model: PreTrainedModel) -> None:
    """Apply heterogeneous per-layer modeling during model construction.

    Called automatically during ``PreTrainedModel.__init__`` when
    ``config.is_heterogeneous`` is ``True``.

    The model must resolve to a ``HeterogeneousModelingSpec`` either by setting
    ``_heterogeneous_modeling_spec`` on the model class, or by having a built-in
    spec factory in ``transformers.heterogeneity.supported_models``.
    The spec defines the decoder layer class to patch, an optional layer-index argument name,
    and optional skip descriptors.

    The mechanism monkey-patches ``layer_cls.__init__`` and stores the per-model
    context in a ``ContextVar``.  The wrapper reads from the ``ContextVar``
    at layer-construction time, so each thread/model naturally gets its own
    context with no shared mutable state.

    1. The patched ``layer_cls.__init__`` determines the current layer index (from the function
       arguments or by walking the call stack).
    2. It passes ``config.per_layer_config[layer_idx]`` to the original ``__init__``.
    3. For layers with a ``skip`` attribute, the
       corresponding sub-modules are replaced with no-op modules according to
       ``HeterogeneousModelingSpec.skip_descriptors``.
    4. For layers with layer-specific attention masks, the layer ``forward``
       method is patched to select the mask matching that layer's configured
       mask key.

    After model construction, ``clean_up_post_heterogeneous_modeling``
    resets the ``ContextVar``.

    The resolved ``HeterogeneousModelingSpec`` contains:
        ``layer_cls``: The layer class to patch, e.g. ``LlamaDecoderLayer``.
        ``layer_idx_variable_name``: Optional name of the layer-index
            argument in ``layer_cls.__init__``, if not one of the common
            defaults (``layer_idx``, ``idx``, ``layer_id``, etc.).
        ``skip_descriptors``: Optional dict mapping skip type names to
            dicts of ``{member_name_or_(name, class): ReplacementModule}``.

    Args:
        model: The model being constructed. Must have a heterogeneous ``config``
            and a resolvable ``HeterogeneousModelingSpec`` with ``layer_cls`` set.
    """
    if _layer_init_context.get() is not None:
        return

    heterogeneous_modeling_spec = get_heterogeneous_modeling_spec(model)

    per_layer_skip_types = [layer_config.skip for layer_config in model.config.per_layer_config]
    skip_descriptors = heterogeneous_modeling_spec.skip_descriptors or {}
    _validate_skip_descriptors(per_layer_skip_types, skip_descriptors)

    ctx = _LayerInitContext(
        per_layer_skip_types=per_layer_skip_types,
        skip_descriptors=skip_descriptors,
        per_layer_attributes=model.config.per_layer_attributes,
        layer_idx_variable_name=heterogeneous_modeling_spec.layer_idx_variable_name,
    )
    model._layer_init_context_token = _layer_init_context.set(ctx)

    _patch_layer_init(heterogeneous_modeling_spec.layer_cls)


def clean_up_post_heterogeneous_modeling(model: PreTrainedModel) -> None:
    if not hasattr(model, "_layer_init_context_token"):
        return

    _layer_init_context.reset(model._layer_init_context_token)
    del model._layer_init_context_token


def _patch_layer_init(layer_cls: type[nn.Module]) -> None:
    if getattr(layer_cls.__init__, "_patched_by_heterogeneity", False):
        return
    with _layer_patching_lock:
        if getattr(layer_cls.__init__, "_patched_by_heterogeneity", False):
            return

        orig_layer_init = layer_cls.__init__

        @wraps(orig_layer_init)
        def _patched_layer_init(self, config, *args, **kwargs):
            ctx = _layer_init_context.get()
            if ctx is None or not getattr(config, "is_heterogeneous", False):
                return orig_layer_init(self, config, *args, **kwargs)

            # --- Resolve layer index ---
            layer_idx_possible_names = (
                [ctx.layer_idx_variable_name] if ctx.layer_idx_variable_name else _LAYER_IDX_POSSIBLE_NAMES
            )
            layer_idx = _get_variable_from_passed_arguments(
                func=orig_layer_init, args=(self, config, *args), kwargs=kwargs, names=layer_idx_possible_names
            )
            if layer_idx is None:
                layer_idx = _get_variable_from_stack(layer_idx_possible_names)

            if layer_idx is None:
                raise RuntimeError(
                    "Could not determine layer index for heterogeneous model initialization. "
                    "Ensure layer_idx is passed as an argument or available in the call stack."
                )

            # --- Apply per-layer config ---
            layer_config = config.per_layer_config[layer_idx]
            orig_layer_init(self, layer_config, *args, **kwargs)

            # --- Replace skipped sublayers ---
            for skip_type, skip_descriptor in ctx.skip_descriptors.items():
                if skip_type in ctx.per_layer_skip_types[layer_idx]:
                    _apply_skip_descriptor(
                        layer=self,
                        skip_descriptor=skip_descriptor,
                        layer_idx=layer_idx,
                    )

            # --- Patch forward for attention mask selection ---
            sliding_window = getattr(layer_config, "sliding_window", None)
            attention_chunk_size = getattr(layer_config, "attention_chunk_size", None)
            mask_key = (
                sliding_window or attention_chunk_size
            )  # Relies on having exclusivity validation in the heterogeneous configuration_utils
            if {"sliding_window", "attention_chunk_size"} & ctx.per_layer_attributes and mask_key:
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
        if isinstance(attention_mask, dict):
            kwargs["attention_mask"] = attention_mask[mask_key]
        return orig_forward(*args, **kwargs)

    layer.forward = MethodType(_patched_forward, layer)


def _validate_skip_descriptors(
    per_layer_skip_types: list[list[str]], skip_descriptors: dict[str, SkipDescriptor]
) -> None:
    skip_types = set(sum(per_layer_skip_types, []))
    missing_descriptors = skip_types - skip_descriptors.keys()
    if missing_descriptors:
        raise ValueError(f"No-op descriptors are missing for the following types: {missing_descriptors}")


def _apply_skip_descriptor(
    *,
    layer,
    skip_descriptor: SkipDescriptor,
    layer_idx: int,
):
    for key, replacement_module in skip_descriptor.items():
        if isinstance(key, tuple):
            member_name, cls = key
        else:
            member_name = key
            cls = None

        if not hasattr(layer, member_name):
            raise AttributeError(
                f"Layer {layer_idx} in class {layer.__class__.__name__} has no attribute {member_name}"
            )

        if cls is None or isinstance(getattr(layer, member_name), cls):
            setattr(layer, member_name, replacement_module())


def _get_variable_from_passed_arguments(*, func: Callable, args: tuple, kwargs: dict, names: list[str]) -> Any | None:
    sig = inspect.signature(func)
    try:
        bound_arguments = sig.bind(*args, **kwargs)
    except TypeError as e:
        raise TypeError(f"{func.__qualname__}() {e}") from None
    bound_arguments.apply_defaults()

    for name in names:
        if name in bound_arguments.arguments:
            return bound_arguments.arguments[name]
    return None


def _get_variable_from_stack(names: list[str]) -> Any:
    f = inspect.currentframe().f_back
    while f:
        for name in names:
            if name in f.f_locals:
                return f.f_locals[name]
        f = f.f_back
    return None
