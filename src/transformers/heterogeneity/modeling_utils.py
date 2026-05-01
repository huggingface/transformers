from __future__ import annotations

import contextvars
import inspect
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from torch import nn

    from transformers import PreTrainedModel

_LAYER_IDX_POSSIBLE_NAMES = ("layer_idx", "idx", "layer_id", "layer_number", "i", "_")


@dataclass
class _LayerInitContext:
    skip_types_per_layer: dict[int, list[str]]
    skip_descriptors: dict
    per_layer_attributes: set[str]
    layer_idx_variable_name: str | None


_layer_init_context: contextvars.ContextVar[_LayerInitContext | None] = contextvars.ContextVar(
    "_layer_init_context", default=None
)
_layer_patching_lock = threading.Lock()


def apply_heterogeneous_modeling(model: PreTrainedModel) -> None:
    """Apply the per-layer configurations during model construction.

    Called automatically during ``PreTrainedModel.__init__`` when
    ``config.is_heterogeneous`` is ``True``.

    The mechanism monkey-patches ``layer_cls.__init__`` and stores the per-model
    context in a ``ContextVar``.  The wrapper reads from the ``ContextVar``
    at layer-construction time, so each thread/model naturally gets its own
    context with no shared mutable state.

    1. The patched ``layer_cls.__init__`` determines the current layer index (from the function
       arguments or by walking the call stack).
    2. It calls ``config.get_full_layer_config(layer_idx)`` to merge the
       global config with the per-layer overrides set up by
       ``apply_heterogeneous_config``.
    3. It passes the resolved config to the original ``__init__``.
    4. For layers with ``skip_<type>`` attributes, the corresponding
       sub-modules are replaced with no-op modules according to the model's
       ``_skip_descriptors``.

    After model construction, ``clean_up_post_heterogeneous_modeling``
    resets the ``ContextVar``.

    The model is expected to define:
        ``_layer_cls``: The layer class to patch (e.g., ``LlamaDecoderLayer``).
        ``_skip_descriptors`` (optional): A dict mapping skip type names to
            dicts of ``{member_name_or_(name, class): ReplacementModule}``.
        ``_layer_idx_variable_name`` (optional): The name of the layer-index
            argument in ``layer_cls.__init__``, if not one of the common
            defaults (``layer_idx``, ``idx``, ``layer_id``, etc.).

    Args:
        model: The model being constructed. Must have ``_layer_cls`` set and
            a heterogeneous ``config``.
    """
    layer_cls = getattr(model, "_layer_cls", None)
    if layer_cls is None:
        raise ValueError("Layer class is not set. Please set it by setting the `_layer_cls` attribute on the model.")

    if _layer_init_context.get() is not None:
        return

    skip_types_per_layer = {
        layer_idx: [attr.removeprefix("skip_") for attr in layer_config.attributes if attr.startswith("skip_")]
        for layer_idx, layer_config in model.config.per_layer_config.items()
    }
    skip_descriptors = getattr(model, "_skip_descriptors", None) or {}
    _validate_skip_descriptors(skip_types_per_layer, skip_descriptors)

    ctx = _LayerInitContext(
        skip_types_per_layer=skip_types_per_layer,
        skip_descriptors=skip_descriptors,
        per_layer_attributes=model.config.per_layer_attributes,
        layer_idx_variable_name=getattr(model, "_layer_idx_variable_name", None),
    )
    model._layer_init_context_token = _layer_init_context.set(ctx)

    _patch_layer_init(layer_cls)


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
            layer_config = config.get_full_layer_config(layer_idx)
            orig_layer_init(self, layer_config, *args, **kwargs)

            # --- Replace skipped sublayers ---
            for skip_type, skip_descriptor in ctx.skip_descriptors.items():
                if skip_type in ctx.skip_types_per_layer.get(layer_idx, []):
                    _apply_skip_descriptor(
                        layer=self,
                        skip_descriptor=skip_descriptor,
                        layer_idx=layer_idx,
                    )

            # --- Patch forward for heterogeneous masks ---
            sliding_window = getattr(layer_config, "sliding_window", None)
            attention_chunk_size = getattr(layer_config, "attention_chunk_size", None)
            mask_key = (
                sliding_window or attention_chunk_size
            )  # Relies on having exclusivity validation in the heterogeneous configuration_utils
            if {"sliding_window", "attention_chunk_size"} & ctx.per_layer_attributes and mask_key:
                _patch_layer_forward_for_heterogeneous_masks(layer=self, mask_key=mask_key)

        _patched_layer_init._patched_by_heterogeneity = True
        layer_cls.__init__ = _patched_layer_init


def _patch_layer_forward_for_heterogeneous_masks(
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


def _validate_skip_descriptors(skip_types_per_layer: dict[int, list[str]], skip_descriptors: dict) -> None:
    skip_types = {skip_type for skip_types in skip_types_per_layer.values() for skip_type in skip_types}
    missing_descriptors = skip_types - skip_descriptors.keys()
    if missing_descriptors:
        raise ValueError(f"No-op descriptors are missing for the following types: {missing_descriptors}")


def _apply_skip_descriptor(
    *,
    layer,
    skip_descriptor: dict[str | tuple[str, type], type[nn.Module]],
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


@dataclass
class ReturnEntry:
    arg_name: str
    transform: Callable


def get_skip_replacement(
    cls: type[nn.Module],
    to_return: ReturnEntry | list[ReturnEntry | None] | None,
) -> type[nn.Module]:
    import torch
    from torch import nn

    class NoOpReplacement(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.empty(0), persistent=False)

        def forward(self, *args, **kwargs):
            if to_return is None:
                return None

            if isinstance(to_return, ReturnEntry):
                local_to_return = [to_return]
                return_tuple = False
            else:
                local_to_return = to_return
                return_tuple = True

            sig = inspect.signature(cls.forward)
            try:
                bound_arguments = sig.bind(self, *args, **kwargs)
            except TypeError as e:
                raise TypeError(f"{cls.__qualname__}.forward() {e}") from None
            bound_arguments.apply_defaults()
            outputs = [None] * len(local_to_return)
            missing_names = []
            for i, return_entry in enumerate(local_to_return):
                if return_entry is None:
                    outputs[i] = None
                    continue

                if return_entry.arg_name not in bound_arguments.arguments:
                    missing_names.append(return_entry.arg_name)
                    continue

                try:
                    outputs[i] = return_entry.transform(bound_arguments.arguments[return_entry.arg_name])
                except Exception as e:
                    arg_value = bound_arguments.arguments[return_entry.arg_name]
                    raise type(e)(
                        f"In the skip replacement for {cls.__qualname__}, failed to apply transform "
                        f"{return_entry.transform!r} to argument '{return_entry.arg_name}' "
                        f"(value type: {type(arg_value).__name__}): {e}"
                    ) from e

            if missing_names:
                raise ValueError(
                    f"In the skip replacement for {cls.__qualname__}, the following return entry arg names "
                    f"are not parameters of {cls.__qualname__}.forward(): {missing_names}"
                )

            return tuple(outputs) if return_tuple else outputs[0]

    return NoOpReplacement
