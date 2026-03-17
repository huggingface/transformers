# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Shared export utilities used by multiple exporter backends.

This module contains helpers that are backend-agnostic and used by more than
one exporter (Dynamo, ONNX, ExecuTorch):

- `get_leaf_tensors`: recursively extract all leaf tensors from nested outputs.
- `prepare_for_export`: configure model config, attention/experts implementations,
  and patch non-exportable module behaviours before any export.
- `decompose_prefill_decode`: run ``model.generate()`` and capture the forward kwargs
  for the prefill and decode steps, returning ``{"prefill": ..., "decode": ...}``.
- `decompose_encoder_decoder`: capture inputs to every known submodule (vision tower,
  projector, language model, lm_head, encoder, decoder, …) via a single forward pass,
  returning one ``(name, module, inputs)`` triplet per component for independent export.
"""

from __future__ import annotations

import contextlib
import copy
import enum
import functools
import inspect
from contextlib import contextmanager
from typing import Any

from ..modeling_utils import PreTrainedModel
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch


# Output flags that should be set on model.config, not passed as forward() kwargs.
_OUTPUT_FLAGS = ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss")


# Types that should not be recursed into when extracting leaf tensors.
# Sym* types carry PyTorch shape_env internals that cause infinite recursion;
# Enums are scalars with no tensor fields.
_LEAF_SKIP_TYPES = (type, enum.Enum, torch.SymInt, torch.SymFloat, torch.SymBool)


# ── Leaf tensors ────────────────────────────────────────────────────────────


def _iter_leaf_tensors(obj: Any, prefix: str = "", default: str = "output"):
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return
    if isinstance(obj, torch.Tensor):
        yield prefix or default, obj
    elif isinstance(obj, (list, tuple, set)):
        for index, item in enumerate(obj):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_leaf_tensors(item, path, default)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from _iter_leaf_tensors(value, path, default)
    elif hasattr(obj, "__dict__"):
        yield from _iter_leaf_tensors(vars(obj), prefix, default)


def get_leaf_tensors(obj: Any, default: str = "output") -> dict[str, torch.Tensor]:
    """Recursively retrieves all leaf tensors from a potentially nested structure."""
    return dict(_iter_leaf_tensors(obj, default=default))


# ── Model patching ──────────────────────────────────────────────────────────
# Backend-agnostic patches applied by prepare_for_export before any export.


def _exportable_update_mask(attention_mask, past_key_values_or_cache_position=None, *args, **kwargs):
    """Export-safe replacement for `_update_mamba_mask` / `_update_linear_attn_mask`.

    The original functions return ``None`` in two cases:
      1. Decode step — ``past_key_values.has_previous_state`` is True, or ``cache_position[0] > 0``
      2. No padding — ``torch.all(attention_mask == 1)``

    Both cases are problematic for ``torch.export``: case 2 uses ``torch.all`` (data-dependent),
    and case 1 with ``cache_position`` (falcon_h1) indexes a tensor value.
    This replacement keeps only the ``has_previous_state`` check (a Python bool, constant at
    trace time). Models that pass ``cache_position`` instead (falcon_h1) fall through to
    returning the attention_mask as-is.
    """
    if getattr(past_key_values_or_cache_position, "has_previous_state", False):
        return None
    return attention_mask


def prepare_for_export(
    model: PreTrainedModel | torch.nn.Module, inputs: dict[str, Any]
) -> tuple[PreTrainedModel | torch.nn.Module, dict[str, Any]]:
    """Configure the model for export. Mutates both ``model`` and ``inputs`` in-place:

    - Sets optimal attention/experts implementations.
    - Patches non-exportable module behaviours (mamba masks, classifier casts, …).
    - Strips label inputs (``labels``, ``future_values``) — loss computation is unsupported.
    - Strips output flags (``use_cache``, ``return_dict``, …) from inputs and bakes non-None
      values into ``model.forward`` via ``functools.partial`` so they are constant at trace time.
    """
    # Strip label inputs — loss computation is not supported during export.
    for label_key in ("labels", "future_values"):
        value = inputs.pop(label_key, None)
        if value is not None:
            raise ValueError(
                f"Found '{label_key}' in inputs. Loss computation is not supported during export. "
                f"Please remove '{label_key}' from your inputs before calling export()."
            )
    if hasattr(model, "config") and getattr(model.config, "return_loss", False):
        raise ValueError(
            "Found 'model.config.return_loss=True'. Loss computation is not supported during export. "
            "Please set 'model.config.return_loss=False' before calling export()."
        )
    if inputs.get("return_loss", False):
        raise ValueError(
            "Found 'return_loss=True' in inputs. Loss computation is not supported during export. "
            "Please remove 'return_loss' from your inputs or set it to False."
        )

    # Strip output flags from inputs. Set on config when possible, otherwise bake into
    # the forward via functools.partial so the value is constant at trace time.
    # Submodule captures often inject these from the parent model's forward; they must not
    # appear as traced kwargs or the exported signature will mismatch at runtime.
    for output_flag in _OUTPUT_FLAGS:
        if output_flag in inputs:
            value = inputs.pop(output_flag)
            if value is not None:
                model.forward = functools.partial(model.forward, **{output_flag: value})

    # set experts implementation to batched_mm for export
    if isinstance(model, PreTrainedModel) and model._can_set_experts_implementation():
        model.set_experts_implementation("batched_mm")

    # set attention implementation to sdpa for export
    if isinstance(model, PreTrainedModel) and model._can_set_attn_implementation():
        try:
            model.set_attn_implementation("sdpa")
        except Exception as e:
            print(f"Could not set attention implementation to sdpa for {model} of type {model.config.model_type}: {e}")

    for module in model.modules():
        if hasattr(module, "config"):
            # disable returning loss for every submodel
            if hasattr(module.config, "return_loss"):
                module.config.return_loss = False
            # disable mamba kernels for every submodel (mamba, jamba)
            if hasattr(module.config, "use_mamba_kernels"):
                module.config.use_mamba_kernels = False
        # disable classifier cast for nllb-moe
        if hasattr(module, "_cast_classifier"):
            module._cast_classifier = lambda *args, **kwargs: None
        # Replace mamba/linear-attn mask update: remove the data-dependent `torch.all(mask == 1)` check
        # but keep the `has_previous_state` check (a Python bool, safe for tracing).
        if hasattr(module, "_update_mamba_mask"):
            module._update_mamba_mask = _exportable_update_mask
        if hasattr(module, "_update_linear_attn_mask"):
            module._update_linear_attn_mask = _exportable_update_mask
        # Reset internal caches that are not part of past_key_values (e.g. DSA indexer in glm_moe_dsa)
        if hasattr(module, "_cached_keys"):
            module._cached_keys = None

    return model, inputs


# ── Model decomposition ──────────────────────────────────────────────────────

# Well-known submodule attribute names across VLM and encoder-decoder architectures.
# All matching names are found and captured by decompose_encoder_decoder.
_SUBMODULE_NAMES = (
    "vision_tower",
    "vision_model",
    "vision_encoder",
    "image_encoder",
    "audio_encoder",
    "multi_modal_projector",
    "connector",
    "encoder",
    "decoder",
    "language_model",
    "text_model",
    "lm_head",
)


def _find_submodules(model: PreTrainedModel) -> dict[str, torch.nn.Module]:
    """Return ``{attr_name: module}`` for all known submodule names found on the model.

    Checks ``model`` first, then ``model.model`` (common wrapper pattern).

    ``encoder`` and ``decoder`` are only included when both are present — a bare
    ``self.encoder`` (e.g. BeitModel, ViT) is an internal component, not a standalone
    exportable submodule.  VLM names (vision_tower, language_model, …) are always
    included individually.
    """
    found: dict[str, torch.nn.Module] = {}
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        for name in _SUBMODULE_NAMES:
            if name not in found and hasattr(root, name):
                found[name] = getattr(root, name)

    # Drop bare encoder/decoder unless both are present.
    if "encoder" in found and "decoder" not in found:
        del found["encoder"]
    if "decoder" in found and "encoder" not in found:
        del found["decoder"]

    return found


@contextmanager
def _capture_forward(module: torch.nn.Module, dest: list, *, capture_once: bool = False):
    """Append captured forward kwargs to ``dest`` on each call.

    Positional args are normalised to kwargs via ``inspect.signature`` so the
    captured dict can be passed directly as ``kwargs=inputs`` to ``torch.export``.
    """
    original = module.forward

    sig = inspect.signature(original)

    @functools.wraps(original)
    def capturing(*args, **kwargs):
        if not capture_once or not dest:
            bound = sig.bind(*args, **kwargs)
            captured = {}
            for param_name, value in bound.arguments.items():
                param = sig.parameters[param_name]
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    captured.update(value)  # unpack **kwargs into top level
                elif param.kind != inspect.Parameter.VAR_POSITIONAL:
                    captured[param_name] = value
            dest.append(copy.deepcopy(captured))
        return original(*args, **kwargs)

    module.forward = capturing
    try:
        yield
    finally:
        module.forward = original


def decompose_prefill_decode(
    model: PreTrainedModel,
    inputs: dict[str, Any],
) -> list[tuple[str, torch.nn.Module, dict]]:
    """Run ``model.generate()`` for 2 tokens and capture prefill and decode inputs.

    Reuses the full generation machinery so every architecture (decoder-only, SSM,
    encoder-decoder, VLM, …) gets correct inputs without reimplementing the loop.

    Returns:
        ``[("prefill", model, prefill_inputs), ("decode", model, decode_inputs)]``
    """
    captured: list[dict] = []

    try:
        with _capture_forward(model, captured), torch.no_grad():
            model.generate(**copy.deepcopy(inputs), max_new_tokens=2, min_new_tokens=2)
    except Exception as e:
        raise RuntimeError(
            f"decompose_prefill_decode failed for {type(model).__name__}. "
            f"Inputs passed: {list(inputs.keys())}. "
            f"Make sure the inputs are compatible with model.generate()."
        ) from e

    return [
        ("prefill", copy.copy(model), captured[0]),
        ("decode", copy.copy(model), captured[1]),
    ]


def decompose_encoder_decoder(
    model: PreTrainedModel, inputs: dict[str, Any]
) -> list[tuple[str, torch.nn.Module, dict]]:
    """Capture inputs to each component submodule via a single forward pass.

    Detects all known submodules by attribute name (vision tower, projector, connector,
    language model, lm_head, encoder, decoder, …) and captures their forward kwargs
    during one ``model(**inputs)`` call.

    Each submodule is returned as a separate ``(name, module, inputs)`` triplet for
    independent export. The token-merge step (e.g. ``masked_scatter`` for VLMs) is
    intentionally left outside the exported graphs — it is the caller's responsibility
    to assemble ``inputs_embeds`` from the encoder outputs before running the decoder.

    Returns:
        A list of ``(attr_name, module, inputs)`` triplets, one per detected submodule,
        in the order they appear in ``_SUBMODULE_NAMES``.

    Raises:
        ValueError: if no known submodules are found.
    """
    submodules = _find_submodules(model)
    if not submodules:
        raise ValueError(
            f"decompose_encoder_decoder found no known submodules on {type(model).__name__}. "
            f"Expected one or more of: {_SUBMODULE_NAMES}."
        )

    per_module_captured: dict[str, list[dict]] = {name: [] for name in submodules}
    ctx_managers = [
        _capture_forward(module, per_module_captured[name], capture_once=True) for name, module in submodules.items()
    ]

    try:
        with contextlib.ExitStack() as stack:
            for cm in ctx_managers:
                stack.enter_context(cm)
            stack.enter_context(torch.no_grad())
            model(**copy.deepcopy(inputs))
    except Exception as e:
        raise RuntimeError(
            f"decompose_encoder_decoder failed for {type(model).__name__}. Inputs passed: {list(inputs.keys())}."
        ) from e

    result = []
    for name, module in submodules.items():
        captured = per_module_captured[name]
        if not captured:
            continue  # submodule not called during this forward (e.g. lm_head may not be called on base models)
        result.append((name, module, captured[0]))
    return result


def is_multicomponent(model: PreTrainedModel) -> bool:
    """Return True if the model has recognisable submodules that should be exported separately."""
    return bool(_find_submodules(model))
