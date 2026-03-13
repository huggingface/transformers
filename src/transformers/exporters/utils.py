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
- `simulate_generation`: run `model.generate()` and capture the forward kwargs
  for the prefill and decode steps.
"""

import copy
import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..utils.import_utils import is_torch_available
from ..utils.logging import get_logger


logger = get_logger(__name__)


if is_torch_available():
    import torch


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel


# Output flags that should be set on model.config, not passed as forward() kwargs.
_OUTPUT_FLAGS = ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss")


# ── Leaf tensors ────────────────────────────────────────────────────────────


def _iter_leaf_tensors(obj: Any, prefix: str = "", default: str = "output"):
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
    model: "PreTrainedModel",
    inputs: dict[str, Any],
) -> tuple["PreTrainedModel", dict[str, Any]]:
    """Configure the model for export (no inference). Sets optimal attention/experts
    implementations and patches non-exportable module behaviours.

    Output flags (use_cache, return_dict, etc.) should be set on model.config before calling
    this function, not passed in inputs. If found in inputs, they are moved to config with a
    warning — callers should fix their input preparation instead of relying on this fallback.
    """
    # Validate inputs: loss computation is not supported during export
    for label_key in ("labels", "future_values"):
        if label_key in inputs:
            raise ValueError(
                f"Found '{label_key}' in inputs. Loss computation is not supported during export. "
                f"Please remove '{label_key}' from your inputs before calling export()."
            )
    if model.config.return_loss:
        raise ValueError(
            "Found 'model.config.return_loss=True'. Loss computation is not supported during export. "
            "Please set 'model.config.return_loss=False' before calling export()."
        )
    if inputs.get("return_loss", False):
        raise ValueError(
            "Found 'return_loss=True' in inputs. Loss computation is not supported during export. "
            "Please remove 'return_loss' from your inputs or set it to False."
        )

    # Fallback: move output flags from inputs to config. Callers should do this beforehand.
    for output_flag in _OUTPUT_FLAGS:
        if output_flag in inputs:
            logger.warning_once(
                f"Found output flag '{output_flag}' in inputs. Moving to model.config.{output_flag} instead. "
                f"Please set output flags on model.config before calling export()."
            )
            setattr(model.config, output_flag, inputs.pop(output_flag))

    # set experts implementation to batched_mm for export
    if model._can_set_experts_implementation():
        model.set_experts_implementation("batched_mm")

    # set attention implementation to sdpa for export
    if model._can_set_attn_implementation() and model.config.model_type != "videomae":
        try:
            model.set_attn_implementation("sdpa")
        except Exception as e:
            print(
                f"Could not set attention implementation to sdpa for {model} of type {model.config.model_type} : {e}"
            )

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


# ── Simulate generation ─────────────────────────────────────────────────────


def simulate_generation(model: "PreTrainedModel", inputs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run generate() for 2 tokens and capture the prefill and decode inputs.

    This reuses the full generation machinery so every model (hybrid, SSM,
    encoder-decoder, …) gets correct inputs without reimplementing the
    generation loop.

    Returns:
        ``(prefill_inputs, decode_inputs)`` — the kwargs the forward received
        on the first (prefill) and second (decode) iterations.
    """
    captured = []

    @contextmanager
    def capture_forward(model):
        original_forward = model.forward

        @functools.wraps(original_forward)
        def capturing_forward(*args, **kwargs):
            captured.append(copy.deepcopy(kwargs))
            return original_forward(*args, **kwargs)

        model.forward = capturing_forward
        yield
        model.forward = original_forward

    try:
        with capture_forward(model), torch.no_grad():
            model.generate(**copy.deepcopy(inputs), max_new_tokens=2, min_new_tokens=2)
    except Exception as e:
        raise RuntimeError(
            f"simulate_generation failed for {type(model).__name__}. "
            f"Inputs passed: {list(inputs.keys())}. "
            f"Make sure the inputs are compatible with model.generate()."
        ) from e

    return captured[0], captured[1]
