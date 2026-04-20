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

Backend-agnostic helpers used by Dynamo, ONNX, and ExecuTorch exporters:

- `get_leaf_tensors`: recursively extract all leaf tensors from nested outputs.
- `prepare_for_export`: configure model config, attention/experts implementations,
  and patch non-exportable module behaviours before any export.
- `decompose_prefill_decode`: run `model.generate()` and capture the forward kwargs
  for the prefill and decode steps.
- `decompose_multimodal`: capture inputs to every known multi-modal submodule (vision
  tower, projector, language model, ...) via a single forward pass, returning one
  `{name: (module, inputs)}` entry per component for independent export.
"""

from __future__ import annotations

import contextlib
import copy
import enum
import functools
import inspect
import sys
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_torch_available


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

    from ..modeling_utils import PreTrainedModel


# Output flags that should be set on model.config, not passed as forward() kwargs.
_OUTPUT_FLAGS = ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss")


_LEAF_SKIP_TYPES = (type,)
if is_torch_available():
    # Types that should not be recursed into when extracting leaf tensors.
    # Sym* types carry PyTorch shape_env internals that cause infinite recursion;
    # Enums are scalars with no tensor fields.
    _LEAF_SKIP_TYPES += (enum.Enum, torch.SymInt, torch.SymFloat, torch.SymBool)


# ── Recursive structure traversal ──────────────────────────────────────────
# All tensor utilities share this traversal. _map_leaf_tensors applies a function
# to every tensor leaf; _iter_leaf_tensors yields (path, tensor) pairs.


def _map_leaf_tensors(obj: Any, fn: callable) -> Any:
    """Apply `fn` to every tensor in a nested structure, preserving container types.

    Traverses dicts, lists, tuples, sets, and objects with `__dict__` (e.g. cache objects).
    Skips non-traversable leaf types (enum, SymInt, etc.).
    """
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return obj
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(_map_leaf_tensors(item, fn) for item in obj)
    if isinstance(obj, dict):
        return type(obj)({k: _map_leaf_tensors(v, fn) for k, v in obj.items()})
    if hasattr(obj, "__dict__"):
        for attr, attr_val in vars(obj).items():
            setattr(obj, attr, _map_leaf_tensors(attr_val, fn))
    return obj


def _iter_leaf_tensors(obj: Any, prefix: str = ""):
    """Yield `(dotted_path, tensor)` for every tensor in a nested structure."""
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return
    if isinstance(obj, torch.Tensor):
        yield prefix or "output", obj
    elif isinstance(obj, (list, tuple, set)):
        for index, item in enumerate(obj):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_leaf_tensors(item, path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from _iter_leaf_tensors(value, path)
    elif hasattr(obj, "__dict__"):
        yield from _iter_leaf_tensors(vars(obj), prefix)


# ── Public tensor utilities ────────────────────────────────────────────────
# Extract or cast tensors from nested model outputs.


def get_leaf_tensors(obj: Any) -> dict[str, torch.Tensor]:
    """Recursively retrieve all leaf tensors from a potentially nested structure.

    Args:
        obj (`Any`):
            A tensor, dataclass, dict, list, tuple, or any nesting thereof.

    Returns:
        `dict[str, torch.Tensor]`: Flat mapping from dotted path strings to tensors.
    """
    return dict(_iter_leaf_tensors(obj))


def duplicate_leaf_tensors(obj: Any) -> Any:
    """Clone tensors that appear more than once in an output structure.

    When a model returns the same tensor under two output names (e.g. `last_hidden_state`
    and `hidden_states[0]`), the ONNX optimizer deduplicates the two output nodes and
    renames one, breaking the expected name mapping. Cloning duplicates gives each output
    leaf a distinct identity so the optimizer has nothing to merge.
    """
    seen = set()

    def _dedup(tensor: torch.Tensor) -> torch.Tensor:
        if id(tensor) in seen:
            return tensor.clone()
        seen.add(id(tensor))
        return tensor

    return _map_leaf_tensors(obj, _dedup)


def cast_leaf_tensors(obj: Any, dtype: torch.dtype, device: torch.device) -> Any:
    """Recursively cast all floating-point tensors to the given dtype and device."""

    def _cast(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(dtype=dtype, device=device) if tensor.is_floating_point() else tensor.to(device=device)

    return _map_leaf_tensors(obj, _cast)


def prepare_for_export(
    model: PreTrainedModel | torch.nn.Module, inputs: dict[str, Any]
) -> tuple[PreTrainedModel | torch.nn.Module, dict[str, Any]]:
    """Configure the model for export. Mutates both `model` and `inputs` in-place:

    - Sets optimal attention/experts implementations.
    - Patches non-exportable module behaviours (mamba masks, classifier casts, …).
    - Strips label inputs (`labels`, `future_values`) — loss computation is unsupported.
    - Strips output flags (`use_cache`, `return_dict`, …) from inputs and bakes non-`None`
      values into `model.forward` via `functools.partial` so they are constant at trace time.
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
            logger.warning("Could not set attention implementation to sdpa for %s: %s", model.config.model_type, e)

    # Idefics2/3's vision encoder uses boolean indexing to filter padding images, creating
    # an unbacked symbolic batch dim. SDPA's CPU kernel guards on Eq(batch, 1) when a mask
    # is provided, which fails with unbacked dims. Keep the vision part on eager.
    if (
        isinstance(model, PreTrainedModel)
        and model._can_set_attn_implementation()
        and model.config.model_type in ("idefics2", "idefics3", "smolvlm")
        and model.device.type == "cpu"
    ):
        model.set_attn_implementation({"vision_config": "eager"})

    for module in model.modules():
        if hasattr(module, "config"):
            # disable returning loss for every submodel
            if hasattr(module.config, "return_loss"):
                module.config.return_loss = False
            # disable mamba kernels for every submodel (mamba, jamba)
            if hasattr(module.config, "use_mamba_kernels"):
                module.config.use_mamba_kernels = False
        # Reset internal caches that are not part of past_key_values (e.g. DSA indexer in glm_moe_dsa)
        if hasattr(module, "_cached_keys"):
            module._cached_keys = None

    # Pre-compute data-dependent vision/audio tensors that use loops, .tolist(),
    # repeat_interleave, or itertools.groupby — untraceable by torch.export.
    with torch.no_grad():
        _precompute_vision_inputs(model, inputs)
        _precompute_audio_inputs(model, inputs)

    # Cast all input tensors to match the model's dtype and device (e.g. cache objects
    # created before the model was moved to bfloat16/CUDA by a backend preparation step).
    try:
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        inputs = cast_leaf_tensors(inputs, dtype=model_dtype, device=model_device)
    except StopIteration:
        pass

    return model, inputs


# ── Multi-modal decomposition ─────────────────────────────────────────────────
# Split multi-modal models into independently exportable submodules (vision encoder,
# projector, language model) by capturing each submodule's forward inputs during a single pass.

# Well-known submodule attribute names for multi-modal architectures.
_MULTIMODAL_LM_NAMES = ("language_model", "text_model", "lm_head")
_MULTIMODAL_PROJECTOR_NAMES = ("multi_modal_projector", "connector", "embed_vision", "embed_audio")
_MULTIMODAL_ENCODER_NAMES = (
    "vision_encoder",
    "image_encoder",
    "audio_encoder",
    "vision_model",
    "vision_tower",
    "audio_tower",
    "visual",
)
_MULTIMODAL_SUBMODULE_NAMES = _MULTIMODAL_ENCODER_NAMES + _MULTIMODAL_PROJECTOR_NAMES + _MULTIMODAL_LM_NAMES


def _find_multimodal_submodules(model: PreTrainedModel) -> dict[str, torch.nn.Module]:
    """Return `{attr_name: module}` for all known multi-modal submodule names found on the model.

    Checks `model` first, then `model.model` (common wrapper pattern).
    Only returns results when at least one modal encoder AND one language model are
    found — otherwise the model is not multi-modal and should be exported as a single unit.
    """
    found: dict[str, torch.nn.Module] = {}
    for root in (model, getattr(model, "model", None)):
        if root is None:
            continue
        for name in _MULTIMODAL_SUBMODULE_NAMES:
            if name not in found and getattr(root, name, None) is not None:
                found[name] = getattr(root, name)

    has_encoder = any(name in found for name in _MULTIMODAL_ENCODER_NAMES)
    has_lm = any(name in found for name in _MULTIMODAL_LM_NAMES)
    if not (has_encoder and has_lm):
        return {}

    return found


def is_multimodal(model: PreTrainedModel) -> bool:
    """Returns `True` if the model is multi-modal with modal encoders and a language model."""
    return bool(_find_multimodal_submodules(model))


def _precompute_vision_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """Pre-compute data-dependent vision tensors and inject them into inputs.

    Vision models use `grid_thw`-based loops, `repeat_interleave`, `itertools.groupby`,
    and `.tolist()` that are not traceable by torch.export. This eagerly computes the
    results and injects them so the forward can skip the untraceable branch.
    """
    # Full-model level: get_rope_index (Qwen-VL, GLM-4V) computes position_ids
    # from input_ids + grid_thw using data-dependent ops (groupby, nonzero).
    if inputs.get("position_ids") is None and hasattr(model, "get_rope_index"):
        input_ids = inputs.get("input_ids")
        attn_mask = inputs.get("attention_mask")
        is_prefill = attn_mask is None or input_ids is None or input_ids.shape[1] == attn_mask.shape[1]
        if is_prefill:
            rope_params = set(inspect.signature(model.get_rope_index).parameters)
            rope_inputs = {k: inputs[k] for k in rope_params if k in inputs}
            position_ids, _ = model.get_rope_index(**rope_inputs)
            inputs["position_ids"] = position_ids

    # Vision submodule level: precompute from grid_thw
    grid_thw = inputs.get("grid_thw")
    if grid_thw is None:
        return

    # Only precompute for models whose forward accepts optional precomputed params
    forward_params = set(inspect.signature(model.forward).parameters)
    precompute_keys = {
        "rotary_pos_ids",
        "cu_seqlens",
        "embed_indices",
        "bilinear_weights",
        "window_index",
        "cu_window_seqlens",
    }
    if not (precompute_keys & forward_params):
        return

    # Import vision functions from the model's module (they're imported at the top of each modeling file)
    model_mod = sys.modules[type(model).__module__]

    # cu_seqlens from repeat_interleave (data-dependent output size)
    if "cu_seqlens" in forward_params and hasattr(model_mod, "get_vision_cu_seqlens"):
        inputs["cu_seqlens"] = model_mod.get_vision_cu_seqlens(grid_thw)

    # rotary pos IDs
    if "rotary_pos_ids" in forward_params and hasattr(model_mod, "get_rotary_pos_ids"):
        inputs["rotary_pos_ids"] = model_mod.get_rotary_pos_ids(grid_thw, model.spatial_merge_size)

    # window_index (Qwen2.5-VL only)
    if "window_index" in forward_params and hasattr(model_mod, "get_window_index"):
        inputs["window_index"], inputs["cu_window_seqlens"] = model_mod.get_window_index(
            grid_thw, model.spatial_merge_size, model.window_size, model.patch_size, model.spatial_merge_unit
        )

    # pos_embed_indices (Qwen3-VL only)
    if "embed_indices" in forward_params and hasattr(model_mod, "get_pos_embed_indices"):
        inputs["embed_indices"], inputs["bilinear_weights"] = model_mod.get_pos_embed_indices(
            grid_thw, model.num_grid_per_side, model.config.spatial_merge_size
        )


def _precompute_audio_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """Precompute audio encoder inputs that use untraceable ops (.tolist(), nonzero(), loops)."""
    model_mod = sys.modules[type(model).__module__]

    if not hasattr(model_mod, "chunk_and_pad_features"):
        return

    if "input_features" not in inputs or "feature_lens" not in inputs:
        return

    feature_lens = inputs.pop("feature_lens")
    input_features = inputs.pop("input_features")

    padded_feature, chunk_lengths = model_mod.chunk_and_pad_features(input_features, feature_lens, model.n_window)
    inputs["padded_feature"] = padded_feature
    inputs["chunk_lengths"] = chunk_lengths

    forward_params = set(inspect.signature(model.forward).parameters)

    if "cu_seqlens" in forward_params and hasattr(model_mod, "get_audio_cu_seqlens"):
        fn = model_mod.get_audio_cu_seqlens
        fn_params = set(inspect.signature(fn).parameters)
        if "feature_lens" in fn_params:
            inputs["cu_seqlens"] = fn(chunk_lengths, feature_lens, model.n_window_infer, model.n_window)
        else:
            inputs["cu_seqlens"] = fn(chunk_lengths)

    if "valid_indices" in forward_params and hasattr(model_mod, "get_valid_indices"):
        inputs["valid_indices"] = model_mod.get_valid_indices(chunk_lengths)

    if "pool_indices" in forward_params and hasattr(model_mod, "get_pool_indices"):
        inputs["pool_indices"] = model_mod.get_pool_indices(feature_lens)


@contextlib.contextmanager
def _capture_forward(module: torch.nn.Module):
    """Capture forward call kwargs into a list (one dict per call).

    Positional args are normalised to kwargs via `inspect.signature` so the
    captured dicts can be passed directly as `kwargs=inputs` to `torch.export`.
    """

    calls: list[dict] = []
    original = module.forward
    sig = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        captured = {}
        bound = sig.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                captured.update(copy.deepcopy(value))
            elif param.kind != inspect.Parameter.VAR_POSITIONAL:
                captured[name] = copy.deepcopy(value)
        calls.append(captured)
        return original(*args, **kwargs)

    module.forward = wrapper
    try:
        yield calls
    finally:
        module.forward = original


def decompose_prefill_decode(
    model: PreTrainedModel,
    inputs: dict[str, Any],
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Run `model.generate()` for 2 tokens and capture prefill and decode inputs.

    Reuses the full generation machinery so every architecture (decoder-only, SSM,
    encoder-decoder, multi-modal, …) gets correct inputs without reimplementing the loop.

    Returns:
        `dict[str, tuple[torch.nn.Module, dict]]`:
        `{"prefill": (model, prefill_inputs), "decode": (model, decode_inputs)}`
    """
    try:
        with _capture_forward(model) as calls, torch.no_grad():
            model.generate(**copy.deepcopy(inputs), max_new_tokens=2, min_new_tokens=2)
    except Exception as e:
        raise RuntimeError(
            f"decompose_prefill_decode failed for {type(model).__name__}. "
            f"Inputs passed: {list(inputs.keys())}. "
            f"Make sure the inputs are compatible with model.generate()."
        ) from e

    return {
        "prefill": (copy.copy(model), calls[0]),
        "decode": (copy.copy(model), calls[1]),
    }


def decompose_multimodal(model: PreTrainedModel, inputs: dict[str, Any]) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Capture inputs to each multi-modal submodule via a single forward pass.

    Detects all known multi-modal submodules by attribute name (vision tower, projector,
    language model, lm_head, …) and captures their forward kwargs during one
    `model(**inputs)` call.

    Each submodule is returned as a separate `name: (module, inputs)` entry for
    independent export. The token-merge step (e.g. `masked_scatter` for multi-modal models)
    is intentionally left outside the exported graphs — it is the caller's responsibility
    to assemble `inputs_embeds` from the encoder outputs before running the decoder.

    Returns:
        `dict[str, tuple[torch.nn.Module, dict]]`: One `name: (module, inputs)`
        entry per detected submodule, in the order they appear in `_MULTIMODAL_SUBMODULE_NAMES`.

    Raises:
        `ValueError`: if no known multi-modal submodules are found on the model.
    """
    submodules = _find_multimodal_submodules(model)
    if not submodules:
        raise ValueError(
            f"decompose_multimodal found no multi-modal submodules on {type(model).__name__}. "
            f"Expected one or more of: {_MULTIMODAL_SUBMODULE_NAMES}."
        )

    try:
        with contextlib.ExitStack() as stack, torch.no_grad():
            submodule_inputs = {
                name: stack.enter_context(_capture_forward(module)) for name, module in submodules.items()
            }
            model(**copy.deepcopy(inputs))
    except Exception as e:
        raise RuntimeError(
            f"decompose_multimodal failed for {type(model).__name__}. Inputs passed: {list(inputs.keys())}."
        ) from e

    return {
        name: (module, submodule_inputs[name][-1])
        for name, module in submodules.items()
        if submodule_inputs[name]  # skip submodules not called (e.g. lm_head on base models)
    }
