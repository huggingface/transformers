# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""The inputs a model would have computed for itself, derived from its config instead.

A registry of `model_type -> (model, inputs)` preparers that precompute the data-dependent tensors
(`cu_seqlens`, position ids, padded audio chunks, ...) a forward computes via `.tolist()` / `nonzero()`, so the
traced forward skips that branch. `get_rope_index_from_config` does the same for M-RoPE positions, for callers
that hold only a config: the export precompute and the runtime driving a saved artifact.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
from collections.abc import Mapping
from typing import Any

from ..utils import logging
from ..utils.generic import get_max_seqlen
from ..utils.import_utils import is_torch_available


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

    from ..configuration_utils import PreTrainedConfig
    from ..vision_utils import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_merged_shape,
        get_vision_nearest_position_ids,
        get_vision_position_ids,
        get_vision_window_index,
    )


def _find_config_attr(config: Any, name: str) -> Any | None:
    """First non-`None` `name` on `config` or any of its (recursive) `sub_configs`.

    A value only the modeling code sets (`num_grid_per_side`) is read off the vision tower instead
    (`_vision_tower_attr`)."""
    value = getattr(config, name, None)
    if value is not None:
        return value
    for sub_key in getattr(config, "sub_configs", {}):
        sub = getattr(config, sub_key, None)
        if sub is not None and (value := _find_config_attr(sub, name)) is not None:
            return value
    return None


@functools.lru_cache(maxsize=8)
def _meta_vision_tower(config_class: type, config_json: str):
    """The tower `config_class` configures, built on the meta device from its JSON (configs aren't hashable)."""
    from ..modeling_utils import PreTrainedModel

    modeling = importlib.import_module(config_class.__module__.replace(".configuration_", ".modeling_"))
    for candidate in vars(modeling).values():
        if (
            isinstance(candidate, type)
            and issubclass(candidate, PreTrainedModel)
            and candidate.config_class is config_class
        ):
            with torch.device("meta"):
                return candidate._from_config(config_class.from_dict(json.loads(config_json)))
    return None


def _vision_tower_attr(config: Any, name: str) -> Any | None:
    """`name` as the vision tower (or the first of its submodules that sets it) holds it, else off the config.

    Read off the module because the modeling code sets some of these in `__init__` rather than the config."""
    vision_config = _find_config_attr(config, "vision_config")
    tower = _meta_vision_tower(type(vision_config), vision_config.to_json_string()) if vision_config else None
    if tower is not None:
        value = next((getattr(m, name) for m in tower.modules() if getattr(m, name, None) is not None), None)
        if value is not None:
            return value
    return _find_config_attr(config, name)


def _resolve_modeling_module(config: Any):
    """The model's `modeling_*` module, from its config's module (`configuration_x` → `modeling_x`)."""
    return importlib.import_module(type(config).__module__.replace(".configuration_", ".modeling_"))


def _lays_out_modality_spans(config: Any) -> bool:
    """Whether `config` describes a model that places modality spans (has a vision/audio sub-config), rather
    than the text model inside it, whose config shares the same module."""
    return bool({"vision_config", "audio_config"} & set(getattr(config, "sub_configs", {}) or ()))


def _rope_index_owner(config: Any):
    """The class that defines `get_rope_index` for `config`'s model, or `None` if none does.

    Among several (qwen3_omni_moe's thinker and talker): by `config_class`, then declared architecture, then first.
    """
    if not _lays_out_modality_spans(config):
        return None
    try:
        module = _resolve_modeling_module(config)
    except ImportError:
        return None
    owners = [obj for obj in vars(module).values() if inspect.isclass(obj) and "get_rope_index" in obj.__dict__]
    if len(owners) <= 1:
        return owners[0] if owners else None
    for owner in owners:
        if isinstance(config, getattr(owner, "config_class", ()) or ()):
            return owner
    for architecture in getattr(config, "architectures", None) or ():
        for base in getattr(getattr(module, architecture, None), "__mro__", ()):
            if base in owners:
                return base
    return owners[0]


def get_rope_index_from_config(config: Any, inputs: Mapping[str, Any]):
    """The model's own `get_rope_index`, run without the model: `(position_ids, rope_deltas)` or `None`.

    The method reads only `self.config`, so it runs on an instance built without `__init__`. `None` (no
    `get_rope_index`, or no span inputs) leaves the caller on standard 1-D positions.
    """
    owner = _rope_index_owner(config)
    if owner is None or inputs.get("input_ids") is None:
        return None
    parameters = inspect.signature(owner.get_rope_index).parameters

    # Processor keys -> model parameter names; the omni thinkers derive `audio_seqlens` from the mel mask.
    candidates = dict(inputs)
    candidates.setdefault("second_per_grids", inputs.get("video_second_per_grid"))
    if inputs.get("audio_feature_lengths") is not None:
        candidates.setdefault("audio_seqlens", inputs["audio_feature_lengths"])
    elif inputs.get("feature_attention_mask") is not None:
        candidates.setdefault("audio_seqlens", inputs["feature_attention_mask"].sum(-1))
    call_kwargs = {name: value for name, value in candidates.items() if name in parameters and value is not None}

    # Layouts want the 2-D padding mask, and some index it unconditionally; anything else (a dict, a
    # `BlockMask`) becomes all-valid.
    if "attention_mask" in parameters:
        mask = call_kwargs.get("attention_mask")
        if not (isinstance(mask, torch.Tensor) and mask.dim() == 2):
            call_kwargs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # Span sources are whatever other tensors the signature takes; none means a text-only prompt.
    spans_from = {
        name
        for name, value in call_kwargs.items()
        if name not in ("input_ids", "attention_mask", "mm_token_type_ids") and isinstance(value, torch.Tensor)
    }
    if not spans_from:
        return None
    # Falling back to 1-D positions would be quietly wrong; the model raises here too.
    if "mm_token_type_ids" in parameters and "mm_token_type_ids" not in call_kwargs:
        raise ValueError(
            "Multi-modal data was passed but `mm_token_type_ids` is missing, so the M-RoPE positions "
            f"{owner.__name__} expects cannot be built. Pass the `mm_token_type_ids` the processor returns "
            "alongside `input_ids`."
        )

    model = owner.__new__(owner)
    object.__setattr__(model, "config", config)
    # A few layouts read a value `__init__` copies off the config (`spatial_merge_size`); fill those on demand.
    while True:
        try:
            return owner.get_rope_index(model, **call_kwargs)
        except AttributeError as missing:
            name = getattr(missing, "name", None)
            value = _find_config_attr(config, name) if name else None
            if value is None or hasattr(model, name):
                raise
            object.__setattr__(model, name, value)


# Marker kwarg tuples -> preparer.
_EXPORT_INPUT_PREPARERS: dict[tuple[str, ...], callable] = {}


def register_export_input_preparer(*markers: str):
    """Register `fn(config, inputs) -> None`, dispatched when every `marker` is in `inputs` and not `None`.

    The preparer reads only `config`, never a live model. Use several markers to narrow an ambiguous match."""

    def decorator(fn):
        _EXPORT_INPUT_PREPARERS[markers] = fn
        return fn

    return decorator


@register_export_input_preparer("image_sizes")
def _prepare_image_sizes_as_ints(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """Replace a tensor `image_sizes` with a list of `(h, w)` int-tuples.

    Encoders crop by it (`image_sizes[i] // patch_size` in Pixtral); as a tensor those bounds become unbacked.
    """
    image_sizes = inputs["image_sizes"]
    if not torch.is_tensor(image_sizes):
        return
    inputs["image_sizes"] = [tuple(int(v) for v in row) for row in image_sizes.tolist()]


@register_export_input_preparer("grid_thw")
def _prepare_grid_thw_vision_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """Precompute `cu_seqlens`, `max_seqlen`, `position_ids` from `grid_thw`, plus optional window-attention and
    interpolation tensors.

    Optional helpers are gated by a config attribute or by the modeling module defining the helper.
    """
    grid_thw = inputs["grid_thw"]
    # The tower's own value: an encoder that runs un-merged and defers the merge (kimi_k25) holds 1.
    spatial_merge_size = _vision_tower_attr(config, "spatial_merge_size")
    if spatial_merge_size is None:
        # Video-Llama-3 carries per-image merge sizes as an input tensor.
        spatial_merge_size = inputs.get("merge_sizes", 1)

    module = _resolve_modeling_module(config)
    merge_temporal = _vision_tower_attr(config, "merge_temporal_attention") is True
    inputs["cu_seqlens"], inputs["max_seqlen"] = get_vision_attention_seqlens(
        grid_thw, config, merge_temporal=merge_temporal, kwargs=inputs
    )
    include_temporal = _vision_tower_attr(config, "include_temporal_position_ids") is True
    inputs["position_ids"] = get_vision_position_ids(grid_thw, spatial_merge_size, include_temporal=include_temporal)

    window_size = _vision_tower_attr(config, "window_size")
    patch_size = _vision_tower_attr(config, "patch_size")
    if window_size is not None and patch_size is not None:
        inputs["window_index"], inputs["cu_window_seqlens"] = get_vision_window_index(
            grid_thw, spatial_merge_size, window_size, patch_size
        )
        inputs["max_window_seqlen"] = get_max_seqlen(
            inputs["cu_window_seqlens"], config, kwargs=inputs, kwarg_name="max_window_seqlen"
        )

    num_grid_per_side = _vision_tower_attr(config, "num_grid_per_side")
    if num_grid_per_side is not None:
        mode = _vision_tower_attr(config, "interpolation_mode") or "bilinear"
        padding = _vision_tower_attr(config, "interpolation_padding") or "border"
        align_corners = _vision_tower_attr(config, "interpolation_align_corners") is True
        inputs["interp_indices"], inputs["interp_weights"] = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side,
            mode=mode,
            align_corners=align_corners,
            spatial_merge_size=spatial_merge_size,
            padding=padding,
        )

    # The module helpers below each replace a per-clip/per-image loop with one gather index.
    if hasattr(module, "get_vision_frame_index"):
        inputs["frame_index"] = module.get_vision_frame_index(grid_thw)

    if hasattr(module, "get_vision_temporal_merge_index"):
        merge_kernel_size = _find_config_attr(config, "merge_kernel_size")
        kernel_height, kernel_width = (
            merge_kernel_size if not isinstance(merge_kernel_size, int) else (merge_kernel_size, merge_kernel_size)
        )
        inputs["temporal_merge_index"] = module.get_vision_temporal_merge_index(grid_thw, kernel_height, kernel_width)

    if hasattr(module, "get_vision_pixel_shuffle_index"):
        merge_size = _find_config_attr(config, "merge_size")
        inputs["pixel_shuffle_index"] = module.get_vision_pixel_shuffle_index(grid_thw, merge_size)

    if hasattr(module, "get_vision_temporal_slice_index"):
        inputs["temporal_slice_index"] = module.get_vision_temporal_slice_index(grid_thw, spatial_merge_size)


@register_export_input_preparer("target_sizes")
def _prepare_navit_vision_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """NaViT-style packed encoders carry per-image `(h, w)` as `target_sizes` instead of `grid_thw`."""
    target_sizes = inputs["target_sizes"]
    num_patches_per_side = _find_config_attr(config, "num_patches_per_side")
    if num_patches_per_side is None:
        # Derived the way the tower does it (minicpmv4_6).
        image_size = _find_config_attr(config, "image_size")
        patch_size = _find_config_attr(config, "patch_size")
        if image_size is not None and patch_size is not None:
            num_patches_per_side = image_size // patch_size
    if num_patches_per_side is not None:
        inputs["position_ids"] = get_vision_nearest_position_ids(target_sizes, num_patches_per_side)

    window_kernel_size = _find_config_attr(config, "window_kernel_size")
    if window_kernel_size is not None:
        grid_thw = torch.nn.functional.pad(target_sizes, (1, 0), value=1)
        inputs["window_index"], inputs["cu_window_seqlens"] = get_vision_window_index(
            grid_thw, spatial_merge_size=1, window_size=window_kernel_size[0], patch_size=1
        )
        inputs["merged_shape"] = get_vision_merged_shape(target_sizes, window_kernel_size)
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(target_sizes[:, 0] * target_sizes[:, 1], dim=0, dtype=torch.int32), (1, 0)
        )
        inputs["max_seqlen"] = get_max_seqlen(cu_seqlens, config, kwargs=inputs)


@register_export_input_preparer("input_features", "feature_lens")
def _prepare_omni_audio_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """Precompute the omni audio encoder's chunking (`padded_feature`, `chunk_lengths`, `cu_seqlens`, …).

    The helpers live in the model's `modeling_*` module; `n_window_infer` selects the Qwen3-Omni-style form.
    """
    feature_lens = inputs["feature_lens"]
    input_features = inputs["input_features"]
    module = _resolve_modeling_module(config)
    n_window = _find_config_attr(config, "n_window")
    n_window_infer = _find_config_attr(config, "n_window_infer")

    chunk_and_pad_features = getattr(module, "chunk_and_pad_features")
    get_audio_cu_seqlens = getattr(module, "get_audio_cu_seqlens")
    get_valid_indices = getattr(module, "get_valid_indices")

    padded_feature, chunk_lengths = chunk_and_pad_features(input_features, feature_lens, n_window)
    inputs["padded_feature"] = padded_feature
    inputs["chunk_lengths"] = chunk_lengths
    if n_window_infer is not None:
        inputs["cu_seqlens"] = get_audio_cu_seqlens(chunk_lengths, feature_lens, n_window_infer, n_window)
        inputs["valid_indices"] = get_valid_indices(chunk_lengths, n_window)
    else:
        inputs["cu_seqlens"] = get_audio_cu_seqlens(chunk_lengths)
        inputs["valid_indices"] = get_valid_indices(chunk_lengths)
        inputs["pool_indices"] = getattr(module, "get_pool_indices")(feature_lens)
    inputs["max_seqlen"] = get_max_seqlen(inputs["cu_seqlens"], config, kwargs=inputs)


@register_export_input_preparer("input_features", "feature_attention_mask")
def _prepare_masked_omni_audio_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """Pack padded omni audio features by their mask and run `_prepare_omni_audio_inputs` on the result.

    The marker pair is not omni-specific (qwen2_audio), so it fires only where the chunked-audio helpers exist."""
    if not hasattr(_resolve_modeling_module(config), "chunk_and_pad_features"):
        return
    mask = inputs["feature_attention_mask"]
    derived = dict(inputs)
    derived["input_features"] = inputs["input_features"].permute(0, 2, 1)[mask.bool()].permute(1, 0)
    derived["feature_lens"] = mask.sum(-1)
    _prepare_omni_audio_inputs(config, derived)
    for key in ("padded_feature", "chunk_lengths", "cu_seqlens", "valid_indices", "pool_indices", "max_seqlen"):
        if key in derived:
            inputs[key] = derived[key]


@register_export_input_preparer("input_features", "input_features_mask")
def _prepare_qwen3_asr_audio_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """Precompute `cu_seqlens` and `max_seqlen` for Qwen3-ASR, mirroring `Qwen3ASREncoder.forward`."""
    from ..models.qwen3_asr.modeling_qwen3_asr import get_audio_cu_seqlens

    n_window = _find_config_attr(config, "n_window")
    n_window_infer = _find_config_attr(config, "n_window_infer")
    if n_window is None or n_window_infer is None:
        return

    input_features_mask = inputs["input_features_mask"]
    batch_size, padded_feature_length = input_features_mask.shape
    num_chunks = padded_feature_length // (n_window * 2)
    feature_lens = input_features_mask.sum(-1).to(torch.long)
    chunk_lengths = input_features_mask.view(batch_size, num_chunks, -1).sum(dim=-1).reshape(-1).to(torch.long)
    inputs["cu_seqlens"] = get_audio_cu_seqlens(chunk_lengths, feature_lens, n_window_infer, n_window)
    inputs["max_seqlen"] = get_max_seqlen(inputs["cu_seqlens"], config, kwargs=inputs)


@register_export_input_preparer("input_values")
def _prepare_acoustic_noise(config: Any, inputs: dict[str, Any]) -> None:
    """Draw the VAE noise an acoustic tokenizer (VibeVoice) adds to its latents, as an input rather than in-graph.

    OpenVINO strips in-graph sampling to zeros and ONNX uses its own generator; the same two calls, in the same
    order as `get_audio_features`, let eager and export share one sample.
    """
    encoder = getattr(config, "acoustic_tokenizer_encoder_config", None)
    if encoder is None or not getattr(encoder, "vae_std", None) or inputs.get("acoustic_noise") is not None:
        return
    values = inputs["input_values"]
    batch = values.shape[0]
    noise_std = encoder.vae_std * torch.randn(batch, device=values.device, dtype=values.dtype)
    latents = (batch, -(-values.shape[-1] // encoder.hop_length), encoder.hidden_size)
    inputs["acoustic_noise"] = noise_std[:, None, None] * torch.randn(
        latents, device=values.device, dtype=values.dtype
    )


def precompute_export_inputs(config: PreTrainedConfig, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of `inputs` plus the tensors a model would otherwise compute data-dependently while tracing.

    Config-only, so export and the runtime share it: M-RoPE positions via [`get_rope_index_from_config`], then
    every preparer whose markers are present (`register_export_input_preparer`).
    """
    inputs = dict(inputs)

    if inputs.get("position_ids") is None and inputs.get("input_ids") is not None:
        # Prefill: ids span the whole mask; a dict of masks (or none) counts as prefill.
        attn_mask = inputs.get("attention_mask")
        is_prefill = not isinstance(attn_mask, torch.Tensor) or inputs["input_ids"].shape[1] == attn_mask.shape[1]
        if is_prefill and (rope_index := get_rope_index_from_config(config, inputs)) is not None:
            inputs["position_ids"] = rope_index[0]

    for markers, preparer in _EXPORT_INPUT_PREPARERS.items():
        if all(inputs.get(m) is not None for m in markers):
            preparer(config, inputs)
    return inputs
