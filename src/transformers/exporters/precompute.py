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

Registry of `model_type -> (model, inputs) -> None` callables that precompute the
data-dependent tensors (cu_seqlens, position_ids, padded audio chunks, …) the model
would otherwise compute in its forward via `.tolist()` / `nonzero()` / etc. Inject
the results into `inputs` so the forward skips the untraceable branch.

The other half of the same job is `get_rope_index_from_config`: M-RoPE positions are laid out per
architecture by a method on the model class, and both callers here hold a config and nothing else — the
export precompute, and the runtime driving a saved artifact.
"""

from __future__ import annotations

import importlib
import inspect
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
    """First non-`None` `name` on `config` or any of its (recursive) `sub_configs` (`vision_config` /
    `audio_config` / `text_config` / …).

    This is how the preparers below read every parameter they need, which is what lets the precompute run
    from a saved config with no model instance: a plain field, or a `@property` where the vision module
    derives the value (`num_grid_per_side`, muse_glimmer's `window_size`). A model whose module hardcodes a
    value a preparer needs should expose it on its config the same way."""
    value = getattr(config, name, None)
    if value is not None:
        return value
    for sub_key in getattr(config, "sub_configs", {}):
        sub = getattr(config, sub_key, None)
        if sub is not None and (value := _find_config_attr(sub, name)) is not None:
            return value
    return None


def _resolve_modeling_module(config: Any):
    """The model's `modeling_*` module, from its config's module (`configuration_x` → `modeling_x`) — the
    model-free counterpart of `sys.modules[type(model).__module__]`, used to reach a model's own precompute
    helpers (`get_vision_frame_index`, `chunk_and_pad_features`, …)."""
    return importlib.import_module(type(config).__module__.replace(".configuration_", ".modeling_"))


def _lays_out_modality_spans(config: Any) -> bool:
    """Whether `config` describes a model that places modality spans, rather than the text model inside it.

    A model's text sub-config is declared in the same module as the multi-modal config it belongs to, so
    reaching the module is not enough: a component exported from the language model alone carries the text
    config, and the spans are not its to lay out. Declaring a vision or audio sub-config is what separates
    the two — including for an omni thinker, which lays out spans under its own config class rather than
    the outer model's.
    """
    return bool({"vision_config", "audio_config"} & set(getattr(config, "sub_configs", {}) or ()))


def _rope_index_owner(config: Any):
    """The class that defines `get_rope_index` for `config`'s model, or `None` if none does.

    A module can hold more than one (qwen3_omni_moe's thinker and talker each define their own). The class
    whose own `config_class` this is wins; failing that, the config's declared architecture picks, and
    failing that the first definition.
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

    M-RoPE lays its modality spans out per architecture, and that layout lives on the model class
    (`Qwen2VLModel.get_rope_index` and its counterparts). Both callers here hold a config and nothing else
    — the export precompute, and `ExportedGenerator` driving a saved artifact — and the method reads its
    geometry off `self.config` alone, so it runs on an instance built without `__init__`: no module tree,
    no weights, no checkpoint.

    `None` means the positions are not this function's to build: the model defines no `get_rope_index`, or
    the inputs it places spans from are absent. That is the same gate the model's own forward applies, and
    it leaves the caller on the standard 1-D positions.
    """
    owner = _rope_index_owner(config)
    if owner is None or inputs.get("input_ids") is None:
        return None
    parameters = inspect.signature(owner.get_rope_index).parameters

    # The parameter names are the model's, the keys are the processor's; `audio_seqlens` is the one the
    # omni thinkers derive from the mel padding mask rather than receiving outright.
    candidates = dict(inputs)
    candidates.setdefault("second_per_grids", inputs.get("video_second_per_grid"))
    if inputs.get("audio_feature_lengths") is not None:
        candidates.setdefault("audio_seqlens", inputs["audio_feature_lengths"])
    elif inputs.get("feature_attention_mask") is not None:
        candidates.setdefault("audio_seqlens", inputs["feature_attention_mask"].sum(-1))
    call_kwargs = {name: value for name, value in candidates.items() if name in parameters and value is not None}

    # `attention_mask` here means the 2-D padding mask the layouts index positions with. `generate`
    # carries the per-layer form instead (a dict, or a `BlockMask`), which is not that, and some layouts
    # index the mask unconditionally with no `None` branch (the omni thinkers) — so anything that is not
    # the 2-D mask becomes the all-valid one, which is what an absent mask meant here all along.
    if "attention_mask" in parameters:
        mask = call_kwargs.get("attention_mask")
        if not (isinstance(mask, torch.Tensor) and mask.dim() == 2):
            call_kwargs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # Spans are placed from whatever modality tensors the layout declares — a grid for most, audio lengths
    # for the omni thinkers, `target_sizes` for minicpm. Read off the signature, since the names differ per
    # architecture and the question does not; none present is a text-only prompt through a multi-modal
    # model. `mm_token_type_ids` says which tokens the spans cover, not where they come from.
    spans_from = {
        name
        for name, value in call_kwargs.items()
        if name not in ("input_ids", "attention_mask", "mm_token_type_ids") and isinstance(value, torch.Tensor)
    }
    if not spans_from:
        return None
    # There is multi-modal data but nothing saying which tokens it covers. The model raises here rather
    # than guessing, and so do we: falling back to 1-D positions would run and be quietly wrong.
    if "mm_token_type_ids" in parameters and "mm_token_type_ids" not in call_kwargs:
        raise ValueError(
            "Multi-modal data was passed but `mm_token_type_ids` is missing, so the M-RoPE positions "
            f"{owner.__name__} expects cannot be built. Pass the `mm_token_type_ids` the processor returns "
            "alongside `input_ids`."
        )

    model = owner.__new__(owner)
    object.__setattr__(model, "config", config)
    # Most layouts read `self.config` alone, but a few reach for a value the model's `__init__` copies off
    # it (the omni thinkers' `spatial_merge_size`). Fill those in as the method asks for them; a name the
    # config does not carry is a genuine error and re-raises, as does one filling did not fix.
    while True:
        try:
            return owner.get_rope_index(model, **call_kwargs)
        except AttributeError as missing:
            name = getattr(missing, "name", None)
            value = _find_config_attr(config, name) if name else None
            if value is None or hasattr(model, name):
                raise
            object.__setattr__(model, name, value)


# Marker kwarg tuples -> preparer. A preparer runs when every marker in its key is present in the inputs
# (`@register_export_input_preparer(*markers)`), so a model gets exactly the precompute its encoder needs.
_EXPORT_INPUT_PREPARERS: dict[tuple[str, ...], callable] = {}


def register_export_input_preparer(*markers: str):
    """Register `fn(config, inputs) -> None`. Dispatched when every `marker` is a key in
    `inputs` with a non-`None` value — no model_type list to maintain. The preparer reads what it needs
    from `config` (via `_precompute_attr` / `_resolve_modeling_module`), never a live model. Use multiple
    markers to narrow the match when a single kwarg is too ambiguous (e.g.
    `("input_features", "feature_lens")` for omni audio encoders)."""

    def decorator(fn):
        _EXPORT_INPUT_PREPARERS[markers] = fn
        return fn

    return decorator


@register_export_input_preparer("image_sizes")
def _prepare_image_sizes_as_ints(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    """Replace a tensor `image_sizes` with a python list of `(h, w)` int-tuples (the `.tolist()` runs here,
    outside the traced graph).

    `image_sizes` is per-image geometry, and encoders crop/split each image by it — e.g.
    `image_sizes[i] // patch_size` (Pixtral) or `int(image_sizes[i] / factor)` (Emu3 VQVAE). As a tensor
    those bounds become unbacked symints under `torch.export`; as python ints they stay static (matching
    each encoder's own `image_sizes is None` fallback, which already builds int-tuples). Models that route
    `image_sizes` around the traced graph (e.g. LLaVA-NeXT resolves anyres before tracing) never hit this.
    """
    image_sizes = inputs["image_sizes"]
    if not torch.is_tensor(image_sizes):
        return
    inputs["image_sizes"] = [tuple(int(v) for v in row) for row in image_sizes.tolist()]


@register_export_input_preparer("grid_thw")
def _prepare_grid_thw_vision_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """Precompute helpers driven by `grid_thw`: `cu_seqlens`, `max_seqlen`, `position_ids`, plus optional
    `window_index`/`cu_window_seqlens`/`max_window_seqlen` (XNet-style window attn) and
    `bilinear_indices`/`bilinear_weights` (interpolation-based merging).

    Optional helpers are gated by a config attribute (`window_size`+`patch_size` for window attention,
    `num_grid_per_side` for interpolation — see `_find_config_attr`) or, for
    model-specific ones, by the encoder's modeling module defining the helper (`get_vision_frame_index` /
    `get_vision_temporal_merge_index` for kimi_k25) — so a model that doesn't use a feature won't get its
    kwarg injected.
    """
    grid_thw = inputs["grid_thw"]
    spatial_merge_size = _find_config_attr(config, "spatial_merge_size")
    if spatial_merge_size is None:
        # Video-Llama-3 carries per-image merge sizes as an input tensor rather than on its config.
        spatial_merge_size = inputs.get("merge_sizes", 1)
    # An encoder that resamples its position grid before merging (kimi_k25, muse_glimmer, paddleocr_vl)
    # builds these tensors at patch resolution — the same value its module passes.
    resample_merge_size = 1 if _find_config_attr(config, "resample_before_merge") is True else spatial_merge_size

    # Whether packed attention spans a whole clip (kimi_k25) or one segment per frame.
    module = _resolve_modeling_module(config)
    merge_temporal = _find_config_attr(config, "merge_temporal_attention") is True
    inputs["cu_seqlens"], inputs["max_seqlen"] = get_vision_attention_seqlens(
        grid_thw, config, merge_temporal=merge_temporal, kwargs=inputs
    )
    # 3-axis (t, h, w) rotary encoders expose an ``axis_dim`` on their rotary_emb (minimax_m3_vl); default
    # 2-axis (h, w) covers qwen2_5_vl / qwen3_vl / glm4v / paddleocr_vl.
    include_temporal = _find_config_attr(config, "include_temporal_position_ids") is True
    inputs["position_ids"] = get_vision_position_ids(grid_thw, resample_merge_size, include_temporal=include_temporal)

    window_size = _find_config_attr(config, "window_size")
    patch_size = _find_config_attr(config, "patch_size")
    if window_size is not None and patch_size is not None:
        inputs["window_index"], inputs["cu_window_seqlens"] = get_vision_window_index(
            grid_thw, spatial_merge_size, window_size, patch_size
        )
        inputs["max_window_seqlen"] = get_max_seqlen(
            inputs["cu_window_seqlens"], config, kwargs=inputs, kwarg_name="max_window_seqlen"
        )

    num_grid_per_side = _find_config_attr(config, "num_grid_per_side")
    if num_grid_per_side is not None:
        # How the vision embedding resamples its learned grid (kimi_k25 bicubic, qwen3_vl / paddleocr_vl
        # bilinear with aligned corners, muse_glimmer grid_sample zeros padding) — each declared on the
        # vision config; the defaults here are what a config that says nothing means.
        mode = _find_config_attr(config, "interpolation_mode") or "bilinear"
        padding = _find_config_attr(config, "interpolation_padding") or "border"
        align_corners = _find_config_attr(config, "interpolation_align_corners") is True
        inputs["interp_indices"], inputs["interp_weights"] = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side,
            mode=mode,
            align_corners=align_corners,
            spatial_merge_size=resample_merge_size,
            padding=padding,
        )

    # Per-frame additive position table (kimi_k25): gathered by frame index instead of a per-clip loop.
    if hasattr(module, "get_vision_frame_index"):
        inputs["frame_index"] = module.get_vision_frame_index(grid_thw)

    # Temporal-pooling spatial merger (kimi_k25): one gather index replaces its per-clip merge loop.
    if hasattr(module, "get_vision_temporal_merge_index"):
        merge_kernel_size = _find_config_attr(config, "merge_kernel_size")
        kernel_height, kernel_width = (
            merge_kernel_size if not isinstance(merge_kernel_size, int) else (merge_kernel_size, merge_kernel_size)
        )
        inputs["temporal_merge_index"] = module.get_vision_temporal_merge_index(grid_thw, kernel_height, kernel_width)

    # Pixel-shuffle spatial merger (muse_glimmer): one gather index replaces its per-image merge loop.
    if hasattr(module, "get_vision_pixel_shuffle_index"):
        merge_size = _find_config_attr(config, "merge_size")
        inputs["pixel_shuffle_index"] = module.get_vision_pixel_shuffle_index(grid_thw, merge_size)

    if hasattr(module, "get_vision_temporal_slice_index"):
        # ernie4_5_vl_moe's merger interleaves even/odd frames through a `range(0, temporal_size, 2)` loop
        # over the grid's values — untraceable, and the indices depend on nothing but the grid.
        inputs["temporal_slice_index"] = module.get_vision_temporal_slice_index(grid_thw, spatial_merge_size)


@register_export_input_preparer("target_sizes")
def _prepare_navit_vision_inputs(config: Any, inputs: dict[str, Any]) -> None:
    """NaViT-style packed encoders carry per-image `(h, w)` as `target_sizes` instead of `grid_thw`.
    Synthesise `grid_thw = [1, h, w]` and run the nearest-position-id / window-index /
    merged-shape / maximum-sequence-length helpers outside the traced graph."""
    target_sizes = inputs["target_sizes"]
    num_patches_per_side = _find_config_attr(config, "num_patches_per_side")
    if num_patches_per_side is None:
        # The tower derives the grid side rather than declaring it (minicpmv4_6's embeddings hold
        # `image_size // patch_size`), and the precompute only ever sees the config — so derive it the same
        # way. Reached only via the `target_sizes` marker, so an anyres model never lands here.
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
    """Replace `input_features`/`feature_lens` with precomputed `padded_feature`, `chunk_lengths`,
    `cu_seqlens`, `max_seqlen`, `valid_indices` (+ `pool_indices` on Qwen2.5-Omni-style encoders) so the
    encoder's `.split(.tolist(), dim=0)` and related data-dependent ops happen outside the
    traced graph.

    The helpers (`chunk_and_pad_features`, `get_audio_cu_seqlens`, …) all live in the model's
    own ``modeling_*.py`` module, resolved from `config`. ``n_window_infer`` selects the Qwen3-Omni-style
    four-arg ``get_audio_cu_seqlens`` over the Qwen2.5-Omni-style single-arg form.
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
    """The Omni `get_audio_features` seam carries the padded features and their padding mask rather than
    the packed `feature_lens` pair — pack them the way the getter's own masking branch does (eagerly,
    outside the trace) and hand the packed pair to `_prepare_omni_audio_inputs`; its precompute rides in
    as extra graph inputs while the graph keeps taking the raw features and mask.

    Unlike `feature_lens`, this marker pair is not omni-specific (qwen2_audio carries it too), so fire
    only for a model whose own modeling module has the chunked-audio helpers."""
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
    """Precompute `cu_seqlens` and `max_seqlen` for Qwen3-ASR so the encoder pops them from
    ``kwargs``. Mirrors the few lines that build ``feature_lens``/``chunk_lengths`` in
    ``Qwen3ASREncoder.forward``.
    """
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
    """Draw the VAE noise an acoustic tokenizer adds to its latents, as an input rather than inside the graph.

    VibeVoice samples `vae_std * randn` on every forward, and no exported graph can match that: OpenVINO strips
    the sampling to zeros, ONNX draws from its own generator. Drawn here — with the same two calls, in the same
    order, that `get_audio_features` makes — eager and export share one sample, and a seeded run repeats it.
    The latents are `(batch, ceil(samples / hop_length), hidden_size)`; chunking keeps that, since a chunk is a
    multiple of `hop_length`.
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
    """Return `inputs` plus the tensors a model would otherwise compute data-dependently while tracing.

    Driven entirely by the config — no model and no weights — so the same call serves the export path and
    the runtime, which only ever has the saved config. `inputs` is not modified; the precomputed tensors
    come back in a new dict.

    Two layers:
    - Outer LLM M-RoPE positions, via [`get_rope_index_from_config`] — the model's own `get_rope_index`,
      called without the model.
    - Per-encoder preparer dispatched by marker kwargs present in `inputs` (e.g. `grid_thw`,
      `target_sizes`, `(input_features, feature_lens)`) — see `register_export_input_preparer`.
      A preparer fires only when every one of its markers is present in `inputs`.
    """
    inputs = dict(inputs)

    # Outer-model M-RoPE positions. Placing the spans reads the token ids, so this is a no-op on
    # encoder-only components (an exported `get_image_features`) that carry no `input_ids`, and on a
    # text-only model, whose class defines no `get_rope_index`.
    if inputs.get("position_ids") is None and inputs.get("input_ids") is not None:
        # Prefill is the step whose ids span the whole mask. A model that takes a *dict* of per-type masks
        # (t5gemma2, the mixed full/sliding models) states no single width to compare against, so it is read
        # the way a missing mask is: nothing there contradicts the prompt.
        attn_mask = inputs.get("attention_mask")
        is_prefill = not isinstance(attn_mask, torch.Tensor) or inputs["input_ids"].shape[1] == attn_mask.shape[1]
        if is_prefill and (rope_index := get_rope_index_from_config(config, inputs)) is not None:
            inputs["position_ids"] = rope_index[0]

    # Encoder-level: dispatch by marker kwargs (preparer fires when every marker is in `inputs`
    # with a non-`None` value).
    for markers, preparer in _EXPORT_INPUT_PREPARERS.items():
        if all(inputs.get(m) is not None for m in markers):
            preparer(config, inputs)
    return inputs
