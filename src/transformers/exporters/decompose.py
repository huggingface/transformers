# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Splitting a generative model into the graphs an export needs.

`generate` runs a model in stages with different shapes — prefill over a whole prompt, decode one token at
a time, a vision or audio tower once up front — and each is its own graph. This captures those stages by
running `generate` and recording what each stage was called with, so the exporters get
`{component: (submodel, forward_kwargs)}` without per-model glue.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
from dataclasses import replace
from typing import Any, NamedTuple

from ..utils import logging
from ..utils.import_utils import is_torch_available
from .cache import (
    check_cache_geometry,
    indexer_layers_of,
    keeps_write_once_state,
    kv_geometry_of,
    materialize_cache_layers,
)
from .components import (
    Component,
    ComponentRole,
    CrossAttentionEncoder,
    ModalityEncoder,
    PatchVisionEncoder,
    TokenEmbedder,
)
from .precompute import _find_config_attr, precompute_export_inputs
from .utils import (
    module_device,
    module_dtype,
    patch_attributes,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

    from ..cache_utils import Cache, DynamicCrossAttentionLayer, EncoderDecoderCache, kv_cache_geometry
    from ..modeling_utils import PreTrainedModel


@contextlib.contextmanager
def _capture_calls(obj: Any, attribute: str):
    """Capture the kwargs of each `obj.<attribute>(...)` call during the block (positional args normalised
    to kwargs), restoring the attribute afterwards."""
    calls: list[dict] = []
    original = getattr(obj, attribute)
    was_instance_attr = attribute in vars(obj)
    sig = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        captured = {}
        for name, value in sig.bind(*args, **kwargs).arguments.items():
            kind = sig.parameters[name].kind
            if kind == inspect.Parameter.VAR_KEYWORD:
                captured.update(copy.deepcopy(value))
            elif kind != inspect.Parameter.VAR_POSITIONAL:
                captured[name] = copy.deepcopy(value)
        calls.append(captured)
        return original(*args, **kwargs)

    setattr(obj, attribute, wrapper)
    try:
        yield calls
    finally:
        # Delete rather than reassign: a bound method pinned in the instance dict would make a later
        # `copy.copy` of the module run with `self` = the original (empty getter capture on qwen2_5_omni).
        if was_instance_attr:
            setattr(obj, attribute, original)
        else:
            delattr(obj, attribute)


@contextlib.contextmanager
def _capture_forward(module: torch.nn.Module):
    """Capture each `module(...)` call's kwargs."""
    with _capture_calls(module, "forward") as calls:
        yield calls


class CrossWriter(NamedTuple):
    """One module that fills the cross-attention cache, and the call that filled it.

    The slots (`("arg", index)` or `("kwarg", name)`) locate the encoder output, the decoder query (`width`
    wide) and the cache, so a replay can swap in live tensors.
    """

    module: Any
    args: tuple
    kwargs: dict
    states_at: tuple
    query_at: tuple | None
    width: int | None
    cache_at: tuple


def _slot_of(args: tuple, kwargs: dict, match) -> tuple | None:
    """Where the first argument satisfying `match` sits, as `("arg", index)` / `("kwarg", name)`."""
    for index, value in enumerate(args):
        if match(value):
            return ("arg", index)
    for name, value in kwargs.items():
        if match(value):
            return ("kwarg", name)
    return None


def _argument(args: tuple, kwargs: dict, slot: tuple):
    """The argument a `CrossWriter` slot points at."""
    kind, key = slot
    return args[key] if kind == "arg" else kwargs[key]


@contextlib.contextmanager
def capture_cross_writers(model: PreTrainedModel):
    """Capture the modules that fill the cross-attention cache, with the calls that filled it.

    A write to the cross half of an `EncoderDecoderCache` is attributed to the innermost running module, and
    the encoder output is found by identity among its arguments, so nothing is looked up by name.

    Yields `{layer index: CrossWriter}`, filled once the model has run.
    """
    writers: dict[int, CrossWriter] = {}
    active: list[tuple] = []
    encoder_states: list = []
    hooks = []

    def opened(module, args, kwargs):
        active.append((module, args, kwargs))

    def closed(module, args, output):
        if active:
            active.pop()

    def encoded(module, args, output):
        encoder_states.append(output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0])

    for module in model.get_decoder().modules():
        hooks.append(module.register_forward_pre_hook(opened, with_kwargs=True))
        hooks.append(module.register_forward_hook(closed))
    hooks.append(model.get_encoder().register_forward_hook(encoded))

    original_update = Cache.update

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if not active or layer_idx in writers or not encoder_states:
            return original_update(self, key_states, value_states, layer_idx, *args, **kwargs)
        module, call_args, call_kwargs = active[-1]
        cache_at = _slot_of(call_args, call_kwargs, lambda value: isinstance(value, EncoderDecoderCache))
        paired = _argument(call_args, call_kwargs, cache_at) if cache_at else None
        if paired is not None and paired.cross_attention_cache is self:
            states = encoder_states[-1]
            states_at = _slot_of(call_args, call_kwargs, lambda value: value is states)
            query_at = _slot_of(
                call_args,
                call_kwargs,
                lambda value: isinstance(value, torch.Tensor) and value is not states and value.dim() == 3,
            )
            if states_at is not None:
                query = _argument(call_args, call_kwargs, query_at) if query_at else None
                writers[layer_idx] = CrossWriter(
                    module,
                    call_args,
                    call_kwargs,
                    states_at,
                    query_at,
                    query.shape[-1] if query is not None else None,
                    cache_at,
                )
        return original_update(self, key_states, value_states, layer_idx, *args, **kwargs)

    Cache.update = update
    try:
        yield writers
    finally:
        Cache.update = original_update
        for hook in hooks:
            hook.remove()


def _merge_decode_calls(decode_calls: list[dict], streamed: str | None = None) -> dict:
    """Merge consecutive single-token decode captures into one multi-token decode input.

    A single-token step specializes the query axis to 1 under `torch.export`; concatenating `N` steps gives
    it a hint `N > 1` so it stays dynamic. The cache is taken from the first step.
    """
    first = decode_calls[0]
    merged = copy.copy(first)

    # With `use_cache` off each step re-runs the whole sequence, already multi-token, so take the last.
    def query_length(call: dict) -> int | None:
        for key in ("input_ids", "inputs_embeds"):
            value = call.get(key)
            if value is not None:
                return value.shape[1]
        return None

    if any(query_length(call) != 1 for call in decode_calls):
        return copy.copy(decode_calls[-1])

    def concat_along(key: str, dim: int) -> None:
        values = [call[key] for call in decode_calls if call.get(key) is not None]
        if len(values) == len(decode_calls):
            merged[key] = torch.cat(values, dim=dim)

    concat_along("input_ids", 1)
    concat_along("inputs_embeds", 1)
    concat_along("cache_position", 0)
    # `[batch, seq]` or `[n_axes, batch, seq]` (m-rope): the sequence axis is last in both.
    concat_along("position_ids", -1)
    # Left at one step it re-specializes the query axis (`Guard failed: token_type_ids.size()[1] == 1`).
    concat_along("token_type_ids", -1)
    # A streamed modality's window is per-token at its own stride; left at one step the graph bakes the ratio.
    if streamed is not None:
        concat_along(streamed, 1)

    masks = [call.get("attention_mask") for call in decode_calls]
    if all(mask is not None for mask in masks):
        merged["attention_mask"] = _merge_step_masks(masks)

    return merged


def _merge_step_masks(masks: list[Any]) -> Any:
    """Merge one attention mask per decode step into a single multi-token mask.

    A 4D causal mask is concatenated along the query axis (one causal row per step); a 2D padding mask keeps
    the last step. Dicts (hybrid attention) merge per entry; `None` stays `None`.
    """
    last_mask = masks[-1]
    if last_mask is None:
        return None
    if isinstance(last_mask, dict):
        return {key: _merge_step_masks([mask[key] for mask in masks]) for key in last_mask}
    if last_mask.dim() == 4 and all(mask.shape[3] == last_mask.shape[3] for mask in masks):
        return torch.cat(masks, dim=2)
    return last_mask


def decompose_prefill_decode(
    model: PreTrainedModel,
    inputs: dict[str, Any],
    generation_config: Any = None,
    multi_token_decode: bool = False,
) -> dict[str, Component]:
    """Run `model.generate()` and capture prefill and decode inputs.

    `generation_config` is forwarded to `generate()`; one with `cache_implementation="static"` and
    `max_cache_len=N` captures a fixed-size cache. With `multi_token_decode`, two decode steps are merged
    (`_merge_decode_calls`) so the decode graph's query axis stays symbolic.

    Returns:
        `{"prefill": Component(model, prefill_inputs), "decode": Component(model, decode_inputs)}`.
    """
    # Encoder-decoders capture from the second decode step: cache length 1 would be 0/1-specialized.
    spec = streaming_embedder_spec(model.config)
    streamed_kwarg = spec.produces if spec is not None else None
    first_decode = 2 if getattr(model.config, "is_encoder_decoder", False) else 1
    num_new_tokens = first_decode + (2 if multi_token_decode else 1)
    capture_config = copy.deepcopy(generation_config if generation_config is not None else model.generation_config)
    capture_config.max_new_tokens = num_new_tokens
    capture_config.min_new_tokens = num_new_tokens
    capture_config.disable_compile = True
    # Captured as plain decode steps: assisted generation would fold them into candidate windows
    capture_config.prompt_lookup_num_tokens = None
    capture_config.assistant_early_exit = None
    no_mm_encoder_outputs = []
    if hasattr(model, "_supports_mm_encoder_outputs"):
        no_mm_encoder_outputs.append((model, "_supports_mm_encoder_outputs", lambda original: lambda: False))
    try:
        with _capture_forward(model) as calls, patch_attributes(no_mm_encoder_outputs):
            model.generate(**copy.deepcopy(inputs), generation_config=capture_config)
    except Exception as e:
        # A cache-shape error means `kv_cache_geometry` disagrees with what the model writes.
        if "slice shapes" in str(e) or "Sizes of tensors must match" in str(e):
            raise RuntimeError(
                f"decompose_prefill_decode failed for {type(model).__name__}: the exporter materialized the "
                f"cache as (heads, key_dim, value_dim)={(kv_cache_geometry(model.config) or [None])[0]}, which is not "
                "what this model caches — `kv_cache_geometry` needs to learn its layout."
            ) from e
        raise RuntimeError(
            f"decompose_prefill_decode failed for {type(model).__name__}. "
            f"Inputs passed: {list(inputs.keys())}. "
            f"Make sure the inputs are compatible with model.generate()."
        ) from e

    if len(calls) < num_new_tokens:
        raise RuntimeError(
            f"decompose_prefill_decode expected at least {num_new_tokens} calls to "
            f"{type(model).__name__}.forward() during generate(max_new_tokens={num_new_tokens}), but "
            f"captured {len(calls)}. This likely means generate() bypasses the top-level forward() "
            "(e.g. delegates to an inner model), so prefill/decode decomposition is not supported "
            "for this architecture."
        )

    # `logits_to_keep` is kept: without it the graph outputs full `[batch, tokens, vocab]` logits (311 MB
    # for a 512-token Qwen3 prompt) when generation reads only the last row.
    prefill_inputs = calls[0]
    decode_inputs = (
        _merge_decode_calls(calls[first_decode:num_new_tokens], streamed=streamed_kwarg)
        if multi_token_decode
        else calls[first_decode]
    )
    # `generate` built this cache, so it carries the real geometry to check the derived one against.
    if (captured_cache := decode_inputs.get("past_key_values")) is not None:
        check_cache_geometry(model.config, captured_cache)

    return {
        "prefill": Component("prefill", copy.copy(model), prefill_inputs, ComponentRole.PREFILL),
        "decode": Component("decode", copy.copy(model), decode_inputs, ComponentRole.DECODE),
    }


def _multimodal_text_decoder(model: PreTrainedModel | torch.nn.Module) -> torch.nn.Module | None:
    """The text decoder of a multi-modal model, or `None` when the model is not one to decompose.

    A modality getter counts as well as an encoder module: vibevoice_asr reports no encoder but has
    `get_audio_features`.
    """
    if not isinstance(model, PreTrainedModel):
        return None
    decoder = model.get_decoder()
    if decoder is None or decoder is model:
        return None
    for modality in ("image", "audio"):
        # `get_encoder` falls back to `self`, and some models keep `self.vision_tower = None`.
        encoder = model.get_encoder(modality=modality)
        if encoder is not None and encoder is not model:
            return decoder
    if any(_modality_owner(model, getter) is not None for _name, getter, *_ in _MODALITY_SPECS):
        return decoder
    return None


def is_multimodal(model: PreTrainedModel | torch.nn.Module) -> bool:
    """Returns `True` if the model is multi-modal with a modality to export and a language model."""
    return _multimodal_text_decoder(model) is not None


# Suffixes marking a modality input as aux (grid, mask, size table): skipped by presence checks and feature
# routing, since `generate` may keep one after dropping the features it described.
_MODALITY_AUX_SUFFIXES = ("_grid_thw", "_position_mask", "_attention_mask", "_sizes", "padding_mask", "_indices")


class StreamingEmbedderSpec(NamedTuple):
    """How one model streams a modality alongside its text."""

    component: str
    """Component name the embedder graph is exported under."""
    path: str
    """Submodule path to the embedder, under the base model."""
    source: str
    """The kwarg it consumes (the raw features)."""
    produces: str
    """The kwarg the decode graph takes a window of its output under."""
    stride: str
    """Config field: how many embedded rows one token spans."""
    encoder_config: str
    """Config field holding the sub-config of the encoder whose cache the decode graph takes, so the runtime
    can build one shaped the way the trace saw it."""


# Modalities embedded once before the decode loop, each step taking a window of the output, so the embedder
# is its own graph and the decode graph takes the window. Keyed by model type.
_STREAMING_EMBEDDERS = {
    "voxtral_realtime": StreamingEmbedderSpec(
        "audio_embedder",
        "audio_tower.embedder",
        "input_features",
        "encoder_inputs_embeds",
        "downsample_factor",
        "audio_config",
    ),
}


def streaming_embedder_spec(config) -> StreamingEmbedderSpec | None:
    """This config's `_STREAMING_EMBEDDERS` entry, or `None`."""
    return _STREAMING_EMBEDDERS.get(getattr(config, "model_type", None))


# (component name, getter, input kwargs signalling the modality, native grid kwarg, placeholder-id field).
# The first input kwarg present is used; the rest of the tuple covers per-model names and aux keys.
_MODALITY_SPECS = (
    (
        "image_encoder",
        "get_image_features",
        (
            "pixel_values",
            "pixel_values_images",
            "flattened_patches",
            "image_patches",
            "image_patches_indices",
            "image_embeds_position_mask",
            "pixel_attention_mask",
            "target_sizes",
        ),
        "image_grid_thw",
        "image_token_id",
    ),
    (
        "video_encoder",
        "get_video_features",
        ("pixel_values_videos", "target_sizes_videos"),
        "video_grid_thw",
        "video_token_id",
    ),
    (
        "audio_encoder",
        "get_audio_features",
        ("input_features", "audio_input_ids", "input_values", "padding_mask"),
        None,
        "audio_token_id",
    ),
)

_MODALITY_GETTERS = {name: getter for name, getter, *_ in _MODALITY_SPECS}


def grid_renamed(key: str) -> str:
    """Map a per-modality grid kwarg (`image_grid_thw`) to the getter's `grid_thw`."""
    return "grid_thw" if key.endswith("_grid_thw") else key


def pack_anyres_features(config, features, image_sizes, outputs) -> torch.Tensor:
    """The `pack_image_features` step `PatchVisionEncoder` leaves out of the graph, using the modeling's own
    grid/unpad helpers."""
    from ..models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image
    from .precompute import _find_config_attr

    newline = next((t for name, t in outputs.items() if name.endswith("image_newline")), None)
    pinpoints = _find_config_attr(config, "image_grid_pinpoints")
    tile = _find_config_attr(config, "image_size")
    # From the projected tensor, not the config: a deepstack projector downsamples the token grid.
    side = round(features.shape[1] ** 0.5)
    packed = []
    for index, feature in enumerate(torch.split(features, anyres_patch_counts(config, image_sizes), dim=0)):
        if feature.shape[0] > 1:
            base, patches = feature[0], feature[1:]
            num_patch_height, num_patch_width = get_anyres_image_grid_shape(image_sizes[index], pinpoints, tile)
            patches = patches.view(num_patch_height, num_patch_width, side, side, -1).permute(4, 0, 2, 1, 3)
            patches = unpad_image(patches.flatten(1, 2).flatten(2, 3), image_sizes[index])
            if newline is not None:
                column = newline[:, None, None].expand(*patches.shape[:-1], 1).to(patches)
                patches = torch.cat((patches, column), dim=-1)
            packed.append(torch.cat((base, patches.flatten(1, 2).transpose(0, 1)), dim=0))
        else:
            feature = feature[0]
            if newline is not None:
                feature = torch.cat((feature, newline[None].to(feature)), dim=0)
            packed.append(feature)
    return torch.cat(packed, dim=0)


def anyres_patch_counts(config: Any, image_sizes) -> list[int]:
    """Per-image patch counts (tiles plus the base patch) from `image_sizes`; config-only."""
    from ..image_processing_utils import select_best_resolution

    pinpoints = _find_config_attr(config, "image_grid_pinpoints")
    tile = _find_config_attr(config, "image_size")
    counts = []
    for size in image_sizes:
        height, width = select_best_resolution(size.tolist() if hasattr(size, "tolist") else list(size), pinpoints)
        counts.append(-(-height // tile) * -(-width // tile) + 1)
    return counts


def flatten_anyres_patches(config: Any, pixel_values, image_sizes):
    """The flat `(total_patches, …)` tensor the tower takes, dropping padding rows outside the graph."""
    if pixel_values.dim() != 5:
        return pixel_values
    counts = anyres_patch_counts(config, image_sizes)
    return torch.cat([pix[:count] for pix, count in zip(pixel_values, counts)], dim=0)


def packs_anyres_features(owner: Any, config: Any) -> bool:
    """Whether this image getter ends in the anyres `pack_image_features` (deepstack towers included)."""
    projectors = ("multi_modal_projector", "layerwise_projectors")
    return (
        _find_config_attr(config, "image_grid_pinpoints") is not None
        and any(hasattr(owner, name) for name in projectors)
        and all(hasattr(owner, attr) for attr in ("vision_tower", "image_newline"))
    )


def _modality_owner(model, getter):
    """Whichever of the model or its base actually defines a modality getter."""
    base = model.base_model
    return base if hasattr(base, getter) else (model if hasattr(model, getter) else None)


def _present_input_key(inputs, input_keys):
    """The modality's input kwarg that this call actually carries, or `None` when the modality is absent."""
    return next((key for key in input_keys if inputs.get(key) is not None), None)


def _embeds_input_ids(decoder: Any) -> bool:
    """Whether `decoder` turns `input_ids` into embeddings with a single module (not musicgen's
    per-codebook `ModuleList`)."""
    try:
        embeddings = decoder.get_input_embeddings() if decoder is not None else None
    except NotImplementedError:
        return False
    return isinstance(embeddings, torch.nn.Module) and not isinstance(embeddings, torch.nn.ModuleList)


def decompose_multimodal(
    model: PreTrainedModel,
    inputs: dict[str, Any],
    recorded_features: dict[str, list] | None = None,
    prompt_ids: torch.Tensor | None = None,
) -> dict[str, Component]:
    """Split a multi-modal model into independently exportable components.

    Components: `embed_tokens`, one `<modality>_encoder` per modality present (`get_<modality>_features`),
    and `text_decoder`. The `masked_scatter` merge stays outside the graphs.

    Raises:
        `ValueError`: if no known multi-modal submodules are found on the model.
    """
    decoder = _multimodal_text_decoder(model)
    if decoder is None:
        raise ValueError(
            f"decompose_multimodal found no multi-modal submodules on {type(model).__name__}. "
            f"Expected an image/audio encoder + language model, found neither."
        )

    # An encoder-decoder consumed its modality inputs before prefill; the caller passes the recorded calls.
    recorded_features = recorded_features or {}
    active_modalities = []
    for name, getter, input_keys, grid_key, _token_field in _MODALITY_SPECS:
        if (owner := _modality_owner(model, getter)) is None:
            continue
        if _present_input_key(inputs, input_keys) is not None or recorded_features.get(name):
            active_modalities.append((name, getter, owner, grid_key))

    try:
        with contextlib.ExitStack() as stack, torch.no_grad():
            decoder_calls = stack.enter_context(_capture_forward(decoder))
            captured_features = {
                name: stack.enter_context(_capture_calls(owner, getter))
                for name, getter, owner, _ in active_modalities
            }
            model(**copy.deepcopy(inputs))
    except Exception as e:
        raise RuntimeError(
            f"decompose_multimodal failed for {type(model).__name__}. Inputs passed: {list(inputs.keys())}."
        ) from e

    components = (
        {"text_decoder": Component("text_decoder", decoder, decoder_calls[-1], ComponentRole.DECODE)}
        if decoder_calls
        else {}
    )

    # Dual-encoders (owlvit) refuse `get_input_embeddings`. `prompt_ids` covers an encoder-decoder, whose
    # prefill kwargs carry `decoder_input_ids` instead.
    token_ids = inputs.get("input_ids") if inputs.get("input_ids") is not None else prompt_ids
    if token_ids is not None and _embeds_input_ids(decoder):
        placeholder_ids = [
            getattr(model.config, spec[-1], None)
            for spec in _MODALITY_SPECS
            if getattr(model.config, spec[-1], None) is not None
        ]
        components["embed_tokens"] = Component(
            "embed_tokens",
            TokenEmbedder(decoder, placeholder_ids),
            {"input_ids": token_ids},
            ComponentRole.EMBED_TOKENS,
        )

    for name, getter, owner, grid_key in active_modalities:
        calls = captured_features.get(name) or recorded_features.get(name) or []
        if not calls:
            continue
        feature_inputs = {
            ("grid_thw" if key == grid_key else key): value for key, value in calls[-1].items() if value is not None
        }
        # Config-derived tensors become graph inputs: a getter reading its grid as data cannot be traced.
        if name == "image_encoder" and packs_anyres_features(owner, model.config):
            tower_inputs = {
                "pixel_values": flatten_anyres_patches(
                    model.config, feature_inputs["pixel_values"], feature_inputs["image_sizes"]
                )
            }
            tower_inputs.update(
                {
                    key: feature_inputs[key]
                    for key in ("vision_feature_layer", "vision_feature_select_strategy")
                    if key in feature_inputs
                }
            )
            components[name] = Component(name, PatchVisionEncoder(owner), tower_inputs, ComponentRole.MODALITY_ENCODER)
            continue
        feature_inputs = precompute_export_inputs(model.config, feature_inputs)
        components[name] = Component(
            name, ModalityEncoder(owner, getter, grid_key), feature_inputs, ComponentRole.MODALITY_ENCODER
        )
    return components


def _needs_prefill_graph(model, components: dict, *, cross_written_without_prompt=None) -> bool:
    """Whether the prompt needs a graph of its own, beside a decode graph that could serve it.

    A second graph duplicates every parameter, so "no" unless the decode graph provably cannot stand in:
    it was traced after a branch the prompt takes (e.g. the cross-cache write), which export does not keep.
    `cross_written_without_prompt` asks hypothetically, as if something else wrote the cross cache.
    """
    # The decode graph only reads the cross cache unless the encoder component writes it.
    encoder = components.get("encoder")
    writes_cross_cache = (
        isinstance(encoder.module if encoder is not None else None, CrossAttentionEncoder)
        if cross_written_without_prompt is None
        else cross_written_without_prompt
    )
    if encoder is not None and not writes_cross_cache:
        return True
    # A streamed modality bakes a query-length branch into a merged decode.
    if streaming_embedder_spec(model.config) is not None:
        return True
    # Written-once layer state has the same writer/reader split; recurrent models keep it in `cache_params`.
    decode_inputs = components["decode"].inputs
    cache = next(
        (decode_inputs[name] for name in ("past_key_values", "cache_params") if decode_inputs.get(name) is not None),
        None,
    )
    layers = getattr(cache, "layers", [])
    if any(
        keeps_write_once_state(layer) or (isinstance(layer, DynamicCrossAttentionLayer) and not writes_cross_cache)
        for layer in layers
    ):
        return True
    # A modality no component took (mllama) runs inside the prompt's forward. Checked per modality, since
    # getters rename their inputs.
    prefill_inputs = components["prefill"].inputs
    return any(
        prefill_inputs.get(key) is not None
        for name, _getter, input_keys, *_ in _MODALITY_SPECS
        if name not in components
        for key in input_keys
    )


def _cross_writing_encoder(encoder, writers: dict, encoder_inputs: dict, components: dict):
    """A `CrossAttentionEncoder` that reproduces this model's cross cache, or `None` if it cannot.

    Replay is not always separable (t5gemma2 merges self- and cross-attention), so the result is checked
    against the cache the model itself filled.
    """
    captured = components["decode"].inputs.get("past_key_values")
    layers = getattr(getattr(captured, "cross_attention_cache", None), "layers", [])
    if not layers:
        return None
    # Copies: the check may write in place into tensors the decode component is exported with.
    copied = {
        index: writer._replace(args=copy.deepcopy(writer.args), kwargs=copy.deepcopy(writer.kwargs))
        for index, writer in writers.items()
    }
    component = CrossAttentionEncoder(encoder, copied).eval()
    try:
        with torch.no_grad():
            produced = component(**copy.deepcopy(encoder_inputs))
    except Exception:
        logger.warning_once(
            f"{type(encoder).__name__} cannot compute the decoder's cross-attention cache on its own, so the "
            "export keeps a separate prompt graph to write it."
        )
        return None
    for index, layer in enumerate(layers):
        for kind, expected in (("keys", layer.keys), ("values", layer.values)):
            actual = produced.get(f"cross_{kind}_{index}")
            if expected is None:
                continue
            if actual is None or actual.shape != expected.shape or not torch.allclose(actual, expected, atol=1e-5):
                logger.warning_once(
                    f"{type(encoder).__name__} reproduces the decoder's cross-attention cache incorrectly at "
                    f"layer {index}, so the export keeps a separate prompt graph to write it."
                )
                return None
    return component


def _fold_cross_cache_into_encoder(model, components: dict, writers: dict) -> dict:
    """Let the encoder component write the decoder's cross-attention cache, when that buys the prompt graph.

    If a prompt graph is needed anyway, it keeps writing the cache: it was traced filling an empty one.
    """
    encoder = components.get("encoder")
    if not writers or encoder is None:
        return components
    if _needs_prefill_graph(model, components, cross_written_without_prompt=True):
        return components
    module = _cross_writing_encoder(encoder.module, writers, encoder.inputs, components)
    return components if module is None else {**components, "encoder": replace(encoder, module=module)}


def _write_cross_cache_in_decoder(components: dict) -> tuple[dict, bool]:
    """Let the decode graph write its own cross-attention cache, from the `encoder_outputs` it is fed.

    Emptying the cross half (`is_updated` off) traces the write instead of the read, for a backend that
    keeps its cache in variables (OpenVINO). Only a growing cross half is emptied. Returns the components
    and whether the decode graph now writes the cross cache.
    """
    decode = components["decode"]
    cache = decode.inputs.get("past_key_values")
    # BLT keeps a self-attention cache in the cross half and takes no `encoder_outputs`.
    if not isinstance(cache, EncoderDecoderCache) or decode.inputs.get("encoder_outputs") is None:
        return components, False
    layers = cache.cross_attention_cache.layers
    if not layers or any(getattr(layer, "is_compileable", False) for layer in layers):
        return components, False
    cache = copy.deepcopy(cache)
    for index, layer in enumerate(cache.cross_attention_cache.layers):
        if layer.keys is not None:
            layer.keys, layer.values = layer.keys[..., :0, :], layer.values[..., :0, :]
        cache.is_updated[index] = False
    return {**components, "decode": replace(decode, inputs={**decode.inputs, "past_key_values": cache})}, True


def _materialize_prefill_cache(model, prefill_inputs: dict, decode_inputs: dict) -> None:
    """Fill in the prompt capture's cache, so it is traced against the cache the runtime will feed it.

    Geometry and indexer slots come from the decode capture's cache, which holds what the model really
    writes; layers left empty there (hy_v4's shared indexers) stay empty so both graphs take the same leaves.
    """
    cache = prefill_inputs.get("past_key_values")
    if cache is None:
        return
    batch_size = next(t for t in prefill_inputs.values() if isinstance(t, torch.Tensor)).shape[0]
    materialize_cache_layers(
        cache,
        batch_size,
        model.config,
        module_dtype(model),
        module_device(model),
        kv_geometry=kv_geometry_of(decode_inputs.get("past_key_values")),
        indexer_layers=indexer_layers_of(decode_inputs.get("past_key_values")),
    )


def _capture_generation(
    model, inputs, generation_config, multi_token_decode
) -> tuple[dict[str, Component], dict, dict]:
    """Run one `generate` and return `(components, recorded_features, cross_writers)`; the last two are empty
    for a decoder-only model."""
    capture = functools.partial(
        decompose_prefill_decode,
        model,
        inputs,
        generation_config=generation_config,
        multi_token_decode=multi_token_decode,
    )
    if not getattr(model.config, "is_encoder_decoder", False):
        return capture(), {}, {}

    # The encoder and modality getters run outside the captured forwards, so record them over the same generate.
    modality_owners = {
        name: owner for name, getter, *_ in _MODALITY_SPECS if (owner := _modality_owner(model, getter)) is not None
    }
    with contextlib.ExitStack() as stack:
        encoder_calls = stack.enter_context(_capture_calls(model.get_encoder(), "forward"))
        cross_writers = stack.enter_context(capture_cross_writers(model))
        live = {
            name: stack.enter_context(_capture_calls(owner, _MODALITY_GETTERS[name]))
            for name, owner in modality_owners.items()
        }
        components = capture()
    encoder_inputs = {key: value for key, value in encoder_calls[0].items() if isinstance(value, torch.Tensor)}
    components = {
        "encoder": Component("encoder", model.get_encoder(), encoder_inputs, ComponentRole.ENCODER),
        **components,
    }
    return components, {name: list(calls) for name, calls in live.items() if calls}, cross_writers


def _streaming_embedder(model, inputs) -> Component | None:
    """The pre-loop embedder of a model whose modality advances with the text (`_STREAMING_EMBEDDERS`)."""
    spec = streaming_embedder_spec(model.config)
    if spec is None or inputs.get(spec.source) is None:
        return None
    module = model.base_model
    for attribute in spec.path.split("."):
        module = getattr(module, attribute, None)
        if module is None:
            return None
    return Component(spec.component, module, {spec.source: inputs[spec.source]}, ComponentRole.STREAMING_EMBEDDER)


def _embedded_inputs(call_inputs: dict, components: dict) -> dict:
    """A text graph's captured kwargs, rewritten to take embeddings instead of token ids and modality inputs,
    so the runtime can scatter features in."""
    call_inputs = copy.copy(call_inputs)
    for _name, _getter, input_keys, grid_key, _token_field in _MODALITY_SPECS:
        for input_key in input_keys:
            call_inputs.pop(input_key, None)
        if grid_key is not None:
            call_inputs.pop(grid_key, None)
    # Only the runtime's anyres packing reads the image sizes.
    call_inputs.pop("image_sizes", None)
    # `mm_token_type_ids` only feeds `get_rope_index`, unused once `position_ids` is given.
    if call_inputs.get("position_ids") is not None:
        call_inputs.pop("mm_token_type_ids", None)
    if call_inputs.get("input_ids") is not None and "embed_tokens" in components:
        with torch.no_grad():
            embedded = components["embed_tokens"].module(call_inputs.pop("input_ids"))
        # Per-layer embeddings, when returned, are decode inputs too.
        call_inputs.update(embedded if isinstance(embedded, dict) else {"inputs_embeds": embedded})
    return call_inputs


def decompose_for_generation(
    model: PreTrainedModel,
    inputs: dict[str, Any],
    generation_config: Any = None,
    multi_token_decode: bool = False,
    decoder_writes_cross_cache: bool = False,
) -> dict[str, Component]:
    """Decompose a generative model into independently exportable components.

    Captures prefill and decode kwargs from a real `generate`, splits a multi-modal prompt into components,
    and drops the prompt's graph wherever the decode graph can serve it.

    Args:
        model: Generative model. Must support `model.generate(**inputs)`.
        inputs: **Generate** kwargs — what you'd pass to `model.generate(**inputs)`.
        generation_config: Optional `GenerationConfig` forwarded to `generate()`; use
            `cache_implementation="static"` + `max_cache_len=N` for a static cache.
        multi_token_decode: Capture `decode` with a dynamic query axis, so it can also serve the prompt.
        decoder_writes_cross_cache: With `multi_token_decode`, have the decode graph write its own
            cross-attention cache instead of the encoder (for backends that keep the cache as state).

    Returns:
        `{component_name: Component}`: `"prefill"` / `"decode"`, plus `"embed_tokens"`, modality encoders
        and `"encoder"` as the architecture calls for. A multi-modal `decode` takes `inputs_embeds`.
    """
    components, recorded_features, cross_writers = _capture_generation(
        model, inputs, generation_config, multi_token_decode
    )

    prompt = components["prefill"]
    multimodal = is_multimodal(prompt.module)
    if multimodal:
        split = decompose_multimodal(prompt.module, prompt.inputs, recorded_features, inputs.get("input_ids"))
        # The decode component serves the text stack; keeping `text_decoder` would duplicate its parameters.
        split.pop("text_decoder", None)
        components.update(split)
        if (embedder := _streaming_embedder(model, inputs)) is not None:
            components[embedder.name] = embedder

    cross_written_by_decoder = False
    if multi_token_decode and decoder_writes_cross_cache:
        components, cross_written_by_decoder = _write_cross_cache_in_decoder(components)
    elif multi_token_decode:
        components = _fold_cross_cache_into_encoder(model, components, cross_writers)

    written = True if cross_written_by_decoder else None
    if not multi_token_decode or _needs_prefill_graph(model, components, cross_written_without_prompt=written):
        _materialize_prefill_cache(model, components["prefill"].inputs, components["decode"].inputs)
    else:
        del components["prefill"]

    # A kept prompt graph takes embeddings only if modality graphs exist; idefics-style runs its tower inline.
    if multimodal:
        scattered = any(name.endswith("_encoder") for name in components)
        for name in ("decode", "prefill") if scattered else ("decode",):
            if (component := components.get(name)) is not None:
                components[name] = replace(component, inputs=_embedded_inputs(component.inputs, components))
    return components
