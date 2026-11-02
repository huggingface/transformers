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
"""Splitting a multi-modal model into the graphs an export needs.

A multi-modal model embeds its text, runs a vision or audio tower on each modality present, and scatters the
features into the text embeddings before its main stack. This records those calls in one forward, so the exporters
get one graph per modality, one for the token embeddings, and one for the model itself on `inputs_embeds`, with the
calls each graph is run with.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import inspect
from typing import Any, NamedTuple

from ..utils.import_utils import is_torch_available
from .components import Component, ModalityEncoder, PatchVisionEncoder, TokenEmbedder
from .precompute import _find_config_attr, precompute_export_inputs


if is_torch_available():
    import torch

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
    if any(_modality_owner(model, spec.getter) is not None for spec in _MODALITY_SPECS):
        return decoder
    return None


def is_multimodal(model: PreTrainedModel | torch.nn.Module) -> bool:
    """Returns `True` if the model is multi-modal with a modality to export and a language model."""
    return _multimodal_text_decoder(model) is not None


class ModalitySpec(NamedTuple):
    """One input modality a model may take, and the generate kwargs that carry it."""

    # The component exporting it, and the model method it is captured from
    component: str
    getter: str
    # The modality's data, under each name a model gives it; a call carrying one has the modality
    feature_keys: tuple[str, ...]
    # Side inputs riding along (masks, sizes, indices), which `generate` may keep after dropping the features
    aux_keys: tuple[str, ...]
    # The native grid kwarg, which the exported graph takes as `grid_thw`
    grid_key: str | None
    # The config field holding the placeholder token id its features scatter into
    token_field: str

    @property
    def input_keys(self) -> tuple[str, ...]:
        return self.feature_keys + self.aux_keys


_MODALITY_SPECS = (
    ModalitySpec(
        "vision_embeddings_model",
        "get_image_features",
        ("pixel_values", "pixel_values_images", "flattened_patches", "image_patches"),
        ("image_patches_indices", "image_embeds_position_mask", "pixel_attention_mask", "target_sizes"),
        "image_grid_thw",
        "image_token_id",
    ),
    ModalitySpec(
        "video_embeddings_model",
        "get_video_features",
        ("pixel_values_videos",),
        ("target_sizes_videos",),
        "video_grid_thw",
        "video_token_id",
    ),
    ModalitySpec(
        "audio_embeddings_model",
        "get_audio_features",
        ("input_features", "audio_input_ids", "input_values"),
        ("padding_mask",),
        None,
        "audio_token_id",
    ),
)


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


def pack_anyres_features(config: Any, features, image_sizes, image_newline=None, per_image: bool = False):
    """The `pack_image_features` step [`PatchVisionEncoder`] leaves out of the graph, with the modeling's own grid and
    unpad helpers: each image's base patch, then its tiles unpadded to the image's aspect ratio. Concatenated, or one
    tensor per image with `per_image`."""
    from ..models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image

    pinpoints = _find_config_attr(config, "image_grid_pinpoints")
    tile = _find_config_attr(config, "image_size")
    # From the projected tensor, not the config: a projector can downsample the token grid.
    side = round(features.shape[1] ** 0.5)
    packed = []
    for index, feature in enumerate(torch.split(features, anyres_patch_counts(config, image_sizes), dim=0)):
        if feature.shape[0] > 1:
            base, patches = feature[0], feature[1:]
            num_patch_height, num_patch_width = get_anyres_image_grid_shape(image_sizes[index], pinpoints, tile)
            patches = patches.view(num_patch_height, num_patch_width, side, side, -1).permute(4, 0, 2, 1, 3)
            patches = unpad_image(patches.flatten(1, 2).flatten(2, 3), image_sizes[index])
            if image_newline is not None:
                column = image_newline[:, None, None].expand(*patches.shape[:-1], 1).to(patches)
                patches = torch.cat((patches, column), dim=-1)
            packed.append(torch.cat((base, patches.flatten(1, 2).transpose(0, 1)), dim=0))
        else:
            feature = feature[0]
            if image_newline is not None:
                feature = torch.cat((feature, image_newline[None].to(feature)), dim=0)
            packed.append(feature)
    return packed if per_image else torch.cat(packed, dim=0)


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


def _embeds_input_ids(decoder: Any) -> bool:
    """Whether `decoder` turns `input_ids` into embeddings with a single module (not musicgen's
    per-codebook `ModuleList`)."""
    try:
        embeddings = decoder.get_input_embeddings() if decoder is not None else None
    except NotImplementedError:
        return False
    return isinstance(embeddings, torch.nn.Module) and not isinstance(embeddings, torch.nn.ModuleList)


def decompose_multimodal(model: PreTrainedModel, inputs: dict[str, Any]) -> dict[str, Component]:
    """Split a multi-modal model into independently exportable components.

    Components, named as Optimum Intel names them: `text_embeddings_model`, one `<modality>_embeddings_model` per
    modality present (`get_<modality>_features`), and `language_model`: `model` itself on `inputs_embeds`, so its head
    and anything after the text stack stay in the graph. The `masked_scatter` merge stays outside the graphs.

    Raises:
        `ValueError`: if no known multi-modal submodules are found on the model.
    """
    decoder = _multimodal_text_decoder(model)
    if decoder is None:
        raise ValueError(
            f"decompose_multimodal found no multi-modal submodules on {type(model).__name__}. "
            f"Expected an image/audio encoder + language model, found neither."
        )

    active_modalities = []
    for spec in _MODALITY_SPECS:
        if (owner := _modality_owner(model, spec.getter)) is None:
            continue
        if any(inputs.get(key) is not None for key in spec.input_keys):
            active_modalities.append((spec.component, spec.getter, owner, spec.grid_key))

    try:
        with contextlib.ExitStack() as stack, torch.no_grad():
            decoder_calls = stack.enter_context(_capture_calls(decoder, "forward"))
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
        {"language_model": Component(model, _main_inputs(model, inputs, decoder_calls[-1]))} if decoder_calls else {}
    )

    # Dual-encoders (owlvit) refuse `get_input_embeddings`.
    token_ids = inputs.get("input_ids")
    if token_ids is not None and _embeds_input_ids(decoder):
        placeholder_ids = [
            token_id
            for spec in _MODALITY_SPECS
            if (token_id := getattr(model.config, spec.token_field, None)) is not None
        ]
        components["text_embeddings_model"] = Component(
            TokenEmbedder(decoder, placeholder_ids), {"input_ids": token_ids}
        )

    for name, getter, owner, grid_key in active_modalities:
        calls = captured_features.get(name)
        if not calls:
            continue
        feature_inputs = {
            ("grid_thw" if key == grid_key else key): value for key, value in calls[-1].items() if value is not None
        }
        # Config-derived tensors become graph inputs: a getter reading its grid as data cannot be traced.
        if name == "vision_embeddings_model" and packs_anyres_features(owner, model.config):
            tower_inputs = {
                "pixel_values": flatten_anyres_patches(
                    model.config, feature_inputs["pixel_values"], feature_inputs["image_sizes"]
                ),
                **{
                    key: feature_inputs[key]
                    for key in ("vision_feature_layer", "vision_feature_select_strategy")
                    if key in feature_inputs
                },
            }
            components[name] = Component(PatchVisionEncoder(owner), tower_inputs)
            continue
        feature_inputs = precompute_export_inputs(model.config, feature_inputs)
        components[name] = Component(ModalityEncoder(owner, getter, grid_key), feature_inputs)
    return components


def embedded_inputs(call_inputs: dict, components: dict) -> dict:
    """A text graph's captured kwargs, rewritten to take embeddings instead of token ids and modality inputs,
    so the runtime can scatter features in."""
    call_inputs = copy.copy(call_inputs)
    for spec in _MODALITY_SPECS:
        for input_key in (*spec.input_keys, spec.grid_key):
            call_inputs.pop(input_key, None)
    # Only the runtime's anyres packing reads the image sizes.
    call_inputs.pop("image_sizes", None)
    # `mm_token_type_ids` only feeds `get_rope_index`, unused once `position_ids` is given.
    if call_inputs.get("position_ids") is not None:
        call_inputs.pop("mm_token_type_ids", None)
    if call_inputs.get("input_ids") is not None and "text_embeddings_model" in components:
        with torch.no_grad():
            embedded = components["text_embeddings_model"].module(call_inputs.pop("input_ids"))
        # Per-layer embeddings, when returned, are decode inputs too.
        call_inputs.update(embedded if isinstance(embedded, dict) else {"inputs_embeds": embedded})
    return call_inputs


# What a deepstack decoder adds at the visual positions inside its layers (Qwen3-VL's, Granite4-Vision's names).
DEEPSTACK_INPUTS = ("visual_pos_masks", "deepstack_visual_embeds", "vision_mask", "deepstack_features")


def _main_inputs(model: PreTrainedModel, inputs: dict[str, Any], decoder_call: dict[str, Any]) -> dict[str, Any]:
    """`model`'s own kwargs on `inputs_embeds`: the embeddings its text stack received, features scattered in,
    replace the token ids and modality inputs. What the model derived from those and can't recompute without them
    comes along: M-RoPE positions, computed by the config-only precompute as the runtime computes them, and deepstack
    features."""
    if decoder_call.get("inputs_embeds") is None:
        return inputs
    main_inputs = copy.copy(inputs)
    accepted = inspect.signature(model.forward).parameters
    if main_inputs.get("position_ids") is None and "position_ids" in accepted:
        position_ids = precompute_export_inputs(model.config, inputs).get("position_ids")
        if position_ids is not None:
            main_inputs["position_ids"] = position_ids
    main_inputs = embedded_inputs(main_inputs, {})
    main_inputs.pop("input_ids", None)
    main_inputs["inputs_embeds"] = decoder_call["inputs_embeds"]
    for name in DEEPSTACK_INPUTS:
        if decoder_call.get(name) is not None:
            main_inputs[name] = decoder_call[name]
    return main_inputs
