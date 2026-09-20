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
from typing import Any, NamedTuple

from ..utils import logging
from ..utils.import_utils import is_torch_available
from .cache import (
    _cache_kv_geometry,
    check_cache_geometry,
    indexer_layers_of,
    kv_geometry_of,
    materialize_cache_layers,
)
from .utils import (
    ModalityEncoder,
    PatchVisionEncoder,
    TokenEmbedder,
    _find_config_attr,
    module_device,
    module_dtype,
    precompute_export_inputs,
)


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

    from ..cache_utils import DynamicCrossAttentionLayer
    from ..modeling_utils import PreTrainedModel


@contextlib.contextmanager
def _capture_calls(obj: Any, attribute: str):
    """Capture the kwargs of each `obj.<attribute>(...)` call during the block (positional args normalised
    to kwargs), restoring the attribute afterwards. Generalises `_capture_forward` to any method — used to
    record exactly what the model passes each `get_*_features`, so we don't hardcode per-model input keys."""
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
        # Restore by *deleting* the wrapper unless the attribute was an instance attribute to begin with:
        # assigning the bound method back would pin it in the instance dict, and a later `copy.copy` of the
        # module (the prefill/decode split) would then carry a method bound to the original -- calls on the
        # copy would run with `self` = the original, silently escaping any patch or capture applied to the
        # copy (that is how the modality-getter capture came back empty on qwen2_5_omni).
        if was_instance_attr:
            setattr(obj, attribute, original)
        else:
            delattr(obj, attribute)


@contextlib.contextmanager
def _capture_forward(module: torch.nn.Module):
    """Capture each `module(...)` call's kwargs -- `_capture_calls` on the method every module has."""
    with _capture_calls(module, "forward") as calls:
        yield calls


def _merge_decode_calls(decode_calls: list[dict], streamed: str | None = None) -> dict:
    """Merge consecutive single-token decode captures into one multi-token decode input.

    Each `model.generate` decode step feeds a single new token, so `torch.export` (with `Dim.AUTO`)
    sees a query-sequence axis of length 1 and specializes it to a constant — the exported decode can
    then only ever run one token. Concatenating `N` consecutive decode steps along that axis yields a
    genuine `N`-token decode input: the traced graph is identical (a KV-cache forward), but the sequence
    axis now has hint `N > 1` so it stays dynamic. The exported decode then handles both a single token
    (ordinary decoding) and many (continuation-from-past for multi-turn, or a plain prefill when the cache is empty).

    The cache (`past_key_values`) is taken from the FIRST step (the state right after prefill, before
    the chunk). The per-token tensors are concatenated along their sequence axis; `attention_mask` is
    handled below (its layout depends on the cache).
    """
    first = decode_calls[0]
    merged = copy.copy(first)

    # Concatenation assumes single-token decode steps. When `use_cache` is off the steps re-run the whole
    # growing sequence (query length > 1) — each is already a valid multi-token forward, so take the last.
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
    # `position_ids` is `[batch, seq]` or `[n_axes, batch, seq]` (m-rope) — the sequence axis is
    # last in both, so a negative dim concatenates it correctly either way.
    concat_along("position_ids", -1)
    # `token_type_ids` is per-token too, and left at one step it specializes the merged graph's query axis
    # back to 1 (`Guard failed: token_type_ids.size()[1] == 1`) — defeating the whole point of the merge.
    concat_along("token_type_ids", -1)
    # A streamed modality's window (`_STREAMING_EMBEDDERS`) is per-token as well, just at its own stride:
    # each step carries the rows its one token spans, so concatenating the steps rebuilds the window the
    # merged query needs. Left at one step, the graph bakes the ratio (`encoder_inputs_embeds.size()[1] //
    # 4 == 1`) and no multi-token step can satisfy it.
    if streamed is not None:
        concat_along(streamed, 1)

    # `attention_mask` is either a 2D padding mask `[batch, kv]` (a growing `DynamicCache`: the model
    # rebuilds the causal mask from `position_ids` / `cache_position` internally, so the last step's
    # mask — spanning the most positions — is all it needs) or a 4D causal mask `[batch, heads, query,
    # kv]` (a static cache passes the mask in explicitly). For the 4D case each single-token step is one
    # causal query row against the fixed-size cache, so concatenating along the query axis rebuilds the
    # correct `N`-token causal mask; taking just the last step would freeze the query axis at 1 and the
    # exported decode could never run more than one token. Hybrid-attention models pass a dict
    # `{attention_type: 4D mask}` instead of a single tensor — merge each entry the same way.
    masks = [call.get("attention_mask") for call in decode_calls]
    if all(mask is not None for mask in masks):
        merged["attention_mask"] = _merge_step_masks(masks)

    return merged


def _merge_step_masks(masks: list[Any]) -> Any:
    """Merge one attention mask per decode step into a single multi-token mask.

    A 4D causal mask `[batch, heads, query, kv]` is concatenated along the query axis (each step is one
    causal row against the fixed cache); a 2D padding mask keeps the last step (it already spans the most
    positions). A dict `{attention_type: mask}` (hybrid-attention models) is merged entry by entry, and a
    `None` mask (an attention type the model leaves unmasked) is preserved as `None`.
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
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Run `model.generate()` and capture prefill and decode inputs.

    Reuses the full generation machinery so every architecture (decoder-only, SSM,
    encoder-decoder, multi-modal, …) gets correct inputs without reimplementing the loop.

    `generation_config` is forwarded to `generate()` (defaulting to the model's own), so the captured
    inputs use whatever cache `generate()` would build. Pass one with `cache_implementation="static"`
    and `max_cache_len=N` to capture a **statically sized** cache in the decode inputs — the basis for
    a static-cache export. `max_cache_len` sizes the cache independently of the capture, so the
    exported decode takes a fixed `[..., N, ...]` cache rather than a growing one.

    When `multi_token_decode`, the `decode` component is captured as a **multi-token** decode — two
    consecutive decode steps merged (see `_merge_decode_calls`) so its query-sequence axis stays
    symbolic (a single-token decode would specialize that axis to 1). It then handles both one token
    (ordinary decoding) and many (continuation-from-past, or a plain prefill when the cache is empty). Otherwise `decode` is the
    classic single-token decode.

    Returns:
        `dict[str, tuple[torch.nn.Module, dict]]`:
        `{"prefill": (model, prefill_inputs), "decode": (model, decode_inputs)}`.
    """
    # 1 prefill forward + 1 decode (or 2 decode steps merged, when `multi_token_decode`) forward to capture.
    # Set the capture window on the config itself, not as generate() kwargs — passing a
    # `generation_config` alongside generation kwargs is deprecated. Base it on the model's own config
    # when none is given (preserving its defaults), and deep-copy into a distinct `capture_config` so
    # the caller's `generation_config` is never mutated.
    # Encoder-decoder decoding is single-token from an (almost) empty self-attention cache: the first
    # decode step runs at cache length 1, which 0/1 specialization would freeze into the graph — capture
    # from the SECOND decode step (cache length 2, symbolic) instead.
    spec = streaming_embedder_spec(model.config)
    streamed_kwarg = spec.produces if spec is not None else None
    first_decode = 2 if getattr(model.config, "is_encoder_decoder", False) else 1
    num_new_tokens = first_decode + (2 if multi_token_decode else 1)
    capture_config = copy.deepcopy(generation_config if generation_config is not None else model.generation_config)
    capture_config.max_new_tokens = num_new_tokens
    capture_config.min_new_tokens = num_new_tokens
    try:
        with _capture_forward(model) as calls:
            model.generate(**copy.deepcopy(inputs), generation_config=capture_config)
    except Exception as e:
        # A cache-shape error here means the geometry `_cache_kv_geometry` derived for the materialized
        # cache disagrees with what the model writes into it — say so, rather than leaving an
        # `index_copy_()` slice error from deep inside the forward.
        if "slice shapes" in str(e) or "Sizes of tensors must match" in str(e):
            raise RuntimeError(
                f"decompose_prefill_decode failed for {type(model).__name__}: the exporter materialized the "
                f"cache as (heads, key_dim, value_dim)={_cache_kv_geometry(model.config, 0)}, which is not "
                "what this model caches — `_cache_kv_geometry` needs to learn its layout."
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

    # Remove `logits_to_keep` from the captured calls — it's a generation-time hint for the model's
    # internal top-k pruning, not a forward input. The export graph should not depend on it.
    for call in calls:
        call.pop("logits_to_keep", None)

    # A single-token decode specializes its query-sequence axis to 1 (never dynamic). When
    # `multi_token_decode`, merge the two decode steps into one multi-token decode so that axis stays
    # symbolic (continuation-from-past, or a plain prefill when the cache is empty, and it still covers seq == 1).
    prefill_inputs = calls[0]
    decode_inputs = (
        _merge_decode_calls(calls[first_decode:num_new_tokens], streamed=streamed_kwarg)
        if multi_token_decode
        else calls[first_decode]
    )
    # `generate` built this cache itself, so its layers carry the model's real geometry — the one thing
    # that can tell us whether the geometry the exporter derives (and materializes for the prefill capture
    # and the runtime) is right for this architecture.
    if (captured_cache := decode_inputs.get("past_key_values")) is not None:
        check_cache_geometry(model.config, captured_cache)

    return {
        "prefill": (copy.copy(model), prefill_inputs),
        "decode": (copy.copy(model), decode_inputs),
    }


def _multimodal_text_decoder(model: PreTrainedModel | torch.nn.Module) -> torch.nn.Module | None:
    """The text decoder of a multi-modal model, or `None` when the model is not one to decompose.

    Multi-modal takes both halves: a modality to export a graph from, and a decoder that is not the model
    itself. The modality half is evidence rather than a module — the components come from the
    `get_<modality>_features` getters (`decompose_multimodal`), never from an encoder module — so an
    encoder is looked for only to answer the question, and a model that composes a modality without naming
    one counts too: vibevoice_asr runs two co-equal audio encoders and so reports none, yet it has the
    getter the split exports from. Without that second reading such a model is declared single-modal and
    its whole audio path, data-dependent asserts and all, stays inside the text prefill graph.

    A non-`PreTrainedModel` (a bare `nn.Module`) has no canonical `get_encoder` / `get_decoder` accessors
    and is trivially not multi-modal.
    """
    if not isinstance(model, PreTrainedModel):
        return None
    decoder = model.get_decoder()
    if decoder is None or decoder is model:
        return None
    for modality in ("image", "audio"):
        # `get_encoder` returns `self` as the "no match" fallback, and some models keep
        # `self.audio_tower = None` / `self.vision_tower = None` when the corresponding sub-config is
        # absent — `hasattr` is True but `getattr` is None.
        encoder = model.get_encoder(modality=modality)
        if encoder is not None and encoder is not model:
            return decoder
    if any(_modality_owner(model, getter) is not None for _name, getter, *_ in _MODALITY_SPECS):
        return decoder
    return None


def is_multimodal(model: PreTrainedModel | torch.nn.Module) -> bool:
    """Returns `True` if the model is multi-modal with a modality to export and a language model."""
    return _multimodal_text_decoder(model) is not None


# One row per input modality: (component name, `get_*_features` method, the input kwarg that signals the
# modality is present, the getter's native grid kwarg — or `None` for audio, the placeholder-id config
# field). Video/audio slot in exactly like image; a modality is exported only when its getter exists and
# its input is passed.
# The input kwarg is a tuple: a model may name the same modality differently (video_llava splits images
# and videos, so its images arrive as `pixel_values_images`). The first name present is the one used.
# A modality the model embeds once *before* the decode loop, handing each step a window of the result: its
# `generate` runs the embedder in `_prepare_model_inputs` and `prepare_inputs_for_generation` then slices the
# output by `past_seen_tokens * <stride>`, so the audio advances with the text. The decode graph therefore
# takes the *window* rather than the raw features, and the runtime needs the embedder as a graph of its own —
# no modality component fits, because there are no placeholder rows to scatter into (the model sums its
# features into the prompt). `(component, submodule path under the base model, source kwarg, the kwarg it
# produces, stride config field)`, keyed by model type.
# Suffixes that mark a modality input as *aux* rather than the features themselves: a grid, a scatter mask,
# a per-sample padding mask, an image-size table. Presence checks and feature routing both skip these —
# `generate` may keep one after dropping the features it described, and routing one onto a tower's feature
# input would feed it a mask where it wants pixels.
_MODALITY_AUX_SUFFIXES = ("_grid_thw", "_position_mask", "_attention_mask", "_sizes", "padding_mask", "_indices")


class StreamingEmbedderSpec(NamedTuple):
    """How one model streams a modality alongside its text — see `_STREAMING_EMBEDDERS`."""

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
    """This config's `_STREAMING_EMBEDDERS` entry, or `None` for a model that carries its modality in the
    prompt like every other."""
    return _STREAMING_EMBEDDERS.get(getattr(config, "model_type", None))


_MODALITY_SPECS = (
    (
        "image_encoder",
        "get_image_features",
        # `image_embeds_position_mask` (kosmos2_5) is the modality's scatter mask, `pixel_attention_mask`
        # (lfm2_vl's NaViT packing) its per-patch padding mask and `target_sizes` (minicpmv4_6) its grid in
        # all but name — aux kwargs, not features: every presence check skips `*_position_mask` /
        # `*_attention_mask` / `*_sizes` keys the way it skips the grids.
        (
            "pixel_values",
            "pixel_values_images",
            "flattened_patches",
            # fuyu hands its patches straight to the getter's `pixel_values` parameter under its own name,
            # with `image_patches_indices` alongside as the aux index tensor.
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
    # `audio_input_ids` (inkling) and `input_values` (vibevoice_asr, a raw waveform rather than a
    # spectrogram) are the same slot as `input_features` — the tensor whose presence means this call carries
    # audio — just named for what their own getter takes. `padding_mask` rides along with the waveform as its
    # per-sample mask, so it is an aux key (`_MODALITY_AUX_SUFFIXES`), never the feature itself.
    (
        "audio_encoder",
        "get_audio_features",
        ("input_features", "audio_input_ids", "input_values", "padding_mask"),
        None,
        "audio_token_id",
    ),
)

_MODALITY_GETTERS = {name: getter for name, getter, *_ in _MODALITY_SPECS}


def anyres_patch_counts(config: Any, image_sizes) -> list[int]:
    """Tiles each image snaps to, plus the base patch — the per-image split sizes `get_image_features`
    derives from `image_sizes`. Config-only, so the export and the runtime agree without a model."""
    from ..image_processing_utils import select_best_resolution

    pinpoints = _find_config_attr(config, "image_grid_pinpoints")
    tile = _find_config_attr(config, "image_size")
    counts = []
    for size in image_sizes:
        height, width = select_best_resolution(size.tolist() if hasattr(size, "tolist") else list(size), pinpoints)
        counts.append(-(-height // tile) * -(-width // tile) + 1)
    return counts


def flatten_anyres_patches(config: Any, pixel_values, image_sizes):
    """The flat `(total_patches, …)` tensor the tower takes, dropping each image's padding rows. The getter
    does this from `image_sizes`; doing it here keeps the sizes out of the graph entirely."""
    if pixel_values.dim() != 5:
        return pixel_values
    counts = anyres_patch_counts(config, image_sizes)
    return torch.cat([pix[:count] for pix, count in zip(pixel_values, counts)], dim=0)


def packs_anyres_features(owner: Any, config: Any) -> bool:
    """Whether this image getter ends in the anyres `pack_image_features`, so the component must stop at the
    projector and the runtime packs instead. Covers a deepstack tower (granite4_vision) too — it just has one
    projector per injected decoder layer rather than one overall."""
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
    """Whether `decoder` turns `input_ids` into embeddings with a single module.

    Models that embed nothing raise rather than return. A multi-codebook decoder (musicgen) answers with a
    `ModuleList` — one embedding per codebook — which is a container with no `forward`, so there is no one
    `input_ids -> inputs_embeds` graph to export for it either."""
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
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Split a multi-modal model into independently exportable `name: (module, inputs)` pairs.

    Exports the model's own composition methods rather than raw submodules, so each component is
    self-contained and the set can be reassembled into a generation runtime:
    - `embed_tokens` — `input_ids -> inputs_embeds` (`get_input_embeddings`, placeholder ids zeroed),
    - `<modality>_encoder` — the modality features (`get_<modality>_features`, i.e. encoder **and**
      projection), one per input modality present (image / video / audio),
    - `text_decoder` — captured from the forward (`get_decoder()`; it produces the logits, the head included).

    The token-merge step (`masked_scatter`) stays outside the graphs — the caller assembles
    `inputs_embeds` from the encoder outputs before running the decoder.

    Raises:
        `ValueError`: if no known multi-modal submodules are found on the model.
    """
    decoder = _multimodal_text_decoder(model)
    if decoder is None:
        raise ValueError(
            f"decompose_multimodal found no multi-modal submodules on {type(model).__name__}. "
            f"Expected an image/audio encoder + language model, found neither."
        )

    # Each active modality's `get_*_features` is invoked on the base model during `forward` (the outer
    # `ForConditionalGeneration` getter just delegates), so capture — and later export — from there.
    # `inputs` says which modalities this forward carries. A model that consumed them earlier — an
    # encoder-decoder feeds its images through the encoder, so by prefill they are gone — has none left to
    # find, and the caller instead hands over what it recorded the getters doing during the same generate.
    recorded_features = recorded_features or {}
    active_modalities = []
    for name, getter, input_keys, grid_key, _token_field in _MODALITY_SPECS:
        if (owner := _modality_owner(model, getter)) is None:
            continue
        if _present_input_key(inputs, input_keys) is not None or recorded_features.get(name):
            active_modalities.append((name, getter, owner, grid_key))

    # the `text_decoder` takes activations, not user inputs, so capture its kwargs; capture each
    # modality getter's call kwargs — all in one real forward.
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

    components = {"text_decoder": (decoder, decoder_calls[-1])} if decoder_calls else {}

    # embed_tokens: `input_ids -> inputs_embeds`, zeroing the placeholder ids (out of the text vocab)
    # first, the way a VLM `forward` does before scattering in encoder features. Only a model whose
    # prompt *is* text gets one: an encoder-decoder (t5gemma, seamless_m4t, …) prompts with the encoded
    # modality and reaches its decoder through cross-attention, with no placeholder rows to scatter into,
    # so there is nothing for this component to do and the runtime drives those graphs directly. A
    # dual-encoder (owlvit, groupvit, …) has a text tower rather than a decoder that embeds ids, and says
    # so by refusing `get_input_embeddings` — take it at its word instead of failing the whole export.
    # `prompt_ids` covers the same gap as `recorded_features`: an encoder-decoder's prefill kwargs carry
    # `decoder_input_ids`, so the prompt this component embeds has to come from the generate inputs.
    token_ids = inputs.get("input_ids") if inputs.get("input_ids") is not None else prompt_ids
    if token_ids is not None and _embeds_input_ids(decoder):
        placeholder_ids = [
            getattr(model.config, spec[-1], None)
            for spec in _MODALITY_SPECS
            if getattr(model.config, spec[-1], None) is not None
        ]
        components["embed_tokens"] = (
            TokenEmbedder(decoder, placeholder_ids),
            {"input_ids": token_ids},
        )

    # One feature graph per modality, from the captured getter call — a `ModalityEncoder` wrapping the
    # owner, whose `forward` runs `get_<modality>_features` and delegates introspection to the model.
    for name, getter, owner, grid_key in active_modalities:
        calls = captured_features.get(name) or recorded_features.get(name) or []
        if not calls:
            continue
        feature_inputs = {
            ("grid_thw" if key == grid_key else key): value for key, value in calls[-1].items() if value is not None
        }
        # Hand the graph the tensors the precompute derives from the config, so it takes them as inputs
        # instead of deriving them itself — the point of the precompute, and the only way a getter that
        # reads its grid or image sizes *as data* can be traced at all.
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
            components[name] = (PatchVisionEncoder(owner), tower_inputs)
            continue
        feature_inputs = precompute_export_inputs(model.config, feature_inputs)
        components[name] = (ModalityEncoder(owner, getter, grid_key), feature_inputs)
    return components


def _prefill_reason(model, stages: dict, components: dict, prefill_inputs: dict) -> str | None:
    """Why the prompt needs a graph of its own, or `None` when the decode graph can serve it.

    The merged decode graph covers a prompt on a fresh cache, so most models need no separate prefill.
    These are the shapes where it cannot:

    - an encoder-decoder's prefill is the one graph that *writes* the cross-attention cache; the decode
      graph is captured after it and only reads it (`is_updated` bakes as trace-time context);
    - a modality that *streams* alongside the text unifies its token count with the query length in a
      merged decode, then bakes a query-length branch, so that graph cannot serve single-token steps;
    - a text stack whose layers keep written-once state (conv / linear-attention / indexer slots, or an
      idefics-style gated cross-attention cache) has the same writer/reader split as the first case;
    - the prompt carries a modality no component took — a model whose images enter through
      cross-attention (mllama, idefics) has no `get_<modality>_features` getter to split out, so its vision
      tower runs inside the prompt's own forward and the merged decode graph never sees pixels.

    The last one asks only about the *features* a modality arrives as. Its other kwargs are a different
    question: the runtime consumes `image_sizes` itself (it sizes the anyres merge, while the packed vision
    graph takes the flattened patches), so a leftover of that kind says nothing about which graph runs.
    """
    if "encoder" in stages:
        return "encoder-decoder: the prefill writes the cross-attention cache"
    if streaming_embedder_spec(model.config) is not None:
        return "a streaming modality bakes a query-length branch into the merged decode"
    layers = getattr(stages["decode"][1].get("past_key_values"), "layers", [])
    if any(
        hasattr(layer, "conv_states")
        or hasattr(layer, "recurrent_states")
        or hasattr(layer, "idx_keys")
        or isinstance(layer, DynamicCrossAttentionLayer)
        for layer in layers
    ):
        return "the text stack keeps state only the prompt's forward writes"
    # Per modality, not per kwarg name: a getter takes its features under its own parameter name (the video
    # one takes `pixel_values`, not `pixel_values_videos`), so the component's presence is the fact to read.
    unserved = sorted(
        key
        for name, _getter, input_keys, *_ in _MODALITY_SPECS
        if name not in components
        for key in input_keys
        if prefill_inputs.get(key) is not None
    )
    if unserved:
        return f"the prompt's own forward computes {unserved}"
    return None


def _materialize_stage_cache(model, stage_inputs: dict, decode_inputs: dict) -> None:
    """Fill in a captured stage's cache, so it is traced against the cache the runtime will feed it.

    What a capture hands back is the pre-forward, lazily-uninitialized cache, while `torch.export` bakes
    real tensors into the graph's input spec. The geometry comes from the *decode* capture's cache, which
    ran after prefill and so carries what the model really writes, per layer — better than any derivation
    from the config. Its sparse-indexer slots come from there for the same reason, and for one more: a
    layer the model leaves empty (hy_v4's "shared" indexer layers reuse another's) must stay empty here too,
    or the prefill graph takes a leaf the decode graph does not.
    """
    cache = stage_inputs.get("past_key_values")
    if cache is None:
        return
    batch_size = next(t for t in stage_inputs.values() if isinstance(t, torch.Tensor)).shape[0]
    materialize_cache_layers(
        cache,
        batch_size,
        model.config,
        module_dtype(model),
        module_device(model),
        kv_geometry=kv_geometry_of(decode_inputs.get("past_key_values")),
        indexer_layers=indexer_layers_of(decode_inputs.get("past_key_values")),
    )


def decompose_for_generation(
    model: PreTrainedModel, inputs: dict[str, Any], generation_config: Any = None, multi_token_decode: bool = False
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Decompose a generative model into independently exportable `(model, forward_inputs)` pairs.

    Runs `decompose_prefill_decode` to capture prefill and decode forward kwargs from a real
    `model.generate(**inputs, max_new_tokens=2)`. If the prefill is multi-modal (per `is_multimodal`),
    further splits it into one entry per submodule (vision/audio encoder, projector, language model,
    `text_decoder`) via `decompose_multimodal`.

    Args:
        model: Generative model. Must support `model.generate(**inputs)`.
        inputs: **Generate** kwargs — what you'd pass to `model.generate(**inputs)`.
        generation_config: Optional `GenerationConfig` forwarded to `generate()` during capture. Pass
            one with `cache_implementation="static"` + `max_cache_len=N` to export against a statically
            sized cache (see `decompose_prefill_decode`).
        multi_token_decode: When `True`, capture the `decode` component as a multi-token decode (dynamic
            query sequence axis: multiple tokens at once — continuation-from-past, or a plain prefill when the cache is empty); a
            single-token decode can't stay dynamic (see `decompose_prefill_decode`).

    Returns:
        `{component_name: (submodel, forward_inputs)}`. Keys are `"prefill"` / `"decode"` for
        plain generative models and `"embed_tokens"` / `"image_encoder"` / `"audio_encoder"` /
        `"text_decoder"` / `"decode"` for multi-modal generative models. For multi-modal
        models the `decode` component takes `inputs_embeds` (not `input_ids`) so the caller can scatter the
        encoder features into the embeddings before running it.
    """
    recorded_features: dict[str, list] = {}
    if getattr(model.config, "is_encoder_decoder", False):
        # `generate` runs the encoder once outside the decoder loop (`get_encoder()(...)`), so it never
        # appears in the captured forwards — capture its call during the same generate to export it as its
        # own component (the runtime serves it back through `get_encoder()`).
        # Record the modality getters over the same generate: this model runs its vision tower once, into
        # the encoder, so the prefill kwargs the split sees below no longer carry the images.
        modality_owners = {
            name: owner
            for name, getter, *_ in _MODALITY_SPECS
            if (owner := _modality_owner(model, getter)) is not None
        }
        with contextlib.ExitStack() as stack:
            encoder_calls = stack.enter_context(_capture_calls(model.get_encoder(), "forward"))
            live = {
                name: stack.enter_context(_capture_calls(owner, _MODALITY_GETTERS[name]))
                for name, owner in modality_owners.items()
            }
            stages = decompose_prefill_decode(
                model, inputs, generation_config=generation_config, multi_token_decode=multi_token_decode
            )
        recorded_features = {name: list(calls) for name, calls in live.items() if calls}
        encoder_inputs = {k: v for k, v in encoder_calls[0].items() if isinstance(v, torch.Tensor)}
        stages = {"encoder": (model.get_encoder(), encoder_inputs), **stages}
    else:
        stages = decompose_prefill_decode(
            model, inputs, generation_config=generation_config, multi_token_decode=multi_token_decode
        )
    prefill_model, prefill_inputs = stages["prefill"]

    if not is_multimodal(prefill_model):
        # Text path only — the multi-modal prefill is discarded after the submodule split, and materializing
        # it would leak cache inputs into the `text_decoder` component's capture. (Decode, captured
        # post-prefill, already holds a materialized cache.)
        _materialize_stage_cache(model, prefill_inputs, stages["decode"][1])
        return stages

    components = decompose_multimodal(prefill_model, prefill_inputs, recorded_features, inputs.get("input_ids"))
    # The pre-loop embedder, for a model whose modality advances with the text (`_STREAMING_EMBEDDERS`): the
    # decode graph takes a window of its output, so without this graph nothing turns the raw features into
    # one and the runtime is handed a feature kwarg no graph declares.
    if (spec := streaming_embedder_spec(model.config)) is not None and inputs.get(spec.source) is not None:
        module = model.base_model
        for attribute in spec.path.split("."):
            module = getattr(module, attribute, None)
            if module is None:
                break
        if module is not None:
            components[spec.component] = (module, {spec.source: inputs[spec.source]})
    # The multi-modal split rebuilds the component set from the prefill, so carry over the stages that
    # belong to the model as a whole — an encoder-decoder's `encoder` runs once outside the decode loop and
    # is captured above, and dropping it leaves the runtime with a decode graph asking for `encoder_outputs`
    # nothing produces.
    if "encoder" in stages:
        components["encoder"] = stages["encoder"]
    # One decision, made once: does the first step need a graph of its own, or can the decode graph serve
    # it? Three branches used to answer it with the same body; the reasons differ, the answer does not.
    if "prefill" in stages and _prefill_reason(model, stages, components, prefill_inputs):
        _materialize_stage_cache(model, stages["prefill"][1], stages["decode"][1])
        components["prefill"] = stages["prefill"]

    # Feed the decode graph `inputs_embeds` (not `input_ids`) so the runtime can scatter the encoder embeds
    # into the embeddings before the text stack; the full forward accepts `inputs_embeds` and — with no
    # modality inputs — skips the encoders. This is what lets the components reassemble into a loop.
    def as_embedded_inputs(call_inputs: dict) -> dict:
        call_inputs = copy.copy(call_inputs)
        for _name, _getter, _input_keys, grid_key, _token_field in _MODALITY_SPECS:
            for _input_key in _input_keys:
                call_inputs.pop(_input_key, None)
            if grid_key is not None:
                call_inputs.pop(grid_key, None)
        # `mm_token_type_ids` only drives the model's internal M-RoPE (`get_rope_index`); once `position_ids`
        # is captured, the forward never reads it. Drop it from the graph's inputs so the runtime (which
        # supplies `position_ids` by running that same `get_rope_index`) needn't thread a per-step
        # token-type tensor.
        if call_inputs.get("position_ids") is not None:
            call_inputs.pop("mm_token_type_ids", None)
        if call_inputs.get("input_ids") is not None and "embed_tokens" in components:
            embedding = components["embed_tokens"][0]
            with torch.no_grad():
                embedded = embedding(call_inputs.pop("input_ids"))
            # A decoder with per-layer embeddings returns those alongside `inputs_embeds`; both are per-token
            # inputs of the decode graph, so they go in as they come out.
            call_inputs.update(embedded if isinstance(embedded, dict) else {"inputs_embeds": embedded})
        return call_inputs

    components["decode"] = (stages["decode"][0], as_embedded_inputs(stages["decode"][1]))
    # Same treatment for a kept prefill when the modality graphs exist: the runtime scatters the features
    # in front of it, and the vision tower — whose data-dependent packing only traces at the getter seam,
    # where the precompute injects its tensors — stays out of the graph. The idefics-style writer (no
    # modality getter) keeps its raw inputs: the tower inline is the point there.
    if "prefill" in components and any(name.endswith("_encoder") for name in components):
        components["prefill"] = (components["prefill"][0], as_embedded_inputs(components["prefill"][1]))
    return components
