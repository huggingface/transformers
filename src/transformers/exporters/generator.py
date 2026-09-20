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
"""Run exported generative models by orchestrating their component graphs through `GenerationMixin.generate`.

`HfExporter.export_for_generation` produces one graph per component; this module plugs the graphs back
together and drives the generation loop from artifacts + configs alone — no model instance, no checkpoint
weights. The model config and the generation config **used at export** are the contract: save them with
the artifacts and hand them back to `ExportedGenerator.from_runners` — the generation config declares the
cache the graphs were traced against (growing `DynamicCache` vs fixed-size `StaticCache` + length), so
nothing is introspected from the graphs. Two public pieces:

- `ModelRunner` — wraps a backend's runtime handle (an `onnxruntime.InferenceSession`, a `torch.export`
  unlifted `module()`, a loaded ExecuTorch program) so it forwards like the module it was exported from:
  `runner(**kwargs) -> {name: tensor}`, torch tensors in and out. One subclass per backend
  (`OnnxModelRunner`, `DynamoModelRunner`, `ExecutorchModelRunner`), each hiding how its backend carries
  tensors (ORT's numpy boundary, executorch's positional buffers) and the KV-cache (flat `input.<name>` /
  `output.<name>` tensors vs the `Cache` pytree).
- `ExportedGenerator` — a `GenerationMixin` over the component runners: a `decode` graph alone is
  decoder-only text generation; add a `text_embed` graph and `Modality` entries (one features graph per
  image / video / audio input) and prefill scatters each modality's features into `inputs_embeds` at its
  placeholder positions, injecting the grid-derived precompute the graphs expect from the config alone.
  `ExportedGenerator.from_runners` assembles either kind from `{component_name: runner}` + the saved
  configs.
"""

from __future__ import annotations

import copy
import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from ..cache_utils import DynamicCache, EncoderDecoderCache, StaticCache
from ..generation import GenerationConfig, GenerationMixin
from ..masking_utils import create_masks_for_generate
from ..modeling_outputs import BaseModelOutput, CausalLMOutputWithPast
from ..models.auto import AutoConfig
from ..utils import GENERATION_CONFIG_NAME, logging
from .base import (
    ModelRunner,
    load_export_runners,
    resolve_export_file,
    split_download_kwargs,
)
from .cache import (
    _advance_cache,
    _cache_length,
    _empty_container,
    _self_attention_layers,
    mask_width,
    materialize_cache_layers,
)
from .decompose import (
    _MODALITY_AUX_SUFFIXES,
    _MODALITY_SPECS,
    anyres_patch_counts,
    flatten_anyres_patches,
    streaming_embedder_spec,
)
from .utils import (
    _find_config_attr,
    cast_leaf_tensors,
    get_rope_index_from_config,
    precompute_export_inputs,
    runner_feed,
)


logger = logging.get_logger(__name__)


# The text-path kwargs `generate` always carries. A modality graph that declares one of these names means
# its own (an audio encoder's `attention_mask` covers mel frames, not prompt tokens), so it is never
# sourced from `generate`'s — those come from the capture or the precompute instead.
_TEXT_KWARGS = frozenset(
    {
        "input_ids",
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "cache_position",
        "past_key_values",
        "cache_params",
        "use_cache",
    }
)


def _grid_renamed(key: str) -> str:
    """`generate` names the grid per modality (`image_grid_thw` / `video_grid_thw`); the exported graph
    takes the one the getter itself declares, `grid_thw`."""
    return "grid_thw" if key.endswith("_grid_thw") else key


@dataclass
class Modality:
    """Routes one input modality (image / video / audio) of an `ExportedGenerator`:

    - `token_id`: the placeholder id in `input_ids` its features scatter into (`config.image_token_id`, …);
      `None` for a model that marks the rows with an explicit `*_position_mask` kwarg instead (kosmos2_5).
    - `runner`: the exported `get_<modality>_features` graph.
    - `input_keys`: the generate kwargs that belong to it (e.g. `("pixel_values", "image_grid_thw")`); the
      first is the presence key — the modality runs only when it's passed.
    """

    token_id: int | None
    runner: ModelRunner
    input_keys: tuple


def _pack_anyres_features(config, features, image_sizes, outputs) -> torch.Tensor:
    """The packing `PatchVisionEncoder` leaves out of the graph: per image, the base patch's tokens followed
    by its patch grid reshaped, unpadded to the image's aspect ratio and given a newline column per row.
    Mirrors `pack_image_features`, calling the modeling's own grid/unpad helpers so the geometry lives in one
    place — the split optimum-intel uses, and the reason the graph is dynamic in image count and resolution."""
    from ..models.llava_next.modeling_llava_next import get_anyres_image_grid_shape, unpad_image
    from .utils import _find_config_attr

    newline = next((t for name, t in outputs.items() if name.endswith("image_newline")), None)
    pinpoints = _find_config_attr(config, "image_grid_pinpoints")
    tile = _find_config_attr(config, "image_size")
    # Tokens per patch, not `image_size // patch_size`: a deepstack projector downsamples the token grid, so
    # the square side has to come off the projected tensor rather than the vision config.
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


@dataclass
class StreamingEmbedder:
    """A modality embedded once before the decode loop, fed to each step as a window of the result.

    `runner` is the embedder graph, `source` the kwarg it consumes (`input_features`), `produces` the kwarg
    the decode graph takes the window under, and `stride` how many embedded rows one token spans — see
    `_STREAMING_EMBEDDERS`, which is where all of that comes from."""

    runner: ModelRunner
    source: str
    produces: str
    stride: int


class _ExportedEncoder:
    """`get_encoder()` stand-in over the exported encoder graph. `generate` filters its kwargs by
    `forward`'s signature (a wildcard here, so nothing is dropped), always adds the `output_*` /
    `return_dict` flags, and expects a `ModelOutput` back — the wrapper feeds the graph only what it
    declares and returns its outputs as the class the decoder graphs were traced with.

    That class is the encoder's own (`ExportMetadata.kwarg_class`), not a normalized `BaseModelOutput`:
    an encoder that returns more than hidden states — parakeet's frame mask, which the canary and
    cohere_asr decoders read off `encoder_outputs.attention_mask` — has nowhere else to put it, and the
    decode graphs take the kwarg as that type anyway."""

    def __init__(self, runner: ModelRunner, merge=None, output_class: type | None = None):
        self._runner = runner
        self._merge = merge
        self._output_class = output_class or BaseModelOutput

    def forward(self, **kwargs):
        # A multi-modal encoder-decoder merges its features once, in front of the text encoder — the graph
        # takes `inputs_embeds`, and `merge` (the generator's own embed-and-scatter) builds them from the
        # prompt and whichever modality inputs this call carries.
        if self._merge is not None:
            merged = self._merge(kwargs.pop("input_ids"), kwargs)
            # The merge keys its primary entry by the embed graph's own output name; this graph declares it
            # under its text input, the way the decode feed maps it.
            primary, *extra = merged
            kwargs[text_input(self._runner)] = merged[primary]
            kwargs.update({name: merged[name] for name in extra})
        feed = runner_feed(self._runner, kwargs)
        outputs = self._runner(**feed)
        # By name when the graph's outputs are the class's own fields, which is what a `ModelOutput`
        # encoder gives. An encoder that returned a bare tensor names its output whatever the trace named
        # it, and that tensor is the hidden states — the first field, and all a decoder reads from it.
        fields = {field.name for field in dataclasses.fields(self._output_class)}
        if outputs.keys() <= fields:
            return self._output_class(**outputs)
        return self._output_class(next(iter(outputs.values())))

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


# ── How the generation loop reads a runner's declared inputs ────────────────
# Pure derivations from `ModelRunner.input_names`, and generation-specific: which input carries the prompt,
# which the cache, which the mask. They live here rather than on the runner so a runner stays what it is —
# a callable graph with named inputs, a device and a dtype — and can serve any task its graph was exported
# for (classification, embeddings, …), not only generation.


def _mask_type(name: str) -> str:
    """The attention type a per-type mask input names, under either backend's flattening."""
    return name.removeprefix("attention_mask.").removeprefix("attention_mask_")


def _declares(runner, name: str, value) -> bool:
    """Whether this graph takes the feed entry `name` — directly, or as the pytree whose leaves it names.

    A pytree kwarg (`encoder_outputs`, a mask dict, the cache) goes in under its *kwarg* name and each runner
    flattens it to whatever its backend calls the leaves (`encoder_outputs.last_hidden_state` for ONNX,
    `encoder_outputs_last_hidden_state` for ExecuTorch, the kwarg itself for dynamo). Only a non-tensor value
    is flattened, so a plain tensor must be named outright — `input_features` is not declared by a graph that
    only takes `input_features_mask`."""
    if name in runner.input_names or name == runner.cache_input:
        return True
    return not isinstance(value, torch.Tensor) and any(
        declared.removeprefix("input.").startswith((f"{name}.", f"{name}_")) for declared in runner.input_names
    )


def _resize_to_traced_lengths(cache, lengths: dict[int, int]) -> None:
    """Size each fixed-size layer to the length the trace recorded for it, before its buffers are made.

    `generate` sizes a fixed cache from the prompt in front of it and from per-model facts — mllama's
    cross-attention layers hold the vision sequence, not the generation length — so re-deriving a size here
    gives a different cache for a different prompt. A graph carries its cache's sizes in the input spec and
    refuses to be called against any other, so the trace's own account is the one that fits.
    """
    if not lengths:
        return
    layers = _self_attention_layers(cache)
    for index, length in lengths.items():
        if index < len(layers) and getattr(layers[index], "max_cache_len", None) not in (None, length):
            layers[index].max_cache_len = length


# ── What a graph's inputs mean to a generation loop ──────────────────────────
# Derivations from what a runner declares, kept here rather than on `ModelRunner`: they answer questions
# only this loop asks ("which input carries the text", "where does the causal mask go"), and a runner
# stays what it is -- the thing that runs a graph.


def text_input(runner) -> str:
    """The graph's text input: `"decoder_input_ids"` (encoder-decoder decode), `"inputs_embeds"`
    (multi-modal decode) or `"input_ids"`."""
    return next((n for n in ("decoder_input_ids", "inputs_embeds") if n in runner.input_names), "input_ids")


def mask_inputs(runner) -> tuple[str, ...]:
    """The graph's attention-mask input name(s) — several for mixed full/sliding attention."""
    return tuple(
        n for n in runner.input_names if n == "attention_mask" or n.startswith(("attention_mask.", "attention_mask_"))
    )


def decoder_mask_input(runner) -> str | None:
    """`"decoder_attention_mask"` when the graph declares one. An encoder-decoder splits the two masks:
    `attention_mask` covers the *encoder's* sequence (what cross-attention reads) while this one covers
    the decoder's own — so the causal mask belongs here, and `generate` does not hand it over (the eager
    model builds it inside the forward the graph starts after)."""
    return "decoder_attention_mask" if "decoder_attention_mask" in runner.input_names else None


def mask_rank(runner) -> int | None:
    """The rank the graph's `attention_mask` was traced with, `None` when it takes none.

    `generate` upgrades a 2D padding mask to the 4D causal mask for any compileable cache, assuming the
    model's forward wants one — but an exported graph starts *after* whatever mask building its model
    does, so only the trace can say which it took. An alibi model (bloom) reads the 2D padding mask
    directly and compares its width to the cache length, so a 4D mask fails a guard rather than
    mismatching a shape.

    From the rank the trace recorded for the kwarg, not from a shape the artifact declares: ONNX reports
    the session's declared shapes, which is a different fact and gave this a different answer per backend.
    """
    return runner.export_metadata.kwargs.get("attention_mask", {}).get("rank")


def mask_dtype(runner) -> torch.dtype | None:
    """The dtype the graph's `attention_mask` was traced with, `None` when it takes none. A model reads a
    bool mask and a float one differently (a keep-mask vs an additive bias), so a mask built here is built
    as the one the graph took."""
    name = runner.export_metadata.kwargs.get("attention_mask", {}).get("dtype")
    return getattr(torch, name, None) if name else None


def mask_dict_ranks(runner) -> dict[str, int] | None:
    """`{attention type: rank}` when the graph took a *dict* of masks instead of one (mixed full/sliding
    attention), which a model builds inside its forward — so the runtime has to hand one in.

    Read off what the export recorded, in kwarg space, because the artifacts themselves do not agree on
    anything else: dynamo takes the dict as one kwarg and keeps the per-type keys in its pytree child spec,
    while ONNX and ExecuTorch flatten it to one input per type."""
    return runner.export_metadata.mask_ranks


class ExportedGenerator(GenerationMixin):
    """Drive exported component graphs through `generate`, from artifacts + configs alone.

    A `decode` runner alone is decoder-only text generation. Pass `text_embed` (the `input_ids -> inputs_embeds`
    graph) and `modalities` for a multi-modal model: prefill computes each modality's features and scatters
    them into `inputs_embeds` at its placeholder positions; decode steps are text-only. Pass `prefill=` to
    drive a fixed query=1 `decode` graph from a separate dynamic prefill graph (the shape CUDA-graph capture
    / io-binding want); when omitted, `decode` serves both (it must then be a multi-token graph, i.e.
    exported with `multi_token_decode=True`).

    `generation_config` must be the one the model was **exported with**: it declares the cache the decode
    graph was traced against (no `cache_implementation` → growing `DynamicCache`; `"static"` → fixed-size
    `StaticCache`, whose `max_cache_len` a static-cache export should pin explicitly).

    Example:
        programs = OnnxExporter().export_for_generation(model, inputs,
                                                        OnnxConfig(dynamic=True, external_data=False),
                                                        generation_config=generation_config,
                                                        multi_token_decode=True)
        session = ort.InferenceSession(programs["decode"].model_proto.SerializeToString())
        runtime = ExportedGenerator(model.config, generation_config, OnnxModelRunner(session))
        # Called like a normal model — the runtime builds the cache the exported graph needs.
        ids = runtime.generate(input_ids=prompt, max_new_tokens=32)
    """

    base_model_prefix = ""
    main_input_name = "input_ids"

    _supports_cache_class = True

    def __init__(
        self,
        config,
        generation_config,
        decode: ModelRunner,
        *,
        prefill: ModelRunner | None = None,
        encoder: ModelRunner | None = None,
        text_embed: ModelRunner | None = None,
        modalities: list[Modality] = (),
        embedder: StreamingEmbedder | None = None,
    ):
        self.config = config
        self.generation_config = generation_config
        self._decode_runner = decode
        # (`_prefill`/`_decode` without the suffix would shadow `GenerationMixin` methods `generate` calls.)
        self._prefill_runner = prefill if prefill is not None else decode
        self._text_embed = text_embed
        self._modalities = list(modalities)
        self._modality_keys = {key for modality in self._modalities for key in modality.input_keys}
        # An encoder-decoder scatters the features in front of its *text encoder* (florence2), not per
        # decode step — the encoder graph says so by taking `inputs_embeds` where a plain encoder-decoder's
        # takes `input_ids`.
        scatters_at_encoder = text_embed is not None and encoder is not None and "inputs_embeds" in encoder.input_names
        self._encoder = (
            _ExportedEncoder(
                encoder,
                merge=self._merge_modalities if scatters_at_encoder else None,
                # From whichever graph takes the object: the prompt's, usually, since a decode graph reads
                # the cross-attention cache the prefill wrote rather than the encoder's output again.
                output_class=next(
                    (
                        traced
                        for runner in (self._prefill_runner, decode)
                        if (traced := runner.export_metadata.kwarg_class("encoder_outputs")) is not None
                    ),
                    None,
                ),
            )
            if encoder is not None
            else None
        )
        # Everything the component graphs declare: `generate` kwargs matching these are fed straight through
        # (a model may take extra per-step tensors, e.g. `token_type_ids`), so they count as consumed.
        self._embedder = embedder
        # The embedded modality, kept across steps: it is computed once from the whole prompt's features and
        # each step reads its own window out of it.
        self._embedded = None
        # Which sub-config shapes the auxiliary cache the decode graph takes (the streaming encoder's own).
        self._encoder_config = (
            getattr(config, spec.encoder_config, None)
            if (spec := streaming_embedder_spec(config)) is not None
            else None
        )
        runners = [decode, self._prefill_runner, encoder, text_embed, *(m.runner for m in self._modalities)]
        if embedder is not None:
            runners.append(embedder.runner)
        self._graph_inputs = {name for runner in runners if runner is not None for name in runner.input_names}
        # `GenerationMixin` bookkeeping (where it builds tensors, at what precision) — read off the decode
        # runner, whose backend already decided where its outputs land.
        self._device = torch.device(decode.device)
        self._dtype = decode.dtype

    @classmethod
    def from_runners(
        cls,
        runners: dict[str, ModelRunner],
        config,
        generation_config: GenerationConfig | None = None,
    ) -> ExportedGenerator:
        """Assemble the generator from `{component_name: runner}` (the names
        `HfExporter.export_for_generation` produces) + the configs — text-only from a `"decode"` runner,
        multi-modal when `"embed_tokens"` and `"<modality>_encoder"` runners are present; each modality's
        precompute is built from `config` alone. `generation_config` must be the one the model was
        **exported with** (it declares the cache the graphs were traced against — save it with the
        artifacts); when `None`, the model config's own generation defaults apply (a growing cache). To
        load from disk, build each `ModelRunner` from its saved artifact (e.g.
        `OnnxModelRunner(onnxruntime.InferenceSession(path))`,
        `DynamoModelRunner(torch.export.load(path).module())`) and pass a config from
        `AutoConfig.from_pretrained(...)`."""
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        # The scatter path applies when a graph takes embeddings where a plain model's takes token ids: the
        # decode graph (decoder-only VLMs) or the encoder graph (a multi-modal encoder-decoder like
        # florence2, whose decode then reads the merged features through `encoder_outputs`). Otherwise it
        # runs as a plain generator even when an embed graph was exported.
        takes_embeds = text_input(runners["decode"]) == "inputs_embeds" or (
            "encoder" in runners and "inputs_embeds" in runners["encoder"].input_names
        )
        text_embed = runners["embed_tokens"] if "embed_tokens" in runners and takes_embeds else None
        modalities = []
        if text_embed is not None:
            for name, _getter, spec_input_keys, grid_key, token_field in _MODALITY_SPECS:
                token_id = getattr(config, token_field, None)
                if token_id is None:  # older configs name it `<modality>_token_index`
                    token_id = getattr(config, f"{token_field[:-3]}_index", None)
                runner = runners.get(name)
                if runner is None:
                    # A model can route one modality through another's getter — perception_lm's videos go
                    # through `get_image_features(pixel_values=pixel_values_videos)` — so the decomposition,
                    # which models one component per getter, exported none of its own for it. Share the image
                    # graph: it *is* the graph that modality runs through. Only with a placeholder token of
                    # its own, which is what tells the scatter where its rows go; without one there is nothing
                    # to key on and the modality is simply not served.
                    if name == "image_encoder" or token_id is None or "image_encoder" not in runners:
                        continue
                    runner = runners["image_encoder"]
                input_keys = tuple(spec_input_keys) + ((grid_key,) if grid_key is not None else ())
                modalities.append(Modality(token_id, runner, input_keys))
        embedder = None
        if (spec := streaming_embedder_spec(config)) is not None and spec.component in runners:
            embedder = StreamingEmbedder(
                runners[spec.component], spec.source, spec.produces, _find_config_attr(config, spec.stride) or 1
            )
        return ExportedGenerator(
            config,
            generation_config,
            runners["decode"],
            prefill=runners.get("prefill"),
            encoder=runners.get("encoder"),
            text_embed=text_embed,
            modalities=modalities,
            embedder=embedder,
        )

    @classmethod
    def from_pretrained(cls, save_directory: str | Path, **kwargs) -> ExportedGenerator:
        """Load a saved export — a local directory or a Hub repo — and return a runnable generator.

        The manifest says which file is which component and which backend wrote them, so each artifact is
        opened by the runner for that format; everything else about the graphs — precision, cache geometry,
        traced shapes — comes from inside the artifacts themselves. The `generation_config` saved alongside
        is the one the model was exported with, which is what makes the cache the graphs were traced against
        the cache this builds.

        Example:
            OnnxExporter().save_pretrained(programs, "out/", config=model.config,
                                           generation_config=generation_config)
            runtime = ExportedGenerator.from_pretrained("out/")
            ids = runtime.generate(input_ids=prompt, max_new_tokens=32)
        """
        download_kwargs, _ = split_download_kwargs(dict(kwargs))
        runners, _ = load_export_runners(save_directory, **kwargs)
        config = AutoConfig.from_pretrained(save_directory, **download_kwargs)
        # The one the model was exported with, saved beside the artifacts: it declares the cache the graphs
        # were traced against, so a load without it would build a different one.
        has_generation_config = (
            resolve_export_file(save_directory, GENERATION_CONFIG_NAME, **download_kwargs) is not None
        )
        generation_config = (
            GenerationConfig.from_pretrained(save_directory, **download_kwargs) if has_generation_config else None
        )
        return cls.from_runners(runners, config, generation_config)

    # ── GenerationMixin plumbing (a real PreTrainedModel provides all of this) ──
    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def can_generate(self):
        return True

    def is_remote_code(self):
        return False

    def get_experts_implementation(self):
        return {}

    def get_output_embeddings(self):
        return None

    def get_encoder(self):
        return self._encoder

    def get_compiled_call(self, compile_config):
        """`generate` swaps in a compiled forward for the decode loop when the cache is compileable. The
        graphs are already compiled (that is what exporting them was), so hand back the plain call."""
        return self.__call__

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    @property
    def _consumed_kwargs(self) -> set[str]:
        """What `generate` may carry that this runtime consumes without naming it on `forward`: whatever the
        component graphs declare as an input, which `forward` feeds through (a model may take extra per-step
        tensors, e.g. `token_type_ids`), plus the text-path kwargs the runtime routes itself — a model that
        derives one internally (ctrl and xlm build their own `token_type_ids`) exports a graph that never
        takes it, and dropping it here is exactly what its own forward did. With modalities, also each
        modality's own inputs and `mm_token_type_ids` (the placeholder map the scatter reads, never a graph
        input of its own)."""
        consumed = self._graph_inputs | _TEXT_KWARGS | self._modality_keys
        if self._embedder is not None:
            consumed = consumed | {self._embedder.source}
        return (consumed | {"mm_token_type_ids"}) if self._text_embed is not None else consumed

    def _validate_model_kwargs(self, model_kwargs):
        super()._validate_model_kwargs({k: v for k, v in model_kwargs.items() if k not in self._consumed_kwargs})

    def _supports_default_dynamic_cache(self) -> bool:  # noqa: D401 (instance form: reads the prototype)
        """Whether `generate` should build it a `DynamicCache`.

        On a real model this is a class-level fact; here it is read off the cache the graphs were traced
        against — a recurrent-only model (mamba, rwkv, …) keeps fixed-size states instead, and handing it a
        `DynamicCache` makes `generate` ask a question (`get_seq_length`) that cache refuses to answer; a
        model with no cache at all (openai-gpt recomputes the whole sequence every step) exports graphs that
        take none.
        """
        return self._decode_runner.cache_input == "past_key_values"

    @property
    def _is_recurrent(self) -> bool:
        """Whether the graphs carry fixed-size recurrent state instead of a growing KV cache — the decode
        graph says which by the kwarg it takes its cache under (`ModelRunner.cache_input`)."""
        return self._decode_runner.cache_input == "cache_params"

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        """Build the cache the exported decode graph expects, so `generate` is called like a normal model
        (`generate(input_ids=..., max_new_tokens=N)` — no hand-rolled cache).

        `generate`'s own builder decides the kind (`EncoderDecoderCache` pairing, a source-length static
        cross cache, …) from the generation config — the same config that built the cache at export capture.
        The *size* does not come from there: `generate` sizes a fixed-size cache from the prompt in front of
        it, so the same config gives a different length for a different prompt, and a graph traced against
        one refuses the other (`max_cache_len` is part of its input spec). The trace recorded the length it
        held, so build to that when the artifact says so. The other addition is materialization:
        `torch.export` bakes real tensors into the graph's input spec, so the lazily-uninitialized layers
        `generate` hands back have to be filled in (`materialize_cache_layers`, the helper the capture uses
        too)."""
        super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        # Nothing to build when the graphs take no cache at all: turn caching off so `generate`'s loop
        # re-feeds the whole sequence each step — what those graphs were traced on — instead of slicing to a
        # single-token step, and drop any cache handed in for graphs that cannot read one.
        if not self._is_recurrent and self._decode_runner.cache_input is None:
            model_kwargs.pop("past_key_values", None)
            generation_config.use_cache = False
            return
        # Beam search (and several returned sequences) expand the batch before the first decode call, so the
        # zero-length tensors materialized below have to be sized the way `generate` sizes a static cache.
        batch_size *= max(generation_config.num_beams, generation_config.num_return_sequences)
        # A recurrent model gets no cache from `generate` (it expects the model's own
        # `prepare_inputs_for_generation` to make one), so build the one its configs describe: `layer_types`
        # gives the same mix of attention and linear-attention layers the model would, and the generation
        # config says whether those were traced growing or at a fixed size.
        if self._is_recurrent:
            text_config = self.config.get_text_config()
            model_kwargs.setdefault(
                "cache_params",
                StaticCache(config=text_config, max_cache_len=max_cache_length)
                if generation_config.cache_implementation == "static"
                else DynamicCache(config=text_config),
            )
        # Cross-attention caches the encoder's states in full, so it is never sliding — but `generate` builds
        # both halves of an `EncoderDecoderCache` from the same decoder config, sliding `layer_types` and all,
        # and each model that cares corrects it in its own `_prepare_cache_for_generation` (t5gemma). The
        # runtime has no model to inherit that from, and the layer *classes* are part of the graph's input
        # spec, so rebuild the cross half the way the traced cache had it.
        cache = model_kwargs.get("past_key_values")
        if isinstance(cache, EncoderDecoderCache):
            cross = cache.cross_attention_cache
            traced_classes = self._decode_runner.export_metadata.cross_layer_classes
            built_classes = tuple(type(layer).__name__ for layer in cross.layers)
            # Against the recorded classes when the artifact carries them; an older one that does not falls
            # back to recognising the kind that is wrong here, which is what this did before it could ask.
            needs_rebuild = (
                built_classes != traced_classes
                if traced_classes
                else any(name.endswith("SlidingWindowLayer") for name in built_classes)
            )
            if needs_rebuild:
                cross_config = copy.deepcopy(self.config.get_text_config(decoder=True))
                cross_config.sliding_window = None
                cross_config.layer_types = ["full_attention"] * cross_config.num_hidden_layers
                cross_kwargs = {"config": cross_config}
                if isinstance(cross, StaticCache):
                    # A static cross cache is sized to the encoder sequence, not the decode length.
                    cross_kwargs["max_cache_len"] = model_kwargs["encoder_outputs"][0].shape[1]
                cache.cross_attention_cache = type(cross)(**cross_kwargs)

        for cache_name in ("past_key_values", "cache_params"):
            if (cache := model_kwargs.get(cache_name)) is not None:
                _resize_to_traced_lengths(cache, self._decode_runner.export_metadata.cache_lengths)
                # The traced prototype is a cache the model filled, so it carries the real per-layer
                # geometry — the config can't always say (see `materialize_cache_layers`).
                materialize_cache_layers(
                    cache,
                    batch_size,
                    self.config,
                    self._dtype,
                    self._device,
                    kv_geometry=self._decode_runner.kv_geometry,
                    indexer_layers=self._decode_runner.export_metadata.indexer_layers,
                )

    # ── decode orchestration ──
    def create_masks_for_generate(self, config, inputs_embeds, attention_mask, **kwargs):
        """Keep the 2D padding mask when that is what the decode graph took.

        `prepare_inputs_for_generation` upgrades a 2D mask to the 4D causal mask for any compileable cache,
        which is right for a model whose forward builds its own mask but wrong for a graph traced *on* the 2D
        mask — the 4D one then fails an internal guard rather than a shape check. The trace is the authority
        (`_mask_rank`), so defer to it and only fall back to the generic builder otherwise.
        """
        if mask_rank(self._decode_runner) == 2 and attention_mask is not None:
            return attention_mask
        return create_masks_for_generate(
            config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def _mask_feed(self, runner, attention_mask, position_ids, cache_len):
        """Feed the decode graph's mask input(s). `generate` hands us either a dict of 4D bool masks (one
        per attention type, for mixed full/sliding models) or a single 4D mask; when it drops a mask as
        redundant (`None` — the whole kwarg or one dict slot) the traced graph still takes a tensor there,
        so rebuild the full causal mask the eager model would have built internally."""
        # A mixed-attention model (nemotron_h, jamba, …) builds its per-layer-type mask dict *inside* its
        # forward, which the graph starts after — so when the graph takes one mask per type and `generate`
        # handed us a single tensor, build the dict here, keyed the way the config declares its layers.
        mask_ranks = mask_dict_ranks(runner)
        if mask_ranks and not isinstance(attention_mask, dict):
            padding_mask = attention_mask if getattr(attention_mask, "dim", lambda: 0)() == 2 else None
            attention_mask = {
                layer_type: padding_mask if rank == 2 else None for layer_type, rank in mask_ranks.items()
            }
        if isinstance(attention_mask, dict):
            # A slot the trace took as `None` stays `None`: it is a leaf of the graph's input spec like any
            # other, and filling it with a causal mask feeds a tensor where the graph declares nothing.
            traced_without_mask = {name for name, rank in (mask_ranks or {}).items() if rank is None}
            attention_mask = {
                layer_type: mask
                if mask is not None or layer_type in traced_without_mask
                else self._causal_mask(position_ids, cache_len)
                for layer_type, mask in attention_mask.items()
            }
            # Mixed full/sliding models: ONNX declares one input per attention type
            # (`attention_mask.<type>`, flattened by the exporter); dynamo takes the whole dict as a single
            # `attention_mask` pytree kwarg.
            # ONNX flattens the dict into one input per type. Feed exactly the names it declares: keying
            # off our own layer types instead offers ones the graph never took and omits ones it needs.
            if declared := [name for name in mask_inputs(runner) if name != "attention_mask"]:
                fallback = None
                feed = {}
                for name in declared:
                    mask = attention_mask.get(_mask_type(name))
                    if mask is None:
                        fallback = fallback if fallback is not None else self._causal_mask(position_ids, cache_len)
                        mask = fallback
                    feed[name] = mask
                return feed
            return {"attention_mask": attention_mask}
        # A graph may take no explicit mask — e.g. a prefill graph builds the causal mask internally from
        # positions on an empty, unpadded cache. Nothing to feed then.
        if not mask_inputs(runner):
            return {}
        if attention_mask is None:
            # `generate` drops the mask when nothing is padded, leaving the runtime to build the one the
            # graph took — and which that is, is the graph's own rank. A graph traced on the 2-D padding
            # mask compares its *width* against the query and the cache, so handing it the 4-D causal mask
            # puts the head axis where the width belongs and trips that comparison as a guard. Everything is
            # attendable either way: the mask is missing precisely because nothing was padded.
            name = mask_inputs(runner)[0]
            if mask_rank(runner) == 2:
                dtype = mask_dtype(runner) or torch.long
                batch = position_ids.shape[-2] if position_ids.dim() == 3 else position_ids.shape[0]
                return {name: torch.ones(batch, cache_len, dtype=dtype, device=self._device)}
            return {name: self._causal_mask(position_ids, cache_len)}
        # A graph traced on the 2D padding mask was traced against a *cache-width* mask: `generate` pads the
        # mask out to a static cache's length, and the model compares the two (bloom's alibi). The runtime's
        # mask tracks the real sequence instead, so pad the tail back — zeros, i.e. the unfilled cache slots
        # masked out, which is what the padded mask meant at capture.
        # Only when this *is* the decoder's own mask. An encoder-decoder's `attention_mask` covers the
        # *encoder* sequence and its width is tied to the encoder output's, not to the decoder cache — and it
        # stays under that name whether or not the graph also takes a `decoder_attention_mask`, so the config
        # is what settles it rather than the presence of the decoder mask input.
        pads_to_cache = mask_rank(runner) == 2 and not self.config.is_encoder_decoder
        if pads_to_cache and attention_mask.dim() == 2:
            if (padding_length := cache_len - attention_mask.shape[-1]) > 0:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        return {mask_inputs(runner)[0]: attention_mask}

    def _causal_mask(self, position_ids, cache_len):
        """Full causal mask `[batch, 1, query, cache_len]` from the positions (per batch row) — what the
        eager model builds internally when `generate` drops the mask as redundant. M-RoPE positions carry
        extra axes in front; the text row (axis 0) is the sequence position. Assumes no left-padding — the
        common single-sequence case."""
        text_positions = position_ids if position_ids.dim() == 2 else position_ids[0]
        positions = torch.arange(cache_len, device=text_positions.device)
        return (positions <= text_positions[..., None])[:, None]

    def _text_feed(self, runner, text_ids, kwargs, image_sizes=None) -> dict:
        """What goes in under the decode graph's text input.

        Text-only, or a decode graph that takes token ids directly (an encoder-decoder's, reading the merged
        features through `encoder_outputs`): the ids themselves. With an embed graph, they go through it
        first and each present modality's features are scattered into the result.

        The embed graph's first output is the embeddings themselves, whatever it named them; they go in under
        the decode graph's own text input. Any further per-token outputs (`per_layer_inputs`) are fed by name,
        and only if this graph declares them. `image_sizes` arrives as an explicit kwarg — no graph declares
        it once the anyres packing moved here, so `generate` would otherwise reject it — and goes only to the
        merge, since putting it in `kwargs` would expose an `(images, 2)` tensor to the per-step slicing.
        """
        if self._text_embed is None or text_input(runner) != "inputs_embeds":
            return {text_input(runner): text_ids}
        merge_kwargs = kwargs if image_sizes is None else {**kwargs, "image_sizes": image_sizes}
        embedded = self._merge_modalities(text_ids, merge_kwargs)
        primary, *extra = embedded
        feed = {text_input(runner): embedded[primary]}
        feed.update({name: embedded[name] for name in extra if name in runner.input_names})
        return feed

    def forward(
        self,
        past_key_values=None,
        input_ids=None,
        decoder_input_ids=None,
        position_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        cache_params=None,
        image_sizes=None,
        **kwargs,
    ):
        # First step (empty cache) is the prefill; subsequent steps are decode. With no dedicated prefill
        # graph the two runners are the same object, so this just always runs `decode`. For an
        # encoder-decoder model the split is load-bearing beyond shapes: the prefill graph *writes* the
        # cross cache from `encoder_outputs`, decode graphs read it (the modeling's `is_updated` python
        # branch bakes at trace time). With caching off (`use_cache=False`) there is no cache at all and
        # every step re-feeds the whole sequence — that is a prefill each time.
        # A recurrent model (mamba, rwkv, …) carries its fixed-size state under `cache_params` instead;
        # it is the same thing to the graphs, so the loop below treats them alike.
        past_key_values = past_key_values if past_key_values is not None else cache_params
        # How much the cache already holds decides which graph runs and, below, how wide the mask and the
        # streamed-modality window are. One answer per step: every reader here means the same thing by it.
        past_len = _cache_length(past_key_values) if past_key_values is not None else 0
        runner = self._prefill_runner if past_len == 0 else self._decode_runner
        text_ids = decoder_input_ids if decoder_input_ids is not None else input_ids
        feed = self._text_feed(runner, text_ids, kwargs, image_sizes)
        text = feed[text_input(runner)]
        if position_ids is not None and "position_ids" in runner.input_names:
            feed["position_ids"] = position_ids
        if encoder_outputs is not None and any(n.startswith("encoder_outputs") for n in runner.input_names):
            feed["encoder_outputs"] = encoder_outputs
        # Extra per-step inputs the graph declares (`token_type_ids`, per-model aux masks, …) come from
        # `generate`'s kwargs. `generate` slices sequence kwargs to the current step only for the ones named
        # on the real model's `forward` — ours takes them generically, so trim them here the same way.
        for name, value in kwargs.items():
            # `_declares`, not a plain name lookup: a backend that flattens a pytree kwarg declares only its
            # leaves, so the kwarg's own name is never in `input_names` — which is how t5gemma's *dict* of
            # per-type decoder masks (`decoder_attention_mask.full_attention`, `.sliding_attention`, built by
            # `generate` and handed to us whole) went unfed on ONNX while dynamo, which takes the dict under
            # one name, got it.
            if name not in feed and _declares(runner, name, value):
                # Rank 2 or 3 only — those have the sequence at axis 1 (higgs_audio_v2's audio ids carry a
                # codebook axis after it). A 4-D mask is `[batch, 1, query, key]`, where axis 1 is heads.
                # A modality's own tensors (`pixel_values`, packed patches, …) are features, not per-token
                # kwargs — their axis 1 is patches or channels, so they go through whole (a prefill graph
                # with the vision tower inline takes them directly).
                is_per_token = isinstance(value, torch.Tensor) and name not in self._modality_keys
                if is_per_token and value.dim() in (2, 3) and value.shape[1] > text.shape[1]:
                    value = value[:, -text.shape[1] :]
                feed[name] = value
        # How wide the graph's key axis is: a fixed-size cache is allocated in full, so the mask has to be
        # padded out to it, while a growing one is only as long as what it already holds plus this step's
        # query. Growing vs fixed-size is the layer's *kind*, not what `get_max_length` reports — a
        # `DynamicSlidingWindowLayer` grows and crops, yet reports its whole window (4096 on a gemma2 text
        # config), padding the mask to a width the graph never declared (`set_inputs` then refuses it).
        cache_len = mask_width(past_key_values, text.shape[1])
        # A modality that advances with the text: embed the whole prompt's features once, then hand this step
        # the window its own tokens span (`past_seen * stride` onwards) — what the model's own
        # `prepare_inputs_for_generation` slices out before each call.
        if self._embedder is not None and self._embedder.produces in runner.input_names:
            features = kwargs.get(self._embedder.source)
            if features is not None and (self._embedded is None or past_len == 0):
                self._embedded = next(iter(self._embedder.runner(**{self._embedder.source: features}).values()))
            if self._embedded is not None:
                stride = self._embedder.stride
                feed[self._embedder.produces] = self._embedded[
                    :, past_len * stride : (past_len + text.shape[1]) * stride
                ]
        # Containers the graph declares besides its own cache, and that the trace saw *empty*: a fresh empty
        # one each step is exactly the structure it was traced with (voxtral_realtime's decode re-derives its
        # encoder state from this step's audio window, and its conv state belongs to the pre-loop embedder).
        # One traced non-empty is a cache with real content and not ours to invent, so it is left alone.
        for name, spec in runner.export_metadata.kwargs.items():
            container = spec.get("container")
            if container is None or spec.get("leaves") or name in feed:
                continue
            empty = _empty_container(
                container, self.config, text.shape[0], self._dtype, self._device, self._encoder_config
            )
            # `_declares`, not a plain name lookup: a backend that flattens the container names only its
            # leaves (`input.encoder_past_key_values.layers.0.keys`), so the kwarg itself is never in
            # `input_names` and the graph would go unfed.
            if empty is not None and _declares(runner, name, empty):
                feed[name] = empty
        feed.update(self._mask_feed(runner, attention_mask, position_ids, cache_len))
        # A graph that names the decoder's mask separately gets the causal one here — `attention_mask` is
        # the *encoder's* on those models. `generate` supplies neither this nor decoder positions (the
        # eager forward derives both inside), so count the positions off the cache.
        decoder_mask = decoder_mask_input(runner)
        if decoder_mask is not None and decoder_mask not in feed:
            decoder_positions = (
                torch.arange(text.shape[1], device=self._device).unsqueeze(0).expand(text.shape[0], -1) + past_len
            )
            feed[decoder_mask] = self._causal_mask(decoder_positions, cache_len)
        # Under the name this graph declares, and only if it declares one: a model whose `generate` hands
        # back a cache the exported graph does not take (xlstm) would otherwise be fed an input it never had.
        if past_key_values is not None and runner.cache_input is not None:
            feed[runner.cache_input] = past_key_values
        # Only what this graph declares: a model whose decode graph takes its text some other way
        # (higgs_audio_v2 embeds it upstream) would otherwise be handed an `input_ids` it never had. Pytree
        # kwargs are the exception — see `_declares`, which knows a graph naming only their leaves still takes
        # them (the cache, `encoder_outputs`, a mask dict).
        outputs = runner(**{name: value for name, value in feed.items() if _declares(runner, name, value)})
        if past_key_values is not None:
            past_key_values = _advance_cache(past_key_values, outputs, num_new_tokens=text.shape[1])
        # A recurrent model's state is not a KV cache and `generate` must not carry it back as one; which
        # kind the graphs hold is `_is_recurrent`, read off the kwarg the decode graph takes it under.
        returned_cache = None if self._is_recurrent else past_key_values
        return CausalLMOutputWithPast(logits=outputs["logits"], past_key_values=returned_cache)

    def _merge_modalities(self, input_ids, kwargs) -> dict[str, torch.Tensor]:
        """Embed `input_ids` and scatter each present modality's features into its placeholder rows.

        Returns every per-token input the embed graph produces, keyed by the decode graph's input name —
        `inputs_embeds` plus, for a decoder with per-layer embeddings, `per_layer_inputs`. Only the
        embeddings take the scattered features; the rest pass through as embedded.
        Which inputs a modality takes besides its features varies per model (`image_position_ids`, `input_features_mask`, …),
        so they come from the names its graph declares rather than a fixed list per modality. The eager
        `get_<modality>_features` computes its grid-derived tensors (`cu_seqlens` / `window_index` / …)
        internally; the export moved them out of the graph, so they're injected here from `self.config`
        alone (`precompute_export_inputs`), after renaming generate's grid kwarg (`image_grid_thw` /
        `video_grid_thw`) to the graph's `grid_thw` input."""

        extras: dict = {}

        embedded = self._text_embed(input_ids=input_ids)
        embeds_name = next(iter(embedded))
        inputs_embeds = embedded[embeds_name]
        for modality in self._modalities:
            # Presence keys on whichever of the modality's own input names this call carries — never the
            # aux keys, which `generate` may keep after dropping the features themselves.
            if all(kwargs.get(key) is None for key in modality.input_keys if not key.endswith(_MODALITY_AUX_SUFFIXES)):
                continue
            if modality.token_id is not None:
                mask = (input_ids == modality.token_id).unsqueeze(-1)
            else:
                # A model with no placeholder token id (kosmos2_5) marks the feature rows with an explicit
                # mask kwarg instead — the same tensor its own forward scatters by. `generate` keeps the
                # full-prompt mask across steps, so align it to this step's ids (its tail) first.
                mask_key = next(key for key in modality.input_keys if key.endswith("_position_mask"))
                mask = (kwargs[mask_key][:, -input_ids.shape[1] :] == 1).unsqueeze(-1)
            # A step with no placeholder rows has nothing to scatter into — a decode step whose feature
            # kwarg `generate` kept around would otherwise re-encode for nothing.
            if not mask.any():
                continue
            feed = self._modality_feed(modality, kwargs, input_ids)
            # An anyres image graph stops at the projector (`PatchVisionEncoder`) because the packing's token
            # count per image is data; so the padding rows come off here and the packing happens here too.
            image_sizes = kwargs.get("image_sizes")
            packs_anyres = (
                image_sizes is not None
                and modality.input_keys[0] == "pixel_values"
                and "image_sizes" not in modality.runner.input_names
                and _find_config_attr(self.config, "image_grid_pinpoints") is not None
            )
            if packs_anyres:
                tower_input = modality.runner.input_names[0]
                feed[tower_input] = flatten_anyres_patches(self.config, feed[tower_input], image_sizes)
            outputs = modality.runner(**feed)
            # A deepstack tower emits one feature tensor per decoder layer it is injected at
            # (`image_features.<layer>`). Those are summed in inside the decoder rather than scattered, so the
            # placeholder rows are zeroed here and the packed tensors go on to the decode graph by name.
            per_layer = {
                int(name.rsplit(".", 1)[-1]): tensor
                for name, tensor in outputs.items()
                if re.fullmatch(r".*image_features\.\d+", name)
            }
            if packs_anyres and per_layer:
                extras["deepstack_features"] = {
                    layer: _pack_anyres_features(self.config, tensor, image_sizes, outputs).to(inputs_embeds.dtype)
                    for layer, tensor in sorted(per_layer.items())
                }
                extras["vision_mask"] = mask
                inputs_embeds = inputs_embeds.masked_fill(mask, 0.0)
                continue
            features = next(iter(outputs.values()))
            if packs_anyres:
                features = _pack_anyres_features(self.config, features, image_sizes, outputs)
            inputs_embeds = inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.dtype))
        return {**embedded, embeds_name: inputs_embeds, **extras}

    def _modality_feed(self, modality, kwargs, input_ids) -> dict:
        """Everything one modality graph declares, sourced in order: `generate`'s kwargs, then the prompt's
        ids if it takes them, then the tensors the precompute derives from the config, then the config
        itself.

        Those three cover the three kinds of input such a graph takes — the modality's own data
        (`pixel_values`, and any aux the model names beside it), the grid-derived tensors the export moved
        out of the graph (`cu_seqlens` / `window_index` / …), and plain settings the eager forward defaults
        from the config (`vision_feature_layer`). Anything the graph does not declare is left out, and
        `generate`'s text kwargs are never a source: an encoder's `attention_mask` is its own. The prompt's
        `input_ids` are the exception, fed only to a graph that names them — some getters read them to place
        their features in time (musicflamingo's rotary timestamps) — and a modality only ever runs on the
        prefill, which is where the capture read them too.

        Sourcing by name alone would cross the modalities over. `generate` names the features per modality
        (`pixel_values_videos`), but a graph takes the name its own getter declared — and a video getter
        often declares the generic `pixel_values` (llava_next_video), which is the *image* kwarg. So route
        this modality's own key onto its graph's feature input, the way `_grid_renamed` routes the grid, and
        drop the keys that belong to another modality.
        """
        declared = set(modality.runner.input_names)
        feature_input = modality.runner.input_names[0]
        present = next(
            (
                key
                for key in modality.input_keys
                if not key.endswith(_MODALITY_AUX_SUFFIXES) and key not in declared and kwargs.get(key) is not None
            ),
            None,
        )
        foreign = {key for other in self._modalities if other is not modality for key in other.input_keys} - set(
            modality.input_keys
        )
        inputs = {}
        for key, value in kwargs.items():
            if value is None or key in _TEXT_KWARGS or key in foreign:
                continue
            # `_declares`, not an exact name match: a *list-valued* input (ernie4_5_vl_moe's
            # `temporal_slice_index`, the even/odd gather pair) is declared by its flattened leaves
            # (`temporal_slice_index.0`, `.1`), and each runner flattens the pytree its own way.
            if _declares(modality.runner, name := _grid_renamed(key), value):
                inputs[name] = value
        # Only when nothing named the graph's feature input: the modality's kwarg is that tensor under
        # another name (`pixel_values_videos` for a video getter that declares `pixel_values`). A getter
        # whose first input is a *different* quantity (inkling's `audio_input_ids`, vibevoice_asr's
        # `input_values`) is fed by name above, so this must not overwrite it.
        if present is not None and feature_input not in inputs:
            inputs[feature_input] = kwargs[present]
        # Cast BEFORE the precompute: the graph was traced with the model's dtype for the modality inputs
        # (`pixel_values` …) but with whatever dtype the precompute itself produces (fp32 interpolation
        # weights), so casting after would hand the graph bf16 weights it never saw.
        inputs = cast_leaf_tensors(inputs, dtype=modality.runner.dtype, device=self._device)
        inputs = precompute_export_inputs(self.config, inputs)
        # Device-only move for what the precompute added — it builds some index tensors on CPU
        # (minicpmv4_6's `window_index` comes out of python list ops). No dtype cast: the graph was traced
        # with whatever dtype the precompute produces (see above).
        inputs = {
            name: value.to(self._device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        feed = {name: value for name, value in inputs.items() if _declares(modality.runner, name, value)}
        if "input_ids" in declared:
            feed["input_ids"] = input_ids
        for name in declared - feed.keys():
            if (value := _find_config_attr(self.config, name)) is not None:
                feed[name] = value
        return feed

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Per-step inputs, with every kwarg the trace took at the step's own width trimmed to it.

        A cross-attention mask grows a row per token generated, and the model that takes one slices it to
        the tokens being processed in its own `prepare_inputs_for_generation` (mllama). This runtime
        inherits no such hook -- it stands in for the model rather than subclassing it, and cannot call the
        model's own, whose zero-argument `super()` refuses a foreign `self` -- so it does the same cut here.

        Named rather than derived from the recorded widths: the merged multi-token decode is a synthetic
        call, built by concatenating single-token steps, and it concatenates the text while leaving this
        mask at one row. Widths read off that capture say the opposite of what the graph wants.
        """
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        text = model_inputs.get("input_ids")
        if text is None:
            text = model_inputs.get("inputs_embeds")
        mask = model_inputs.get("cross_attention_mask")
        if isinstance(mask, torch.Tensor) and text is not None:
            model_inputs["cross_attention_mask"] = mask[:, -text.shape[1] :]
        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """Multi-modal M-RoPE: build the `[text; 3 vision]` 4-axis `position_ids` the exported decode graph
        expects — what `generate` normally gets from a VLM's own override of this method. Runs that same
        override without the model: the text row from `super()` (GenerationMixin), the 3 vision rows from
        the model class's own `get_rope_index` (see `get_rope_index_from_config`), and the decode step
        advances the text row by the cached rope-delta. Models that lay out no modality spans (plain
        decoders, VLMs with 1D text positions like Llava) keep the standard positions."""
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        cache = model_kwargs.get("past_key_values")
        past_length = _cache_length(cache)
        if past_length != 0 and getattr(self, "_rope_deltas", None) is not None:
            return text_positions[None, ...] + self._rope_deltas

        if model_kwargs.get("input_ids") is not None and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]
        # No `attention_mask`: unpadded single sequence, and `generate` has already turned the mask into the
        # per-layer form the attention needs, not the 2D form `get_rope_index` wants. `None` = all valid.
        rope_index = get_rope_index_from_config(
            self.config, {**model_kwargs, "input_ids": inputs_tensor, "attention_mask": None}
        )
        if rope_index is None:
            return text_positions
        vision_positions, self._rope_deltas = rope_index
        return torch.cat([text_positions[None, ...], vision_positions], dim=0)
