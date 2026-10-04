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
"""Run exported generative models through `GenerationMixin.generate`.

`ExportedGenerator` plugs the component graphs of `export_for_generation` back together and drives the
generation loop from artifacts and configs alone, with no model instance or weights. The model config and the
generation config used at export are the contract: the generation config says which cache the graphs were
traced against (growing `DynamicCache` or fixed-size `StaticCache`), so nothing is introspected from the graphs.

A `decode` graph alone is text generation. With an `embed_tokens` graph and one `Modality` per image / video /
audio encoder, prefill scatters each modality's features into `inputs_embeds` at its placeholder positions. The
graphs are called through the backends' `ModelRunner`s (`runner_*.py`).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

from ..generation import GenerationConfig, GenerationMixin
from ..models.auto import AutoConfig
from ..utils import GENERATION_CONFIG_NAME, logging
from ..utils.import_utils import is_torch_available
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
    advance_cache_length,
    is_fixed_size,
    mask_width,
    materialize_cache_layers,
    resize_to_traced_lengths,
)
from .decompose import (
    _MODALITY_SPECS,
    ModalitySpec,
    flatten_anyres_patches,
    grid_renamed,
    pack_anyres_features,
    streaming_embedder_spec,
)
from .precompute import _find_config_attr, get_rope_index_from_config, precompute_export_inputs
from .utils import (
    cast_leaf_tensors,
    get_leaf_tensors,
    runner_feed,
)


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

    from ..cache_utils import DynamicCache, EncoderDecoderCache, StaticCache
    from ..masking_utils import create_masks_for_generate
    from ..modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, CausalLMOutputWithPast

# Text-path kwargs `generate` always carries. A modality graph declaring one means its own (an audio
# encoder's `attention_mask` covers mel frames), so these are never sourced from `generate`.
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


@dataclass
class Modality:
    """Routes one input modality (image / video / audio) of an `ExportedGenerator`.

    - `token_id`: placeholder id in `input_ids` its features scatter into; `None` when the model marks rows
      with a `*_position_mask` kwarg instead (kosmos2_5).
    - `runner`: the exported `get_<modality>_features` graph.
    - `spec`: which generate kwargs carry it.
    """

    token_id: int | None
    runner: ModelRunner
    spec: ModalitySpec

    @property
    def input_keys(self) -> tuple[str, ...]:
        """Every generate kwarg it owns, the grid included."""
        return self.spec.input_keys + ((self.spec.grid_key,) if self.spec.grid_key else ())

    @property
    def kind(self) -> str:
        """Its key in `generate`'s `mm_encoder_outputs`: `"image"`, `"video"` or `"audio"`."""
        return self.spec.component.removesuffix("_encoder")


@dataclass
class StreamingEmbedder:
    """A modality embedded once before the decode loop, fed to each step as a window of the result
    (`stride` embedded rows per token)."""

    runner: ModelRunner
    source: str
    produces: str
    stride: int


class _ExportedEncoder:
    """`get_encoder()` stand-in over the exported encoder graph, returning the encoder's own output class.

    The class is the traced one, not `BaseModelOutput`: some encoders return more than hidden states
    (parakeet's frame mask, read by canary / cohere_asr decoders). Cross keys/values from a
    `CrossAttentionEncoder` graph are kept in `cross_states` for `_prepare_cache_for_generation` to seed."""

    def __init__(self, runner: ModelRunner, merge=None, output_class: type | None = None):
        self._runner = runner
        self._merge = merge
        self._output_class = output_class or BaseModelOutput
        # `{layer index: (keys, values)}` from the last call, when this graph writes the cross cache.
        self.cross_states: dict[int, tuple] = {}

    def forward(self, **kwargs):
        # A multi-modal encoder-decoder merges features in front of the text encoder.
        if self._merge is not None:
            merged = self._merge(kwargs.pop("input_ids"), kwargs)
            primary, *extra = merged
            kwargs[text_input(self._runner)] = merged[primary]
            kwargs.update({name: merged[name] for name in extra})
        feed = runner_feed(self._runner, kwargs)
        outputs = self._runner(**feed)
        # Taken out before building the output: leaving them in would change the `encoder_outputs` pytree the
        # decode graph declares.
        self.cross_states = {}
        for name in [name for name in outputs if name.startswith(("cross_keys_", "cross_values_"))]:
            kind, _, index = name.rpartition("_")
            keys, values = self.cross_states.get(int(index), (None, None))
            tensor = outputs.pop(name)
            self.cross_states[int(index)] = (tensor, values) if kind == "cross_keys" else (keys, tensor)
        # A bare-tensor encoder output carries a trace-chosen name; it is the hidden states (first field).
        fields = {field.name for field in dataclasses.fields(self._output_class)}
        if outputs.keys() <= fields:
            return self._output_class(**outputs)
        return self._output_class(next(iter(outputs.values())))

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


# Input-role derivations only the generation loop needs; kept off `ModelRunner` so runners stay task-agnostic.


def _mask_type(name: str) -> str:
    """The attention type a per-type mask input names, under either backend's flattening."""
    return name.removeprefix("attention_mask.").removeprefix("attention_mask_")


def text_input(runner) -> str:
    """The graph's text input: `"decoder_input_ids"`, `"inputs_embeds"` or `"input_ids"`."""
    return next((n for n in ("decoder_input_ids", "inputs_embeds") if n in runner.input_names), "input_ids")


def mask_inputs(runner) -> tuple[str, ...]:
    """The graph's attention-mask input name(s) — several for mixed full/sliding attention."""
    return tuple(
        n for n in runner.input_names if n == "attention_mask" or n.startswith(("attention_mask.", "attention_mask_"))
    )


def decoder_mask_input(runner) -> str | None:
    """`"decoder_attention_mask"` when the graph declares one. On encoder-decoders `attention_mask` covers the
    encoder sequence, so the causal mask goes here; `generate` does not supply it."""
    return "decoder_attention_mask" if "decoder_attention_mask" in runner.input_names else None


class ExportedGenerator(GenerationMixin):
    """Drive exported component graphs through `generate`, from artifacts + configs alone.

    A `decode` runner alone is decoder-only text generation. Pass `text_embed` and `modalities` for a
    multi-modal model: prefill scatters each modality's features into `inputs_embeds`; decode steps are
    text-only. Pass `prefill=` to drive a fixed query=1 `decode` graph from a separate dynamic prefill graph;
    otherwise `decode` serves both and must be multi-token.

    `generation_config` must be the one the model was **exported with**: it declares the cache the decode
    graph was traced against (`DynamicCache`, or `StaticCache` for `cache_implementation="static"`).

    Example:
        exported_artifacts = OnnxExporter().export_for_generation(model, inputs,
                                                        OnnxConfig(dynamic=True, external_data=False),
                                                        generation_config=generation_config)
        runtime = exported_artifacts.runtime()
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
        # An encoder taking `inputs_embeds` means features scatter in front of the text encoder (florence2).
        scatters_at_encoder = text_embed is not None and encoder is not None and "inputs_embeds" in encoder.input_names
        self._encoder = (
            _ExportedEncoder(
                encoder,
                merge=self._merge_modalities if scatters_at_encoder else None,
                # Usually the prefill's: a decode graph reads the cross cache rather than the encoder output.
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
        self._embedder = embedder
        # Computed once from the whole prompt; each step reads its own window.
        self._embedded = None
        # The sub-config shaping the auxiliary cache the decode graph takes (the streaming encoder's own).
        self._encoder_config = (
            getattr(config, spec.encoder_config, None)
            if (spec := streaming_embedder_spec(config)) is not None
            else None
        )
        runners = [decode, self._prefill_runner, encoder, text_embed, *(m.runner for m in self._modalities)]
        if embedder is not None:
            runners.append(embedder.runner)
        self._graph_inputs = {name for runner in runners if runner is not None for name in runner.input_names}
        self._device = torch.device(decode.device)
        self._dtype = decode.dtype

    @classmethod
    def from_runners(
        cls,
        runners: dict[str, ModelRunner],
        config,
        generation_config: GenerationConfig | None = None,
    ) -> ExportedGenerator:
        """Assemble the generator from `{component_name: runner}` (as `HfExporter.export` names them) + configs.

        Text-only from a `"decode"` runner, multi-modal when `"embed_tokens"` and `"<modality>_encoder"`
        runners are present. `generation_config` must be the one used at export; when `None`, the model
        config's generation defaults apply (a growing cache). Runners must carry the export's recorded
        metadata, as [`~ExportArtifacts.runtime`] and [`~ExportedGenerator.from_pretrained`] build them."""
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        # Scatter applies only when the decode graph (decoder-only VLMs) or the encoder graph (florence2) takes
        # embeddings; otherwise run as a plain generator even if an embed graph was exported.
        takes_embeds = text_input(runners["decode"]) == "inputs_embeds" or (
            "encoder" in runners and "inputs_embeds" in runners["encoder"].input_names
        )
        text_embed = runners["embed_tokens"] if "embed_tokens" in runners and takes_embeds else None
        modalities = []
        if text_embed is not None:
            for spec in _MODALITY_SPECS:
                token_id = getattr(config, spec.token_field, None)
                if token_id is None:  # older configs name it `<modality>_token_index`
                    token_id = getattr(config, f"{spec.token_field[:-3]}_index", None)
                runner = runners.get(spec.component)
                if runner is None:
                    # A modality routed through another's getter (perception_lm's videos via
                    # `get_image_features`) shares the image graph, if it has its own placeholder token.
                    if spec.component == "image_encoder" or token_id is None or "image_encoder" not in runners:
                        continue
                    runner = runners["image_encoder"]
                modalities.append(Modality(token_id, runner, spec))
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

        The manifest maps files to components and backends; precision, cache geometry and traced shapes come
        from the artifacts. The saved `generation_config` is the export-time one, which fixes the cache built.

        Example:
            exported_artifacts = OnnxExporter().export_for_generation(model, inputs, config, generation_config=generation_config)
            exported_artifacts.save_pretrained("out/")
            runtime = ExportedGenerator.from_pretrained("out/")
            ids = runtime.generate(input_ids=prompt, max_new_tokens=32)
        """
        download_kwargs, _ = split_download_kwargs(dict(kwargs))
        runners, _ = load_export_runners(save_directory, **kwargs)
        config = AutoConfig.from_pretrained(save_directory, **download_kwargs)
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
        """The graphs are already compiled, so return the plain call."""
        return self.__call__

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    @property
    def _consumed_kwargs(self) -> set[str]:
        """Kwargs this runtime consumes without naming them on `forward`: every graph input, the text-path
        kwargs (a model deriving one internally, e.g. ctrl's `token_type_ids`, exports a graph without it),
        and with modalities their inputs plus `mm_token_type_ids`."""
        consumed = self._graph_inputs | _TEXT_KWARGS | self._modality_keys
        if self._embedder is not None:
            consumed = consumed | {self._embedder.source}
        return (consumed | {"mm_token_type_ids"}) if self._text_embed is not None else consumed

    def _validate_model_kwargs(self, model_kwargs):
        super()._validate_model_kwargs({k: v for k, v in model_kwargs.items() if k not in self._consumed_kwargs})

    def _supports_default_dynamic_cache(self) -> bool:  # noqa: D401 (instance form: reads the prototype)
        """Whether `generate` should build a `DynamicCache`, read off the traced cache input.

        Recurrent-only models (mamba, rwkv) keep fixed-size states and `DynamicCache.get_seq_length` fails on
        them; cacheless models (openai-gpt) take none.
        """
        return self._decode_runner.cache_input == "past_key_values"

    @property
    def _is_recurrent(self) -> bool:
        """Whether the graphs carry fixed-size recurrent state (`cache_params`) instead of a KV cache."""
        return self._decode_runner.cache_input == "cache_params"

    @property
    def _is_stateful(self) -> bool:
        """Whether the cache can't be rolled back, which assisted decoding needs: recurrent state, or a cache the
        backend keeps in its own variables."""
        return self._is_recurrent or self._decode_runner.owns_state

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        """Build the cache the exported decode graph expects, so `generate` is called like a normal model.

        `generate`'s builder picks the cache kind; the size comes from the trace, since `generate` sizes a
        fixed-size cache from the prompt and `max_cache_len` is part of the graph's input spec. Lazy layers are
        materialized because `torch.export` bakes real tensors into the input spec."""
        # Backend-owned state outlives the loop's cache; reset it so each `generate` starts fresh.
        for runner in (self._prefill_runner, self._decode_runner):
            if runner.owns_state:
                runner.reset_state()
        super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        # Cacheless graphs: turn caching off so the loop re-feeds the whole sequence, as traced.
        if not self._is_recurrent and self._decode_runner.cache_input is None:
            model_kwargs.pop("past_key_values", None)
            generation_config.use_cache = False
            return
        # Beam search / several returned sequences expand the batch before the first decode call.
        batch_size *= max(generation_config.num_beams, generation_config.num_return_sequences)
        # `generate` builds no cache for recurrent models (their own `prepare_inputs_for_generation` does).
        if self._is_recurrent:
            text_config = self.config.get_text_config()
            model_kwargs.setdefault(
                "cache_params",
                StaticCache(config=text_config, max_cache_len=max_cache_length)
                if generation_config.cache_implementation == "static"
                else DynamicCache(config=text_config),
            )
        cache = model_kwargs.get("past_key_values")
        if isinstance(cache, EncoderDecoderCache):
            self._seed_cross_cache(cache, batch_size)

        for cache_name in ("past_key_values", "cache_params"):
            if (cache := model_kwargs.get(cache_name)) is not None:
                resize_to_traced_lengths(cache, self._decode_runner.export_metadata.cache_lengths)
                # The traced geometry, since the config can't always give per-layer shapes.
                materialize_cache_layers(
                    cache,
                    batch_size,
                    self.config,
                    self._dtype,
                    self._device,
                    kv_geometry=self._decode_runner.kv_geometry,
                    indexer_layers=self._decode_runner.export_metadata.indexer_layers,
                )

    def _seed_cross_cache(self, cache, batch_size: int) -> None:
        """Fill the cross half from what the encoder graph produced, and mark it written.

        The decode graph only reads the cross cache, so seeding it here lets that one graph serve the prompt.
        No-op when the encoder graph produced no cross states.
        """
        cross_states = getattr(self._encoder, "cross_states", None)
        if not cross_states:
            return
        cross = cache.cross_attention_cache
        for index, (keys, values) in sorted(cross_states.items()):
            if index >= len(cross.layers) or keys is None or values is None:
                continue
            # `generate` expands the batch for beams after the encoder has run.
            if keys.shape[0] != batch_size and (repeats := batch_size // keys.shape[0]) > 1:
                keys, values = keys.repeat_interleave(repeats, dim=0), values.repeat_interleave(repeats, dim=0)
            cross.layers[index].lazy_initialization(keys, values)
            cross.layers[index].keys, cross.layers[index].values = keys, values
            if hasattr(cache, "is_updated"):
                cache.is_updated[index] = True
        # A state-owning graph is never fed the loop's cache, so seed its variables directly.
        runner = self._prefill_runner
        if runner.owns_state:
            runner.adopt_state(get_leaf_tensors({runner.cache_input: {"cross_attention_cache": cross}}), 0)

    # ── decode orchestration ──
    def create_masks_for_generate(self, config, inputs_embeds, attention_mask, **kwargs):
        """Keep the 2D padding mask when that is what the decode graph took.

        `prepare_inputs_for_generation` upgrades it to 4D for compileable caches, which fails an internal guard
        on a graph traced on the 2D mask (`ExportMetadata.mask_rank`).
        """
        if self._decode_runner.export_metadata.mask_rank == 2 and attention_mask is not None:
            return attention_mask
        return create_masks_for_generate(
            config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

    def _mask_feed(self, runner, attention_mask, position_ids, cache_len):
        """Feed the graph's mask input(s), rebuilding the causal mask where `generate` dropped one as
        redundant but the graph still takes a tensor."""
        # Mixed-attention models (nemotron_h, jamba) build their per-type mask dict inside forward.
        mask_ranks = runner.export_metadata.mask_ranks
        if mask_ranks and not isinstance(attention_mask, dict):
            padding_mask = (
                attention_mask if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2 else None
            )
            attention_mask = {
                layer_type: padding_mask if rank == 2 else None for layer_type, rank in mask_ranks.items()
            }
        if isinstance(attention_mask, dict):
            # A slot traced as `None` is a leaf of the input spec and must stay `None`.
            traced_without_mask = {name for name, rank in (mask_ranks or {}).items() if rank is None}
            attention_mask = {
                layer_type: mask
                if mask is not None or layer_type in traced_without_mask
                else self._causal_mask(position_ids, cache_len)
                for layer_type, mask in attention_mask.items()
            }
            # ONNX flattens the dict to `attention_mask.<type>` inputs; dynamo takes it whole.
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
        if not mask_inputs(runner):
            return {}
        if attention_mask is None:
            # Nothing is padded; build the rank the graph took. A 2-D-traced graph guards on mask width, so a
            # 4-D causal mask would trip it.
            name = mask_inputs(runner)[0]
            if runner.export_metadata.mask_rank == 2:
                dtype = runner.export_metadata.mask_dtype or torch.long
                batch = position_ids.shape[-2] if position_ids.dim() == 3 else position_ids.shape[0]
                return {name: torch.ones(batch, cache_len, dtype=dtype, device=self._device)}
            return {name: self._causal_mask(position_ids, cache_len)}
        # A 2D-traced graph saw a cache-width mask (bloom's alibi compares them), so zero-pad the tail. Not on
        # encoder-decoders, whose `attention_mask` is tied to the encoder sequence.
        pads_to_cache = runner.export_metadata.mask_rank == 2 and not self.config.is_encoder_decoder
        if pads_to_cache and attention_mask.dim() == 2:
            if (padding_length := cache_len - attention_mask.shape[-1]) > 0:
                attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
        return {mask_inputs(runner)[0]: attention_mask}

    def _causal_mask(self, position_ids, cache_len):
        """Full causal mask `[batch, 1, query, cache_len]` from the positions (M-RoPE: text row 0). Assumes no
        left-padding."""
        text_positions = position_ids if position_ids.dim() == 2 else position_ids[0]
        positions = torch.arange(cache_len, device=text_positions.device)
        return (positions <= text_positions[..., None])[:, None]

    def _text_feed(self, runner, text_ids, kwargs, image_sizes=None) -> dict:
        """What goes in under the graph's text input: the ids, or their embeddings with modality features
        scattered in.

        The embed graph's first output is the embeddings; further outputs (`per_layer_inputs`) are fed by name
        if declared. `image_sizes` goes only to the merge, since in `kwargs` it would hit per-step slicing.
        """
        if self._text_embed is None or text_input(runner) != "inputs_embeds":
            return {text_input(runner): text_ids}
        merge_kwargs = kwargs if image_sizes is None else {**kwargs, "image_sizes": image_sizes}
        embedded = self._merge_modalities(text_ids, merge_kwargs)
        primary, *extra = embedded
        feed = {text_input(runner): embedded[primary]}
        feed.update({name: embedded[name] for name in extra if name in runner.input_names})
        return feed

    def _step_kwargs(self, runner, kwargs: dict, feed: dict, query_length: int) -> dict:
        """Extra per-step inputs the graph declares (`token_type_ids`, auxiliary masks), trimmed to the step
        the way `generate` trims kwargs named on a real model's `forward`."""
        step = {}
        for name, value in kwargs.items():
            # `declares`: flattening backends declare only a pytree kwarg's leaves (t5gemma's mask dict on ONNX).
            if name in feed or not runner.declares(name, value):
                continue
            # Rank 2/3 have the sequence at axis 1; 4-D masks and modality features do not.
            is_per_token = isinstance(value, torch.Tensor) and name not in self._modality_keys
            if is_per_token and value.dim() in (2, 3) and value.shape[1] > query_length:
                value = value[:, -query_length:]
            step[name] = value
        return step

    def _streamed_window(self, runner, kwargs: dict, past_len: int, query_length: int) -> dict:
        """This step's window (`past_len * stride` onwards) of a modality embedded once for the whole prompt."""
        embedder = self._embedder
        if embedder is None or embedder.produces not in runner.input_names:
            return {}
        features = kwargs.get(embedder.source)
        if features is not None and (self._embedded is None or past_len == 0):
            self._embedded = next(iter(embedder.runner(**{embedder.source: features}).values()))
        if self._embedded is None:
            return {}
        window = self._embedded[:, past_len * embedder.stride : (past_len + query_length) * embedder.stride]
        return {embedder.produces: window}

    def _empty_containers(self, runner, feed: dict, batch_size: int) -> dict:
        """Fresh empty containers for those the graph declares besides its cache and the trace saw empty
        (voxtral_realtime's encoder state). Ones traced non-empty are left alone."""
        containers = {}
        for name, spec in runner.export_metadata.kwargs.items():
            container = spec.get("container")
            if container is None or spec.get("leaves") or name in feed:
                continue
            empty = _empty_container(
                container, self.config, batch_size, self._dtype, self._device, self._encoder_config
            )
            if empty is not None and runner.declares(name, empty):
                containers[name] = empty
        return containers

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
        logits_to_keep=None,
        **kwargs,
    ):
        # Empty cache runs the prefill runner, otherwise decode (the same object without a prefill graph).
        # Recurrent state under `cache_params` is treated like a KV cache.
        past_key_values = past_key_values if past_key_values is not None else cache_params
        decode_runner = self._decode_runner
        past_len = self._past_length(past_key_values)
        runner = self._prefill_runner if past_len == 0 else decode_runner
        text_ids = decoder_input_ids if decoder_input_ids is not None else input_ids
        feed = self._text_feed(runner, text_ids, kwargs, image_sizes)
        text = feed[text_input(runner)]
        if position_ids is not None and "position_ids" in runner.input_names:
            feed["position_ids"] = position_ids
        # Declaring it makes `generate` supply it; unfed, the graph runs the LM head over the whole prompt.
        if logits_to_keep is not None and "logits_to_keep" in runner.input_names:
            feed["logits_to_keep"] = logits_to_keep
        if encoder_outputs is not None and any(n.startswith("encoder_outputs") for n in runner.input_names):
            feed["encoder_outputs"] = encoder_outputs
        feed.update(self._step_kwargs(runner, kwargs, feed, text.shape[1]))
        feed.update(self._streamed_window(runner, kwargs, past_len, text.shape[1]))
        feed.update(self._empty_containers(runner, feed, text.shape[0]))
        # Key-axis width follows the layer kind, not `get_max_length`: a `DynamicSlidingWindowLayer` grows but
        # reports its whole window (4096 on gemma2).
        fixed_size = is_fixed_size(past_key_values)
        if runner.owns_state and not fixed_size:
            cache_len = past_len + text.shape[1]
        else:
            cache_len = mask_width(past_key_values, text.shape[1])
        feed.update(self._mask_feed(runner, attention_mask, position_ids, cache_len))
        # `generate` supplies neither the decoder mask nor decoder positions, so count them off the cache.
        decoder_mask = decoder_mask_input(runner)
        if decoder_mask is not None and decoder_mask not in feed:
            decoder_positions = (
                torch.arange(text.shape[1], device=self._device).unsqueeze(0).expand(text.shape[0], -1) + past_len
            )
            feed[decoder_mask] = self._causal_mask(decoder_positions, cache_len)
        # Only when declared: some graphs take no cache even though `generate` builds one (xlstm).
        if past_key_values is not None and runner.cache_input is not None and not runner.owns_state:
            feed[runner.cache_input] = past_key_values
        outputs = runner(**{name: value for name, value in feed.items() if runner.declares(name, value)})
        # Hand the prompt graph's cache to a state-owning decode graph once, after the step that wrote it.
        if decode_runner is not runner and decode_runner.owns_state and not decode_runner.state_length:
            written = runner.state_tensors(decode_runner.state_paths) if runner.owns_state else outputs
            decode_runner.adopt_state(written, text.shape[1])
        if past_key_values is not None and not runner.owns_state and not decode_runner.owns_state:
            past_key_values = _advance_cache(past_key_values, outputs, num_new_tokens=text.shape[1])
        elif fixed_size:
            # `generate` builds a fixed-size cache's 4D mask from its length; unadvanced, each query masks its
            # own slot.
            advance_cache_length(past_key_values, text.shape[1])
        returned_cache = None if self._is_recurrent else past_key_values
        return CausalLMOutputWithPast(logits=outputs["logits"], past_key_values=returned_cache)

    def _past_length(self, cache) -> int:
        """How much of the sequence has been run; a state-owning runtime counts it itself, since the loop's
        cache never grows."""
        if self._decode_runner.owns_state:
            return self._decode_runner.state_length
        return _cache_length(cache) if cache is not None else 0

    def _maybe_prepare_encoder_kwargs_for_generation(
        self, inputs_tensor, model_kwargs, model_input_name, generation_config
    ):
        """Encode each modality once, before `generate` expands the batch for beams, into the
        `mm_encoder_outputs` it expands and hands to the prefill."""
        model_kwargs = super()._maybe_prepare_encoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )
        if self.config.is_encoder_decoder or self._text_embed is None or model_input_name != "input_ids":
            return model_kwargs
        encoded = model_kwargs.setdefault("mm_encoder_outputs", {})
        for modality in self._modalities:
            feature_keys = modality.spec.feature_keys
            # `generate` pre-encodes images and videos only; deepstack per-layer features don't fit that form.
            if (
                modality.kind not in ("image", "video")
                or modality.kind in encoded
                or all(model_kwargs.get(key) is None for key in feature_keys)
                or getattr(self.config, f"{modality.kind}_token_id", None) is None
                or any(
                    re.fullmatch(r".*image_features\.\d+", name)
                    for name in modality.runner.export_metadata.output_names
                )
            ):
                continue
            features = self._modality_features(modality, model_kwargs, inputs_tensor).flatten(0, -2)
            rows_per_sample = (inputs_tensor == modality.token_id).sum(-1).tolist()
            encoded[modality.kind] = BaseModelOutputWithPooling(pooler_output=list(features.split(rows_per_sample)))
            for key in feature_keys:
                model_kwargs.pop(key, None)
        return model_kwargs

    def _modality_features(self, modality, kwargs, input_ids):
        """Run one modality's graph: features for its placeholder rows, or `{layer: features}` for deepstack."""
        feed = self._modality_feed(modality, kwargs, input_ids)
        # Anyres graphs stop at the projector since per-image token counts are data; pack here.
        image_sizes = kwargs.get("image_sizes")
        packs_anyres = (
            image_sizes is not None
            and modality.spec.feature_keys[0] == "pixel_values"
            and "image_sizes" not in modality.runner.input_names
            and _find_config_attr(self.config, "image_grid_pinpoints") is not None
        )
        if packs_anyres:
            tower_input = modality.runner.input_names[0]
            feed[tower_input] = flatten_anyres_patches(self.config, feed[tower_input], image_sizes)
        outputs = modality.runner(**feed)
        per_layer = {
            int(name.rsplit(".", 1)[-1]): tensor
            for name, tensor in outputs.items()
            if re.fullmatch(r".*image_features\.\d+", name)
        }
        if packs_anyres and per_layer:
            return {
                layer: pack_anyres_features(self.config, tensor, image_sizes, outputs)
                for layer, tensor in sorted(per_layer.items())
            }
        features = next(iter(outputs.values()))
        if packs_anyres:
            features = pack_anyres_features(self.config, features, image_sizes, outputs)
        return features

    def _merge_modalities(self, input_ids, kwargs) -> dict[str, torch.Tensor]:
        """Embed `input_ids` and scatter each present modality's features into its placeholder rows.

        Returns every per-token output of the embed graph (`inputs_embeds`, plus e.g. `per_layer_inputs`);
        only the embeddings take the features."""

        extras: dict = {}
        encoded = kwargs.get("mm_encoder_outputs") or {}

        embedded = self._text_embed(input_ids=input_ids)
        embeds_name = next(iter(embedded))
        inputs_embeds = embedded[embeds_name]
        for modality in self._modalities:
            # Never presence-key on aux keys, which `generate` may keep after dropping the features.
            pre_encoded = encoded.get(modality.kind)
            if pre_encoded is None and all(kwargs.get(key) is None for key in modality.spec.feature_keys):
                continue
            if modality.token_id is not None:
                mask = (input_ids == modality.token_id).unsqueeze(-1)
            else:
                # kosmos2_5 marks rows with a mask kwarg that `generate` keeps full-length; take this step's tail.
                mask_key = next(key for key in modality.spec.aux_keys if key.endswith("_position_mask"))
                mask = (kwargs[mask_key][:, -input_ids.shape[1] :] == 1).unsqueeze(-1)
            if not mask.any():
                continue
            if pre_encoded is not None:
                features = torch.cat(pre_encoded.pooler_output)
            else:
                features = self._modality_features(modality, kwargs, input_ids)
            # Deepstack features are summed in inside the decoder, so zero the placeholder rows here.
            if isinstance(features, dict):
                extras["deepstack_features"] = {
                    layer: tensor.to(inputs_embeds.dtype) for layer, tensor in features.items()
                }
                extras["vision_mask"] = mask
                inputs_embeds = inputs_embeds.masked_fill(mask, 0.0)
                continue
            inputs_embeds = inputs_embeds.masked_scatter(mask, features.to(inputs_embeds.dtype))
        return {**embedded, embeds_name: inputs_embeds, **extras}

    def _modality_feed(self, modality, kwargs, input_ids) -> dict:
        """Everything one modality graph declares, from `generate`'s kwargs, the prompt ids, the precompute
        (grid-derived `cu_seqlens` / `window_index`), then the config itself.

        This modality's feature key is routed onto the graph's feature input, and other modalities' keys are
        dropped: a video getter often declares the image kwarg `pixel_values` (llava_next_video).
        """
        declared = set(modality.runner.input_names)
        feature_input = modality.runner.input_names[0]
        present = next(
            (key for key in modality.spec.feature_keys if key not in declared and kwargs.get(key) is not None),
            None,
        )
        foreign = {key for other in self._modalities if other is not modality for key in other.input_keys} - set(
            modality.input_keys
        )
        inputs = {}
        for key, value in kwargs.items():
            if value is None or key in _TEXT_KWARGS or key in foreign:
                continue
            # Own kwargs stay even when undeclared: the precompute derives from them, and OpenVINO drops an
            # unread omni `input_features`.
            if modality.runner.declares(name := grid_renamed(key), value) or key in modality.input_keys:
                inputs[name] = value
        # Must not overwrite a differently-meant first input fed by name above (vibevoice_asr's `input_values`).
        if present is not None and feature_input not in inputs:
            inputs[feature_input] = kwargs[present]
        # Cast before the precompute: the graph was traced with the precompute's own dtypes (fp32 weights).
        inputs = cast_leaf_tensors(inputs, dtype=modality.runner.dtype, device=self._device)
        inputs = precompute_export_inputs(self.config, inputs)
        # Device-only: the precompute builds some index tensors on CPU (minicpmv4_6's `window_index`).
        inputs = {
            name: value.to(self._device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        feed = {name: value for name, value in inputs.items() if modality.runner.declares(name, value)}
        if "input_ids" in declared:
            feed["input_ids"] = input_ids
        for name in declared - feed.keys():
            if (value := _find_config_attr(self.config, name)) is not None:
                feed[name] = value
        return feed

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Per-step inputs, with `cross_attention_mask` trimmed to the step (mllama's own hook does this).

        The model's hook can't be called (its zero-argument `super()` refuses a foreign `self`), and recorded
        widths can't be used: the merged decode capture leaves this mask at one row.
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
        """Multi-modal M-RoPE `position_ids`, as a VLM's own override of this method builds them.

        Text row from `super()`, modality rows from `get_rope_index_from_config`; decode advances the text row
        by the cached rope delta. The row count is per architecture (`ExportMetadata.position_axes`)."""
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        past_length = self._past_length(model_kwargs.get("past_key_values"))
        runner = self._prefill_runner if past_length == 0 else self._decode_runner
        axes = runner.export_metadata.position_axes
        if past_length != 0 and getattr(self, "_rope_deltas", None) is not None:
            positions = text_positions[None, ...] + self._rope_deltas
            # Every axis (hunyuan_vl) or the single broadcast row (qwen2_vl), as traced.
            return positions.expand(axes, -1, -1) if axes and axes > positions.shape[0] else positions

        if model_kwargs.get("input_ids") is not None and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]
        # `generate` has already turned the mask into per-layer form, not the 2D one `get_rope_index` wants.
        rope_index = get_rope_index_from_config(
            self.config, {**model_kwargs, "input_ids": inputs_tensor, "attention_mask": None}
        )
        if rope_index is None:
            return text_positions
        vision_positions, self._rope_deltas = rope_index
        # A layout covering only the modality axes gets the text row in front, as the model's forward does.
        if axes == vision_positions.shape[0]:
            return vision_positions
        return torch.cat([text_positions[None, ...], vision_positions], dim=0)
