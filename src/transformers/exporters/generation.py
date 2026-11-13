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
"""Running a generation export: `generate` drives its decode graph as it drives the model."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from ..generation import GenerationConfig, GenerationMixin
from ..modeling_outputs import BaseModelOutput
from ..utils import is_torch_available
from .base import ExportedModel
from .cache import advance_cache_length, causal_mask, mask_width, materialize_cache_layers
from .decompose import _MODALITY_SPECS, DEEPSTACK_INPUTS, decode_graph_name
from .metadata import load_treespec
from .multimodal import ExportedMultimodalModel
from .precompute import get_rope_index_from_config
from .runners.base import ModelRunner


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..configuration_utils import PreTrainedConfig
    from .base import ExportedComponent
    from .configs import ExportFormat


def _with_signature(function, names) -> Any:
    """`function`, with keyword parameters `names` as its signature: `generate` reads a model's `forward` signature to
    decide which kwargs to build."""
    parameters = [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in names]
    parameters.append(inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD))
    function.__signature__ = inspect.Signature(parameters)
    return function


class _ExportedEncoder:
    """An encoder-decoder's `encoder_model` graph, called by `generate` as the model's encoder: it returns the
    encoder outputs, and keeps the cross-attention keys and values the graph computed for the decode cache."""

    main_input_name = "input_ids"

    def __init__(self, model: ExportedModelForSeq2SeqLM):
        # Its runner is looked up on each call: `ExportedModel.to` replaces them.
        self.model = model
        self.cross_states = {}
        traced = model.components["encoder_model"].metadata.kwargs
        self.forward = _with_signature(lambda **kwargs: self(**kwargs), traced)

    def __call__(self, **kwargs) -> BaseModelOutput:
        runner = self.model.runners["encoder_model"]
        traced = runner.metadata.kwargs
        outputs = runner(**{name: value for name, value in kwargs.items() if name in traced})
        self.cross_states = {name: value for name, value in outputs.items() if name.startswith("cross_")}
        output_class = self.model._encoder_output_class()
        return output_class(**{name: value for name, value in outputs.items() if not name.startswith("cross_")})


class ExportedModelForCausalLM(ExportedModel, GenerationMixin):
    """A causal LM's generation export ([`~HfExporter.export_for_generation`]) that `generate` drives as it drives
    the model it came from: each step calls the decode graph with the kwargs `generate` builds for the model's own
    forward, which are what the graph was traced with.

    ```python
    >>> exported = OpenVINOExporter().export_for_generation(model, inputs, OpenVINOConfig(dynamic=True))
    >>> exported.save_pretrained("exported")
    >>> model = ExportedModel.from_pretrained("exported")
    >>> generated = model.generate(**inputs, max_new_tokens=32)
    ```

    [`ExportedMultimodalModelForCausalLM`] and [`ExportedModelForSeq2SeqLM`] add a multimodal prompt's embedding and
    an encoder-decoder's encoder on top.
    """

    main_input_name = "input_ids"
    base_model_prefix = ""
    _is_stateful = False

    def __init__(
        self,
        components: dict[str, ExportedComponent],
        export_format: ExportFormat | str,
        config: PreTrainedConfig | None = None,
        generation_config: GenerationConfig | None = None,
        generation: bool = True,
    ):
        super().__init__(components, export_format, config, generation_config, generation=True)
        if self.generation_config is None:
            self.generation_config = GenerationConfig.from_model_config(config)
        self.forward = _with_signature(lambda **kwargs: self._step(**kwargs), self._forward_names())
        self._generating = False

    @property
    def runner(self) -> ModelRunner:
        """The decode graph's runner."""
        return self.runners[decode_graph_name(self.components)]

    def _decode_metadata(self):
        return self.components[decode_graph_name(self.components)].metadata

    def _forward_names(self) -> list[str]:
        """What `forward` declares, so `generate` builds them: the kwargs the decode graph was traced with, and the
        positions a graph traced on `cache_position` takes it from (`generate` builds them, not it)."""
        names = list(self._decode_metadata().kwargs)
        if "cache_position" in names and "position_ids" not in names:
            names.append("position_ids")
        return names

    def __call__(self, **kwargs) -> Any:
        return self.forward(**kwargs)

    def can_generate(self) -> bool:
        return True

    def is_remote_code(self) -> bool:
        return False

    def get_output_embeddings(self):
        return None

    def get_experts_implementation(self) -> dict:
        return {}

    def get_compiled_call(self, compile_config):
        """The graphs are already compiled."""
        return self.__call__

    def prepare_inputs_for_generation(self, *args, attention_mask=None, **kwargs):
        """[`~GenerationMixin.prepare_inputs_for_generation`], with the padding mask as `generate` keeps it for a graph
        traced on a 4D mask: `_causal_mask` builds that one, not `generate` from a config it may not match."""
        model_inputs = super().prepare_inputs_for_generation(*args, attention_mask=attention_mask, **kwargs)
        padding_mask = attention_mask
        traced = self._decode_metadata().kwargs.get("attention_mask", {})
        if not self.config.is_encoder_decoder and traced.get("rank") == 4 and getattr(padding_mask, "ndim", 0) == 2:
            model_inputs["attention_mask"] = padding_mask
        return model_inputs

    def generate(self, *args, **kwargs):
        """[`~GenerationMixin.generate`]. A stateful decode graph (OpenVINO's) hands out its own output buffers instead
        of copies meanwhile: it only returns the logits (the cache is its state), and `generate` copies the row it
        reads before the next step overwrites them."""
        self._generating = True
        try:
            return super().generate(*args, **kwargs)
        finally:
            self._generating = False

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        """`generate`'s cache, materialized the way the decode graph was traced on it; a backend keeping the cache as
        state starts it over."""
        if self.runner.stateful:
            # Before `generate` makes its own: one here was passed in to continue from.
            if model_kwargs.get("cache") is not None:
                raise ValueError(
                    "This export keeps its cache as its own state, which a cache passed in can't restore: generate from "
                    "the whole conversation as the prompt instead."
                )
            self.runner.reset_state()
        # A fixed-size cache is the size the decode graph was traced on, whatever this call generates.
        traced_length = self.runner.metadata.cache_geometry.get("max_cache_len")
        if traced_length is not None:
            if max_cache_length > traced_length:
                raise ValueError(
                    f"This export's static cache holds {traced_length} positions and this call needs "
                    f"{max_cache_length}: export it again with a generation config asking for more tokens."
                )
            max_cache_length = traced_length
        super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        cache = model_kwargs.get("cache")
        if cache is not None:
            # Beam search and several returned sequences expand the batch before the first step.
            batch_size *= max(generation_config.num_beams, generation_config.num_return_sequences)
            # Saved as JSON: layer indices come back as strings, shapes as lists.
            geometry = self.runner.metadata.cache_geometry
            materialize_cache_layers(
                cache,
                batch_size,
                self.config,
                self.dtype,
                self.device,
                kv_geometry={int(index): tuple(shape) for index, shape in geometry.get("kv", {}).items()},
                indexer_layers={int(index): present for index, present in geometry.get("indexers", {}).items()},
                linear_states={
                    int(index): {name: [tuple(shape) for shape in shapes] for name, shapes in states.items()}
                    for index, states in geometry.get("linear_states", {}).items()
                },
            )

    def _step(self, **kwargs) -> Any:
        traced = self.runner.metadata.kwargs
        text_name = next(
            name
            for name in ("inputs_embeds", "decoder_input_ids", "input_ids")
            if name in traced and kwargs.get(name) is not None
        )
        query_length = kwargs[text_name].shape[1]
        # A multi-token decode graph is traced on two tokens; one traced on a single token (any static export, or a
        # dynamic one whose model can't decode several at once) takes the prompt a token at a time.
        if query_length == 1 or traced[text_name].get("shape", [0, 0])[1] != 1:
            return self._call(**kwargs)
        outputs = None
        for index in range(query_length):
            outputs = self._call(**self._token(kwargs, index, query_length))
            if getattr(outputs, "cache", None) is not None:
                kwargs["cache"] = outputs.cache
        return outputs

    @staticmethod
    def _token(kwargs: dict, index: int, query_length: int) -> dict:
        """The `index`-th token's call out of a `query_length`-token step: per-token inputs sliced to it, and the
        padding mask to the positions seen so far."""

        def mask_for_token(mask):
            # A padding mask covers the positions seen so far, a 4D one the token's own query row.
            if mask is None or mask.dim() not in (2, 4):
                return mask
            return (
                mask[:, : mask.shape[1] - query_length + index + 1]
                if mask.dim() == 2
                else mask[:, :, index : index + 1]
            )

        token = dict(kwargs)
        features = {spec.features for spec in DEEPSTACK_INPUTS.values()}
        for name, value in kwargs.items():
            if name in features and value is not None:
                # One row per position of the step (`dense_deepstack`): the token's.
                def rows(dense):
                    return dense.view(-1, query_length, dense.shape[-1])[:, index]

                is_dict = isinstance(value, dict)
                token[name] = {key: rows(item) for key, item in value.items()} if is_dict else [rows(v) for v in value]
                continue
            if name in ("attention_mask", "decoder_attention_mask"):
                is_dict = isinstance(value, dict)
                token[name] = (
                    {key: mask_for_token(mask) for key, mask in value.items()} if is_dict else mask_for_token(value)
                )
                continue
            if not isinstance(value, torch.Tensor):
                continue
            elif name in ("position_ids", "decoder_position_ids"):
                token[name] = value[..., index : index + 1]
            elif name == "cache_position":
                token[name] = value[index : index + 1]
            elif value.dim() >= 2 and value.shape[1] == query_length:
                token[name] = value[:, index : index + 1]
        return token

    def _call(self, **kwargs) -> Any:
        traced = self.runner.metadata.kwargs
        # `generate` drops a full causal mask as redundant (SDPA's `is_causal`); a graph traced on one still takes it.
        for mask_name in ("attention_mask", "decoder_attention_mask"):
            spec, mask = traced.get(mask_name, {}), kwargs.get(mask_name)
            if spec.get("rank") == 4 and (mask is None or mask.dim() == 2):
                kwargs[mask_name] = self._causal_mask(kwargs, mask, spec["dtype"])
            elif spec.get("rank") == 2 and mask is None:
                # A padding mask over the cache and the step (an optimum-intel graph takes one): nothing is padded.
                text = next(kwargs[name] for name in ("input_ids", "inputs_embeds") if kwargs.get(name) is not None)
                cache = kwargs.get("cache")
                width = (cache.get_seq_length() if cache is not None else 0) + text.shape[1]
                kwargs[mask_name] = torch.ones(text.shape[0], width, dtype=getattr(torch, spec["dtype"]))
            elif "leaves" in spec and isinstance(mask, dict):
                # Per layer type (mixed full / sliding attention): only the slots `generate` left empty.
                kwargs[mask_name] = {
                    layer_type: self._causal_mask(kwargs, None, spec["leaves"][layer_type]["dtype"])
                    if value is None and spec["leaves"].get(layer_type, {}).get("rank") == 4
                    else value
                    for layer_type, value in mask.items()
                }
        if "cache_position" in traced and kwargs.get("cache_position") is None:
            # The step's cache slots, as its positions: such a graph (optimum-executorch's) takes a batch of one.
            kwargs["cache_position"] = kwargs["position_ids"][0]
        stateful = self.runner.stateful
        inputs = {name: value for name, value in kwargs.items() if name in traced}
        outputs = self.runner(share_outputs=stateful and self._generating, **inputs)
        # A backend keeping the cache as state (OpenVINO's stateful model) never hands the loop's cache back advanced.
        if stateful and kwargs.get("cache") is not None:
            text = next(
                kwargs[name]
                for name in ("inputs_embeds", "decoder_input_ids", "input_ids")
                if kwargs.get(name) is not None
            )
            advance_cache_length(kwargs["cache"], text.shape[1])
        return outputs

    def _causal_mask(self, kwargs: dict, padding_mask, dtype: str):
        """The 4D causal mask `[batch, 1, query, cache width]` over the step's positions, with the padding mask's
        zeros masked out."""
        if padding_mask is not None and padding_mask.dim() == 4:
            return padding_mask
        text = next(
            kwargs[name]
            for name in ("input_ids", "decoder_input_ids", "inputs_embeds")
            if kwargs.get(name) is not None
        )
        cache = kwargs.get("cache")
        cache = getattr(cache, "self_attention_cache", cache)
        width = mask_width(cache, text.shape[1])
        # The cache slots the step writes, not `position_ids`, which left padding shifts back.
        past = width - text.shape[1] if cache is None else cache.get_seq_length()
        slots = torch.arange(past, past + text.shape[1], device=text.device).expand(text.shape[0], -1)
        mask = causal_mask(slots, width)
        if padding_mask is not None:
            padding = torch.nn.functional.pad(
                padding_mask.bool(), (0, max(width - padding_mask.shape[1], 0)), value=True
            )
            mask = mask & padding[:, None, None, :width]
        dtype = getattr(torch, dtype)
        if dtype.is_floating_point:
            # Added to the scores: zero where a key is seen, the lowest value where it isn't.
            return torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, torch.finfo(dtype).min)
        return mask.to(dtype)


# What the prompt's embedding consumes: laid out by item (image, video, audio clip), not by prompt
_MODALITY_INPUTS = {key for spec in _MODALITY_SPECS for key in (*spec.input_keys, spec.grid_key)} | {"image_sizes"}


class ExportedMultimodalModelForCausalLM(ExportedModelForCausalLM, ExportedMultimodalModel):
    """A multimodal causal LM's generation export: the prompt's token ids are embedded and each modality's features
    scattered into their placeholder rows, as [`ExportedMultimodalModel`] does, before the decode graph runs on
    `inputs_embeds`."""

    def _forward_names(self) -> list[str]:
        # The step's ids, embedded here: the decode graph takes `inputs_embeds`.
        return [*super()._forward_names(), "input_ids"]

    def _validate_model_kwargs(self, model_kwargs: dict):
        """Also accept the modality inputs, which the prompt's embedding consumes."""
        consumed = _MODALITY_INPUTS | {"mm_token_type_ids"}
        super()._validate_model_kwargs({key: value for key, value in model_kwargs.items() if key not in consumed})

    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False, input_ids=None, **model_kwargs):
        """Embeds the prompt before `generate` repeats it for each beam or returned sequence: the modality inputs are
        laid out by item (an image's patches, a grid per image), not by prompt, so only their embeddings repeat."""
        if expand_size > 1 and input_ids is not None and model_kwargs.get("inputs_embeds") is None:
            embedded, _ = self.embed(input_ids=input_ids, **model_kwargs)
            model_kwargs = {key: value for key, value in model_kwargs.items() if key not in _MODALITY_INPUTS}

            def repeat_rows(features):
                # Deepstack features cover every `(prompt, position)` row: `generate` only repeats top-level tensors.
                features = features.unflatten(0, (input_ids.shape[0], -1)).repeat_interleave(expand_size, dim=0)
                return features.flatten(0, 1)

            for key, value in embedded.items():
                if isinstance(value, dict):
                    value = {layer: repeat_rows(features) for layer, features in value.items()}
                elif isinstance(value, list):
                    value = [repeat_rows(features) for features in value]
                model_kwargs[key] = value
        return super()._expand_inputs_for_generation(expand_size, is_encoder_decoder, input_ids, **model_kwargs)

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """M-RoPE positions for the prompt, as a VLM's own override builds them; `generate` advances them per step."""
        positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        input_ids = model_kwargs.get("input_ids", inputs_tensor)
        rope_index = get_rope_index_from_config(self.config, {**model_kwargs, "input_ids": input_ids})
        if rope_index is None:
            return positions
        rope_positions = rope_index[0]
        # A graph traced with one more row takes the text positions in front, as the model's forward builds them.
        traced_shape = self._decode_metadata().kwargs.get("position_ids", {}).get("shape", [])
        if len(traced_shape) == 3 and traced_shape[0] == rope_positions.shape[0] + 1:
            return torch.cat([positions[None].to(rope_positions), rope_positions], dim=0)
        return rope_positions

    def _step(self, **kwargs) -> Any:
        if kwargs.get("input_ids") is not None:
            embedded, _ = self.embed(**kwargs)
            kwargs.update(embedded)
        return super()._step(**kwargs)


class ExportedModelForSeq2SeqLM(ExportedModelForCausalLM):
    """An encoder-decoder's generation export: `generate` runs the `encoder_model` graph once, which also computes
    the decoder's cross-attention keys and values, and the cache starts with them."""

    def __init__(
        self,
        components: dict[str, ExportedComponent],
        export_format: ExportFormat | str,
        config: PreTrainedConfig | None = None,
        generation_config: GenerationConfig | None = None,
        generation: bool = True,
    ):
        super().__init__(components, export_format, config, generation_config, generation)
        self.encoder = _ExportedEncoder(self)

    def get_encoder(self):
        return self.encoder

    def _encoder_output_class(self) -> type:
        """The class the decode graph took its `encoder_outputs` as: `torch.export` checks it."""
        kwargs_spec = load_treespec(self._decode_metadata().in_spec).children_specs[1]
        traced = dict(zip(kwargs_spec.context, kwargs_spec.children_specs))
        return traced["encoder_outputs"].type if "encoder_outputs" in traced else BaseModelOutput

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        """Also fill the cross-attention cache with what the encoder graph computed, unless the decode graph writes
        it itself (`decoder_writes_cross_cache`)."""
        super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        cache = model_kwargs.get("cache")
        if cache is None or not self.encoder.cross_states:
            return
        batch_size *= max(generation_config.num_beams, generation_config.num_return_sequences)
        for index, layer in enumerate(cache.cross_attention_cache.layers):
            keys = self.encoder.cross_states[f"cross_keys_{index}"]
            values = self.encoder.cross_states[f"cross_values_{index}"]
            if keys.shape[0] != batch_size:
                keys = keys.repeat_interleave(batch_size // keys.shape[0], dim=0)
                values = values.repeat_interleave(batch_size // values.shape[0], dim=0)
            layer.lazy_initialization(keys, values)
            layer.keys, layer.values = keys, values
            cache.is_updated[index] = True
