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

from ...generation import GenerationConfig, GenerationMixin
from ...modeling_outputs import BaseModelOutput
from ...utils import is_torch_available
from ..cache import advance_cache_length, causal_mask, mask_width, materialize_cache_layers
from ..decompose import _MODALITY_SPECS, DEEPSTACK_INPUTS, decode_graph_name
from ..metadata import load_treespec
from ..precompute import get_rope_index_from_config
from .base import ExportedModel, ModelRunner
from .multimodal import ExportedMultimodalModel


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ...configuration_utils import PreTrainedConfig


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

    def __init__(self, runner: ModelRunner, output_class: type = BaseModelOutput):
        self.runner = runner
        # The class the decode graph was traced with: `torch.export` checks it.
        self.output_class = output_class
        self.cross_states = {}
        self.forward = _with_signature(lambda **kwargs: self(**kwargs), runner.metadata.kwargs)

    def __call__(self, **kwargs) -> BaseModelOutput:
        traced = self.runner.metadata.kwargs
        outputs = self.runner(**{name: value for name, value in kwargs.items() if name in traced})
        self.cross_states = {name: value for name, value in outputs.items() if name.startswith("cross_")}
        return self.output_class(**{name: value for name, value in outputs.items() if not name.startswith("cross_")})


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
        runners: dict[str, ModelRunner],
        config: PreTrainedConfig,
        generation_config: GenerationConfig | None = None,
    ):
        super().__init__(runners[decode_graph_name(runners)], config)
        self.generation_config = (
            generation_config if generation_config is not None else GenerationConfig.from_model_config(config)
        )
        self.forward = _with_signature(lambda **kwargs: self._step(**kwargs), self._forward_names())
        self._generating = False

    def _forward_names(self) -> list[str]:
        """What `forward` declares, so `generate` builds them: the kwargs the decode graph was traced with."""
        return list(self.runner.metadata.kwargs)

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
        if hasattr(self.runner, "reset_state"):
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
        cache = model_kwargs.get("past_key_values")
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
            if getattr(outputs, "past_key_values", None) is not None:
                kwargs["past_key_values"] = outputs.past_key_values
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
            elif "leaves" in spec and isinstance(mask, dict):
                # Per layer type (mixed full / sliding attention): only the slots `generate` left empty.
                kwargs[mask_name] = {
                    layer_type: self._causal_mask(kwargs, None, spec["leaves"][layer_type]["dtype"])
                    if value is None and spec["leaves"].get(layer_type, {}).get("rank") == 4
                    else value
                    for layer_type, value in mask.items()
                }
        stateful = "beam_idx" in self.runner.metadata.input_names
        inputs = {name: value for name, value in kwargs.items() if name in traced}
        outputs = self.runner(share_outputs=stateful and self._generating, **inputs)
        # A backend keeping the cache as state (OpenVINO's stateful model) never hands the loop's cache back advanced.
        if "beam_idx" in self.runner.metadata.input_names and kwargs.get("past_key_values") is not None:
            text = next(
                kwargs[name]
                for name in ("inputs_embeds", "decoder_input_ids", "input_ids")
                if kwargs.get(name) is not None
            )
            advance_cache_length(kwargs["past_key_values"], text.shape[1])
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
        cache = kwargs.get("past_key_values")
        cache = getattr(cache, "self_attention_cache", cache)
        width = mask_width(cache, text.shape[1])
        positions = kwargs.get("position_ids")
        if positions is None:
            past = width - text.shape[1] if cache is None else cache.get_seq_length()
            positions = torch.arange(past, past + text.shape[1], device=text.device).expand(text.shape[0], -1)
        mask = causal_mask(positions, width)
        if padding_mask is not None:
            padding = torch.nn.functional.pad(
                padding_mask.bool(), (0, max(width - padding_mask.shape[1], 0)), value=True
            )
            mask = mask & padding[:, None, None, :width]
        return mask.to(getattr(torch, dtype))


class ExportedMultimodalModelForCausalLM(ExportedModelForCausalLM):
    """A multimodal causal LM's generation export: the prompt's token ids are embedded and each modality's features
    scattered into their placeholder rows, as [`ExportedMultimodalModel`] does, before the decode graph runs on
    `inputs_embeds`."""

    def __init__(
        self,
        runners: dict[str, ModelRunner],
        config: PreTrainedConfig,
        generation_config: GenerationConfig | None = None,
    ):
        self.embedder = ExportedMultimodalModel(runners, config)
        super().__init__(runners, config, generation_config)

    def _forward_names(self) -> list[str]:
        # The step's ids, embedded here: the decode graph takes `inputs_embeds`.
        return [*super()._forward_names(), "input_ids"]

    def _validate_model_kwargs(self, model_kwargs: dict):
        """Also accept the modality inputs, which the prompt's embedding consumes."""
        consumed = {key for spec in _MODALITY_SPECS for key in (*spec.input_keys, spec.grid_key)}
        consumed |= {"image_sizes", "mm_token_type_ids"}
        super()._validate_model_kwargs({key: value for key, value in model_kwargs.items() if key not in consumed})

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        """M-RoPE positions for the prompt, as a VLM's own override builds them; `generate` advances them per step."""
        positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        input_ids = model_kwargs.get("input_ids", inputs_tensor)
        rope_index = get_rope_index_from_config(self.config, {**model_kwargs, "input_ids": input_ids})
        if rope_index is None:
            return positions
        rope_positions = rope_index[0]
        # A graph traced with one more row takes the text positions in front, as the model's forward builds them.
        traced_shape = self.runner.metadata.kwargs.get("position_ids", {}).get("shape", [])
        if len(traced_shape) == 3 and traced_shape[0] == rope_positions.shape[0] + 1:
            return torch.cat([positions[None].to(rope_positions), rope_positions], dim=0)
        return rope_positions

    def _step(self, **kwargs) -> Any:
        if kwargs.get("input_ids") is not None:
            embedded, _ = self.embedder.embed(**kwargs)
            kwargs.update(embedded)
        return super()._step(**kwargs)


class ExportedModelForSeq2SeqLM(ExportedModelForCausalLM):
    """An encoder-decoder's generation export: `generate` runs the `encoder_model` graph once, which also computes
    the decoder's cross-attention keys and values, and the cache starts with them."""

    def __init__(
        self,
        runners: dict[str, ModelRunner],
        config: PreTrainedConfig,
        generation_config: GenerationConfig | None = None,
    ):
        super().__init__(runners, config, generation_config)
        kwargs_spec = load_treespec(self.runner.metadata.in_spec).children_specs[1]
        traced = dict(zip(kwargs_spec.context, kwargs_spec.children_specs))
        output_class = traced["encoder_outputs"].type if "encoder_outputs" in traced else BaseModelOutput
        self.encoder = _ExportedEncoder(runners["encoder_model"], output_class)

    def get_encoder(self):
        return self.encoder

    def _prepare_cache_for_generation(
        self, generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
    ):
        """Also fill the cross-attention cache with what the encoder graph computed, unless the decode graph writes
        it itself (`decoder_writes_cross_cache`)."""
        super()._prepare_cache_for_generation(
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
        )
        cache = model_kwargs.get("past_key_values")
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
