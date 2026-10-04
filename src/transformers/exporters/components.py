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
"""What a component is, shared by the three stages that pass one along.

A decomposition produces components, an exporter turns each into a graph, and a runtime drives them. Components
are keyed by name (`"decode"`, `"prefill"`, `"encoder"`, `"embed_tokens"`, `"<modality>_encoder"`), and an export
with a `"decode"` graph is driven through `generate`. This holds the `Component` / `ExportedComponent` pair, then
the modules a decomposition wraps a model's own methods in so each can be exported on its own.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from ..utils.import_utils import is_torch_available
from .metadata import ExportMetadata
from .utils import get_leaf_tensors


if is_torch_available():
    import torch

    from ..modeling_utils import PreTrainedModel


@dataclass
class Component:
    """One piece of a decomposed model, ready to export: what to trace, and what to trace it with."""

    module: Any
    inputs: dict[str, Any]


@dataclass
class ExportedComponent:
    """One exported graph, with its trace metadata."""

    artifact: Any
    metadata: ExportMetadata


class _ModelComponent(torch.nn.Module):
    """Wraps a model so a single method can be exported on its own. Missing attributes fall through to the
    model, so the export precompute introspects the component exactly as it would the model."""

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        # `super().__getattr__("model")`, not `self.model`, to avoid re-entering this hook.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)


class ModalityEncoder(_ModelComponent):
    """Runs one modality's `get_<modality>_features` and returns a single `[num_tokens, hidden]` tensor."""

    def __init__(self, model: PreTrainedModel, getter: str, grid_kwarg: str | None = None):
        super().__init__(model)
        self._getter = getter
        self._grid_kwarg = grid_kwarg

    def forward(self, **kwargs):
        if self._grid_kwarg is not None and "grid_thw" in kwargs:
            kwargs[self._grid_kwarg] = kwargs.pop("grid_thw")
        # The precompute may offer kwargs the getter never takes (`window_index` on minicpmv4_6).
        getter = getattr(self.model, self._getter)
        parameters = inspect.signature(getter).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            kwargs = {name: value for name, value in kwargs.items() if name in parameters}
        outputs = getter(**kwargs)
        # Some outputs declare both fields and fill neither (granite4_vision), so fall through on `None`.
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = getattr(outputs, "last_hidden_state", None)
        if features is None:
            features = outputs
        return torch.cat(features) if isinstance(features, (tuple, list)) else features


class PatchVisionEncoder(_ModelComponent):
    """An anyres vision tower + projector, cut before `pack_image_features`.

    Tracing the packing would bake one `(image count, sizes)` pair into the graph, so the runtime packs
    instead. `image_newline` is returned too, since the packing needs it.
    """

    def projector_specs(self) -> list[tuple[int, Any, Any]] | None:
        """`(llm_layer, vision_layer, projector)` per projector of a deepstack tower (granite4_vision), or
        `None` for the single-projector case."""
        config = self.model.config
        layer_map = getattr(config, "deepstack_layer_map", None)
        if not layer_map:
            return None
        specs = [
            (llm_layer, vision_layer, self.model.layerwise_projectors[index])
            for index, (vision_layer, llm_layer) in enumerate(layer_map)
        ]
        specs += [
            (llm_layer, config.spatial_vision_layer, self.model.spatial_projectors[index])
            for index, llm_layer in enumerate(config.spatial_target_layers)
        ]
        return specs

    def forward(self, pixel_values, vision_feature_layer=None, vision_feature_select_strategy=None):
        outputs = self.model.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)

        def project(layer, projector):
            if isinstance(layer, int):
                selected = outputs.hidden_states[layer]
            else:
                selected = torch.cat([outputs.hidden_states[index] for index in layer], dim=-1)
            if vision_feature_select_strategy == "default":
                selected = selected[:, 1:]
            return projector(selected)

        specs = self.projector_specs()
        if specs is None:
            features = {"image_features": project(vision_feature_layer, self.model.multi_modal_projector)}
        else:
            # Keyed by injection layer, so the runtime rebuilds `deepstack_features` without the config.
            features = {f"image_features.{llm}": project(layer, proj) for llm, layer, proj in specs}
        features["image_newline"] = self.model.image_newline
        return features


class CrossAttentionEncoder(_ModelComponent):
    """The encoder, plus the cross-attention keys and values its output determines.

    A decode graph traced after the first step only reads the cross cache, so the encoder graph computes it
    instead: `forward` returns the encoder outputs alongside `cross_keys_<layer>` / `cross_values_<layer>`.
    The writers (`capture_cross_writers`) are replayed with their captured arguments and a one-token zero
    query; the attention output is discarded.
    """

    def __init__(self, encoder, writers: dict):
        super().__init__(encoder)
        self.writers = torch.nn.ModuleList(writer.module for _index, writer in sorted(writers.items()))
        self._calls = [writer for _index, writer in sorted(writers.items())]

    def forward(self, **encoder_inputs):
        from ..cache_utils import DynamicCache, EncoderDecoderCache

        encoded = self.model(**encoder_inputs)
        states = encoded.last_hidden_state if hasattr(encoded, "last_hidden_state") else encoded[0]
        cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        for writer in self._calls:
            args, kwargs = list(writer.args), dict(writer.kwargs)
            for slot, value in (
                (writer.states_at, states),
                (writer.cache_at, cache),
                # Live batch, captured width: a query at the captured batch would not broadcast.
                (
                    writer.query_at,
                    None if writer.width is None else states.new_zeros(states.shape[0], 1, writer.width),
                ),
            ):
                if slot is None or value is None:
                    continue
                if slot[0] == "arg":
                    args[slot[1]] = value
                else:
                    kwargs[slot[1]] = value
            writer.module(*args, **kwargs)
        # All encoder outputs, not just hidden states: parakeet's decoder reads its frame mask.
        outputs = dict(get_leaf_tensors(encoded))
        for index, layer in enumerate(cache.cross_attention_cache.layers):
            outputs[f"cross_keys_{index}"] = layer.keys
            outputs[f"cross_values_{index}"] = layer.values
        return outputs


class TokenEmbedder(_ModelComponent):
    """`input_ids -> inputs_embeds`, zeroing the placeholder ids first as a VLM `forward` does. Wraps the
    text decoder, never the outer VLM, so the precompute's `get_rope_index` branch stays off.

    A decoder with per-layer embeddings (gemma3n) otherwise recovers them from `inputs_embeds` by a
    data-dependent reverse lookup that breaks once features are scattered in, so `per_layer_inputs` is
    returned too, with placeholders mapped to the pad id as in the eager forward.
    """

    def __init__(self, decoder: PreTrainedModel, placeholder_ids: list[int]):
        super().__init__(decoder)
        self._placeholder_ids = placeholder_ids

    def _placeholder_mask(self, input_ids):
        placeholder = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self._placeholder_ids:
            placeholder = placeholder | (input_ids == token_id)
        return placeholder

    def forward(self, input_ids):
        placeholder = self._placeholder_mask(input_ids)
        inputs_embeds = self.model.get_input_embeddings()(input_ids.masked_fill(placeholder, 0))
        if not hasattr(self.model, "get_per_layer_inputs"):
            return inputs_embeds
        pad_token_id = self.model.config.get_text_config().pad_token_id or 0
        per_layer_ids = input_ids.masked_fill(placeholder, pad_token_id)
        # gemma4 takes `(input_ids, inputs_embeds)` with no defaults, gemma3n only `(input_ids)`.
        takes_embeds = len(inspect.signature(self.model.get_per_layer_inputs).parameters) > 1
        per_layer_inputs = (
            self.model.get_per_layer_inputs(per_layer_ids, None)
            if takes_embeds
            else (self.model.get_per_layer_inputs(per_layer_ids))
        )
        return {"inputs_embeds": inputs_embeds, "per_layer_inputs": per_layer_inputs}
