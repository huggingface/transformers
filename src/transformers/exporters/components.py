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

A decomposition produces components, an exporter turns each into a graph, and a runtime drives them. The
vocabulary lives here so all three agree by construction rather than by recognising each other's strings:
the `Component` / `ExportedComponent` pair and the `ComponentRole` they carry, then the modules a
decomposition wraps a model's own methods in so each can be exported on its own.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..utils.import_utils import is_torch_available
from .metadata import ExportMetadata
from .utils import get_leaf_tensors


if is_torch_available():
    import torch

    from ..modeling_utils import PreTrainedModel


class ComponentRole(str, Enum):
    """What a component *is* to a runtime, as opposed to what it is called.

    A decomposition names its components for the reader (`"image_encoder"`, `"text_decoder"`); the role is
    what the runtime and the loader dispatch on, so neither has to recognise names. `DECODE` decides the
    shape of the whole export: one that has a decode graph is driven through `generate`.
    """

    MODEL = "model"
    # The text stack, driven once per step. `prefill` and `decode` name the *call* a graph was traced at,
    # not the module it wraps — for a decoder-only model both wrap the whole model, and only the inputs
    # differ (a prompt on an empty cache, a continuation on a full one). A decode graph whose query axis
    # stayed symbolic serves the prompt as well, and then it is the only text graph an export ships.
    DECODE = "decode"
    # The prompt's own graph, shipped only where the decode graph cannot stand in for it.
    PREFILL = "prefill"
    # An encoder-decoder's encoder, which may also compute the decoder's cross-attention cache.
    ENCODER = "encoder"
    # `input_ids -> inputs_embeds`, so the runtime can scatter modality features into the embeddings.
    EMBED_TOKENS = "embed_tokens"
    # One modality's `get_<modality>_features` — its tower and projector (`image_encoder`, `audio_encoder`).
    MODALITY_ENCODER = "modality_encoder"
    # A modality embedded once ahead of the loop, each step reading its own window (voxtral_realtime).
    STREAMING_EMBEDDER = "streaming_embedder"


@dataclass
class Component:
    """One piece of a decomposed model, ready to export: what to trace, and what to trace it with."""

    name: str
    module: Any
    inputs: dict[str, Any]
    role: ComponentRole


@dataclass
class ExportedComponent:
    """One exported graph, with what the trace recorded about it and what it is for.

    The three travel together because they are one thing: a graph whose metadata went missing is a graph a
    runtime has to guess about (precision, cache kwarg, mask layout — each guess has been wrong at least
    once), and a graph whose role went missing is one the loader has to recognise by name.
    """

    name: str
    artifact: Any
    metadata: ExportMetadata
    role: ComponentRole


class _ModelComponent(torch.nn.Module):
    """Base for the standalone export/runtime components a multi-modal model decomposes into. Wraps a
    model (the full VLM, its base, or the text decoder) so a single method can be exported on its own;
    missing attributes fall through to it, so the export precompute introspects the component (`config`,
    submodules, `get_rope_index`, device) exactly as it would the real model."""

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        # nn.Module owns params/buffers/submodules (incl. `model`); anything else delegates to the
        # wrapped model. `super().__getattr__("model")` (not `self.model`) avoids re-entering this hook.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("model"), name)


class ModalityEncoder(_ModelComponent):
    """Wraps one modality's `get_<modality>_features` method.

    `forward` runs `model.<getter>(**kwargs)` and normalises the result to a single
    `[num_tokens, hidden]` tensor — concatenating per-item `pooler_output` lists, else the bare
    `pooler_output` / `last_hidden_state` / tensor — remapping the precompute marker `grid_thw` back to
    the getter's native grid kwarg.
    """

    def __init__(self, model: PreTrainedModel, getter: str, grid_kwarg: str | None = None):
        super().__init__(model)
        self._getter = getter
        self._grid_kwarg = grid_kwarg

    def forward(self, **kwargs):
        if self._grid_kwarg is not None and "grid_thw" in kwargs:
            kwargs[self._grid_kwarg] = kwargs.pop("grid_thw")
        # `precompute_export_inputs` derives its tensors from the config alone, so it offers whatever
        # the config implies — a windowed vision config yields `window_index` even for a getter that
        # never takes one (minicpmv4_6). Keep only what this getter actually declares.
        getter = getattr(self.model, self._getter)
        parameters = inspect.signature(getter).parameters
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            kwargs = {name: value for name, value in kwargs.items() if name in parameters}
        outputs = getter(**kwargs)
        # Most getters put the features in `pooler_output` or `last_hidden_state`. Some declare both
        # and fill neither (granite4_vision returns its features as `hidden_states` +
        # `deepstack_features`), so fall through to the whole output rather than the `None` those
        # fields hold — a default on `getattr` only covers a *missing* attribute, not a null one.
        features = getattr(outputs, "pooler_output", None)
        if features is None:
            features = getattr(outputs, "last_hidden_state", None)
        if features is None:
            features = outputs
        return torch.cat(features) if isinstance(features, (tuple, list)) else features


class PatchVisionEncoder(_ModelComponent):
    """An anyres vision tower + projector, cut *before* `pack_image_features`.

    The packing decides how many tokens each image contributes from that image's own size, so tracing it
    bakes one `(image count, sizes)` pair into the graph. Everything up to the projector is plain batched
    compute over a flat `(total_patches, channels, height, width)` tensor, so the component stops there
    and the runtime packs the result — the same split optimum-intel's `OVModelForVisualCausalLM` uses.
    `image_newline` rides along as a second output: the packing needs that weight and the runtime holds
    no module to read it off.
    """

    def projector_specs(self) -> list[tuple[int, Any, Any]] | None:
        """`(llm_layer, vision_layer, projector)` per projector this tower feeds, or `None` for the
        single-projector case. A deepstack tower (granite4_vision) runs one projector per
        `deepstack_layer_map` entry and one per `spatial_target_layers` group, each injected into the
        decoder at its own layer — both loop counts come from the config, so they unroll legitimately."""
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
            # Keyed by the decoder layer each one is injected at, so the runtime rebuilds the
            # `deepstack_features` map without needing the config's ordering again.
            features = {f"image_features.{llm}": project(layer, proj) for llm, layer, proj in specs}
        features["image_newline"] = self.model.image_newline
        return features


class CrossAttentionEncoder(_ModelComponent):
    """The encoder, plus the cross-attention keys and values its output determines.

    An encoder-decoder's decoder fills its cross cache on the *first* step and reads it on every later
    one, so a decode graph traced after that step holds the read and not the projections that filled it.
    That used to cost a second graph over the whole decoder — every decoder parameter shipped twice — to
    have something that writes them. They are a function of the encoder's output, so they belong to the
    graph that produces it: `forward` returns `last_hidden_state` alongside `cross_keys_<layer>` /
    `cross_values_<layer>`, and the runtime seeds the cross cache with those before the first step.

    The writers are whichever modules were seen filling that cache (`capture_cross_writers`), replayed
    here with the arguments they were called with, so nothing names a projection or an attention class.
    The replayed query is one zero token of the width that call used: the attention it computes is
    discarded, only the keys and values it caches are wanted, and one token is the cheapest way to ask a
    module for them through its own code path.
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
                # Live batch, captured width: the keys and values follow the encoder's batch, and a query
                # left at the captured one would not broadcast against them.
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
        # Everything the encoder itself returned, not just the hidden states: parakeet attaches the
        # frame mask its decoder reads, and dropping it would change the output the decode graph takes.
        outputs = dict(get_leaf_tensors(encoded))
        for index, layer in enumerate(cache.cross_attention_cache.layers):
            outputs[f"cross_keys_{index}"] = layer.keys
            outputs[f"cross_values_{index}"] = layer.values
        return outputs


class TokenEmbedder(_ModelComponent):
    """`input_ids -> inputs_embeds`, zeroing the placeholder ids (out of the text vocab) first, the way
    a VLM `forward` does before scattering in encoder features. Wraps the text decoder (never the outer
    VLM), so the export precompute's `get_rope_index` branch stays off on the `input_ids` it carries.

    A decoder with per-layer embeddings (gemma3n, gemma4) reads a *second* per-token embedding straight
    from `input_ids`, and recovers them by an exact reverse lookup when handed `inputs_embeds` alone —
    data-dependent, and it fails outright once features are scattered in. So this returns that tensor
    too, under the `per_layer_inputs` kwarg the decoder's `forward` already takes to skip the lookup.
    Its placeholder rows survive into the decoder untouched (nothing scatters over them), so they use
    the pad id the eager forward substitutes rather than the zero standing in for the text embedding.
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
        # The signature differs by model: gemma4 takes `(input_ids, inputs_embeds)` with no defaults,
        # gemma3n only `(input_ids)`. Pass the ids, and the embeds slot only if there is one.
        takes_embeds = len(inspect.signature(self.model.get_per_layer_inputs).parameters) > 1
        per_layer_inputs = (
            self.model.get_per_layer_inputs(per_layer_ids, None)
            if takes_embeds
            else (self.model.get_per_layer_inputs(per_layer_ids))
        )
        return {"inputs_embeds": inputs_embeds, "per_layer_inputs": per_layer_inputs}
