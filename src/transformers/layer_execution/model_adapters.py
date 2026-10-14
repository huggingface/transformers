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

"""Adapters for native cross-layer KV dependencies and externally owned recurrent state."""

from collections import UserDict

import torch

from ..masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ..modeling_outputs import BaseModelOutputWithPast
from .adapters import DecoderLayerExecutionAdapter


class Gemma3nExecutionAdapter(DecoderLayerExecutionAdapter):
    """Preserve all AltUp streams, per-source-layer inputs, and explicit native KV producer bindings."""

    def kv_dependencies(self, decoder):
        first_shared = decoder.config.num_hidden_layers - decoder.config.num_kv_shared_layers
        if not decoder.config.num_kv_shared_layers:
            return {}
        producers = {kind: index for index, kind in enumerate(decoder.config.layer_types[:first_shared])}
        return {
            index: producers[kind] for index, kind in enumerate(decoder.config.layer_types) if index >= first_shared
        }

    def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
        per_layer_inputs = kwargs.get("per_layer_inputs")
        input_ids = kwargs.get("input_ids_for_adapter")
        if input_ids is not None:
            per_layer_inputs = decoder.get_per_layer_inputs(input_ids)
        per_layer_inputs = decoder.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        if position_ids is None:
            offset = cache.get_seq_length() if cache is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + offset).unsqueeze(0)
        if not isinstance(attention_mask, dict):
            masks = {
                "config": decoder.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": cache,
                "position_ids": position_ids,
            }
            attention_mask = {
                "full_attention": create_causal_mask(**masks),
                "sliding_attention": create_sliding_window_causal_mask(**masks),
            }
        magnitude = inputs_embeds.square().mean(dim=-1, keepdim=True).sqrt()
        streams = [inputs_embeds]
        for projection in decoder.altup_projections:
            value = projection(inputs_embeds).to(inputs_embeds)
            streams.append(value * magnitude / value.square().mean(dim=-1, keepdim=True).clamp_min(1e-5).sqrt())
        hidden_states = torch.stack(streams)
        positions = {
            kind: decoder.rotary_emb(hidden_states, position_ids, kind) for kind in set(decoder.config.layer_types)
        }
        return hidden_states, {
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "position_embeddings": positions,
            "per_layer_inputs": per_layer_inputs,
        }

    def step_kwargs(self, decoder, step, context, shared_states):
        kind = decoder.config.layer_types[step.source_index]
        states = UserDict()
        if step.kv_producer is not None:
            producer = decoder._layer_execution_steps[step.kv_producer].source_index
            states[producer] = shared_states[step.kv_producer][producer]
        return {
            "position_ids": context["position_ids"],
            "attention_mask": context["attention_mask"][kind],
            "position_embeddings": context["position_embeddings"][kind],
            "per_layer_input": context["per_layer_inputs"][:, :, step.source_index, :],
            "shared_kv_states": states,
        }

    def finish_step(self, step, layer_output, layer_kwargs, shared_states):
        if step.kv_producer is None and layer_kwargs["shared_kv_states"]:
            shared_states[step.execution_index] = dict(layer_kwargs["shared_kv_states"])

    def finalize(self, decoder, hidden_states, cache, context):
        magnitude = hidden_states[0].square().mean(dim=-1, keepdim=True).sqrt()
        streams = [hidden_states[0]]
        for index, projection in enumerate(decoder.altup_unembed_projections, 1):
            value = projection(hidden_states[index]).to(hidden_states)
            streams.append(value * magnitude / value.square().mean(dim=-1, keepdim=True).clamp_min(1e-5).sqrt())
        return BaseModelOutputWithPast(
            last_hidden_state=decoder.norm(torch.stack(streams).mean(dim=0)),
            past_key_values=cache,
        )


class RecurrentGemmaExecutionAdapter(DecoderLayerExecutionAdapter):
    """Pass recurrent state explicitly; parameter modules retain no request-specific mutable state."""

    def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
        if position_ids is None:
            offset = cache.get_seq_length() if cache is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + offset).unsqueeze(0)
        return inputs_embeds * decoder.normalizer.to(inputs_embeds.dtype), {
            "position_ids": position_ids,
            "attention_mask": create_sliding_window_causal_mask(
                decoder.config,
                inputs_embeds,
                attention_mask,
                cache,
                position_ids=position_ids,
            ),
        }

    def execute_layer(self, layer, hidden_states, layer_kwargs, cache, use_cache):
        # A fresh empty state also makes checkpoint recomputation pure when training without a persistent cache.
        return layer(
            hidden_states,
            **layer_kwargs,
            past_key_values=cache,
            use_cache=use_cache,
            execution_state=cache.state if cache is not None else {},
        )

    def finalize(self, decoder, hidden_states, cache, context):
        return BaseModelOutputWithPast(last_hidden_state=decoder.final_norm(hidden_states), past_key_values=cache)
