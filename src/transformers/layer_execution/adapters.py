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

"""Model-specific preparation and output handling for constant-width decoders."""

from collections.abc import Sequence
from typing import Any

import torch

from ..masking_utils import create_causal_mask, create_recurrent_attention_mask
from ..modeling_outputs import BaseModelOutputWithPast
from ..utils import ModelOutput


class DecoderLayerExecutionAdapter:
    """Adapter protocol for causal decoder-only stacks whose layers preserve the hidden-state shape.

    Subclasses prepare model-specific layer kwargs and may override output construction. Parameters remain registered
    on the original model. Register an adapter with `register_layer_execution_adapter` before enabling a new model type.
    """

    layer_container_name = "layers"
    cache_name = "past_key_values"
    supports_compile = False

    def kv_dependencies(self, decoder) -> dict[int, int]:
        """Map native consumer source layers to their producer source layers, or return an empty mapping."""
        return {}

    def step_kwargs(self, decoder, step, context, shared_states):
        """Prepare immutable kwargs for this logical step. Override to bind cross-layer dependencies."""
        return self.layer_kwargs(decoder, step.source_index, context)

    def execute_step(self, layer, state, layer_kwargs, cache, use_cache):
        """Execute all state streams. Structured-state adapters may override this and `extract_state`."""
        return self.execute_layer(layer, state, layer_kwargs, cache, use_cache)

    def extract_state(self, layer_output, previous_state):
        """Keep the default single-tensor protocol compatible with existing adapters."""
        return self.extract_hidden_states(layer_output)

    def finish_step(self, step, layer_output, layer_kwargs, shared_states):
        """Publish explicit dependencies for subsequent steps, without storing state on parameter modules."""
        return None

    def slice_token_kwargs(self, kwargs, token_index, sequence_length):
        """Slice sequence-dependent auxiliary inputs during speculative state recording. Override for other layouts."""
        result = dict(kwargs)
        for name in ("per_layer_inputs", "token_type_ids", "input_ids_for_adapter"):
            value = result.get(name)
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == sequence_length:
                result[name] = value[:, token_index : token_index + 1]
        value = result.get("cache_position")
        if isinstance(value, torch.Tensor) and value.shape[-1] == sequence_length:
            result["cache_position"] = value[..., token_index : token_index + 1]
        return result

    def get_layers(self, decoder) -> Sequence[torch.nn.Module]:
        """Return the original registered blocks; override this for nested or unusual layer containers."""
        return getattr(decoder, self.layer_container_name)

    def validate(self, decoder) -> None:
        """Validate constant decoder width from configuration metadata, independently of embedding projections."""
        layer_configs = decoder.config.per_layer_config
        if any(config.hidden_size != layer_configs[0].hidden_size for config in layer_configs):
            raise ValueError("Layer execution requires the same hidden size in every decoder layer.")

    def prepare(
        self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the initial hidden states and shared layer kwargs, preparing the stack input only once."""
        raise NotImplementedError

    def layer_kwargs(self, decoder, source_index, context) -> dict[str, Any]:
        """Return kwargs for one source layer using this forward's context."""
        return context

    def execute_layer(self, layer, hidden_states, layer_kwargs, cache, use_cache):
        """Call the original module with a per-execution cache. Override this for other block signatures.

        `layer_kwargs` combines caller kwargs with prepared kwargs, which take precedence. Always call `layer(...)`
        rather than `layer.forward(...)` so that checkpointing, sharding, and output hooks remain active.
        """
        return layer(hidden_states, **{**layer_kwargs, "past_key_values": cache, "use_cache": use_cache})

    def extract_hidden_states(self, layer_output) -> torch.Tensor:
        """Extract the hidden states from a block output; override this for tuple or structured outputs."""
        return layer_output

    def finalize(self, decoder, hidden_states, cache, context) -> ModelOutput:
        """Apply stack output transformations once and construct the decoder output, using per-forward context."""
        return BaseModelOutputWithPast(last_hidden_state=decoder.norm(hidden_states), past_key_values=cache)


class _LlamaAdapter(DecoderLayerExecutionAdapter):
    supports_compile = True

    def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
        if isinstance(attention_mask, dict):
            attention_mask = attention_mask["full_attention"]
        if position_ids is None:
            offset = cache.get_seq_length() if cache is not None else 0
            position_ids = (torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + offset).unsqueeze(0)
        return inputs_embeds, {
            "position_embeddings": decoder.rotary_emb(inputs_embeds, position_ids=position_ids),
            "position_ids": position_ids,
            "attention_mask": create_causal_mask(
                config=decoder.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=cache,
                position_ids=position_ids,
            ),
        }


class _Qwen3_5Adapter(DecoderLayerExecutionAdapter):
    supports_compile = True

    def prepare(self, decoder, inputs_embeds, attention_mask, position_ids, cache, **kwargs):
        if position_ids is None:
            offset = cache.get_seq_length() if cache is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + offset
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None
        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": decoder.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": cache,
                "position_ids": text_position_ids,
            }
            executed_types = {decoder.config.layer_types[index] for index in decoder.config.layer_execution_plan}
            attention_mask = {}
            if "full_attention" in executed_types:
                attention_mask["full_attention"] = create_causal_mask(**mask_kwargs)
            if "linear_attention" in executed_types:
                attention_mask["linear_attention"] = create_recurrent_attention_mask(**mask_kwargs)
        return inputs_embeds, {
            "position_embeddings": decoder.rotary_emb(inputs_embeds, position_ids),
            "position_ids": text_position_ids,
            "attention_mask": attention_mask,
        }

    def layer_kwargs(self, decoder, source_index, context):
        return {**context, "attention_mask": context["attention_mask"][decoder.config.layer_types[source_index]]}

    def finalize(self, decoder, hidden_states, cache, context):
        from ..models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

        return Qwen3_5ModelOutputWithPast(last_hidden_state=decoder.norm(hidden_states), past_key_values=cache)


_ADAPTERS = {"llama": _LlamaAdapter, "qwen3_5_text": _Qwen3_5Adapter}
_BUILTIN_MODEL_ADAPTERS = {"gemma3n_text", "recurrent_gemma"}


def _get_adapter_class(model_type):
    if model_type in _BUILTIN_MODEL_ADAPTERS and model_type not in _ADAPTERS:
        from .model_adapters import Gemma3nExecutionAdapter, RecurrentGemmaExecutionAdapter

        _ADAPTERS.update({"gemma3n_text": Gemma3nExecutionAdapter, "recurrent_gemma": RecurrentGemmaExecutionAdapter})
    return _ADAPTERS.get(model_type)


def register_layer_execution_adapter(model_type: str, adapter: type[DecoderLayerExecutionAdapter]) -> None:
    """Register a constant-width decoder-only adapter for a text configuration's `model_type`.

    Args:
        model_type (`str`): Text model type served by the adapter.
        adapter (`type[DecoderLayerExecutionAdapter]`): Adapter class implementing the decoder protocol.
    """
    if model_type in _ADAPTERS or model_type in _BUILTIN_MODEL_ADAPTERS:
        raise ValueError(f"A layer execution adapter is already registered for {model_type}.")
    if not isinstance(adapter, type) or not issubclass(adapter, DecoderLayerExecutionAdapter):
        raise TypeError("adapter must be a DecoderLayerExecutionAdapter subclass.")
    _ADAPTERS[model_type] = adapter
