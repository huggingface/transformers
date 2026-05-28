# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""GraniteSWA: Granite with Sliding Window Attention and learnable attention sinks.

The sink mechanism uses post-attention LSE scaling:
    sink_scale = sigmoid(lse - sinks)
    attn_output = attn_output * sink_scale

This is different from GPT-OSS which concatenates sinks into the softmax denominator.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..granite.modeling_granite import (
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteModel,
    GranitePreTrainedModel,
)
from ..llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb
from .configuration_granite_swa import GraniteSWAConfig


logger = logging.get_logger(__name__)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    """Eager attention that also computes LSE for sink scaling."""
    key_states = key
    value_states = value

    num_key_value_groups = module.num_key_value_groups
    if num_key_value_groups > 1:
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Compute LSE before softmax and store on the module for sink scaling
    module._lse = torch.logsumexp(attn_weights, dim=-1)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


class GraniteSWAAttention(LlamaAttention):
    """Attention with per-layer sliding window and learnable attention sinks (LSE-based)."""

    def __init__(self, config: GraniteSWAConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.scaling = config.attention_multiplier
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        # Learnable per-head attention sink parameter
        self.sinks = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Attention dispatch
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # Get LSE for sink scaling
        # For eager: stored on self._lse by eager_attention_forward
        # For FA3: passed via s_aux (handled by kernel) — but we use LSE route instead
        lse = getattr(self, "_lse", None)
        if lse is None and hasattr(self, "_fa3_lse"):
            lse = self._fa3_lse

        # Apply sink scaling: sink_scale = sigmoid(lse - sinks)
        if lse is not None:
            sink_scale = torch.sigmoid((lse - self.sinks.view(1, -1, 1)).to(torch.float32)).to(attn_output.dtype)
            # attn_output: (B, S, H*D) -> (B, S, H, D) for per-head scaling
            B = input_shape[0]
            S = input_shape[1] if len(input_shape) > 1 else attn_output.shape[0]
            attn_output = attn_output.view(B, S, self.config.num_attention_heads, self.head_dim)
            sink_scale = sink_scale.transpose(1, 2).unsqueeze(-1)  # (B, S, H, 1)
            attn_output = attn_output * sink_scale
            attn_output = attn_output.view(B, S, -1)
            # Clean up
            self._lse = None

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GraniteSWADecoderLayer(GraniteDecoderLayer):
    def __init__(self, config: GraniteSWAConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GraniteSWAAttention(config=config, layer_idx=layer_idx)


class GraniteSWAPreTrainedModel(GranitePreTrainedModel):
    _supports_sdpa = False
    _compatible_flash_implementations = ["flash_attention_3"]
    _can_record_outputs = {
        "hidden_states": GraniteSWADecoderLayer,
        "attentions": GraniteSWAAttention,
    }

    def _init_weights(self, module):
        pass


class GraniteSWAModel(GraniteModel):
    def __init__(self, config: GraniteSWAConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteSWADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Create per-layer-type causal masks
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GraniteSWAForCausalLM(GraniteForCausalLM):
    pass


__all__ = ["GraniteSWAConfig", "GraniteSWAForCausalLM", "GraniteSWAModel", "GraniteSWAPreTrainedModel"]
