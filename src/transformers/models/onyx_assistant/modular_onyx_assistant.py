# Copyright 2026 the HuggingFace Team. All rights reserved.
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


from collections.abc import Callable

import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..exaone4.modeling_exaone4 import (
    Exaone4Attention,
    Exaone4DecoderLayer,
    Exaone4Model,
    Exaone4RMSNorm,
    eager_attention_forward,
    rotate_half,
)
from .configuration_onyx_assistant import OnyxAssistantConfig


class OnyxAssistantRMSNorm(Exaone4RMSNorm):
    pass


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Due to the added context from the main model, k/v and q do not have the same seq_len, so we have to slice here
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class OnyxAssistantAttention(Exaone4Attention):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)
        del self.sliding_window_pattern
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.FloatTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # The total k/v states in Dflash are the concatenation of the previous `context_hidden_states` (same for every layer)
        # and the actual projections on the diffusion window (the actual `hidden_states` input). Everything gets appended to the
        # `past_key_values`, and then the diffusion window will be evicted from it so that the cache is effectively only the context
        # from the main model
        if context_hidden_states is not None:
            kv_hidden_states = torch.cat([context_hidden_states, hidden_states], dim=1)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(kv_hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(kv_hidden_states).view(hidden_shape).transpose(1, 2)

        # We use QK-norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        # We use global NoPE for hybrid attention model
        if self.sliding_window is None or self.is_sliding:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window if self.is_sliding else None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OnyxAssistantDecoderLayer(Exaone4DecoderLayer):
    def __init__(self, config: OnyxAssistantConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.input_layernorm = OnyxAssistantRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        del self.post_feedforward_layernorm

    # override: apply pre-LN not post-LM, and pass target states!
    def forward(
        self,
        hidden_states: torch.Tensor,
        context_hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            context_hidden_states=context_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class OnyxTargetEncoder(nn.Module):
    def __init__(self, config: OnyxAssistantConfig):
        super().__init__()
        # fuse concatenated target hidden states -> hidden_size
        self.target_layer_ids = config.target_layer_ids
        encoder_input_size = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(encoder_input_size, config.hidden_size, bias=False)
        self.output_norm_enc = OnyxAssistantRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, target_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        target_hidden_states (`torch.Tensor` of shape `[batch, seq_len, target_hidden_size]`):
            concatenated list of tensors, one per `config.target_layer_ids` entry in that order
        """
        target_hidden_states = self.fc(target_hidden_states)
        target_hidden_states = self.output_norm_enc(target_hidden_states)
        return target_hidden_states


class OnyxAssistantModel(Exaone4Model):
    def __init__(self, config: OnyxAssistantConfig):
        super().__init__(config)
        del self.embed_tokens
        del self.padding_idx
        del self.vocab_size
        self.encoder = OnyxTargetEncoder(config)
        self.layers = nn.ModuleList(
            [OnyxAssistantDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        noise_embeds: torch.FloatTensor,
        context_hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # project targets to the same hidden dim as the model
        context_hidden_states = self.encoder(context_hidden_states)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(noise_embeds.shape[1], device=noise_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": noise_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }

        hidden_states = noise_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                context_hidden_states=context_hidden_states,
                attention_mask=mask_mapping[layer_type],
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


__all__ = ["OnyxAssistantModel"]
