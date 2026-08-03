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


import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from ...modeling_outputs import BaseModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..exaone4.modeling_exaone4 import Exaone4DecoderLayer, Exaone4Model, Exaone4RMSNorm
from .configuration_onyx_assistant import OnyxAssistantConfig


class OnyxAssistantRMSNorm(Exaone4RMSNorm):
    pass


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    r"""
    Applies interleaved Rotary Position Embedding to the query and key tensors.

    DeepSeek lays the rotary dimensions out in interleaved pairs `(x0, x1), (x2, x3), ...`, each rotated by a
    single frequency. We compute that rotation directly on the even/odd slices instead of de-interleaving with a
    `view`/`transpose`/`reshape`; the output is bit-identical to the de-interleaved `rotate_half` formulation while
    avoiding the extra contiguous copy.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # `cos`/`sin` are `cat(freqs, freqs)`; the first half holds the per-pair angle.
    cos = cos[..., : cos.shape[-1] // 2].unsqueeze(unsqueeze_dim)
    sin = sin[..., : sin.shape[-1] // 2].unsqueeze(unsqueeze_dim)

    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]

    q_embed = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_embed = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_embed, k_embed

class OnyxAssistantDecoderLayer(Exaone4DecoderLayer):
    def __init__(self, config: OnyxAssistantConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.attention_layernorm = OnyxAssistantRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layernorm = OnyxAssistantRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        del self.post_attention_layernorm
        del self.post_feedforward_layernorm

    # override: apply pre-LN not post-LM
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.feedforward_layernorm(hidden_states)
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

    def update_cache_with_target_states(
        self,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_values: Cache = None,
    ) -> Cache:
        target_hidden_states = self.encoder(target_hidden_states)
        position_embeddings = self.rotary_emb(target_hidden_states, position_ids)
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": target_hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        mask_mapping = {
            "full_attention": create_bidirectional_mask(**mask_kwargs),
            "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
        }
        for i, layer in enumerate(self.layers):
            # kinda wasteful, we just need the cache KV
            layer.self_attn(
                hidden_states=target_hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=mask_mapping[self.config.layer_types[i]],
                past_key_values=past_key_values,
            )

        return past_key_values

    def forward(
        self,
        inputs_embeds: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            mask_mapping = {
                "full_attention": create_bidirectional_mask(**mask_kwargs),
                "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            layer_type = self.config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
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
            past_key_values=past_key_values if use_cache else None,
        )


__all__ = ["OnyxAssistantModel"]
