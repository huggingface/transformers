# coding=utf-8
# Copyright 2025 The OpenAI Authors and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Dia model."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import RMSNorm

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationConfig, GenerationMixin
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, logging
from .configuration_dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DiaConfig"
_CHECKPOINT_FOR_DOC = "nari-labs/Dia-1.6B"


def apply_rotary_pos_emb(
    tensor: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], unsqueeze_dim: int = 1
) -> torch.Tensor:
    cos = position_embeddings[0]
    sin = position_embeddings[1]
    first_half, second_half = torch.chunk(tensor.to(torch.float32), 2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return torch.cat((first_part.to(tensor.dtype), second_part.to(tensor.dtype)), dim=-1)


class DiaRotaryEmbedding(nn.Module):
    def __init__(self, config: DiaEncoderConfig | DiaDecoderConfig):
        super().__init__()
        self.embedding_dims = config.head_dim
        self.min_timescale = config.rope_min_timescale
        self.max_timescale = config.rope_max_timescale

        half_embedding_dim = self.embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / self.embedding_dims
        freqs = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        position_ids_expanded = position_ids[:, :, None, None].float().repeat(x.shape[0], 1, 1, 1)
        half_embedding_dim = self.embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / self.embedding_dims
        freqs = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)

        full_freqs = position_ids_expanded.float() / freqs.to(position_ids_expanded.device)
        cos, sin = full_freqs.cos(), full_freqs.sin()
        return cos, sin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, query.shape[1] // key.shape[1])
    value_states = repeat_kv(value, query.shape[1] // key.shape[1])

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights.masked_fill(causal_mask.bool(), torch.finfo(attn_weights.dtype).min)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DiaSelfAttention(nn.Module):  # Modular : LlamaAttentions
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DiaEncoderConfig | DiaDecoderConfig, layer_idx: int, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads or self.num_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.layer_idx = layer_idx
        self.scaling = 1
        self.is_causal = is_causal
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = apply_rotary_pos_emb(query_states, position_embeddings, -2).transpose(1, 2)
        key_states = apply_rotary_pos_emb(key_states, position_embeddings, -2).transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_positions": cache_position}
            )

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class DiaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: DiaDecoderConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.num_heads = self.config.cross_num_attention_heads
        self.num_key_value_heads = self.config.cross_num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.cross_head_dim
        self.layer_idx = layer_idx
        self.scaling = 1
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.num_key_value_heads * self.head_dim, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.num_key_value_heads * self.head_dim, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.num_key_value_heads = 16

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)

        if cross_attention_states is not None:
            cross_shape = (*cross_attention_states.shape[:-1], -1, self.head_dim)
            key_states = self.k_proj(cross_attention_states).view(cross_shape)
            value_states = self.v_proj(cross_attention_states).view(cross_shape).transpose(1, 2)
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_positions": cache_position}
                )
        elif past_key_values is not None:  # not prefill, make it compile compatible
            key_states = past_key_values.key_cache[self.layer_idx]  # ty: ignore[unresolved-attribute]
            value_states = past_key_values.value_cache[self.layer_idx]  # ty: ignore[unresolved-attribute]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape((*input_shape, -1)).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class DiaEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaEncoderConfig, layer_idx: int):
        super().__init__()
        self.pre_sa_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attention = DiaSelfAttention(config, layer_idx)
        self.post_sa_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attention(
            hidden_states=normed_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        normed_states = self.post_sa_norm(hidden_states)
        mlp_out = self.mlp(normed_states)
        hidden_states = residual + mlp_out
        return hidden_states, self_attn_weights


class DiaPreTrainedModel(PreTrainedModel):
    config_class = DiaConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DiaEncoderLayer", "DiaDecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = getattr(self.config, "init_std", 0.2)
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()


class DiaEncoder(nn.Module):
    def __init__(self, config: DiaEncoderConfig, **kwargs):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DiaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            dtype=torch.float32,
        )
        self.rotary_embeddings = DiaRotaryEmbedding(config)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # TODO can the auudio codes be "padded"?
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutput | Tuple[torch.Tensor, ...]:
        hidden_states = self.embedding(input_features)
        position_embeddings = self.rotary_embeddings(hidden_states, cache_position)  # type: ignore
        _attention_mask = torch.ones(
            (input_features.shape[0], 1, hidden_states.shape[1], hidden_states.shape[1]), device=hidden_states.device
        )
        _attention_mask = (_attention_mask.long() & attention_mask[:, None, None, :]) & attention_mask[  # type: ignore
            :, None, :, None
        ]

        all_hidden_states = () if output_hidden_states else None
        all_self_attn_weights = () if output_attentions else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            hidden_states, self_attn_weights = layer(
                hidden_states=hidden_states,
                attention_mask=_attention_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )

            if all_self_attn_weights is not None:
                all_self_attn_weights += (self_attn_weights,)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        hidden_states = self.norm(hidden_states)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attn_weights] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attn_weights,
        )


class DiaMLP(nn.Module):  # Modular GlmMLP
    def __init__(self, config):
        super().__init__()

        self.config = config
        # TODO gate_up_proj and down_proj name
        self.wi_fused = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.wi_fused(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.wo(up_states)


class DiaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaDecoderConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.dropout = config.dropout
        self.self_attention = DiaSelfAttention(config, layer_idx)
        self.cross_attention = DiaCrossAttention(config, layer_idx)
        self.pre_sa_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.pre_ca_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.pre_mlp_norm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        cache_position: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[EncoderDecoderCache] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)
        self_mask = torch.tril(torch.ones_like(normed_states))
        self_mask[..., : cache_position + 1] = 0
        hidden_states, self_attn_weights = self.self_attention(
            hidden_states=normed_states,
            attention_mask=self_mask[:, None, ...],
            position_embeddings=position_embeddings,
            past_key_values=past_key_values.self_attention_cache if past_key_values is not None else None,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_ca_norm(hidden_states)
        cross_states, cross_attn_weights = self.cross_attention(
            hidden_states=hidden_states,
            cross_attention_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values.cross_attention_cache if past_key_values is not None else None,
            output_attentions=output_attentions,
        )
        cross_states = nn.functional.dropout(cross_states, p=self.dropout, training=self.training)
        hidden_states = residual + cross_states

        residual = hidden_states
        x_norm = self.pre_mlp_norm(hidden_states)
        mlp_out = self.mlp(x_norm)
        hidden_states = residual + mlp_out
        return hidden_states, cross_attn_weights, self_attn_weights


class DiaMultiChannelEmbed(nn.Module):
    """In order to efficiently compute the audio embedding from the 9 different channels
    we vectorize the embedding process by using a single embedding layer, and an offset.
    Example:
    - num_embeds = 3
    - vocab_size = 8
    - num_chanels = 4
    We would have offsets = [0, 256, 512]
    If audio_codes = [0, 1, 2, 3], [1, 3, 4, 7], [5, 6, 7, 8]
    then tokens = audio_codes + offsets
                = [0, 1, 2, 3, 257, 259, 260, 263, 517, 518, 519, 520]
    This allows us to use a single embedding layer for all channels.
    """

    def __init__(self, config: DiaDecoderConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size * config.num_channels, config.hidden_size)
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        offsets = torch.arange(config.num_channels, dtype=torch.long) * config.vocab_size  # (C,)
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        tokens = (audio_codes + self.offsets.to(audio_codes.device)).squeeze(1)
        embeds = self.embed(tokens).view(tokens.shape[0], audio_codes.shape[1], -1, self.hidden_size)
        return embeds.sum(dim=2)


class DiaDecoder(nn.Module):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaDecoderConfig, **kwargs):
        super().__init__()
        self.num_channels = config.num_channels
        self.vocab_size = config.vocab_size
        self.embeddings = DiaMultiChannelEmbed(config)
        self.rotary_embeddings = DiaRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [DiaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.logits_dense = nn.Linear(config.hidden_size, (self.num_channels * self.vocab_size), bias=False)

    def forward(
        self,
        audio_codes: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPastAndCrossAttentions | Tuple[torch.Tensor, ...]:
        if past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        hidden_states = self.embeddings(audio_codes)

        if cache_position is None:
            past_length = 0
            if past_key_values is not None:
                past_length = past_key_values.get_seq_length()

            cache_position = torch.arange(  # ty: ignore[invalid-assignment]
                past_length, past_length + hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
            )[None, :]

        position_embeddings = self.rotary_embeddings(hidden_states, cache_position)

        all_hidden_states = () if output_hidden_states else None
        all_self_attn_weights = () if output_attentions else None
        all_cross_attn_weights = () if output_attentions else None

        for i, layer in enumerate(self.layers):
            hidden_states, cross_attn_weights, self_attn_weights = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)

            if all_self_attn_weights is not None:
                all_self_attn_weights += (self_attn_weights,)

            if all_cross_attn_weights is not None:
                all_cross_attn_weights += (cross_attn_weights,)

        hidden_states = self.norm(hidden_states)
        last_hidden_states = self.logits_dense(hidden_states).view(-1, self.num_channels, self.vocab_size)

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_states,
                    past_key_values.self_attention_cache,
                    all_hidden_states,
                    all_self_attn_weights,
                    all_cross_attn_weights,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states,
            past_key_values=past_key_values,  # ty: ignore[invalid-argument-type]
            hidden_states=all_hidden_states,
            attentions=all_self_attn_weights,
            cross_attentions=all_cross_attn_weights,
        )


class DiaModel(DiaPreTrainedModel):
    def __init__(self, config: DiaConfig, **kwargs):
        super().__init__(config)
        self.encoder = DiaEncoder(config.encoder_config)
        self.decoder = DiaDecoder(config.decoder_config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def forward(
        self,
        encoder_input_ids: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqModelOutput:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, DiaModel
         >>> from datasets import load_dataset

         >>> model = DiaModel.from_pretrained("nari-labs/Dia-1.6B")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("nari-labs/Dia-1.6B")
         >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
         >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
         >>> input_features = inputs.input_features
         >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
         >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
         >>> list(last_hidden_state.shape)
         [1, 2, 512]
         ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if encoder_outputs is None:
            if encoder_input_ids is None:
                raise ValueError("Either `encoder_input_ids` or `encoder_outputs` must be provided.")
            encoder_input_features_prepared = encoder_input_ids

            encoder_cache_position: torch.LongTensor
            if cache_position is None:
                encoder_cache_position = torch.arange(encoder_input_ids.shape[-1], device=encoder_input_ids.device)[
                    None, :
                ]
            else:
                encoder_cache_position = cache_position

            encoder_outputs = self.encoder(
                input_features=encoder_input_features_prepared,
                attention_mask=encoder_attention_mask,
                cache_position=encoder_cache_position,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        encoder_hidden_states_tensor: torch.Tensor
        if isinstance(encoder_outputs, tuple):
            encoder_hidden_states_tensor = encoder_outputs[0]
        elif hasattr(encoder_outputs, "last_hidden_state") and encoder_outputs.last_hidden_state is not None:
            encoder_hidden_states_tensor = encoder_outputs.last_hidden_state
        else:
            raise TypeError(
                f"Unexpected type or missing last_hidden_state for encoder_outputs: {type(encoder_outputs)}"
            )

        processed_decoder_audio_codes: torch.LongTensor
        if decoder_input_ids is None:
            batch_size = encoder_input_features_prepared.shape[0]
            processed_decoder_audio_codes = torch.full((batch_size, 1, 9), 1026, device=self.device, dtype=torch.long)
        else:
            if decoder_input_ids.dtype != torch.long:
                temp_audio_codes = decoder_input_ids.to(torch.long)
            else:
                temp_audio_codes = decoder_input_ids

            if temp_audio_codes.shape[-1] != 9:
                num_dims_to_pad_from_end = temp_audio_codes.ndim * 2
                padding_config = [0] * num_dims_to_pad_from_end
                padding_config[1] = 9 - temp_audio_codes.shape[-1]

                padded_audio_codes = torch.nn.functional.pad(temp_audio_codes, tuple(padding_config), value=1026)

                if padded_audio_codes.ndim == 2:
                    processed_decoder_audio_codes = padded_audio_codes[:, None, :]
                elif padded_audio_codes.ndim == 3:
                    processed_decoder_audio_codes = padded_audio_codes
                else:
                    raise ValueError(f"Padded audio_codes has unexpected ndim: {padded_audio_codes.ndim}")
            else:
                processed_decoder_audio_codes = temp_audio_codes

        if past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        decoder_outputs = self.decoder(
            audio_codes=processed_decoder_audio_codes,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states_tensor,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class DiaForConditionalGeneration(GenerationMixin, DiaPreTrainedModel):
    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.config = config
        self.model = DiaModel(config)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def _prepare_generation_config(
        self, generation_config: Optional[GenerationConfig], use_model_defaults: Optional[bool] = None, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )

        # encoder part
        _encoder_length = self.config.encoder_config.max_length
        _encoder_pad = 0
        encoder_input_ids: Optional[torch.Tensor] = model_kwargs.get("encoder_input_ids", None)
        if encoder_input_ids is None:
            raise ValueError("encoder_input_ids must be provided")
        if encoder_input_ids.ndim == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        if encoder_input_ids.ndim != 2:
            raise ValueError("encoder_input_ids must be a 1D or 2D tensor")
        if encoder_input_ids.shape[-1] == _encoder_length:
            raise ValueError(f"encoder_input_ids length must be {_encoder_length} for generation")

        batch_size = encoder_input_ids.shape[0]

        encoder_padding_mask = (encoder_input_ids != _encoder_pad).to(self.device).repeat_interleave(2, dim=0)
        model_kwargs["encoder_attention_mask"] = (
            encoder_padding_mask.unsqueeze(2) & encoder_padding_mask.unsqueeze(1)
        ).unsqueeze(1)

        encoder_uncond_input_ids = torch.zeros_like(encoder_input_ids, dtype=torch.long, device=self.device)
        encoder_input_ids = torch.stack([encoder_uncond_input_ids, encoder_input_ids], dim=1).view(2 * batch_size, -1)

        model_kwargs["encoder_input_ids"] = encoder_input_ids

        # decoder part
        decoder_padding_mask = torch.ones((2 * batch_size, 1), dtype=torch.bool, device=self.device)
        model_kwargs["decoder_attention_mask"] = (
            decoder_padding_mask.unsqueeze(2) & encoder_padding_mask.unsqueeze(1)
        ).unsqueeze(1)

        # decoder eos stopping criteria
        _eos = self.config.eos_token_id
        _pad = self.config.pad_token_id
        _channel = self.config.decoder_config.num_channels
        generation_config._eos_token_tensor = torch.tensor([_pad for _ in range(_channel - 1)] + [_eos])

        # cfg scale
        if model_kwargs.get("cfg_scale", None) is None:
            model_kwargs["cfg_scale"] = 3.0

        # cfg filter top k
        if model_kwargs.get("cfg_filter_top_k", None) is None:
            model_kwargs["cfg_filter_top_k"] = 50

        # audio eos value
        model_kwargs["audio_eos_value"] = _eos
        model_kwargs["audio_pad_value"] = _pad

        return generation_config, model_kwargs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values, attention_mask, inputs_embeds, cache_position, **model_kwargs
        )

        batch_size = input_ids.shape[0]
        eos_detected_Bx = model_kwargs["eos_detected_Bx"]
        eos_countdown_Bx = model_kwargs["eos_countdown_Bx"]
        finished_step_Bx = model_kwargs["finished_step_Bx"]

        if eos_detected_Bx is None:
            eos_detected_Bx = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
        if eos_countdown_Bx is None:
            eos_countdown_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        if finished_step_Bx is None:
            finished_step_Bx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

        model_kwargs["eos_detected_Bx"] = eos_detected_Bx
        model_kwargs["eos_countdown_Bx"] = eos_countdown_Bx
        model_kwargs["finished_step_Bx"] = finished_step_Bx

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )
        model_kwargs["encoder_outputs"] = outputs.encoder_last_hidden_state
        return model_kwargs

    def forward(
        self,
        encoder_input_ids: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        eos_detected_Bx: Optional[torch.Tensor] = None,
        eos_countdown_Bx: Optional[torch.Tensor] = None,
        finished_step_Bx: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        cfg_scale: float = 3.0,
        cfg_filter_top_k: int = 50,
        audio_eos_value: int = 1024,
        audio_pad_value: int = 1025,
    ) -> Seq2SeqLMOutput:
        """
        Forward method for DiaForConditionalGeneration, following WhisperForConditionalGeneration style.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        lm_logits: torch.Tensor = outputs.last_hidden_state

        loss = None
        # TODO: add loss
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     labels = labels.to(lm_logits.device)  # ty: ignore[invalid-assignment]
        #     loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        # cfg
        logits_last = lm_logits[:, -1].view(lm_logits.shape[0] // 2, 2, *lm_logits.shape[1:])
        uncond_logits = logits_last[:, 0, :]
        cond_logits = logits_last[:, 1, :]
        logits = cond_logits + cfg_scale * (cond_logits - uncond_logits)

        # cfg filter top k
        _, top_k_indices = torch.topk(logits, k=cfg_filter_top_k, dim=-1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask = mask.scatter(dim=-1, index=top_k_indices, value=False)
        logits = logits.masked_fill(mask, -torch.inf)

        # eos filter
        logits[:, :, audio_eos_value + 1 :] = torch.full_like(
            logits[:, :, audio_eos_value + 1 :],
            fill_value=-torch.inf,
        )
        logits[:, 1:, audio_eos_value:] = torch.full_like(
            logits[:, 1:, audio_eos_value:],
            fill_value=-torch.inf,
        )
        logits[:, 0, audio_eos_value] *= torch.tensor(0.8, device=self.device)
        logits_flat = logits.view(-1, logits.shape[-1])

        top_logit_indices = torch.argmax(logits_flat, dim=-1)
        eos_not_highest_mask = top_logit_indices != audio_eos_value
        mask_eos_unless_highest = torch.zeros_like(logits_flat, dtype=torch.bool)
        mask_eos_unless_highest[eos_not_highest_mask, audio_eos_value] = True
        logits_flat = logits_flat.masked_fill(mask_eos_unless_highest, -torch.inf)
        eos_highest_mask = top_logit_indices == audio_eos_value
        mask_eos_highest = torch.zeros_like(logits_flat, dtype=torch.bool)
        mask_eos_highest[eos_highest_mask, :audio_eos_value] = True
        logits_flat = logits_flat.masked_fill(mask_eos_highest, -torch.inf)

        logits = logits_flat.view(logits.shape)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


__all__ = [
    "DiaModel",
    "DiaPreTrainedModel",
    "DiaForConditionalGeneration",
]
