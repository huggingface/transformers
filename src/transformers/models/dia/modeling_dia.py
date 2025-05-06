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

from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import RMSNorm

from ...activations import ACT2FN
from ...cache_utils import Cache, EncoderDecoderCache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    Seq2SeqModelOutput,
)
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    logging,
)
from .configuration_dia import DiaConfig, DiaDecoderConfig, DiaEncoderConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DiaConfig"
_CHECKPOINT_FOR_DOC = "nari-labs/Dia-1.6B"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(tensor, position_embeddings, unsqueeze_dim=1):
    cos = position_embeddings[0].unsqueeze(unsqueeze_dim)
    sin = position_embeddings[1].unsqueeze(unsqueeze_dim)
    embed_tensor = (tensor * cos) + (rotate_half(tensor) * sin)
    return embed_tensor


class DiaRotaryEmbedding(nn.Module):
    def __init__(self, config: DiaConfig, device=None):
        super().__init__()
        self.embedding_dims = config.hidden_size
        self.min_timescale = config.min_timescale
        self.max_timescale = config.max_timescale

        half_embedding_dim = self.embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / self.embedding_dims
        timescale = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)
        self.register_buffer("timescale", timescale, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        expanded_timescale = self.timescale[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (position_ids_expanded.float() / expanded_timescale).transpose(1, 2)
            cos, sin = freqs.cos(), freqs.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


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
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class DiaSelfAttention(nn.Module):  # Modular : LlamaAttentions
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads or self.num_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx


        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = apply_rotary_pos_emb(query_states, position_embeddings)
        key_states = apply_rotary_pos_emb(key_states, position_embeddings)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_position=cache_position
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
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiaCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Optional[DiaDecoderConfig],
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states = apply_rotary_pos_emb(query_states, position_embeddings)
        if cross_attention_states is not None:
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = apply_rotary_pos_emb(key_states, position_embeddings)
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_position=cache_position
                )
        elif cache_position[0].shape != 0:  # not prefill, make it compile compatible
            key_states = past_key_value.key_cache[self.layer_idx]
            value_states = past_key_value.value_cache[self.layer_idx]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DiaEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaConfig, layer_idx):
        super().__init__()
        self.pre_sa_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps
        )
        self.self_attention = DiaSelfAttention(config, layer_idx)
        self.post_sa_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps
        )
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attention(
            hidden_states=normed_states,
            cache_position=cache_position,
            attention_mask=attention_mask,  # I don't mind if this never changes
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        normed_states = self.post_sa_norm(hidden_states)
        mlp_out = self.mlp(normed_states)
        hidden_states = residual + mlp_out
        return hidden_states


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


class DiaEncoder(DiaPreTrainedModel):

    def __init__(self, config: DiaEncoderConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DiaEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps, dtype=torch.float32,
        )
        self.rotary_embeddings = DiaRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ) -> torch.Tensor:
        hidden_states = self.embedding(hidden_states)
        position_embeddings = self.rotary_embeddings(hidden_states, cache_position)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cache_position,position_embeddings, past_key_values)

        hidden_states = self.norm(hidden_states).to(self.compute_dtype)
        return hidden_states


class DiaMLP(nn.Module):  # Modular GlmMLP
    def __init__(self, config):
        super().__init__()

        self.config = config
        # TODO gate_up_proj and down_proj name
        self.wi_fused = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


class DiaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: DiaDecoderConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attention = DiaSelfAttention(config, layer_idx)
        self.cross_attention = DiaCrossAttention(config, layer_idx)
        self.pre_sa_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps,
        )
        self.pre_ca_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps,
        )
        self.pre_mlp_norm = RMSNorm(
            config.hidden_size, eps=config.norm_eps,
        )
        self.mlp = DiaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        cache_position: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        normed_states = self.pre_sa_norm(hidden_states).to(self.compute_dtype)

        hidden_states, self_attn_weights = self.self_attention(
            hidden_states=normed_states,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.pre_ca_norm(hidden_states).to(self.compute_dtype)
            hidden_states, cross_attn_weights = self.cross_attention(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        residual = hidden_states
        x_norm = self.pre_mlp_norm(hidden_states).to(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        hidden_states = residual + mlp_out
        return hidden_states


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
                = [0, 1, 2, 3, 256, 259, 260, 263, 517, 5128, 519, 520]
    This allows us to use a single embedding layer for all channels.
    """

    def __init__(self, config: DiaConfig):
        super().__init__()
        self.embed = nn.Embedding(config.hidden_size * config.vocab_size, config.num_channels)
        offsets = torch.arange(config.num_channels, dtype=torch.long) * config.hidden_size  # (C,)
        self.register_buffer("offsets", offsets, persistent=False)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        tokens = (audio_codes + self.offsets).squeeze(1)
        embeds = self.embed(tokens)
        return embeds.sum(dim=1, keepdim=True)


class DiaDecoder(DiaPreTrainedModel):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaDecoderConfig):
        super().__init__(config)
        self.num_channels = config.num_channels
        self.vocab_size = config.vocab_size
        self.embeddings = DiaMultiChannelEmbed(config)
        self.rotary_embeddings = DiaRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [DiaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.logits_dense = nn.Linear(config.hidden_size, (self.num_channels * self.vocab_size), bias=False)

    def forward(self, audio_codes: torch.Tensor, cache_position, attention_mask, past_key_values) -> torch.Tensor:
        hidden_states = self.embeddings(audio_codes)
        position_embeddings = self.rotary_embeddings(hidden_states, audio_codes)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                cache_position,
                attention_mask,
                past_key_values=past_key_values,
            )

        hidden_states = self.norm(hidden_states)
        last_hidden_states = self.logits_dense(hidden_states).view(-1, self.num_channels, self.vocab_size)
        return last_hidden_states.to(torch.float32)


class DiaModel(DiaPreTrainedModel):
    def __init__(self, config: DiaConfig):
        super().__init__(config)
        self.encoder = DiaEncoder(config.encoder_config)
        self.decoder = DiaDecoder(config.decoder_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        position_ids: Optional[Tuple[torch.LongTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:
         ```python
         >>> import torch
         >>> from transformers import AutoFeatureExtractor, DiaModel
         >>> from datasets import load_dataset

         >>> model = DiaModel.from_pretrained("openai/dia-base")
         >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/dia-base")
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_features is None:
            input_features = torch.zeros_like(input_ids)
        input_features = torch.cat([input_features, input_ids], dim=0)

        if cache_position is None:
            cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)[None,:]

        if cache_position.shape[1] != 1: # prefill computes encoder kv
            encoder_outputs = self.encoder(
                input_features,
                cache_position=cache_position,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            encoder_outputs = None

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

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


__all__ = [
    "DiaModel",
    "DiaPreTrainedModel",
]
