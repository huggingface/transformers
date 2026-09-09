# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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
#
# Original implementation: https://github.com/pytorch/fairseq/tree/master/examples/wmt19
# Authors:
# - @alexeib Alexei Baevski
# - @edunov Sergey Edunov
# - @michaelauli Michael Auli
# - @myleott Myle Ott
# - @nng555 Nathan Ng
# - David Grangier
# - Kyra Yee
#
# Paper: Facebook FAIR's WMT19 News Translation Task Submission https://huggingface.co/papers/1907.06616
#
"""PyTorch Fairseq model, ported from https://github.com/pytorch/fairseq/tree/master/examples/wmt19"""

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, EncoderDecoderCache
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_causal_mask
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
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from .configuration_fsmt import FSMTConfig


logger = logging.get_logger(__name__)


# FSMT is modelled after BART. The architectural differences that remain are:
# - separate source and target vocabularies, and untied token embeddings
# - sinusoidal (non-learned) position embeddings, generated the fairseq way
# - a projection layer at the end of the decoder instead of an `lm_head`
# - no `final_logits_bias`


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class SinusoidalPositionalEmbedding(nn.Embedding):
    """
    This module produces sinusoidal positional embeddings of any length.

    We don't want to save the weight of this embedding since it's not trained (deterministic) and it can be huge.

    Padding symbols are ignored.

    These embeddings get automatically extended in forward if more positions is needed.
    """

    def __init__(self, num_positions, embedding_dim, padding_idx):
        super().__init__(num_positions, embedding_dim, padding_idx)

    def make_weight(self, num_positions, embedding_dim, padding_idx):
        weight = self.get_embedding(num_positions, embedding_dim, padding_idx)
        # in forward put the weights on the correct dtype and device of the param
        weight = weight.to(dtype=self.weight.dtype, device=self.weight.device)
        self.weight = nn.Parameter(weight)
        self.weight.detach_()
        self.weight.requires_grad = False

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @staticmethod
    def make_positions(tensor, padding_idx: int):
        """
        Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def forward(self, input: torch.Tensor, past_key_values_length: int = 0):
        """Input is expected to be of size [bsz x seqlen]."""
        seq_len = input.shape[1]
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weight.size(0):
            # expand embeddings if needed
            self.make_weight(max_pos, self.embedding_dim, self.padding_idx)
        positions = self.make_positions(input, self.padding_idx)
        if past_key_values_length > 0:
            # when decoding incrementally, the positions of the tokens we are given continue where the cache stopped
            positions = positions + past_key_values_length * input.ne(self.padding_idx).long()
        return super().forward(positions)


# Copied from transformers.models.bart.modeling_bart.eager_attention_forward
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    if scaling is None:
        scaling = query.size(-1) ** -0.5

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->FSMT
class FSMTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: FSMTConfig | None = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal
        self.layer_idx = layer_idx
        if layer_idx is None and self.is_decoder:
            logger.warning_once(
                f"Instantiating a decoder {self.__class__.__name__} without passing `layer_idx` is not recommended and "
                "will lead to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        # TODO: we need a refactor so that the different attention modules can get their specific kwargs
        # ATM, we have mixed things encoder, decoder, and encoder-decoder attn
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        # determine input shapes
        input_shape = hidden_states.shape[:-1]

        hidden_shape = (*input_shape, -1, self.head_dim)

        # get query proj
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        is_updated = False
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_states from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            kv_shape = (*current_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(kv_shape).transpose(1, 2)
            value_states = value_states.view(kv_shape).transpose(1, 2)

            if past_key_values is not None:
                key_states, value_states = curr_past_key_values.update(key_states, value_states, self.layer_idx)
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# Copied from transformers.models.bart.modeling_bart.BartEncoderLayer with Bart->FSMT
class FSMTEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FSMTConfig, layer_idx: int | None = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = FSMTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and not torch.isfinite(hidden_states).all():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states


# Copied from transformers.models.bart.modeling_bart.BartDecoderLayer with Bart->FSMT
class FSMTDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: FSMTConfig, layer_idx: int | None = None):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = FSMTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = FSMTAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            layer_idx=layer_idx,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, _ = self.encoder_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


@auto_docstring
class PretrainedFSMTModel(PreTrainedModel):
    config: FSMTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [r"FSMTEncoderLayer", r"FSMTDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, SinusoidalPositionalEmbedding):
            weight = module.get_embedding(*module.weight.shape, module.padding_idx)
            init.copy_(module.weight, weight)


class FSMTEncoder(PretrainedFSMTModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`FSMTEncoderLayer`].

    Args:
        config: FSMTConfig
    """

    _can_record_outputs = {
        "hidden_states": FSMTEncoderLayer,
        "attentions": FSMTAttention,
    }

    def __init__(self, config: FSMTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.src_vocab_size, config.d_model, config.pad_token_id)
        embed_dim = self.embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx
        )
        self.layers = nn.ModuleList([FSMTEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutput:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Tokens in the source language.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            embed_pos = self.embed_positions(input_ids)
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        else:
            # We assume zeros hidden states correspond to padding tokens
            # and create `position_ids` where inputs_embeds[:, :, 0] == 0
            position_ids = inputs_embeds[:, :, 0].masked_fill(
                inputs_embeds[:, :, 0].eq(0), self.embed_positions.padding_idx
            )
            embed_pos = self.embed_positions(position_ids)
            inputs_embeds = inputs_embeds * self.embed_scale

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        for encoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask,
                    **kwargs,
                )

        return BaseModelOutput(last_hidden_state=hidden_states)


class FSMTDecoder(PretrainedFSMTModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`FSMTDecoderLayer`]

    Args:
        config: FSMTConfig
    """

    _can_record_outputs = {
        "hidden_states": FSMTDecoderLayer,
        "attentions": OutputRecorder(FSMTAttention, index=1, layer_name="self_attn"),
        "cross_attentions": OutputRecorder(FSMTAttention, index=1, layer_name="encoder_attn"),
    }

    def __init__(self, config: FSMTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.tgt_vocab_size, config.d_model, self.padding_idx)
        embed_dim = self.embed_tokens.embedding_dim
        self.embed_positions = SinusoidalPositionalEmbedding(
            config.max_position_embeddings + self.padding_idx + 1, embed_dim, self.padding_idx
        )
        self.layers = nn.ModuleList([FSMTDecoderLayer(config, layer_idx=i) for i in range(config.decoder_layers)])
        self.output_projection = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    # NOTE: `last_hidden_state` is the output of `output_projection` (i.e. the logits), so it must not be
    # tied back into `hidden_states[-1]`, which stays the last decoder layer output.
    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Previous decoder outputs for teacher forcing.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, source_sequence_length, hidden_size)`, *optional*):
            Output of the encoder, used for cross-attention.
        encoder_attention_mask (`torch.Tensor` of shape `(batch_size, source_sequence_length)`, *optional*):
            Mask over the encoder outputs, to ignore padding tokens on the source side.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of decoder_input_ids or decoder_inputs_embeds")

        # initialize `past_key_values`
        if use_cache and past_key_values is None:
            past_key_values = (
                EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))
                if encoder_hidden_states is not None or self.config.is_encoder_decoder
                else DynamicCache(config=self.config)
            )

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if input_ids is not None:
            positions = self.embed_positions(input_ids, past_key_values_length)
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        else:
            # We assume zeros hidden states correspond to padding tokens
            # and create `position_ids` where inputs_embeds[:, :, 0] == 0
            position_ids = inputs_embeds[:, :, 0].masked_fill(
                inputs_embeds[:, :, 0].eq(0), self.embed_positions.padding_idx
            )
            positions = self.embed_positions(position_ids, past_key_values_length)
            inputs_embeds = inputs_embeds * self.embed_scale

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        self_attn_cache = (
            past_key_values.self_attention_cache
            if isinstance(past_key_values, EncoderDecoderCache)
            else past_key_values
        )
        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=self_attn_cache,
        )
        encoder_attention_mask = create_bidirectional_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        for decoder_layer in self.layers:
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.output_projection(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class FSMTModel(PretrainedFSMTModel):
    _tied_weights_keys = {
        "encoder.embed_tokens.weight": "decoder.embed_tokens.weight",
        "decoder.output_projection.weight": "decoder.embed_tokens.weight",
    }

    def __init__(self, config: FSMTConfig):
        super().__init__(config)
        self.encoder = FSMTEncoder(config)
        self.decoder = FSMTDecoder(config)
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: tuple[torch.FloatTensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | Seq2SeqModelOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            FSMT uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
        decoder_attention_mask (`torch.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        """
        # different to other models, FSMT automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if encoder_outputs is None:
            encoder_outputs: BaseModelOutput = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            **kwargs,
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

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.decoder.embed_tokens

    def set_output_embeddings(self, value):
        self.decoder.embed_tokens = value


@auto_docstring(
    custom_intro="""
    The FSMT Model with a language modeling head. Can be used for summarization.
    """
)
class FSMTForConditionalGeneration(PretrainedFSMTModel, GenerationMixin):
    base_model_prefix = "model"

    def __init__(self, config: FSMTConfig):
        super().__init__(config)
        self.model = FSMTModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        encoder_outputs: tuple[torch.FloatTensor] | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor] | Seq2SeqLMOutput:
        r"""
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            FSMT uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).
        decoder_attention_mask (`torch.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example Translation:

        ```python
        >>> from transformers import AutoTokenizer, FSMTForConditionalGeneration

        >>> mname = "facebook/wmt19-ru-en"
        >>> model = FSMTForConditionalGeneration.from_pretrained(mname)
        >>> tokenizer = AutoTokenizer.from_pretrained(mname)

        >>> src_text = "Машинное обучение - это здорово, не так ли?"
        >>> input_ids = tokenizer(src_text, return_tensors="pt").input_ids
        >>> outputs = model.generate(input_ids, num_beams=5, num_return_sequences=3)
        >>> tokenizer.decode(outputs[0], skip_special_tokens=True)
        "Machine learning is great, isn't it?"
        ```
        """
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs: Seq2SeqModelOutput = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        lm_logits = outputs[0]

        masked_lm_loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.tgt_vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def get_output_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_output_embeddings(self, value):
        self.model.decoder.embed_tokens = value


__all__ = ["FSMTForConditionalGeneration", "FSMTModel", "PretrainedFSMTModel"]
