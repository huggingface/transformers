# coding=utf-8
# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch HT Demucs model."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel


class HtdemucsAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class HtdemucsEncoderBlock(nn.Module):
    def __init__(self, config: HtdemucsConfig, is_cross_attn=False):
        super().__init__()
        self.embed_dim = config.d_model
        self.is_cross_attn = is_cross_attn

        self.attn = HtdemucsAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )

        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if is_cross_attn:
            self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.layer_scale_1 = (
            nn.Parameter(torch.ones(self.embed_dim) * config.layer_scale_init_value)
            if config.attn_layer_scale
            else nn.Identity()
        )  # TODO(SG): can remove attn_layer_scale from config
        self.layer_scale_2 = (
            nn.Parameter(torch.ones(self.embed_dim) * config.layer_scale_init_value)
            if config.attn_layer_scale
            else nn.Identity()
        )

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.group_norm = nn.GroupNorm(num_groups=1, num_channels=self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`):
                Inputs to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch, 1, tgt_len, src_len)`, *optional*):
                Attention mask, where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`, *optional*):
                Cross attention input to the layer. Only used for cross attention layers.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.attn_layer_norm(hidden_states)

        if encoder_hidden_states is not None:
            encoder_hidden_states = self.cross_attn_layer_norm(encoder_hidden_states)

        # Cross attention
        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = self.layer_scale_1 * hidden_states
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = self.layer_scale_2 * hidden_states
        hidden_states = residual + hidden_states

        hidden_states = self.group_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class HtdemucsEncoderLayer(nn.Module):
    def __init__(self, config: HtdemucsConfig):
        self.freq_self_attn = HtdemucsEncoderBlock(config)
        self.temp_self_attn = HtdemucsEncoderBlock(config)

        self.freq_cross_attn = HtdemucsEncoderBlock(config, is_cross_attn=True)
        self.temp_self_attn = HtdemucsEncoderBlock(config, is_cross_attn=True)

    def forward(
        self,
        freq_hidden_states: torch.FloatTensor,
        temp_hidden_states: torch.FloatTensor,
        freq_attention_mask: torch.LongTensor = None,
        temp_attention_mask: torch.LongTensor = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
    ):
        freq_layer_outputs = self.freq_self_attn(
            freq_hidden_states, attention_mask=freq_attention_mask, output_attentions=output_attentions
        )
        temp_layer_outputs = self.temp_self_attn(
            temp_hidden_states, attention_mask=temp_attention_mask, output_attentions=output_attentions
        )

        freq_residual = freq_hidden_states

        freq_hidden_states = freq_layer_outputs[0]
        temp_hidden_states = temp_hidden_states[0]

        freq_layer_outputs = self.freq_cross_attn(
            freq_hidden_states,
            attention_mask=freq_attention_mask,
            encoder_hidden_states=temp_hidden_states,
            output_attentions=output_attentions,
        )
        temp_layer_outputs = self.temp_cross_attn(
            temp_hidden_states,
            attention_mask=temp_attention_mask,
            encoder_hidden_states=freq_residual,
            output_attentions=output_attentions,
        )




class HtdemucsPreTrainedModel(PreTrainedModel):
    config_class = HtdemucsConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, HtdemucsEncoder):
            module.gradient_checkpointing = value


def create_sin_embedding(length: int, dim: int, shift: int = 0, device="cpu", max_period=10000):
    # We aim for TBC format
    assert dim % 2 == 0
    pos = shift + torch.arange(length, device=device).view(-1, 1, 1)
    half_dim = dim // 2
    adim = torch.arange(dim // 2, device=device).view(1, 1, -1)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return torch.cat(
        [
            torch.cos(phase),
            torch.sin(phase),
        ],
        dim=-1,
    )


def create_2d_sin_embedding(d_model, height, width, device="cpu", max_period=10000):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(max_period) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe[None, :].to(device)


class HtdemucsEncoder(HtdemucsPreTrainedModel):
    def __init__(self, config: HtdemucsConfig):
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model

        # spectrogram layers
        self.freq_layers = nn.ModuleList()
        # temporal layers
        self.temp_layers = nn.ModuleList()

        # set indices for self/cross attn layers
        self.classic_parity = 1 if config.cross_first else 0

        for idx in range(config.num_hidden_layers):
            if idx % 2 == self.classic_parity:
                self.freq_layers.append(HtdemucsEncoderBlock(config))
                self.temp_layers.append(HtdemucsEncoderBlock(config))
            else:
                self.freq_layers.append(HtdemucsEncoderBlock(config, is_cross_attn=True))
                self.temp_layers.append(HtdemucsEncoderBlock(config, is_cross_attn=True))

        self.layernorm_embedding = nn.LayerNorm(embed_dim)
        self.layer_norm = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: torch.FloatTensor = None,
        input_values: torch.FloatTensor = None,
        freq_attention_mask: torch.LongTensor = None,
        temp_attention_mask: torch.LongTensor = None,
        freq_embeds: Optional[torch.FloatTensor] = None,
        temp_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
                loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
                the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
                [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
                tensor of type `torch.FloatTensor`. See [`~HtdemucsProcessor.__call__`]
            input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
                into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
                soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
                conversion into a tensor of type `torch.FloatTensor`. See [`HtdemucsProcessor.__call__`] for details.
            freq_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on spatial (frequency) padding token indices. Mask values selected
                in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            temp_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on temporal padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            freq_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_features` you can choose to directly pass an embedded representation
                for the spatial (frequency) inputs. This is useful if you want more control over how to convert
                `input_features` into associated vectors than the model's internal embedding lookup matrix.
            temp_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_values` you can choose to directly pass an embedded representation
                for the temporal inputs. This is useful if you want more control over how to convert `input_values`
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
