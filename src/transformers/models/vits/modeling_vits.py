# coding=utf-8
# Copyright 2023 The VITS Authors and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch VITS model."""

import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss

from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqSpectrogramOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VITSConfig


logger = logging.get_logger(__name__)


# General docstring
_CONFIG_FOR_DOC = "VITSConfig"


VITS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # "microsoft/speecht5_asr",
    # "microsoft/speecht5_tts",
    # "microsoft/speecht5_vc",
    # See all VITS models at https://huggingface.co/models?filter=vits
]


# # Copied from transformers.models.bart.modeling_bart.shift_tokens_right
# def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
#     """
#     Shift input ids one token to the right.
#     """
#     shifted_input_ids = input_ids.new_zeros(input_ids.shape)
#     shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
#     shifted_input_ids[:, 0] = decoder_start_token_id

#     if pad_token_id is None:
#         raise ValueError("self.model.config.pad_token_id has to be defined.")
#     # replace possible -100 values in labels by `pad_token_id`
#     shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

#     return shifted_input_ids


# def shift_spectrograms_right(input_values: torch.Tensor, reduction_factor: int = 1):
#     """
#     Shift input spectrograms one timestep to the right. Also applies the reduction factor to the sequence length.
#     """
#     # thin out frames for reduction factor
#     if reduction_factor > 1:
#         input_values = input_values[:, reduction_factor - 1 :: reduction_factor]

#     shifted_input_values = input_values.new_zeros(input_values.shape)
#     shifted_input_values[:, 1:] = input_values[:, :-1].clone()

#     # replace possible -100 values in labels by zeros
#     shifted_input_values.masked_fill_(shifted_input_values == -100.0, 0.0)

#     return shifted_input_values


# # Copied from transformers.models.bart.modeling_bart._make_causal_mask
# def _make_causal_mask(
#     input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
# ):
#     """
#     Make causal mask used for bi-directional self-attention.
#     """
#     bsz, tgt_len = input_ids_shape
#     mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
#     mask_cond = torch.arange(mask.size(-1), device=device)
#     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
#     mask = mask.to(dtype)

#     if past_key_values_length > 0:
#         mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
#     return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)



# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer with Wav2Vec2->SpeechT5
# class SpeechT5NoLayerNormConvLayer(nn.Module):
#     def __init__(self, config, layer_id=0):
#         super().__init__()
#         self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
#         self.out_conv_dim = config.conv_dim[layer_id]

#         self.conv = nn.Conv1d(
#             self.in_conv_dim,
#             self.out_conv_dim,
#             kernel_size=config.conv_kernel[layer_id],
#             stride=config.conv_stride[layer_id],
#             bias=config.conv_bias,
#         )
#         self.activation = ACT2FN[config.feat_extract_activation]

#     def forward(self, hidden_states):
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         return hidden_states


# Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer with Wav2Vec2->SpeechT5
# class SpeechT5LayerNormConvLayer(nn.Module):
#     def __init__(self, config, layer_id=0):
#         super().__init__()
#         self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
#         self.out_conv_dim = config.conv_dim[layer_id]

#         self.conv = nn.Conv1d(
#             self.in_conv_dim,
#             self.out_conv_dim,
#             kernel_size=config.conv_kernel[layer_id],
#             stride=config.conv_stride[layer_id],
#             bias=config.conv_bias,
#         )
#         self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
#         self.activation = ACT2FN[config.feat_extract_activation]

#     def forward(self, hidden_states):
#         hidden_states = self.conv(hidden_states)

#         hidden_states = hidden_states.transpose(-2, -1)
#         hidden_states = self.layer_norm(hidden_states)
#         hidden_states = hidden_states.transpose(-2, -1)

#         hidden_states = self.activation(hidden_states)
#         return hidden_states


# # Copied from transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer with Wav2Vec2->SpeechT5
# class SpeechT5GroupNormConvLayer(nn.Module):
#     def __init__(self, config, layer_id=0):
#         super().__init__()
#         self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
#         self.out_conv_dim = config.conv_dim[layer_id]

#         self.conv = nn.Conv1d(
#             self.in_conv_dim,
#             self.out_conv_dim,
#             kernel_size=config.conv_kernel[layer_id],
#             stride=config.conv_stride[layer_id],
#             bias=config.conv_bias,
#         )
#         self.activation = ACT2FN[config.feat_extract_activation]

#         self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

#     def forward(self, hidden_states):
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.layer_norm(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         return hidden_states


# Copied from transformers.models.speech_to_text.modeling_speech_to_text.Speech2TextSinusoidalPositionalEmbedding with Speech2Text->SpeechT5
# class SpeechT5SinusoidalPositionalEmbedding(nn.Module):
#     """This module produces sinusoidal positional embeddings of any length."""

#     def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
#         super().__init__()
#         self.offset = 2
#         self.embedding_dim = embedding_dim
#         self.padding_idx = padding_idx
#         self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

#     def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
#         emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
#         if hasattr(self, "weights"):
#             # in forward put the weights on the correct dtype and device of the param
#             emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

#         self.weights = nn.Parameter(emb_weights)
#         self.weights.requires_grad = False
#         self.weights.detach_()

#     @staticmethod
#     def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
#         """
#         Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
#         description in Section 3.5 of "Attention Is All You Need".
#         """
#         half_dim = embedding_dim // 2
#         emb = math.log(10000) / (half_dim - 1)
#         emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
#         emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
#         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
#         if embedding_dim % 2 == 1:
#             # zero pad
#             emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
#         if padding_idx is not None:
#             emb[padding_idx, :] = 0
#         return emb.to(torch.get_default_dtype())

#     @torch.no_grad()
#     def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
#         bsz, seq_len = input_ids.size()
#         # Create the position ids from the input token ids. Any padded tokens remain padded.
#         position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
#             input_ids.device
#         )

#         # expand embeddings if needed
#         max_pos = self.padding_idx + 1 + seq_len
#         if max_pos > self.weights.size(0):
#             self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

#         return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

#     def create_position_ids_from_input_ids(
#         self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0
#     ):
#         """
#         Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
#         symbols are ignored. This is modified from fairseq's `utils.make_positions`.

#         Args:
#             x: torch.Tensor x:
#         Returns: torch.Tensor
#         """
#         # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
#         mask = input_ids.ne(padding_idx).int()
#         incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
#         return incremental_indices.long() + padding_idx



# class SpeechT5RelativePositionalEncoding(torch.nn.Module):
#     def __init__(self, dim, max_length=1000):
#         super().__init__()
#         self.dim = dim
#         self.max_length = max_length
#         self.pe_k = torch.nn.Embedding(2 * max_length, dim)

#     def forward(self, hidden_states):
#         seq_len = hidden_states.shape[1]
#         pos_seq = torch.arange(0, seq_len).long().to(hidden_states.device)
#         pos_seq = pos_seq[:, None] - pos_seq[None, :]

#         pos_seq[pos_seq < -self.max_length] = -self.max_length
#         pos_seq[pos_seq >= self.max_length] = self.max_length - 1
#         pos_seq = pos_seq + self.max_length

#         return self.pe_k(pos_seq)



# class SpeechT5BatchNormConvLayer(nn.Module):
#     def __init__(self, config, layer_id=0):
#         super().__init__()

#         if layer_id == 0:
#             in_conv_dim = config.num_mel_bins
#         else:
#             in_conv_dim = config.speech_decoder_postnet_units

#         if layer_id == config.speech_decoder_postnet_layers - 1:
#             out_conv_dim = config.num_mel_bins
#         else:
#             out_conv_dim = config.speech_decoder_postnet_units

#         self.conv = nn.Conv1d(
#             in_conv_dim,
#             out_conv_dim,
#             kernel_size=config.speech_decoder_postnet_kernel,
#             stride=1,
#             padding=(config.speech_decoder_postnet_kernel - 1) // 2,
#             bias=False,
#         )
#         self.batch_norm = nn.BatchNorm1d(out_conv_dim)

#         if layer_id < config.speech_decoder_postnet_layers - 1:
#             self.activation = nn.Tanh()
#         else:
#             self.activation = None

#         self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

#     def forward(self, hidden_states):
#         hidden_states = self.conv(hidden_states)
#         hidden_states = self.batch_norm(hidden_states)
#         if self.activation is not None:
#             hidden_states = self.activation(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         return hidden_states


# class SpeechT5TextEncoderPrenet(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
#         self.encode_positions = SpeechT5ScaledPositionalEncoding(
#             config.positional_dropout,
#             config.hidden_size,
#             config.max_text_positions,
#         )

#     def get_input_embeddings(self):
#         return self.embed_tokens

#     def set_input_embeddings(self, value):
#         self.embed_tokens = value

#     def forward(self, input_ids: torch.Tensor):
#         inputs_embeds = self.embed_tokens(input_ids)
#         inputs_embeds = self.encode_positions(inputs_embeds)
#         return inputs_embeds



#   self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
#   def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):

# TODO? # Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Wav2Vec2
class VITSAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with relative position bias (see
    https://aclanthology.org/N18-2074.pdf)
    """
# TODO?!

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
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
        self.is_decoder = is_decoder

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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        # position_bias: Optional[torch.Tensor] = None,  #MIH????
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

#TODO: can probably use a simpler one here since we never have cross attention / decoder / past
        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )


    # TODO: search for "relative attention"
    # if self.window_size is not None:
    #   assert t_s == t_t, "Relative attention is only available for self-attention."
    #   key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
    #   rel_logits = self._matmul_with_relative_keys(query /math.sqrt(self.k_channels), key_relative_embeddings)
    #   scores_local = self._relative_position_to_absolute_position(rel_logits)
    #   scores = scores + scores_local


        # relative attention bias
        # if position_bias is not None:
        #     reshape_q = query_states.contiguous().view(bsz * self.num_heads, -1, self.head_dim).transpose(0, 1)
        #     rel_pos_bias = torch.matmul(reshape_q, position_bias.transpose(-2, -1))
        #     rel_pos_bias = rel_pos_bias.transpose(0, 1).view(
        #         bsz * self.num_heads, position_bias.size(0), position_bias.size(1)
        #     )
        #     attn_weights += rel_pos_bias


    # TODO: how is this different? oh, they use a different fill value
    # block_length is None
    # ---> it looks like maybe the mask is different at this point too since I can't get
    # the results to match up...
    #
    # if mask is not None:
    #   scores = scores.masked_fill(mask == 0, -1e4)
    #   if self.block_length is not None:
    #     assert t_s == t_t, "Local attention is only available for self-attention."
    #     block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
    #     scores = scores.masked_fill(block_mask == 0, -1e4)

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        # return (attn_weigths, None, None) #MIH

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

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
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class VITSFeedForward(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(config.hidden_size, intermediate_size, config.ffn_kernel_size)
        self.conv_2 = nn.Conv1d(intermediate_size, config.hidden_size, config.ffn_kernel_size)
        self.dropout = nn.Dropout(config.activation_dropout)

        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    def forward(self, hidden_states, padding_mask):
        hidden_states = hidden_states.permute(0, 2, 1)
        padding_mask = padding_mask.permute(0, 2, 1)

        hidden_states = hidden_states * padding_mask
        print(hidden_states.shape)
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states * padding_mask
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states * padding_mask

        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states


class VITSEncoderLayer(nn.Module):
    def __init__(self, config: VITSConfig):
        super().__init__()
        self.attention = VITSAttention(
            embed_dim=config.hidden_size,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = VITSFeedForward(config, config.encoder_ffn_dim)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        # position_bias: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, hidden_size)`
            padding_mask
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            # position_bias (`torch.FloatTensor`):
            #     relative position embeddings of size `(seq_len, seq_len, hidden_size // encoder_attention_heads)`
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            # position_bias=position_bias,
            output_attentions=output_attentions,
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(residual + hidden_states)

        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states, padding_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.final_layer_norm(residual + hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class VITSEncoder(nn.Module):

    # self.encoder = Encoder(
    #   hidden_channels,
    #   filter_channels,
    #   n_heads,
    #   n_layers,
    #   kernel_size,
    #   p_dropout)

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.config = config

        # self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout)

        self.layers = nn.ModuleList([VITSEncoderLayer(config) for _ in range(config.encoder_layers)])

        # self.embed_positions = VITSRelativePositionalEncoding(
        #     config.hidden_size // config.encoder_attention_heads, config.encoder_max_relative_position
        # )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        padding_mask: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        hidden_states = hidden_states * padding_mask

    #TODO: or use W2V2?
        # if attention_mask is not None:
        #     # make sure padded tokens output 0
        #     expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
        #     hidden_states[~expand_attention_mask] = 0

        #     # extend attention_mask
        #     attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        #     attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
        #     attention_mask = attention_mask.expand(
        #         attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        #     )

        # hidden_states = self.layer_norm(hidden_states)
        # hidden_states = self.dropout(hidden_states)

        # position_bias = self.embed_positions(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for encoder_layer in self.layers:
            print("***")
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        padding_mask,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        output_attentions=output_attentions,
                    )
                hidden_states = layer_outputs[0]

                #break #MIH ##################

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        hidden_states = hidden_states * padding_mask

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class VITSTextEncoder(nn.Module):
    """
    """

    def __init__(self, config: VITSConfig):
        super().__init__()
        self.config = config

        #TODO: init for the embedding layer
        # self.emb = nn.Embedding(n_vocab, hidden_channels)
        # nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.encoder = VITSEncoder(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

        # TODO: may not be needed for final model but is needed to get same outputs
        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(-1)
        else:
            padding_mask = torch.ones_like(input_ids).unsqueeze(-1)

        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        # TODO: make this work even if return_dict enabled etc
    # stats = self.proj(x) * x_mask

    # m, logs = torch.split(stats, self.out_channels, dim=1)
    # return x, m, logs, x_mask

        # TODO: special kind of output class?

        return sequence_output
        return outputs


class VITSPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VITSConfig
    base_model_prefix = "vits"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (VITSTextEncoder)):
            module.gradient_checkpointing = value


VITS_BASE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VITSConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        encoder ([`VITSEncoderWithSpeechPrenet`] or [`VITSEncoderWithTextPrenet`] or `None`):
            The Transformer encoder module that applies the appropiate speech or text encoder prenet. If `None`,
            [`VITSEncoderWithoutPrenet`] will be used and the `input_values` are assumed to be hidden states.
        decoder ([`VITSDecoderWithSpeechPrenet`] or [`VITSDecoderWithTextPrenet`] or `None`):
            The Transformer decoder module that applies the appropiate speech or text decoder prenet. If `None`,
            [`VITSDecoderWithoutPrenet`] will be used and the `decoder_input_values` are assumed to be hidden
            states.
"""


VITS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VITSConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


VITS_INPUTS_DOCSTRING = r"""
    Args:
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            <Tip warning={true}>

            `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
            True`. For all models whose processor has `config.return_attention_mask == False`, `attention_mask` should
            **not** be passed to avoid degraded performance when doing batched inference. For such models
            `input_values` should simply be padded with 0 and passed without `attention_mask`. Be aware that these
            models also yield slightly different results depending on whether `input_values` is padded or not.

            </Tip>

        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_values`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read [`VITSDecoder._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

        head_mask (`torch.FloatTensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_values` (those
            that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_values` of shape `(batch_size, sequence_length)`. decoder_inputs_embeds (`torch.FloatTensor`
            of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
            `decoder_input_values` you can choose to directly pass an embedded representation. If `past_key_values` is
            used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is
            useful if you want more control over how to convert `decoder_input_values` indices into associated vectors
            than the model's internal embedding lookup matrix.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare VITS Encoder-Decoder Model outputting raw hidden-states without any specific pre- or post-nets.",
    VITS_BASE_START_DOCSTRING,
)
class VITSModel(VITSPreTrainedModel):
    def __init__(
        self,
        config: VITSConfig,
    ):
        super().__init__(config)
        self.config = config
        self.text_encoder = VITSTextEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    # def get_input_embeddings(self):
    #     if isinstance(self.encoder, VITSEncoderWithTextPrenet):
    #         return self.encoder.get_input_embeddings()
    #     if isinstance(self.decoder, VITSDecoderWithTextPrenet):
    #         return self.decoder.get_input_embeddings()
    #     return None

    # def set_input_embeddings(self, value):
    #     if isinstance(self.encoder, VITSEncoderWithTextPrenet):
    #         self.encoder.set_input_embeddings(value)
    #     if isinstance(self.decoder, VITSDecoderWithTextPrenet):
    #         self.decoder.set_input_embeddings(value)

    def get_encoder(self):
        return self.text_encoder

    # def get_decoder(self):
    #     return self.decoder

    @add_start_docstrings_to_model_forward(VITS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Depending on which encoder is being used, the `input_values` are either: float values of the input raw
            speech waveform, or indices of input sequence tokens in the vocabulary, or hidden states.

        decoder_input_values (`torch.Tensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Depending on which decoder is being used, the `decoder_input_values` are either: float values of log-mel
            filterbank features extracted from the raw speech waveform, or indices of decoder input sequence tokens in
            the vocabulary, or hidden states.

        speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
            Tensor containing the speaker embeddings.

        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # TODO!!!!!!!!!!


        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # # downsample encoder attention mask (only for encoders with speech input)
        # if attention_mask is not None and isinstance(self.encoder, VITSEncoderWithSpeechPrenet):
        #     encoder_attention_mask = self.encoder.prenet._get_feature_vector_attention_mask(
        #         encoder_outputs[0].shape[1], attention_mask
        #     )
        # else:
        #     encoder_attention_mask = attention_mask

        # if isinstance(self.decoder, VITSDecoderWithSpeechPrenet):
        #     decoder_args = {"speaker_embeddings": speaker_embeddings}
        # else:
        #     decoder_args = {}

        decoder_outputs = self.decoder(
            input_values=decoder_input_values,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            # encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # **decoder_args,
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


# @add_start_docstrings(
#     """VITS Model with a text encoder and a speech decoder.""",
#     VITS_START_DOCSTRING,
# )
# class VITSForTextToSpeech(VITSPreTrainedModel):
#     _keys_to_ignore_on_load_missing = []
#     _keys_to_ignore_on_save = []

#     main_input_name = "input_ids"

#     def __init__(self, config: VITSConfig):
#         super().__init__(config)

#         if config.vocab_size is None:
#             raise ValueError(
#                 f"You are trying to instantiate {self.__class__} with a configuration that does not define the"
#                 " vocabulary size of the language model head. Please instantiate the model as follows:"
#                 " `VITSForTextToSpeech.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of"
#                 " your model's configuration."
#             )

#         text_encoder = VITSEncoderWithTextPrenet(config)
#         speech_decoder = VITSDecoderWithSpeechPrenet(config)
#         self.VITS = VITSModel(config, text_encoder, speech_decoder)

#         self.speech_decoder_postnet = VITSSpeechDecoderPostnet(config)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_encoder(self):
#         return self.VITS.get_encoder()

#     def get_decoder(self):
#         return self.VITS.get_decoder()

#     @add_start_docstrings_to_model_forward(VITS_INPUTS_DOCSTRING)
#     @replace_return_docstrings(output_type=Seq2SeqSpectrogramOutput, config_class=_CONFIG_FOR_DOC)
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.LongTensor] = None,
#         decoder_input_values: Optional[torch.FloatTensor] = None,
#         decoder_attention_mask: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         decoder_head_mask: Optional[torch.FloatTensor] = None,
#         cross_attn_head_mask: Optional[torch.Tensor] = None,
#         encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         speaker_embeddings: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.FloatTensor] = None,
#         stop_labels: Optional[torch.Tensor] = None,
#     ) -> Union[Tuple, Seq2SeqSpectrogramOutput]:
#         r"""
#         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary. The `batch_size` should be 1 currently.

#             Indices can be obtained using [`VITSTokenizer`]. See [`~PreTrainedTokenizer.encode`] and
#             [`~PreTrainedTokenizer.__call__`] for details.

#             [What are input IDs?](../glossary#input-ids)
#         decoder_input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`):
#             Float values of input mel spectrogram.

#             VITS uses an all-zero spectrum as the starting token for `decoder_input_values` generation. If
#             `past_key_values` is used, optionally only the last `decoder_input_values` have to be input (see
#             `past_key_values`).
#         speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
#             Tensor containing the speaker embeddings.
#         labels (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_mel_bins)`, *optional*):
#             Float values of target mel spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss
#             computation. Spectrograms can be obtained using [`VITSProcessor`]. See [`VITSProcessor.__call__`]
#             for details.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import VITSProcessor, VITSForTextToSpeech, VITSHifiGan, set_seed
#         >>> import torch

#         >>> processor = VITSProcessor.from_pretrained("microsoft/VITS_tts")
#         >>> model = VITSForTextToSpeech.from_pretrained("microsoft/VITS_tts")
#         >>> vocoder = VITSHifiGan.from_pretrained("microsoft/VITS_hifigan")

#         >>> inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
#         >>> speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file

#         >>> set_seed(555)  # make deterministic

#         >>> # generate speech
#         >>> speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
#         >>> speech.shape
#         torch.Size([15872])
#         ```
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if stop_labels is not None:
#             warnings.warn(
#                 "The argument `stop_labels` is deprecated and will be removed in version 4.30.0 of Transformers",
#                 FutureWarning,
#             )

#         if labels is not None:
#             if decoder_input_values is None:
#                 decoder_input_values = shift_spectrograms_right(labels, self.config.reduction_factor)
#             if self.config.use_guided_attention_loss:
#                 output_attentions = True

#         outputs = self.vits(
#             input_values=input_ids,
#             attention_mask=attention_mask,
#             decoder_input_values=decoder_input_values,
#             decoder_attention_mask=decoder_attention_mask,
#             head_mask=head_mask,
#             decoder_head_mask=decoder_head_mask,
#             cross_attn_head_mask=cross_attn_head_mask,
#             encoder_outputs=encoder_outputs,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             speaker_embeddings=speaker_embeddings,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=True,
#         )

#         outputs_before_postnet, outputs_after_postnet, logits = self.speech_decoder_postnet(outputs[0])

#         loss = None
#         if labels is not None:
#             criterion = VITSSpectrogramLoss(self.config)
#             loss = criterion(
#                 attention_mask,
#                 outputs_before_postnet,
#                 outputs_after_postnet,
#                 logits,
#                 labels,
#                 outputs.cross_attentions,
#             )

#         if not return_dict:
#             output = (outputs_after_postnet,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return Seq2SeqSpectrogramOutput(
#             loss=loss,
#             spectrogram=outputs_after_postnet,
#             past_key_values=outputs.past_key_values,
#             decoder_hidden_states=outputs.decoder_hidden_states,
#             decoder_attentions=outputs.decoder_attentions,
#             cross_attentions=outputs.cross_attentions,
#             encoder_last_hidden_state=outputs.encoder_last_hidden_state,
#             encoder_hidden_states=outputs.encoder_hidden_states,
#             encoder_attentions=outputs.encoder_attentions,
#         )

#     @torch.no_grad()
#     def generate_speech(
#         self,
#         input_ids: torch.LongTensor,
#         speaker_embeddings: Optional[torch.FloatTensor] = None,
#         threshold: float = 0.5,
#         minlenratio: float = 0.0,
#         maxlenratio: float = 20.0,
#         vocoder: Optional[nn.Module] = None,
#         output_cross_attentions: bool = False,
#     ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
#         r"""
#         Converts a sequence of input tokens into a sequence of mel spectrograms, which are subsequently turned into a
#         speech waveform using a vocoder.

#         Args:
#             input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#                 Indices of input sequence tokens in the vocabulary. The `batch_size` should be 1 currently.

#                 Indices can be obtained using [`SpeechT5Tokenizer`]. See [`~PreTrainedTokenizer.encode`] and
#                 [`~PreTrainedTokenizer.__call__`] for details.

#                 [What are input IDs?](../glossary#input-ids)
#             speaker_embeddings (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_dim)`, *optional*):
#                 Tensor containing the speaker embeddings.
#             threshold (`float`, *optional*, defaults to 0.5):
#                 The generated sequence ends when the predicted stop token probability exceeds this value.
#             minlenratio (`float`, *optional*, defaults to 0.0):
#                 Used to calculate the minimum required length for the output sequence.
#             maxlenratio (`float`, *optional*, defaults to 20.0):
#                 Used to calculate the maximum allowed length for the output sequence.
#             vocoder (`nn.Module`, *optional*, defaults to `None`):
#                 The vocoder that converts the mel spectrogram into a speech waveform. If `None`, the output is the mel
#                 spectrogram.
#             output_cross_attentions (`bool`, *optional*, defaults to `False`):
#                 Whether or not to return the attentions tensors of the decoder's cross-attention layers.

#         Returns:
#             `tuple(torch.FloatTensor)` comprising various elements depending on the inputs:
#             - **spectrogram** (*optional*, returned when no `vocoder` is provided) `torch.FloatTensor` of shape
#               `(output_sequence_length, config.num_mel_bins)` -- The predicted log-mel spectrogram.
#             - **waveform** (*optional*, returned when a `vocoder` is provided) `torch.FloatTensor` of shape
#               `(num_frames,)` -- The predicted speech waveform.
#             - **cross_attentions** (*optional*, returned when `output_cross_attentions` is `True`) `torch.FloatTensor`
#               of shape `(config.decoder_layers, config.decoder_attention_heads, output_sequence_length,
#               input_sequence_length)` -- The outputs of the decoder's cross-attention layers.
#         """
#         return _generate_speech(
#             self,
#             input_ids,
#             speaker_embeddings,
#             threshold,
#             minlenratio,
#             maxlenratio,
#             vocoder,
#             output_cross_attentions,
#         )

