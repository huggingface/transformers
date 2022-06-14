# coding=utf-8
# Copyright 2022 The OFA-Sys Team. All rights reserved.
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
""" PyTorch OFA model."""

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from ...activations import ACT2FN
from ...file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput, Seq2SeqModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_ofa import OFAConfig
from .resnet import ResNet


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "OFA-Sys/OFA-base"
_CONFIG_FOR_DOC = "OFAConfig"
_TOKENIZER_FOR_DOC = "OFATokenizer"

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

OFA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "OFA-Sys/OFA-tiny",
    "OFA-Sys/OFA-medium",
    "OFA-Sys/OFA-base",
    "OFA-Sys/OFA-large",
]

try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True

    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    """
    Layer normalization. If apex is available, use `FusedLayerNorm` instead.
    """
    if torch.jit.is_scripting():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def make_token_bucket_position(bucket_size, max_position=DEFAULT_MAX_SOURCE_POSITIONS):
    """
    Make relative position indices for the text.
    """
    context_pos = torch.arange(max_position, dtype=torch.long)[:, None]
    memory_pos = torch.arange(max_position, dtype=torch.long)[None, :]
    relative_pos = context_pos - memory_pos
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where((relative_pos < mid) & (relative_pos > -mid), mid - 1, torch.abs(relative_pos))
    log_pos = torch.ceil(torch.log(abs_pos / mid) / math.log((max_position - 1) / mid) * (mid - 1)) + mid
    log_pos = log_pos.int()
    bucket_pos = torch.where(abs_pos.le(mid), relative_pos, log_pos * sign).long()
    return bucket_pos + bucket_size - 1


def make_image_bucket_position(bucket_size, num_relative_distance):
    """
    Make relative position indices for the image.
    """
    coords_h = torch.arange(bucket_size)
    coords_w = torch.arange(bucket_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += bucket_size - 1  # shift to start from 0
    relative_coords[:, :, 1] += bucket_size - 1
    relative_coords[:, :, 0] *= 2 * bucket_size - 1
    relative_position_index = torch.zeros(size=(bucket_size * bucket_size + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return relative_position_index


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x. If size is empty, using the size of the
    variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for uni-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    return expanded_mask.masked_fill(expanded_mask.bool(), torch.finfo(dtype).min)


def Embedding(num_embeddings, embedding_dim, padding_idx=None, zero_init=False):
    """Embedding for tokens"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    if zero_init:
        nn.init.constant_(m.weight, 0)
    return m


def Linear(in_features, out_features, bias=True):
    """
    Implementation of linear projection with xavier initialization
    """
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation.

    Args:
        p (`float`): probability of dropping out each layer
        modules (`nn.Module`, *optional*): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p

    def __iter__(self):
        dropout_probs = torch.empty(len(self)).uniform_()
        for i, m in enumerate(super().__iter__()):
            if not self.training or (dropout_probs[i] > self.p):
                yield m


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        x (`nn.Modules`): input nn layers.
        drop_prob (`float`): drop path ratio.
        training (`bool`): whether is training or inference.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (1, x.shape[1], 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        drop_prob: drop path ratio.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class OFAAttention(nn.Module):
    """
    Multi-headed attention, with additional implementation for NormFormer.

    Args:
        embed_dim (`int`): embedding dimension.
        num_heads (`int`): the number of attention heads.
        dropout (`float32`): the ratio for dropout.
        is_decoder (`bool`): whether or not decoder attention.
        bias (`bool`): whether to add bias.
        scale_heads (`bool`): whether to learn scaling heads, only for Normformer.
        scale_factor (`float32`): scaling factor for head.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        scale_heads: bool = True,
        scale_factor=2.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
            f" {num_heads})."
        )
        self.scaling = float(self.head_dim * scale_factor) ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.c_attn = nn.Parameter(torch.ones((self.num_heads,)), requires_grad=True) if scale_heads else None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """
        Reshape tensors for multi-head attention.
        """
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(bsz, tgt_len, embed_dim)`)`: input states.
            key_value_states (`torch.FloatTensor` of shape (bsz, tgt_len, embed_dim), *optional*): key value states.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key value states for fast inference.
            attention_mask (`torch.FloatTensor` of shape `(bsz, 1, tgt_len, seq_len)`, *optional*): attention mask.
            output_attentions (`bool`, *optional*): whether to output attention weights of all layers.
            attn_bias (`torch.FloatTensor` of shape `(bsz, 1, tgt_len, src_len)`, *optional*):
                the attention bias for positional information.

        Returns:
            attn_output (`torch.FloatTensor` of shape `(bsz, tgt_len, embed_dim)`): attention outputs.
            attn_weights_reshaped (`torch.FloatTensor`, *optional*): attention weights of all layers. past_key_value
            (`torch.FloatTensor`, *optional*): cached key value states for fast inference.
        """

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

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

        # Add attention bias for positional information
        if attn_bias is not None:
            attn_weights += attn_bias

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = self.attn_dropout(attn_weights)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        if self.c_attn is not None:
            attn_output = attn_output.view(bsz, tgt_len, self.num_heads, self.head_dim)
            attn_output = torch.einsum("bthd,h->bthd", attn_output, self.c_attn)
            attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OFAEncoderLayer(nn.Module):
    """
    OFA encoder layer implementation.

    Args:
        config: configuration for OFA.
        drop_path_rate: the ratio for drop path.
    """

    def __init__(self, config: OFAConfig, drop_path_rate=0.0):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = OFAAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.self_attn_mid_layer_norm = LayerNorm(self.embed_dim) if config.normformer else None
        self.dropout = nn.Dropout(config.dropout)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.ffn_layer_norm = LayerNorm(config.encoder_ffn_dim) if config.normformer else None
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.normalize_before = config.encoder_normalize_before
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def residual_connection(self, x, residual):
        """
        Residual connection with drop path.
        """
        return residual + self.drop_path(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        attn_bias: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(bsz, src_len, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(bsz, 1, src_len, src_len)* where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                whether to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            attn_bias (`torch.FloatTensor`): bias for positional information.

        Returns:
            outputs (`tuple(torch.FloatTensor)`):
                output hidden states of size (bsz, src_len, embed_dim), optionally with attention weights.
        """

        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            attn_bias=attn_bias,
        )
        if self.self_attn_mid_layer_norm:
            hidden_states = self.self_attn_mid_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)
        if self.ffn_layer_norm:
            hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class OFADecoderLayer(nn.Module):
    """
    OFA decoder layer implementation.

    Args:
        config: configuration for OFA.
        drop_path_rate: the ratio for drop path.
    """

    def __init__(self, config: OFAConfig, drop_path_rate=0.0):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = OFAAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = nn.Dropout(p=config.dropout)
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = nn.Dropout(p=config.activation_dropout)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.self_attn_mid_layer_norm = LayerNorm(self.embed_dim) if config.normformer else None
        self.cross_attn = OFAAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.cross_attn_layer_norm = LayerNorm(self.embed_dim)
        self.cross_attn_mid_layer_norm = LayerNorm(self.embed_dim) if config.normformer else None
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.ffn_layer_norm = LayerNorm(config.decoder_ffn_dim) if config.normformer else None
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.normalize_before = config.decoder_normalize_before
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def residual_connection(self, x, residual):
        """
        Residual connection with drop path.
        """
        return residual + self.drop_path(x)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        self_attn_bias: Optional[torch.Tensor] = None,
        cross_attn_bias: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`): input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(bsz, 1, tgt_len, src_len)`):
                attention mask where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`):
                cross attention input to the layer.
            encoder_attention_mask (`torch.FloatTensor` of shape `(bsz, 1, tgt_len, src_len)`):
                encoder attention mask where padding elements are indicated by very large negative values.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*): whether to return the attentions tensors of all attention layers.
            use_cache (`bool`, *optional*): whether to use cache
            self_attn_bias (`torch.FloatTensor`): self attention bias for positional information.
            cross_attn_bias (`torch.FloatTensor`): cross attention bias for positional information.
        """

        # Self attention with intermediate layernorm
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to position 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            attn_bias=self_attn_bias,
        )
        if self.self_attn_mid_layer_norm:
            hidden_states = self.self_attn_mid_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross attention with intermediate layernorm
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            if self.normalize_before:
                hidden_states = self.cross_attn_layer_norm(hidden_states)
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                attn_bias=cross_attn_bias,
            )
            if self.cross_attn_mid_layer_norm:
                hidden_states = self.cross_attn_mid_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.residual_connection(hidden_states, residual)
            if not self.normalize_before:
                hidden_states = self.cross_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # FFN with intermediate layernorm
        residual = hidden_states
        if self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)
        if self.ffn_layer_norm:
            hidden_states = self.ffn_layer_norm(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.residual_connection(hidden_states, residual)
        if not self.normalize_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OFAPreTrainedModel(PreTrainedModel):
    """Base class OFA"""

    config_class = OFAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """
        Weight initialization which follows BERT.
        """
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
        """
        Turn on the switch of gradient checkpointing.
        """
        if isinstance(module, (OFADecoder, OFAEncoder)):
            module.gradient_checkpointing = value


@dataclass
class OFAEncoderOutput(ModelOutput):
    """
    Base class for OFA's outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`):
            Sequence of hidden-states at the output of the last layer of the model.

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):

            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(bsz, seq_len, hidden)`. Hidden-states of the model at the output of each layer plus the initial
            embedding outputs.

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):

            Tuple of `torch.FloatTensor` (one for each layer) of shape `(bsz, num_heads, seq_len, seq_len)`. Attentions
            weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        position_embedding (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`):
            postional embeddings of the inputs.
    """

    last_hidden_state: torch.FloatTensor = None
    padding_mask: torch.Tensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    position_embedding: Optional[torch.FloatTensor] = None


OFA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`~OFAConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


OFA_GENERATION_EXAMPLE = r"""
    Image captioning example:

    ```python
    >>> model = OFAForConditionalGeneration.from_pretrained(ckpt_dir)
    >>> tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

    >>> txt = " what is the description of the image?"
    >>> inputs = tokenizer([txt], max_length=1024, return_tensors="pt")["input_ids"]
    >>> img = Image.open(path_to_image)
    >>> patch_img = patch_resize_transform(img).unsqueeze(0)

    >>> gen = model.generate(inputs, patch_img=patch_img, num_beams=4)
    >>> print(tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    ```
"""


OFA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`):
            indices of input sequence tokens in the vocabular, and padding will be ignored by default;

            indices can be obtained using [`~OFATokenizer`].

        patch_images (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
            the resized image, which are transformed by the default operations.
        patch_images_2 (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
            the second (if it exists) image.
        patch_masks (`torch.BoolTensor`): the patches to be masked.
        token_embeddings (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`): token embeddings.
        sample_patch_num (`int`): the number of patches to sample.
        decoder_input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`): indices of the sequence in the vocabulary.
        code_masks (`torch.Tensor` of shape `(bsz, seq_len)`): masks only for code generation.
        attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): attention mask for decoding.
        encoder_outputs (`OFAEncoderOutput`):
            encoder outputs with hidden states, positional embeddings, and padding masks.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(bsz, num_heads, tgt_len, head_size)`) and 2 additional tensors of shape `(bsz, num_heads, src_len,
            head_size)`.
        use_cache (`bool`): whether to use cache for faster inference.
        output_attentions (`bool`): whether to output attention weights.
        output_hidden_states (`bool`): whether to output hidden states.
        return_dict (`bool`): unused. Keep it for generation only.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
"""


class OFAEncoder(OFAPreTrainedModel):
    """
    OFA encoder consisting of layers of [`OFAEncoderLayer`].

    Args:
        config: OFAConfig
        embed_tokens (`nn.Embedding`, *optional*): output embedding
    """

    def __init__(self, config: OFAConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = nn.Dropout(config.dropout)
        self.encoder_layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.num_attention_heads = config.encoder_attention_heads

        if getattr(config, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        if config.add_type_embedding:
            self.type_embedding = Embedding(2, embed_dim, padding_idx=None)
        else:
            self.type_embedding = None

        if config.resnet_type == "resnet18":
            self.embed_images = ResNet([2, 2, 2], drop_path_rate=config.resnet_drop_path_rate)
        elif config.resnet_type == "resnet34":
            self.embed_images = ResNet([3, 4, 6], drop_path_rate=config.resnet_drop_path_rate)
        elif config.resnet_type == "resnet50":
            self.embed_images = ResNet([3, 4, 6], drop_path_rate=config.resnet_drop_path_rate)
        elif config.resnet_type == "resnet101":
            self.embed_images = ResNet([3, 4, 23], drop_path_rate=config.resnet_drop_path_rate)
        elif config.resnet_type == "resnet152":
            self.embed_images = ResNet([3, 8, 36], drop_path_rate=config.resnet_drop_path_rate)
        else:
            raise NotImplementedError
        self.image_proj = nn.Linear(1024, embed_dim)

        if config.resnet_model_path:
            print("load resnet {}".format(config.resnet_model_path))
            resnet_state_dict = torch.load(config.resnet_model_path)
            self.embed_images.load_state_dict(resnet_state_dict)
        if config.patch_layernorm_embedding:
            self.patch_layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.patch_layernorm_embedding = None

        self.embed_positions = Embedding(self.max_source_positions + 2, embed_dim)
        self.embed_image_positions = Embedding(config.image_bucket_size**2 + 1, embed_dim)
        self.pos_ln = LayerNorm(embed_dim)
        self.image_pos_ln = LayerNorm(embed_dim)
        self.pos_scaling = float(embed_dim / self.num_attention_heads * config.attn_scale_factor) ** -0.5
        self.pos_q_linear = nn.Linear(embed_dim, embed_dim)
        self.pos_k_linear = nn.Linear(embed_dim, embed_dim)

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, config.encoder_drop_path_rate, config.encoder_layers)]
        self.layers.extend([OFAEncoderLayer(config, drop_path_rate=dpr[i]) for i in range(config.encoder_layers)])
        self.num_layers = len(self.layers)

        if config.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.token_bucket_size = config.token_bucket_size
        token_num_rel_dis = 2 * config.token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(config.token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [
                Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True)
                for _ in range(config.encoder_layers)
            ]
        )

        self.image_bucket_size = config.image_bucket_size
        image_num_rel_dis = (2 * config.image_bucket_size - 1) * (2 * config.image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(config.image_bucket_size, image_num_rel_dis)
        self.image_rel_pos_table_list = nn.ModuleList(
            [
                Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True)
                for _ in range(config.encoder_layers)
            ]
        )

        if config.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)
        self.entangle_position_embedding = config.entangle_position_embedding

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Get the embedding weight.
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the weight of embedding with the given tensor.
        """
        self.embed_tokens = value

    def get_rel_pos_bias(self, x, idx):
        """
        Get the relative positional bias of the text, for attention.
        """

        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        values = values.permute([0, 3, 1, 2])
        return values.contiguous()

    def get_image_rel_pos_bias(self, image_position_ids, idx):
        """
        Get the relative positional bias of the image, for attention.
        """

        bsz, seq_len = image_position_ids.shape
        rp_bucket_size = self.image_rp_bucket.size(1)

        rp_bucket = (
            self.image_rp_bucket.unsqueeze(0)
            .expand(bsz, rp_bucket_size, rp_bucket_size)
            .gather(1, image_position_ids[:, :, None].expand(bsz, seq_len, rp_bucket_size))
            .gather(2, image_position_ids[:, None, :].expand(bsz, seq_len, seq_len))
        )
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(0, 3, 1, 2)
        return values

    def get_patch_images_info(self, patch_images, sample_patch_num, device):
        """
        Get the basic information of the resized image.

        Args:
            patch_images (`torch.FloatTensor` of shape `(bsz, 3, height, width)`): the resized image.
            sample_patch_num (`int`):
                the number of patches to sample. If it is equal to -1, no sampling will be performed.
            device: GPU device.

        Returns:
            image_embed (`torch.FloatTensor` of shape `(bsz, h * w, hidden)`): the output of the visual encoder.
            image_num_patches (`int`, equal to `h * w`): the number of patches. image_padding_mask
            (`torch.BooleanTensor` of shape `(bsz, h*w)`): image padding mask. image_position_ids (`torch.LongTensor`
            of shape `(bsz, h*w)`): image position ids. image_pos_embed (`torch.FloatTensor` of shape (bsz, h*w,
            hidden)): the positional embedding.
        """

        image_embed = self.embed_images(patch_images)
        h, w = image_embed.shape[-2:]
        image_num_patches = h * w
        image_padding_mask = patch_images.new_zeros((patch_images.size(0), image_num_patches)).bool()
        image_position_idx = (
            torch.arange(w).unsqueeze(0).expand(h, w) + torch.arange(h).unsqueeze(1) * self.image_bucket_size + 1
        )
        image_position_idx = image_position_idx.view(-1).to(device)
        image_position_ids = image_position_idx[None, :].expand(patch_images.size(0), image_num_patches)

        image_embed = image_embed.flatten(2).transpose(1, 2)
        if sample_patch_num is not None:
            patch_orders = [
                random.sample(range(image_num_patches), k=sample_patch_num) for _ in range(patch_images.size(0))
            ]
            patch_orders = torch.LongTensor(patch_orders).to(device)
            image_embed = image_embed.gather(1, patch_orders.unsqueeze(2).expand(-1, -1, image_embed.size(2)))
            image_num_patches = sample_patch_num
            image_padding_mask = image_padding_mask.gather(1, patch_orders)
            image_position_ids = image_position_ids.gather(1, patch_orders)
        image_pos_embed = self.embed_image_positions(image_position_ids)

        return image_embed, image_num_patches, image_padding_mask, image_position_ids, image_pos_embed

    def forward_embedding(
        self,
        input_ids,
        image_embed: Optional[torch.Tensor] = None,
        image_embed_2: Optional[torch.Tensor] = None,
        token_embedding: Optional[torch.Tensor] = None,
        pos_embed: Optional[torch.Tensor] = None,
        image_pos_embed: Optional[torch.Tensor] = None,
        image_pos_embed_2: Optional[torch.Tensor] = None,
    ):
        """
        Generate embeddings of both the image and the text. Actually since OFA unifies both unimodal and multimodal
        data, image inputs are optional.

        Args:
            input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`): indices of the tokens in the vocabulary.
            image_embed (`torch.FloatTensor` of shape `(bsz, h*w, embed_dim)`, *optional*): image embeddings.
            image_embed_2 (`torch.FloatTensor` of shape `(bsz, h*w, embed_dim)`, *optional*):
                image embeddings of the second image (if it exists).
            token_embedding (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`, *optional*):
                input token embeddings to replace the embeddings of input ids.
            image_pos_embed (`torch.FloatTensor` of shape `(bsz, h*w, embed_dim)`, *optional*):
                positional embeddings of the image.
            image_pos_embed_2 (`torch.FloatTensor` of shape `(bsz, h*w, embed_dim)`, *optional*):
                positional embeddings of the second image.

        Returns:
            x (`torch.FloatTensor` of shape `(bsz, h*w+seq_len, embed_dim)`): embeddings of the input. embed
            (`torch.FloatTensor` of shape `(bsz, h*w+seq_len, embed_dim)`):
                embeddings without adding positional and type embeddings.
        """

        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(input_ids)
        x = embed = self.embed_scale * token_embedding
        if self.entangle_position_embedding and pos_embed is not None:
            x += pos_embed
        if self.type_embedding is not None:
            x += self.type_embedding(input_ids.new_zeros(x.size()[:2]))
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout(x)

        # embed raw images
        if image_embed is not None:
            image_embed = self.image_proj(image_embed)
            image_x = image_embed = self.embed_scale * image_embed
            if self.entangle_position_embedding and image_pos_embed is not None:
                image_x += image_pos_embed
            if self.type_embedding is not None:
                image_x += self.type_embedding(input_ids.new_ones(image_x.size()[:2]))
            if self.patch_layernorm_embedding is not None:
                image_x = self.patch_layernorm_embedding(image_x)
            image_x = self.dropout(image_x)
            x = torch.cat([image_x, x], dim=1)
            embed = torch.cat([image_embed, embed], dim=1)

        if image_embed_2 is not None:
            assert self.type_embedding is not None
            image_embed_2 = self.image_proj(image_embed_2)
            image_x_2 = image_embed_2 = self.embed_scale * image_embed_2
            if self.entangle_position_embedding and image_pos_embed_2 is not None:
                image_x_2 += image_pos_embed_2
            if self.type_embedding is not None:
                image_x_2 += self.type_embedding(input_ids.new_full(image_x_2.size()[:2], fill_value=2))
            if self.patch_layernorm_embedding is not None:
                image_x_2 = self.patch_layernorm_embedding(image_x_2)
            image_x_2 = self.dropout(image_x_2)
            if self.quant_noise is not None:
                image_x_2 = self.quant_noise(image_x_2)
            x = torch.cat([image_x_2, x], dim=1)
            embed = torch.cat([image_embed_2, embed], dim=1)

        return x, embed

    def forward(
        self,
        input_ids=None,
        patch_images: Optional[torch.Tensor] = None,
        patch_images_2: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`):
                indices of input sequence tokens in the vocabular, and padding will be ignored by default;

                indices can be obtained using [`~OFATokenizer`].

            patch_images (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
                the resized image, which are transformed by the default operations.
            patch_images_2 (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
                the second (if it exists) image.
            patch_masks (`torch.BoolTensor`): the patches to be masked.
            output_attentions (`bool`): whether to return all attention weights,
            output_hidden_states (`bool`): whether to return all hidden states.
            token_embeddings (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`): token embeddings.
            sample_patch_num (`int`): the number of patches to sample.

        Returns:
            [`OFAEncoderOutput`]:
                last_hidden_state (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`):
                    the states of the last layer.
                padding_mask (`torch.BoolTensor` of shape `(bsz, seq_len)`):
                    the padding mask of the source context.
                hidden_states (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`):
                    the states of all layers including the embeddings.
                attentions (`torch.FloatTensor` of shape `(bsz, num_heads, seq_len, seq_len)`):
                    the attention weights of all layers.
                position_embedding (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`):
                    positional embeddings of the input image and tokens.
        """

        image_embed = None
        image_embed_2 = None
        image_pos_embed = None
        image_pos_embed_2 = None
        if patch_images is not None:
            (
                image_embed,
                image_num_patches,
                image_padding_mask,
                image_position_ids,
                image_pos_embed,
            ) = self.get_patch_images_info(patch_images, sample_patch_num, input_ids.device)
            image_padding_mask[~patch_masks] = True
        if patch_images_2 is not None:
            (
                image_embed_2,
                image_num_patches_2,
                image_padding_mask_2,
                image_position_ids_2,
                image_pos_embed_2,
            ) = self.get_patch_images_info(patch_images_2, sample_patch_num, input_ids.device)
            image_padding_mask_2[~patch_masks] = True

        encoder_padding_mask = input_ids.eq(self.padding_idx)
        if patch_images is not None:
            encoder_padding_mask = torch.cat([image_padding_mask, encoder_padding_mask], dim=1)
        if patch_images_2 is not None:
            encoder_padding_mask = torch.cat([image_padding_mask_2, encoder_padding_mask], dim=1)
        has_pads = encoder_padding_mask.any()

        pos_embed = self.embed_positions(new_arange(input_ids))
        x, encoder_embedding = self.forward_embedding(
            input_ids, image_embed, image_embed_2, token_embeddings, pos_embed, image_pos_embed, image_pos_embed_2
        )

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        pos_embed = self.pos_ln(pos_embed)
        if patch_images is not None:
            image_pos_embed = self.image_pos_ln(image_pos_embed)
            pos_embed = torch.cat([image_pos_embed, pos_embed], dim=1)
        if patch_images_2 is not None:
            image_pos_embed_2 = self.image_pos_ln(image_pos_embed_2)
            pos_embed = torch.cat([image_pos_embed_2, pos_embed], dim=1)

        pos_q = (
            self.pos_q_linear(pos_embed).view(x.size(0), x.size(1), self.num_attention_heads, -1).transpose(1, 2)
            * self.pos_scaling
        )
        pos_k = self.pos_k_linear(pos_embed).view(x.size(0), x.size(1), self.num_attention_heads, -1).transpose(1, 2)
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

        # expand attention_mask
        if has_pads:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(encoder_padding_mask, dtype=x.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if output_hidden_states:
            encoder_states.append(x)

        # encoder layers
        for idx, layer in enumerate(self.layers):
            self_attn_bias = abs_pos_bias.clone()
            self_attn_bias[:, :, -input_ids.size(1) :, -input_ids.size(1) :] += self.get_rel_pos_bias(input_ids, idx)
            if patch_images_2 is not None:
                self_attn_bias[:, :, :image_num_patches_2, :image_num_patches_2] += self.get_image_rel_pos_bias(
                    image_position_ids_2, idx
                )
                self_attn_bias[
                    :,
                    :,
                    image_num_patches_2 : image_num_patches_2 + image_num_patches,
                    image_num_patches_2 : image_num_patches_2 + image_num_patches,
                ] += self.get_image_rel_pos_bias(image_position_ids, idx)
            elif patch_images is not None:
                self_attn_bias[
                    :, :, : x.size(1) - input_ids.size(1), : x.size(1) - input_ids.size(1)
                ] += self.get_image_rel_pos_bias(image_position_ids, idx)
            self_attn_bias = self_attn_bias.reshape(-1, x.size(1), x.size(1))

            hidden_outputs = layer(x, attention_mask if has_pads else None, attn_bias=self_attn_bias)
            x = hidden_outputs[0]
            if output_hidden_states:
                encoder_states = encoder_states + (x,)
            if output_attentions:
                attention = hidden_outputs[1]
                all_attentions = all_attentions + (attention,)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return OFAEncoderOutput(
            last_hidden_state=x,
            padding_mask=encoder_padding_mask,
            hidden_states=encoder_states,
            attentions=all_attentions,
            position_embedding=pos_embed,
        )


class OFADecoder(OFAPreTrainedModel):
    """
    OFA decoder consisting of layers of [`OFADecoderLayer`]

    Args:
        config: OFAConfig
        embed_tokens (`nn.Embedding`, *optional*): output embedding
    """

    def __init__(self, config: OFAConfig, embed_tokens: Optional[nn.Embedding] = None, output_projection=None):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout)
        self.decoder_layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self._future_mask = torch.empty(0)
        self.share_input_output_embed = config.share_decoder_input_output_embed
        self.num_attention_heads = config.decoder_attention_heads

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_dim = config.d_model
        self.output_embed_dim = config.d_model

        self.layers = nn.ModuleList([OFADecoderLayer(config) for _ in range(config.decoder_layers)])
        if config.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

        self.window_size = config.code_image_size // 8

        self.embed_positions = Embedding(self.max_target_positions + 2, self.embed_dim)
        self.embed_image_positions = Embedding(config.image_bucket_size**2 + 1, self.embed_dim)
        self.pos_ln = LayerNorm(self.embed_dim)
        self.image_pos_ln = LayerNorm(self.embed_dim)
        self.pos_scaling = float(self.embed_dim / self.num_attention_heads * config.attn_scale_factor) ** -0.5
        self.self_pos_q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.self_pos_k_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_pos_q_linear = nn.Linear(self.embed_dim, self.embed_dim)
        self.cross_pos_k_linear = nn.Linear(self.embed_dim, self.embed_dim)

        if config.code_layernorm_embedding:
            self.code_layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.code_layernorm_embedding = None

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])

        dpr = [x.item() for x in torch.linspace(0, config.decoder_drop_path_rate, config.decoder_layers)]
        self.layers.extend([OFADecoderLayer(config, drop_path_rate=dpr[i]) for i in range(config.decoder_layers)])
        self.num_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(config)

        self.token_bucket_size = config.token_bucket_size
        token_num_rel_dis = 2 * config.token_bucket_size - 1
        token_rp_bucket = make_token_bucket_position(config.token_bucket_size)
        self.token_rel_pos_table_list = nn.ModuleList(
            [
                Embedding(token_num_rel_dis, self.num_attention_heads, zero_init=True)
                for _ in range(config.decoder_layers)
            ]
        )

        self.image_bucket_size = config.image_bucket_size
        image_num_rel_dis = (2 * config.image_bucket_size - 1) * (2 * config.image_bucket_size - 1) + 3
        image_rp_bucket = make_image_bucket_position(config.image_bucket_size, image_num_rel_dis)
        image_position_idx = (
            torch.arange(self.window_size).unsqueeze(0).expand(self.window_size, self.window_size)
            + torch.arange(self.window_size).unsqueeze(1) * config.image_bucket_size
            + 1
        )
        image_position_idx = torch.cat([torch.tensor([0]), image_position_idx.view(-1)])
        image_position_idx = torch.cat([image_position_idx, torch.tensor([1024] * 768)])
        self.image_rel_pos_table_list = nn.ModuleList(
            [
                Embedding(image_num_rel_dis, self.num_attention_heads, zero_init=True)
                for _ in range(config.decoder_layers)
            ]
        )

        self.register_buffer("token_rp_bucket", token_rp_bucket)
        self.register_buffer("image_rp_bucket", image_rp_bucket)
        self.register_buffer("image_position_idx", image_position_idx)
        self.entangle_position_embedding = config.entangle_position_embedding

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def build_output_projection(self, config):
        """generate output projection layer"""
        if self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(self.output_embed_dim, config.vocab_size, bias=False)
            nn.init.normal_(self.output_projection.weight, mean=0, std=self.output_embed_dim**-0.5)

    def get_rel_pos_bias(self, x, idx):
        """
        Get the relative positional bias of the text, for attention.
        """

        seq_len = x.size(1)
        rp_bucket = self.token_rp_bucket[:seq_len, :seq_len]
        values = F.embedding(rp_bucket, self.token_rel_pos_table_list[idx].weight)
        values = values.permute([2, 0, 1])
        return values.contiguous()

    def get_image_rel_pos_bias(self, x, idx):
        """
        Get the relative positional bias of the image, for attention.
        """

        seq_len = x.size(1)
        image_position_idx = self.image_position_idx[:seq_len]
        rp_bucket = self.image_rp_bucket[image_position_idx][:, image_position_idx]
        values = F.embedding(rp_bucket, self.image_rel_pos_table_list[idx].weight)
        values = values.permute(2, 0, 1)
        return values

    def get_pos_info(self, tgt_pos_embed, src_pos_embed=None, use_image=False):
        """
        Get the positional information.

        Args:
            tgt_pos_embed (`torch.FloatTensor` of shape `(bsz, tgt_len, embed_dim)`):
                the target-side positional embeddings.
            src_pos_embed (`torch.FloatTensor` of shape `(bsz, src_len, embed_dim)`, *optional*):
                the source-side positional embeddings.
            use_image (`bool`): whether to use image.

        Returns:
            abs_pos_bias (`torch.FloatTensor` of shape `(bsz, src_len, tgt_len, src_len)`):
                absolute positional bias for attention.
        """

        batch_size = tgt_pos_embed.size(0)
        tgt_len = tgt_pos_embed.size(1)
        tgt_pos_embed = self.image_pos_ln(tgt_pos_embed) if use_image else self.pos_ln(tgt_pos_embed)

        if src_pos_embed is not None:
            src_len = src_pos_embed.size(1)
            pos_q = (
                self.cross_pos_q_linear(tgt_pos_embed)
                .view(batch_size, tgt_len, self.num_attention_heads, -1)
                .transpose(1, 2)
                * self.pos_scaling
            )
            pos_k = (
                self.cross_pos_k_linear(src_pos_embed)
                .view(batch_size, src_len, self.num_attention_heads, -1)
                .transpose(1, 2)
            )
        else:
            src_len = tgt_pos_embed.size(1)
            pos_q = (
                self.self_pos_q_linear(tgt_pos_embed)
                .view(batch_size, tgt_len, self.num_attention_heads, -1)
                .transpose(1, 2)
                * self.pos_scaling
            )
            pos_k = (
                self.self_pos_k_linear(tgt_pos_embed)
                .view(batch_size, src_len, self.num_attention_heads, -1)
                .transpose(1, 2)
            )
        abs_pos_bias = torch.matmul(pos_q, pos_k.transpose(2, 3))

        return abs_pos_bias

    def get_input_embeddings(self):
        """
        Get the input embeddings
        """
        return self.embed_tokens

    def set_input_embeddings(self, value):
        """
        Set the weights of the embeddings with the given tensor.
        """
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, past_key_values_length):
        """
        Create causal mask for unidirectional decoding. [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        """
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        code_masks: Optional[torch.Tensor] = None,
        src_pos_embed: torch.Tensor = None,
        past_key_values: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`): indices of the sequence in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): mask to avoid attention on padding tokens.
            encoder_hidden_states (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`):
                the last hidden state of the encoder.
            encoder_attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): the padding mask of the source side.
            code_masks (`torch.Tensor` of shape `(bsz, seq_len)`): masks only for code generation.
            src_pos_embed (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`):
                the positional embeddings of the source side.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(bsz, num_heads, tgt_len, head_size)`) and 2 additional tensors of shape `(bsz, num_heads,
                src_len, head_size)`.
            use_cache (`bool`): whether to use cache for faster inference.
            output_attentions (`bool`): whether to output attention weights.
            output_hidden_states (`bool`): whether to output hidden states.

        Returns:
            BaseModelOutputWithPastAndCrossAttentions or a plain tuple:
                last_hidden_state (`torch.FloatTensor` of shape `(bsz, seq_len, hidden)`): the last hidden states.
                past_key_values (`tuple(tuple(torch.FloatTensor)): past keys and values for faster inference.
                hidden_states (`tuple(torch.FloatTensor)`): hidden states of all layers. attentions
                (`tuple(torch.FloatTensor)): self attention weights of all layers. cross_attentions
                (`tuple(torch.FloatTensor)): cross attention weights of all layers.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if past_key_values is not None:
            size = past_key_values[0][0].size()
            bsz, tgt_len = size[0], size[-2] + 1
            token_position_idx = torch.arange(tgt_len, device=input_ids.device).expand([bsz, tgt_len]).contiguous()
        else:
            bsz, tgt_len = input_ids.shape
            token_position_idx = new_arange(input_ids)
        tgt_pos_embed = self.embed_positions(token_position_idx)
        if code_masks is not None and torch.any(code_masks):
            image_position_idx = self.image_position_idx[: input_ids.size(1)].unsqueeze(0).expand(bsz, tgt_len)
            tgt_pos_embed[code_masks] = self.embed_image_positions(image_position_idx)[code_masks]

        # self attn position bias
        self_abs_pos_bias = self.get_pos_info(tgt_pos_embed, use_image=False)
        if code_masks is not None and torch.any(code_masks):
            self_image_abs_pos_bias = self.get_pos_info(tgt_pos_embed, use_image=True)
            self_abs_pos_bias[code_masks] = self_image_abs_pos_bias[code_masks]
        # cross attn position bias
        cross_abs_pos_bias = self.get_pos_info(tgt_pos_embed, src_pos_embed=src_pos_embed)
        if code_masks is not None and torch.any(code_masks):
            cross_image_abs_pos_bias = self.get_pos_info(tgt_pos_embed, src_pos_embed=src_pos_embed, use_image=True)
            cross_abs_pos_bias[code_masks] = cross_image_abs_pos_bias[code_masks]
        cross_abs_pos_bias = cross_abs_pos_bias.reshape(-1, *cross_abs_pos_bias.size()[-2:])

        all_prev_output_tokens = input_ids.clone()
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            cross_abs_pos_bias = cross_abs_pos_bias[:, -1:, :]
            tgt_pos_embed = tgt_pos_embed[:, -1:, :]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(input_ids)

        if self.entangle_position_embedding is not None:
            x += tgt_pos_embed

        if self.layernorm_embedding is not None:
            if code_masks is None or not code_masks.any() or not self.code_layernorm_embedding:
                x = self.layernorm_embedding(x)
            elif code_masks is not None and code_masks.all():
                x = self.code_layernorm_embedding(x)
            else:
                x[~code_masks] = self.layernorm_embedding(x[~code_masks])
                x[code_masks] = self.code_layernorm_embedding(x[code_masks])

        hidden_states = self.dropout(x)

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        shape, dtype = input_ids.shape, hidden_states.dtype
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, shape, dtype, past_key_values_length)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # decoder layers
        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            self_attn_bias = self_abs_pos_bias.clone()
            if code_masks is None or not code_masks.any():
                self_attn_bias += self.get_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            elif code_masks is not None and code_masks.all():
                self_attn_bias += self.get_image_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            else:
                self_attn_bias[~code_masks] += self.get_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
                self_attn_bias[code_masks] += self.get_image_rel_pos_bias(all_prev_output_tokens, idx).unsqueeze(0)
            self_attn_bias = self_attn_bias.reshape(-1, *self_attn_bias.size()[-2:])
            if past_key_value is not None:
                self_attn_bias = self_attn_bias[:, -1:, :]

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                self_attn_bias=self_attn_bias,
                cross_attn_bias=cross_abs_pos_bias,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        if self.output_projection is not None:
            hidden_states = self.output_projection(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    "The bare OFA Model outputting raw hidden-states without any specific head on top.",
    OFA_START_DOCSTRING,
)
class OFAModel(OFAPreTrainedModel):
    """
    The OFA model built with an encoder and a decoder only, without any classification head.

    Args:
        config (OFAConfig): OFA configuration.
    """

    def __init__(self, config: OFAConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = OFAEncoder(config, shared)
        self.decoder = OFADecoder(config, shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve input embeddings.
        """
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        """
        Set values for input embeddings
        """
        shared = value
        self.encoder.embed_tokens = shared
        self.decoder.embed_tokens = shared

    def get_encoder(self):
        """
        Retrieve the encoder
        """
        return self.encoder

    def get_decoder(self):
        """
        Retrieve the decoder
        """
        return self.decoder

    @add_start_docstrings_to_model_forward(OFA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        patch_images=None,
        patch_images_2=None,
        patch_masks=None,
        token_embeddings=None,
        sample_patch_num=None,
        decoder_input_ids=None,
        code_masks=None,
        attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`):
                indices of input sequence tokens in the vocabular, and padding will be ignored by default;

                indices can be obtained using [`~OFATokenizer`].

            patch_images (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
                the resized image, which are transformed by the default operations.
            patch_images_2 (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
                the second (if it exists) image.
            patch_masks (`torch.BoolTensor`): the patches to be masked.
            token_embeddings (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`): token embeddings.
            sample_patch_num (`int`): the number of patches to sample.
            decoder_input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`):
                indices of the sequence in the vocabulary.
            code_masks (`torch.Tensor` of shape `(bsz, seq_len)`): masks only for code generation.
            attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): attention mask for decoding.
            encoder_outputs (`OFAEncoderOutput`):
                encoder outputs with hidden states, positional embeddings, and padding masks.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(bsz, num_heads, tgt_len, head_size)`) and 2 additional tensors of shape `(bsz, num_heads,
                src_len, head_size)`.
            use_cache (`bool`): whether to use cache for faster inference.
            output_attentions (`bool`): whether to output attention weights.
            output_hidden_states (`bool`): whether to output hidden states.
        """

        output_attentions = output_attentions if output_attentions else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                patch_images=patch_images,
                patch_images_2=patch_images_2,
                patch_masks=patch_masks,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                token_embeddings=token_embeddings,
                sample_patch_num=sample_patch_num,
            )

        if decoder_input_ids.eq(self.config.pad_token_id).any():
            attention_mask = decoder_input_ids.eq(self.padding_idx)

        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_attention_mask = _expand_mask(
            encoder_outputs.padding_mask, encoder_hidden_states.dtype, decoder_input_ids.shape[-1]
        )
        src_pos_embed = encoder_outputs.position_embedding

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            code_masks=code_masks,
            src_pos_embed=src_pos_embed,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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


@add_start_docstrings(
    "The OFA Model with a language modeling head. Can be used for conditional generation.", OFA_START_DOCSTRING
)
class OFAForConditionalGeneration(OFAPreTrainedModel):
    """
    The OFA model for conditional generation, which can adapt to most tasks due to the unified nature of OFA.

    Args:
        config (OFAConfig): OFA configuration.
    """

    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: OFAConfig):
        super().__init__(config)
        self.model = OFAModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        """
        Retrieve the encoder
        """
        return self.model.get_encoder()

    def get_decoder(self):
        """
        Retrieve the decoder
        """
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """
        Resize token embeddings if necessary
        """
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def get_output_embeddings(self):
        """
        retrieve the output embeddings
        """
        return self.model.decoder.output_projection

    def set_output_embeddings(self, new_embeddings):
        """
        Set values of the output embeddings
        """
        self.model.decoder.output_projection = new_embeddings

    @add_start_docstrings_to_model_forward(OFA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(OFA_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,
        patch_images=None,
        patch_images_2=None,
        patch_masks=None,
        token_embeddings=None,
        sample_patch_num=None,
        decoder_input_ids=None,
        code_masks=None,
        attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        labels=None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`):
                indices of input sequence tokens in the vocabulary, and padding will be ignored by default;

                indices can be obtained using [`~OFATokenizer`].

            patch_images (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
                the resized image, which are transformed by the default operations.
            patch_images_2 (`torch.FloatTensor` of shape `(bsz, 3, height, width)`):
                the second (if it exists) image.
            patch_masks (`torch.BoolTensor`): the patches to be masked.
            token_embeddings (`torch.FloatTensor` of shape `(bsz, seq_len, embed_dim)`): token embeddings.
            sample_patch_num (`int`): the number of patches to sample.
            decoder_input_ids (`torch.LongTensor` of shape `(bsz, seq_len)`):
                indices of the sequence in the vocabulary.
            code_masks (`torch.Tensor` of shape `(bsz, seq_len)`): masks only for code generation.
            attention_mask (`torch.Tensor` of shape `(bsz, seq_len)`): attention mask for decoding.
            encoder_outputs (`OFAEncoderOutput`):
                encoder outputs with hidden states, positional embeddings, and padding masks.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(bsz, num_heads, tgt_len, head_size)`) and 2 additional tensors of shape `(bsz, num_heads,
                src_len, head_size)`.
            use_cache (`bool`): whether to use cache for faster inference.
            output_attentions (`bool`): whether to output attention weights.
            output_hidden_states (`bool`): whether to output hidden states.
            return_dict (`bool`): unused. Keep it for generation only.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Seq2SeqModelOutput (`ModelOutput`): model outputs with hidden states and attentions.
        """

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids=input_ids,
            patch_images=patch_images,
            patch_images_2=patch_images_2,
            patch_masks=patch_masks,
            token_embeddings=token_embeddings,
            sample_patch_num=sample_patch_num,
            decoder_input_ids=decoder_input_ids,
            code_masks=code_masks,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        lm_logits = outputs[0]

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

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

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids=None,
        past=None,
        attention_mask=None,
        code_masks=None,
        use_cache=False,
        encoder_outputs=None,
        **kwargs
    ):
        # if attention_mask is None:
        attention_mask = decoder_input_ids.new_zeros(decoder_input_ids.shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,
            "patch_images": None,
            "patch_images_2": None,
            "patch_masks": None,
            "token_embeddings": None,
            "sample_patch_num": None,
            "attention_mask": attention_mask,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "code_masks": code_masks,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache", "attention_mask"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        if encoder_kwargs.get("patch_masks") is None:
            encoder_kwargs["patch_masks"] = torch.tensor([True])

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["attention_mask"] = None

        return model_kwargs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            encoder_outputs["position_embedding"] = encoder_outputs.position_embedding.index_select(
                0, expanded_return_idx.to(encoder_outputs.position_embedding.device)
            )
            encoder_outputs["padding_mask"] = encoder_outputs.padding_mask.index_select(
                0, expanded_return_idx.to(encoder_outputs.padding_mask.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
