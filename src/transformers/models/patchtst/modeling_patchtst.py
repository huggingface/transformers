# coding=utf-8
# Copyright 2023 IBM & Hugging Face. All rights reserved.
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
""" PyTorch PatchTST model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...trainer_utils import set_seed
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSTConfig"

PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtst-etth1-pretrain",
    # See all PatchTST models at https://huggingface.co/models?filter=patchtst
]


# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->PatchTST
class PatchTSTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
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
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PatchTSTTranspose(nn.Module):
    """
    Transpose the tensor to the dimension defined in **dims**

    Parameters:
        dims (`list`): list of dimensions to be transposed
        contiguous (`bool`, default to False): if True, the transposed tensor is contiguous
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = dims

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor`): input to be transposed
        Returns:
            `torch.Tensor`: transposed tensor
        """
        if self.contiguous:
            return inputs.transpose(*self.dims).contiguous()
        else:
            return inputs.transpose(*self.dims)


class PatchTSTBatchNorm(nn.Module):
    """
    Parameters:
    Compute batch normalization
        d_model (`int`): model dimension
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.transpose = PatchTSTTranspose(1, 2)
        self.batchnorm = nn.BatchNorm1d(self.d_model)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor`: tensor
        """
        output = self.transpose(inputs)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        output = self.transpose(output)  # output: (batch_size, sequence_length, d_model)
        return output


def positional_encoding(position_embedding_type, learned, q_len, d_model):
    # Positional encoding
    if position_embedding_type is None:
        # position_embedding_type = None and learned = False can be used to measure impact of positional encoding
        position_enc = torch.empty((q_len, d_model))
        nn.init.uniform_(position_enc, -0.02, 0.02)
        learned = False
    elif position_embedding_type == "zeros":
        position_enc = torch.empty((q_len, d_model))
        nn.init.uniform_(position_enc, -0.02, 0.02)
    elif position_embedding_type == "normal":
        position_enc = torch.zeros((q_len, 1))
        nn.init.normal_(position_enc, mean=0.0, std=0.1)
    elif position_embedding_type == "uniform":
        position_enc = torch.zeros((q_len, 1))
        nn.init.uniform_(position_enc, a=0.0, b=0.1)
    elif position_embedding_type == "sincos":
        position_enc = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        position_enc[:, 0::2] = torch.sin(position * div_term)
        position_enc[:, 1::2] = torch.cos(position * div_term)
        position_enc = position_enc - position_enc.mean()
        position_enc = position_enc / (position_enc.std() * 10)
    else:
        raise ValueError(
            f"{position_embedding_type} is not a valid positional encoder. Available types are 'normal', 'zeros', 'zero', uniform', 'sincos', None."
        )
    return nn.Parameter(position_enc, requires_grad=learned)


def random_masking(
    inputs: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False,
    mask_value: int = 0,
    seed_number: Optional[int] = None,
):
    """random_masking: Mask the input considering the control variables.

    Args:
        inputs (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, num_features)`):
            The input tensor to mask.
        mask_ratio (`float`):
            Mask ratio.
        unmasked_channel_indices (list, *optional*):
            indices of unmasked channels. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, *optional* defaults to False):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels. Defaults to False.
        mask_value (int, *optional* defaults to 0):
            Value to use for masking.
        seed_number (int, *optional*):
            Value to set for the random seed.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as input Tensor and mask tensor of shape [bs x c x
        n]
    """
    if seed_number:
        set_seed(seed_number)

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    device = inputs.device

    len_keep = int(sequence_length * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(batch_size, 1, sequence_length, device=device)  # noise in [0, 1], bs x 1 x  L
        noise = noise.repeat(1, num_channels, 1)  # bs x num_channels x time
    else:
        noise = torch.rand(
            batch_size, num_channels, sequence_length, device=device
        )  # noise in [0, 1], bs x num_channels x L

    mask = torch.ones(
        batch_size, num_channels, sequence_length, device=device
    )  # mask: [bs x num_channels x num_patch]
    mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x num_channels x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


def forecast_masking(
    inputs: torch.Tensor,
    forecast_mask_patches: list,
    forecast_mask_ratios: list = None,
    unmasked_channel_indices: list = None,
    mask_value: int = 0,
    seed_number: Optional[int] = None,
):
    """Forecast masking that masks the last K patches where K is from the forecast_mask_patches list.
    For every batch, distribute the patch lengths based on forecast_mask_ratios and ignore masks for column indices
    mentioned in unmasked_channel_indices.

    Parameters:
        inputs (`torch.Tensor`):
            Input of shape `(bs, num_channels, num_patch, patch_len)` or `(bs, tsg1, tag2, num_channels, num_patch,
            patch_len)`
        forecast_mask_patches (`list`): [2, 4]
            List of patch lengths to mask in the end of the data.
        forecast_mask_ratios (`list`, *optional*): [0.7, 0.3]
            List of weights to use for each patch length. For Ex. if forecast_mask_patches is [5,4] and
            forecast_mask_ratios is [1,1], then equal weights to both patch lengths. Defaults to None.
        unmasked_channel_indices (`list`, *optional*):
            Control Variable channel indices. These channels will not be masked. Defaults to None.
        mask_value (`int`, *optional* defaults to 0):
            Value to use for masking. Defaults to 0.
        seed_number (`int`, *optional*):
            Value to set for the random seed.

    Returns:
        `tuple(torch.Tensor)`: inputs_mask, masked input, same shape as inputs Tensor and Mask tensor of shape `(bs,
        num_channels , num_patch)` or `(bs, tsg1, tsg2, num_channels, num_patch)`
    """
    if seed_number:
        set_seed(seed_number)

    if forecast_mask_ratios is None:
        forecast_mask_ratios = [1 for _ in forecast_mask_patches]

    batch_size, num_channels, sequence_length, num_features = inputs.shape
    mask = torch.zeros(batch_size, num_channels, sequence_length, device=inputs.device)

    t_list = []
    total_length = 0
    total_ratio = sum(forecast_mask_ratios)

    for patch_length, ratio in zip(forecast_mask_patches, forecast_mask_ratios):
        if patch_length <= 0 or patch_length >= sequence_length:
            raise Exception("masked_patch_len should be greater than 0 and less than total patches.")
        temp_len = int(batch_size * ratio / total_ratio)
        t_list.append([patch_length, ratio, temp_len])
        total_length += temp_len

    t_list = sorted(t_list, key=lambda x: x[2])

    if total_length < batch_size:
        t_list[0][2] = t_list[0][2] + (batch_size - total_length)
    elif total_length > batch_size:
        t_list[-1][2] = t_list[-1][2] + (total_length - batch_size)

    batch1 = 0
    for patch_len, _, temp_len in t_list:
        batch2 = batch1 + temp_len
        mask[batch1:batch2, :, -patch_len:] = 1
        batch1 = batch2

    perm = torch.randperm(mask.shape[0])
    mask = mask[perm]

    mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_features)  # mask: [bs x num_channels x num_patch x patch_len]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    inputs_mask = inputs.masked_fill(mask.bool(), mask_value)
    return inputs_mask, mask[..., 0]


class PatchTSTPatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.stride = config.stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.stride + 1
        new_sequence_length = self.patch_length + self.stride * (num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input to be patchified

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )

        output = past_values[:, self.sequence_start :, :]  # output: [bs x new_sequence_length x num_channels]
        output = output.unfold(
            dimension=-2, size=self.patch_length, step=self.stride
        )  # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.transpose(
            -2, -3
        ).contiguous()  # output: [bs x num_input_channels x num_patches x patch_length]
        return output


class PatchTSTMasking(nn.Module):
    """
    Class to perform random or forecast masking.

    Parameters:
        mask_type (`str`, *optional*): Masking type. Allowed values are random, forecast. Defaults to random.
        random_mask_ratio (`float`, *optional*): Mask ratio for random pretraining.
        forecast_mask_patches (`list`, *optional*): List of patch lengths to mask in the end of the data.
        forecast_mask_ratios (`list`, *optional*):
            List of weights to use for each patch length. For Ex. if forecast_mask_patches is [5,4] and
            forecast_mask_ratios is [1,1], then equal weights to both patch lengths. Defaults to None.
        unmasked_channel_indices (`list`, *optional*):
            Define what channels not to mask. These channels will not be masked during pretrainin. Defaults to None.
        channel_consistent_masking (`bool`, *optional*):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels. Defaults to True.
        mask_value (`int`, *optional*): Value to use for masking. Defaults to 0.
        seed_number (`int`, *optional*): Random seed, when None seed is not set. Defaults to None.

    Returns:
        x_mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
        mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
            Bool tensor indicating True on masked points

    """

    def __init__(
        self,
        mask_type: str = "random",
        random_mask_ratio: float = 0.5,
        forecast_mask_patches: list = [2, 3],
        forecast_mask_ratios: list = [1, 1],
        channel_consistent_masking: bool = False,
        unmasked_channel_indices: list = None,
        mask_value: int = 0,
        seed_number: Optional[int] = None,
    ):
        super().__init__()
        self.random_mask_ratio = random_mask_ratio
        self.channel_consistent_masking = channel_consistent_masking
        self.mask_type = mask_type
        self.forecast_mask_patches = forecast_mask_patches
        self.forecast_mask_ratios = forecast_mask_ratios
        self.unmasked_channel_indices = unmasked_channel_indices
        self.mask_value = mask_value
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices.sort()
        self.seed_number = seed_number

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input

        Return:
            masked_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`)
                Masked patched input
            mask (`torch.Tensor` of shape `(batch_size, num_channels, num_patches)`)
                Bool tensor indicating True on masked points

        """

        if self.mask_type == "random":
            masked_input, mask = random_masking(
                inputs=patch_input,
                mask_ratio=self.random_mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
                seed_number=self.seed_number,
            )
        elif self.mask_type == "forecast":
            masked_input, mask = forecast_masking(
                inputs=patch_input,
                forecast_mask_patches=self.forecast_mask_patches,
                forecast_mask_ratios=self.forecast_mask_ratios,
                unmasked_channel_indices=self.unmasked_channel_indices,
                mask_value=self.mask_value,
                seed_number=self.seed_number,
            )
        else:
            raise Exception("Invalid mask type")

        mask = mask.bool()  # mask: [bs x num_input_channels x num_patch]

        return masked_input, mask


class PatchTSTEncoderLayer(nn.Module):
    """
    PatchTST encoder layer
    """
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.channel_attention = config.channel_attention

        # Multi-Head attention
        self.self_attn = PatchTSTAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )

        # Add & Norm of the sublayer 1
        self.dropout_path1 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer1 = PatchTSTBatchNorm(config.d_model)
        else:
            self.norm_sublayer1 = nn.LayerNorm(config.d_model)

        # Add & Norm of the sublayer 2
        if self.channel_attention:
            self.dropout_path2 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
            if "batch" in config.norm.lower():
                self.norm_sublayer2 = PatchTSTBatchNorm(config.d_model)
            else:
                self.norm_sublayer2 = nn.LayerNorm(config.d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.encoder_ffn_dim, bias=config.bias),
            ACT2CLS[config.activation_function](),
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(config.encoder_ffn_dim, config.d_model, bias=config.bias),
        )

        # Add & Norm of sublayer 3
        self.dropout_path3 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer3 = PatchTSTBatchNorm(config.d_model)
        else:
            self.norm_sublayer3 = nn.LayerNorm(config.d_model)

        self.pre_norm = config.pre_norm

    def forward(self,
                hidden_state: torch.Tensor,
                output_attentions: Optional[bool] = None):
        """
        Parameters:
            hidden_state (`torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`, *required*):
                Past values of the time series
        Return:
            `torch.Tensor` of shape `(batch_size, num_channels, sequence_length, d_model)`

        """
        batch_size, num_input_channels, sequence_length, d_model = hidden_state.shape

        # First sublayer: attention across time
        hidden_state = hidden_state.view(
            batch_size * num_input_channels, sequence_length, d_model
        )  # hidden_states: [(bs*num_channels) x sequence_length x d_model]

        if self.pre_norm:
            ## Norm and Multi-Head attention and Add residual connection
            attn_output, attn_weights, _ = self.self_attn(
                hidden_states=self.norm_sublayer1(hidden_state),
                output_attentions=output_attentions
            )
            hidden_state = hidden_state + self.dropout_path1(
                attn_output
            )  # Add: residual connection with residual dropout
        else:
            ## Multi-Head attention and Add residual connection and Norm - Standard Transformer from BERT
            attn_output, attn_weights, _ = self.self_attn(
                hidden_states=hidden_state,
                output_attentions=output_attentions
            )
            hidden_state = self.norm_sublayer1(
                hidden_state + self.dropout_path1(attn_output)
            )  # hidden_states: [(bs*num_channels) x sequence_length x d_model]

        hidden_state = hidden_state.reshape(
            batch_size, num_input_channels, sequence_length, d_model
        )  # [bs x num_channels x sequence_length x d_model]

        # second sublayer: attention across variable at any given time
        # [bs x num_channels x sequence_length x d_model] -> [bs x sequence_length x num_channels x d_model]
        #                                                 -> [(bs*sequence_length) x num_channels x d_model]
        if self.channel_attention:
            hidden_state = (
                hidden_state.transpose(2, 1)
                .contiguous()
                .view(batch_size * sequence_length, num_input_channels, d_model)
            )  # [(bs*sequence_length) x num_channels x d_model]
            if self.pre_norm:
                ## Norm and Multi-Head attention and Add residual connection
                attn_output, channel_attn_weights, _ = self.self_attn(
                    hidden_states=self.norm_sublayer2(hidden_state),
                    output_attentions=output_attentions
                )
                hidden_state = hidden_state + self.dropout_path2(
                    attn_output
                )  # Add: residual connection with residual dropout
            else:
                ## Multi-Head attention and Add residual connection and Norm
                attn_output, channel_attn_weights, _ = self.self_attn(
                    hidden_states=hidden_state, output_attentions=output_attentions
                )
                hidden_state = self.norm_sublayer2(
                    hidden_state + self.dropout_path2(attn_output)
                )  # hidden_states: [(bs*sequence_length) x num_channels x d_model]

            hidden_state = (
                hidden_state.reshape(batch_size, sequence_length, num_input_channels, d_model)
                .transpose(1, 2)
                .contiguous()
            )  # src: [bs x num_channels x sequence_length x d_model]

        # Third sublayer: mixing across hidden
        hidden_state = hidden_state.view(
            batch_size * num_input_channels, sequence_length, d_model
        )  # src: [(batch_size*num_channels) x sequence_length x d_model]
        if self.pre_norm:
            ## Norm and Position-wise Feed-Forward and Add residual connection
            hidden_state = hidden_state + self.dropout_path3(
                self.ff(self.norm_sublayer3(hidden_state))
            )  # Add: residual connection with residual dropout
        else:
            ## Position-wise Feed-Forward and Add residual connection and Norm
            hidden_state = self.norm_sublayer3(
                hidden_state + self.dropout_path3(self.ff(hidden_state))
            )  # Add: residual connection with residual dropout

        hidden_state = hidden_state.reshape(
            batch_size, num_input_channels, sequence_length, d_model
        )  # [bs x num_channels x sequence_length x d_model]

        outputs = (hidden_state,)

        if output_attentions:
            outputs += (attn_weights, channel_attn_weights)

        return outputs


class PatchTSTPreTrainedModel(PreTrainedModel):
    config_class = PatchTSTConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""
        if self.config.use_cls_token:
            nn.init.normal_(self.config.cls_token, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (PatchTSTEncoder)):
            module.gradient_checkpointing = value


class PatchTSTEmbedding(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not config.shared_embedding:
            self.input_embedding = nn.ModuleList()
            for _ in range(config.num_input_channels):
                self.input_embedding.append(nn.Linear(config.patch_length, config.d_model))
        else:
            self.input_embedding = nn.Linear(config.patch_length, config.d_model)

    def forward(self, patch_input: torch.Tensor):
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Patch input for embedding
        return:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, d_model)`
        """
        # Input encoding
        num_input_channels = patch_input.shape[1]
        if isinstance(self.input_embedding, nn.ModuleList):
            embeddings = [ self.input_embedding[i](patch_input[:, i, :, :]) for i in range(num_input_channels)]
            embeddings = torch.stack(embeddings, dim=1)
        else:
            embeddings = self.input_embedding(patch_input)  # x: [bs x num_channels  x num_patches x d_model]
        return embeddings


class PatchTSTEncoder(PatchTSTPreTrainedModel):
    """
    PatchTST Encoder
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.num_input_channels = config.num_input_channels
        self.num_patches = config.num_patches
        self.patch_length = config.patch_length
        self.d_model = config.d_model
        self.shared_embedding = config.shared_embedding
        self.use_cls_token = config.use_cls_token
        self.gradient_checkpointing = False

        # Input embedding: projection of feature vectors onto a d-dim vector space
        self.embedder = PatchTSTEmbedding(config)

        # Positional encoding
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
            self.position_enc = positional_encoding(
                config.positional_encoding, config.learn_pe, config.num_patches + 1, config.d_model
            )
        else:
            self.position_enc = positional_encoding(
                config.positional_encoding, config.learn_pe, config.num_patches, config.d_model
            )

        # Positional dropout
        self.positional_dropout = (
            nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()
        )

        # Encoder
        self.layers = nn.ModuleList([PatchTSTEncoderLayer(config) for i in range(config.encoder_layers)])

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        patch_input: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> BaseModelOutput:
        """
        Parameters:
            patch_input (`torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`, *required*):
                Past values of the time series
            output_hidden_states (bool, optional): Indicates if hidden states should be outputted.
            output_attentions (bool, optional): Indicates if attentions should be outputted.

        return:
            `BaseModelOutput`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # Input embedding
        patch_input = self.embedder(patch_input)

        if self.use_cls_token:
            # x: [bs x num_channels x num_patches x d_model]
            patch_input = self.positional_dropout(patch_input + self.position_enc[1:, :])
            # append cls token
            cls_token = self.cls_token + self.position_enc[:1, :]  # cls_token: [1 x 1 x 1 x d_model]
            cls_tokens = cls_token.expand(patch_input.shape[0], -1, -1)  # get the same copy for all the batch samples
            hidden_state = torch.cat(
                (cls_tokens, patch_input), dim=1
            )  # x: [bs x num_channels x (num_patches+1) x d_model]
        else:
            hidden_state = self.positional_dropout(
                patch_input + self.position_enc
            )  # x: [bs x num_channels x num_patches x d_model]

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_state,)

            layer_outputs = encoder_layer(
                hidden_state=hidden_state,
                output_attentions=output_attentions,
            )
            # get hidden state
            hidden_state = layer_outputs[0]  # hidden_states: [bs x num_channels x num_patches x d_model]
            # or [bs x num_channels x (num_patches+1) x d_model] if use cls_token
            # append layer attention
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # return past_values, hidden_states
        return BaseModelOutput(last_hidden_state=hidden_state,
                               hidden_states=encoder_states,
                               attentions=all_attentions)


PATCHTST_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PatchTSTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PATCHTST_INPUTS_DOCSTRING = r"""
    Parameters:
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, num_input_channels)`):
            Past values of the time series, that serve as context in order to predict the future. The sequence size of
            this tensor must be larger than the `context_length` of the model, since the model will use the larger size
            to construct lag features, i.e. additional values from the past which are added in order to serve as "extra
            context".

            The `sequence_length` here is equal to `config.context_length`

            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
            `static_categorical_features`, `static_real_features`).

            For multivariate time series, the `num_input_channels` > 1 dimension is required and corresponds to the
            number of variates in the time series per time step.

        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, num_input_channels)`, *optional*):
            Future values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`.

            The sequence length here is equal to `prediction_length`.

            See the demo notebook and code snippets for details.

            For multivariate time series, the `num_input_channels` > 1 dimension is required and corresponds to the
            number of variates in the time series per time step.

        output_hidden_states (`bool`, *optional*, default to False):
            Whether or not to return the hidden states of all layers.
"""


@dataclass
class PatchTSTModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Parameters:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            patched input to the Transformer
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*)
            Bool masked tensor indicating which patches are masked
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*)
            mean of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*)
            std of the input data (batch_size, sequence_length, num_channels) over the sequence_length
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    patch_input: torch.FloatTensor = None
    mask: torch.FloatTensor = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class PatchTSTForPretrainingOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPretraining`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction outputs of the time series modeling heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PatchTSTForPredictionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPredictiontion`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction outputs of the time series modeling heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PatchTSTForRegressionOutput(ModelOutput):
    """
    Output type of [`PatchTSTForRegression`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.
        prediction_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction outputs of the time series modeling heads.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class PatchTSTForForecastingOutput(ModelOutput):
    """
    Output type of [`PatchTSTForForecasting`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            MSE loss.

        forecast_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Forecasting outputs of the time series modeling heads.

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    forecast_outputs: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class PatchTSTForClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSTForClassification`].

    Parameters:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SamplePatchTSTPredictionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Parameters:
        sequences `(batch_size, num_samples, prediction_length, num_output_channels)`):
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


@dataclass
class SamplePatchTSTForecastOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Parameters:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, prediction_length)` or `(batch_size,
        num_samples, prediction_length, number_channels)`):
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


@dataclass
class SamplePatchTSTRegressionOutput(ModelOutput):
    """
    Base class for time series model's predictions outputs that contains the sampled values from the chosen
    distribution.

    Parameters:
        sequences (`torch.FloatTensor` of shape `(batch_size, num_samples, num_output_channels)`
                Sampled values from the chosen distribution.
    """

    sequences: torch.FloatTensor = None


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.nll
def nll(input: torch.distributions.Distribution, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the negative log likelihood loss from input distribution with respect to target.
    """
    return -input.log_prob(target)


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.weighted_average
def weighted_average(input_tensor: torch.Tensor, weights: Optional[torch.Tensor] = None, dim=None) -> torch.Tensor:
    """
    Computes the weighted average of a given tensor across a given `dim`, masking values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        input_tensor (`torch.FloatTensor`):
            Input tensor, of which the average must be computed.
        weights (`torch.FloatTensor`, *optional*):
            Weights tensor, of the same shape as `input_tensor`.
        dim (`int`, *optional*):
            The dim along which to average `input_tensor`.

    Returns:
        `torch.FloatTensor`: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, input_tensor * weights, torch.zeros_like(input_tensor))
        sum_weights = torch.clamp(weights.sum(dim=dim) if dim else weights.sum(), min=1.0)
        return (weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()) / sum_weights
    else:
        return input_tensor.mean(dim=dim)


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesStdScaler with TimeSeries->PatchTST
class PatchTSTStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along some given dimension `dim`, and then normalizes it
    by subtracting from the mean and dividing by the standard deviation.

    Args:
        dim (`int`):
            Dimension along which to calculate the mean and standard deviation.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        minimum_scale (`float`, *optional*, defaults to 1e-5):
            Default scale that is used for elements that are constantly zero along dimension `dim`.
    """

    def __init__(self, dim: int, keepdim: bool = False, minimum_scale: float = 1e-5):
        super().__init__()
        if not dim > 0:
            raise ValueError("Cannot compute scale along dim = 0 (batch dimension), please provide dim > 0")
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    @torch.no_grad()
    def forward(self, data: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denominator = weights.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * weights).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * weights) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesMeanScaler with TimeSeries->PatchTST
class PatchTSTMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along dimension `dim`, and scales the data
    accordingly.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
        default_scale (`float`, *optional*, defaults to `None`):
            Default scale that is used for elements that are constantly zero. If `None`, we use the scale of the batch.
        minimum_scale (`float`, *optional*, defaults to 1e-10):
            Default minimum possible scale that is used for any item.
    """

    def __init__(
        self, dim: int = -1, keepdim: bool = True, default_scale: Optional[float] = None, minimum_scale: float = 1e-10
    ):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale
        self.default_scale = default_scale

    @torch.no_grad()
    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: (N, [C], T=1)
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


# Copied from transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesNOPScaler with TimeSeries->PatchTST
class PatchTSTNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along dimension `dim`, and therefore applies no scaling to the input data.

    Args:
        dim (`int`):
            Dimension along which to compute the scale.
        keepdim (`bool`, *optional*, defaults to `False`):
            Controls whether to retain dimension `dim` (of length 1) in the scale tensor, or suppress it.
    """

    def __init__(self, dim: int, keepdim: bool = False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


@add_start_docstrings(
    "The bare PatchTST Model outputting raw hidden-states without any specific head.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        if config.scaling == "mean" or config.scaling is True:
            self.scaler = PatchTSTMeanScaler(dim=1, keepdim=True)
        elif config.scaling == "std":
            self.scaler = PatchTSTStdScaler(dim=1, keepdim=True)
        else:
            self.scaler = PatchTSTNOPScaler(dim=1, keepdim=True)

        self.patchifier = PatchTSTPatchify(config)
        self.mask_input = config.mask_input

        if self.mask_input:
            self.masking = PatchTSTMasking(
                mask_type=config.mask_type,
                random_mask_ratio=config.random_mask_ratio,
                forecast_mask_patches=config.forecast_mask_patches,
                forecast_mask_ratios=config.forecast_mask_ratios,
                channel_consistent_masking=config.channel_consistent_masking,
                unmasked_channel_indices=config.unmasked_channel_indices,
                mask_value=config.mask_value,
                seed_number=config.seed_number,
            )
        else:
            self.masking = nn.Identity()
        self.encoder = PatchTSTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if past_observed_mask is None:
            past_observed_mask = torch.ones_like(past_values)

        # x: tensor [bs x sequence_length x num_input_channels]
        scaled_past_values, loc, scale = self.scaler(past_values, past_observed_mask)

        # patched_values: [bs x num_input_channels x num_patches x patch_length] for pretrain
        patched_values = self.patchifier(scaled_past_values)
        if self.mask_input:
            masked_values, mask = self.masking(patched_values)
        else:
            masked_values, mask = self.masking(patched_values), None

        encoder_output = self.encoder(
            patch_input=masked_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        if not return_dict:
            outputs = (encoder_output.last_hidden_state, encoder_output.hidden_states, encoder_output.attentions)
            outputs = outputs + (patched_values, mask, loc, scale)
            return tuple(v for v in outputs if v is not None)

        return PatchTSTModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            attentions=encoder_output.attentions,
            patch_input=patched_values,
            mask=mask,
            loc=loc,
            scale=scale,
        )


class MaskPretrainHead(nn.Module):
    """
    Pretraining head for mask modelling
    """

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.d_model, config.patch_length)
        self.use_cls_token = config.use_cls_token

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)`
                    or `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                    Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                            `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True

        """
        embedding = self.linear(self.dropout(embedding))  # [bs x num_channels x num_patches x patch_length]
        if self.use_cls_token:
            embedding = embedding[:, :, 1:, :]  # remove the first cls token
        return embedding


class PatchTSTForPretraining(PatchTSTPreTrainedModel):
    """
    Mask pretrain model: PatchTST model + pretrain head
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        config.mask_input = True
        self.model = PatchTSTModel(config=config)
        self.head = MaskPretrainHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTForPretrainingOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForPretrainingOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # past_values: [bs x num_channels x num_patches x d_model] or
        # [bs x num_channels x (num_patches+1) x d_model] if use cls_token
        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        # model_output[0]: [bs x num_channels x num_patches x patch_length] or
        # [bs x num_channels x (num_patches+1) x patch_length] if use cls_token
        x_hat = self.head(model_output[0])

        # calculate masked_loss
        loss = nn.MSELoss(reduction="none")
        loss_val = loss(x_hat, model_output.patch_input)
        masked_loss = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        encoder_states = model_output.hidden_states
        if not return_dict:
            outputs = (masked_loss, x_hat, model_output.hidden_states, model_output.attentions)
            return tuple(v for v in outputs if v is not None)
        return PatchTSTForPretrainingOutput(loss=masked_loss,
                                            prediction_output=x_hat,
                                            hidden_states=encoder_states,
                                            attentions=model_output.attentions
                                            )


class PatchTSTForClassification(PatchTSTPreTrainedModel):
    """
    PatchTST model for classification. The model contains PatchTST model + classification head
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.model = PatchTSTModel(config)
        self.head = PatchTSTClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        labels: torch.Tensor = None,
        past_observed_mask: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, PatchTSTForClassificationOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            labels (`torch.Tensor`, *optional*): labels associates with the `past_values`
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForClassificationOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        y_hat = self.head(model_output[0])

        loss_val = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()
            loss_val = loss(y_hat, labels)

        if not return_dict:
            outputs = (loss_val, y_hat, model_output.hidden_states, model_output.attentions)
            return tuple(v for v in outputs if v is not None)
        return PatchTSTForClassificationOutput(loss=loss_val,
                                               prediction_logits=y_hat,
                                               hidden_states=model_output.hidden_states,
                                               attentions=model_output.attentions
                                               )


class PatchTSTClassificationHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(config.num_input_channels * config.d_model, config.num_labels)

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)`
                    or `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                    Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_labels)`

        """
        if self.use_cls_token:
            x = embedding[:, :, 0, :]  # use the first output token, x: bs x num_channels x d_model
        elif self.pooling == "mean":
            x = embedding.mean(dim=2)  # x: [bs x num_channels x d_model]
        elif self.pooling == "max":
            x = embedding.max(dim=2)  # x: [bs x num_channels x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")

        x = self.flatten(x)  # x: bs x num_channels * d_model
        y = self.linear(self.dropout(x))  # y: bs x n_classes
        return y


class PatchTSTPredictionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig, distribution_output=None):
        super().__init__()

        self.num_output_channels = config.num_output_channels
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling

        head_dim = config.num_input_channels * config.d_model

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

        if distribution_output is None:
            self.projection = nn.Linear(head_dim, config.prediction_length * config.num_output_channels)
        else:
            self.projection = distribution_output.get_parameter_projection(head_dim)

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)`
                    or `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                    Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, pred_len, num_output_channels)`

        """
        batch_size = embedding.shape[0]
        if self.use_cls_token:
            x = embedding[:, :, 0, :]  # use the first output token, x: [bs x num_channels x d_model]
        elif self.pooling == "mean":
            x = embedding.mean(dim=2)  # x: [bs x num_channels x d_model]
        elif self.pooling == "max":
            x = embedding.max(dim=2)  # x: [bs x num_channels x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")

        # flatten the input
        x = self.dropout(self.flatten(x))  # x: bs x (num_channels * d_model)
        # projection
        y = self.projection(x)
        # reshape y
        if isinstance(y, tuple):  # for distribution head
            y = (
                z.reshape(batch_size, -1, self.num_output_channels) for z in y
            )  # tuple of [bs x prediction_len x num_output_channels]
        else:  # for linear head
            y = y.reshape(batch_size, -1, self.num_output_channels)  # [bs x prediction_len x num_output_channels]
        return y


class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    """
    PatchTST model for prediction. The model contains PatchTST model + prediction head
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.model = PatchTSTModel(config)
        if config.loss == "mse":
            self.distribution_output = None
        else:
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.prediction_length * config.num_output_channels)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.prediction_length * config.num_output_channels)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(
                    dim=config.prediction_length * config.num_output_channels
                )
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.head = PatchTSTPredictionHead(config, self.distribution_output)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, PatchTSTForPredictionOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            future_values (`torch.Tensor` of shape `(bs, pred_len, num_output_channels)`, *optional*):
                future target values associates with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForPredictionOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get model output
        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )

        # get output head. y_hat is of shape [bs x pred_len x num_output_channels] or tuple of this shape
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None
        if future_values is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(y_hat)
                loss_val = nll(distribution, future_values)
                # take average of the loss
                loss_val = weighted_average(loss_val)
            else:
                loss = nn.MSELoss(reduction="mean")
                loss_val = loss(y_hat, future_values)

        if not return_dict:
            outputs = (loss_val, y_hat, model_output.hidden_states, model_output.attentions)
            return tuple(v for v in outputs if v is not None)
        return PatchTSTForPredictionOutput(loss=loss_val,
                                           prediction_output=y_hat,
                                           hidden_states=model_output.hidden_states,
                                           attentions=model_output.attentions
                                           )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> SamplePatchTSTPredictionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSTPredictionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, prediction_length, num_output_channels)`
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            target_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )

        # get distribution
        distribution = self.distribution_output.distribution(outputs.prediction_output)
        # get samples
        samples = [
            distribution.sample() for _ in range(num_parallel_samples)
        ]  # samples: list of [bs x pred_len x num_output_channels]
        # stack tensors
        samples = torch.stack(samples, dim=1)  # [bs x num_samples x pred_len x num_output_channels]
        return SamplePatchTSTPredictionOutput(sequences=samples)


class PatchTSTForecastHead(nn.Module):
    def __init__(self, config: PatchTSTConfig, distribution_output=None):
        super().__init__()

        self.shared_projection = config.shared_projection
        self.num_input_channels = config.num_input_channels
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        head_dim = config.d_model if self.pooling else config.d_model * config.num_patches

        if not self.shared_projection:
            # if each channel has its own head
            self.projections = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.num_input_channels):
                self.flattens.append(nn.Flatten(start_dim=2))
                if distribution_output is None:
                    # use linear head
                    self.projections.append(nn.Linear(head_dim, config.prediction_length))
                else:
                    # use distribution head
                    self.projections.append(distribution_output.get_parameter_projection(head_dim))
                self.dropouts.append(nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity())
        else:
            # all the channels share the same head
            self.flatten = nn.Flatten(start_dim=2)
            if distribution_output is None:
                # use linear head
                self.projection = nn.Linear(head_dim, config.prediction_length)
            else:
                # use distribution head
                self.projection = distribution_output.get_parameter_projection(head_dim)
            self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)`
                    or `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                    Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, forecast_len, num_channels)`

        """
        if self.use_cls_token:
            y = embedding[:, :, 0, :]  # y: [bs x num_channels x d_model]
        else:
            if self.pooling == "mean":
                y = embedding.mean(dim=2)  # y: [bs x num_channels x d_model]
            elif self.pooling == "max":
                y = embedding.max(dim=2)  # y: [bs x num_channels x d_model]
            else:
                y = embedding  # y: [bs x num_channels x num_patches x d_model]

        if not self.shared_projection:
            x_out = []
            for i in range(self.num_input_channels):
                z = self.flattens[i](y[:, i, :])  # y: [bs x (d_model * num_patches)] or [bs x d_model)]
                z = self.dropouts[i](z)
                z = self.projections[i](
                    z
                )  # z: [bs x forecast_len]  or tuple ([bs x forecast_len], [bs x forecast_len]) if using distribution head
                x_out.append(z)
            output = torch.stack(x_out, dim=1)  # x: [bs x num_channels x forecast_len]
        else:
            z = self.flatten(y)  # z: [bs x num_channels x (d_model * num_patches)] or [bs x num_channels x d_model)]
            z = self.dropout(z)
            output = self.projection(z)  # output: [bs x num_channels x forecast_len]
            # or tuple ([bs x num_channels x forecast_len], [bs x num_channels x forecast_len]) if using distribution head

        if isinstance(output, tuple):
            output = tuple(
                z.transpose(2, 1) for z in output
            )  # ([bs x forecast_len x num_channels], [bs x forecast_len x num_channels])
        else:
            output = output.transpose(2, 1)  # [bs x forecast_len x num_channels]

        return output


class PatchTSTForForecasting(PatchTSTPreTrainedModel):
    """
    PatchTST for forecasting. The model contains PatchTST model + Forecasting head
    """

    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.model = PatchTSTModel(config)

        if config.loss == "mse":
            self.distribution_output = None
        else:
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.prediction_length)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.prediction_length)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(dim=config.prediction_length)
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.head = PatchTSTForecastHead(config, self.distribution_output)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PatchTSTForForecastingOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            future_values (`torch.Tensor` of shape `(bs, forecast_len, num_input_channels)`, *optional*):
                future target values associates with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForForecastingOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get model output
        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        # get output head
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None

        if future_values is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(
                    y_hat, loc=model_output.loc, scale=model_output.scale
                )
                loss_val = nll(distribution, future_values)
                # take average of the loss
                loss_val = weighted_average(loss_val)
                # for testing
                # loss_val = nn.MSELoss(reduction='none')(distribution.mean, future_values)
                # loss_val = weighted_average(loss_val)
            else:
                y_hat = y_hat * model_output.scale + model_output.loc
                loss = nn.MSELoss(reduction="mean")
                loss_val = loss(y_hat, future_values)

        loc = model_output.loc
        scale = model_output.scale

        if not return_dict:
            outputs = (loss_val, y_hat, model_output.hidden_states, model_output.attentions, loc, scale)
            return tuple(v for v in outputs if v is not None)
        return PatchTSTForForecastingOutput(
            loss=loss_val,
            forecast_outputs=y_hat,
            hidden_states=model_output.hidden_states,
            attentions=model_output.attentions,
            loc=loc,
            scale=scale,
        )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> SamplePatchTSTForecastOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSTForecastOutput`] where the outputs `sequences` tensor will have shape `(batch_size, number
            of samples, prediction_length, 1)` or `(batch_size, number of samples, prediction_length,
            num_input_channels)` for multivariate predictions.
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            future_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )

        # get distribution
        distribution = self.distribution_output.distribution(
            outputs.forecast_outputs, loc=outputs.loc, scale=outputs.scale
        )
        # get samples
        samples = [
            distribution.sample() for _ in range(num_parallel_samples)
        ]  # samples: list of [bs x forecast_len x num_channels]
        # stack tensors
        samples = torch.stack(samples, dim=1)  # [bs x num_samples x forecast_len x num_channels]
        return SamplePatchTSTForecastOutput(sequences=samples)


class PatchTSTRegressionHead(nn.Module):
    """
    Regression head
    """

    def __init__(self, config: PatchTSTConfig, distribution_output=None):
        super().__init__()
        self.y_range = config.prediction_range
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        self.distribution_output = distribution_output

        head_dim = config.num_input_channels * config.d_model

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

        if distribution_output is None:
            self.projection = nn.Linear(head_dim, config.num_output_channels)
        else:
            self.projection = distribution_output.get_parameter_projection(head_dim)

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)`
                    or `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                    Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, output_dim)`

        """
        if self.use_cls_token:
            x = embedding[:, :, 0, :]  # use the first output token, x: [bs x num_channels x d_model]
        elif self.pooling == "mean":
            x = embedding.mean(dim=2)  # x: [bs x num_channels x d_model]
        elif self.pooling == "max":
            x = embedding.max(dim=2)  # x: [bs x num_channels x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")
        # flatten the input
        x = self.dropout(self.flatten(x))  # x: bs x (num_channels * d_model)
        # projection
        y = self.projection(x)  # y: bs x output_dim or a tuple of this shape for distribution head
        #
        if (self.distribution_output is None) & (self.y_range is not None):  # linear head
            y = torch.sigmoid(y) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        return y


class PatchTSTForRegression(PatchTSTPreTrainedModel):
    # PatchTST model + Regression head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.model = PatchTSTModel(config)

        self.model = PatchTSTModel(config)
        if config.loss == "mse":
            self.distribution_output = None
        else:
            if config.distribution_output == "student_t":
                self.distribution_output = StudentTOutput(dim=config.prediction_length * config.num_output_channels)
            elif config.distribution_output == "normal":
                self.distribution_output = NormalOutput(dim=config.prediction_length * config.num_output_channels)
            elif config.distribution_output == "negative_binomial":
                self.distribution_output = NegativeBinomialOutput(
                    dim=config.prediction_length * config.num_output_channels
                )
            else:
                raise ValueError(f"Unknown distribution output {config.distribution_output}")

        self.head = PatchTSTRegressionHead(config, self.distribution_output)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        target_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, PatchTSTForRegressionOutput]:
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(bs, sequence_length, num_input_channels)`, *required*):
                Input sequence to the model
            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
            target_values (`torch.Tensor` of shape `(bs, num_input_channels)`):
                target values associates with the `past_values`
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers
            return_dict (`bool`, *optional*): Whether or not to return a `ModelOutput` instead of a plain tuple.

        Returns:
            `PatchTSTForRegressionOutput` or tuple of `torch.Tensor` (if `return_dict`=False or
            `config.return_dict`=False)

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_output = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        # get output head. y_hat is of shape [bs x num_output_channels] or tuple of this shape
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None
        if target_values is not None:
            if self.distribution_output:
                distribution = self.distribution_output.distribution(y_hat)
                loss_val = nll(distribution, target_values)
                # take average of the loss
                loss_val = weighted_average(loss_val)
            else:
                loss = nn.MSELoss(reduction="mean")
                loss_val = loss(y_hat, target_values)

        if not return_dict:
            outputs = (loss_val, y_hat, model_output.hidden_states, model_output.attentions)
            return tuple(v for v in outputs if v is not None)
        return PatchTSTForRegressionOutput(loss=loss_val,
                                           prediction_output=y_hat,
                                           hidden_states=model_output.hidden_states,
                                           attentions=model_output.attentions
                                           )

    def generate(
        self,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> SamplePatchTSTRegressionOutput:
        """
        Generate sequences of sample predictions from a model with a probability distribution head.

        Parameters:
            past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Past values of the time series that serves as context in order to predict the future.

            past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
                Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
                in `[0, 1]`:

                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Return:
            [`SamplePatchTSTRegressionOutput`] where the outputs `sequences` tensor will have shape `(batch_size,
            number of samples, num_output_channels)`.
        """
        # get number of samples
        num_parallel_samples = self.config.num_parallel_samples

        # get model output
        outputs = self(
            past_values=past_values,
            target_values=None,
            past_observed_mask=past_observed_mask,
            output_hidden_states=False,
        )

        # get distribution
        distribution = self.distribution_output.distribution(outputs.prediction_output)
        # get samples
        samples = [
            distribution.sample() for _ in range(num_parallel_samples)
        ]  # samples: list of [bs x num_output_channels]
        # stack tensors
        samples = torch.stack(samples, dim=1)  # [bs x num_samples x num_output_channels]
        return SamplePatchTSTRegressionOutput(sequences=samples)
