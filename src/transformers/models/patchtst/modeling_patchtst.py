# coding=utf-8
# Copyright 2023 TSFM team. All rights reserved.
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
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.modules.activation import MultiheadAttention

from transformers.modeling_outputs import BaseModelOutputWithNoAttention
from transformers.modeling_utils import PreTrainedModel
from transformers.models.patchtst.configuration_patchtst import PatchTSTConfig
from transformers.utils import ModelOutput, add_start_docstrings, logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PatchTSTConfig"

PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ibm/patchtst-base",
    # See all PatchTST models at https://huggingface.co/models?filter=patchtst
]


class PatchTSTAttention(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.self_attn = MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            bias=config.bias,
            add_bias_kv=True,
            add_zero_attn=False,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: Tensor [bs x q_len x d_model]
        """
        src, _ = self.self_attn(src, src, src, need_weights=False)
        return src


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe is None:
        w_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        w_pos = torch.empty((q_len, 1))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "zeros":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        w_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        w_pos = coord1d_pos_encoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        w_pos = coord1d_pos_encoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        w_pos = coord2d_pos_encoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == "exp2d":
        w_pos = coord2d_pos_encoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == "sincos":
        pos_enc = torch.zeros(q_len, d_model)
        position = torch.arange(0, q_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc - pos_enc.mean()
        pos_enc = pos_enc / (pos_enc.std() * 10)
        w_pos = pos_enc
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
        )
    return nn.Parameter(w_pos, requires_grad=learn_pe)


def coord2d_pos_encoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
            2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )

        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def coord1d_pos_encoding(q_len, exponential=False, normalize=True):
    cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1)) - 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def set_seed(x=42):
    random.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(x)


def random_masking(
    xb: torch.Tensor,
    mask_ratio: float,
    unmasked_channel_indices: list = None,
    channel_consistent_masking: bool = False,
    mask_value=0,
    seed_number: Optional[int] = None,
):
    """random_masking: Mask the input considering the control variables.

    Args:
        xb (Tensor): Input to mask [ bs x nvars x num_patches x patch_length]
        mask_ratio (float): Mask ratio.
        unmasked_channel_indices (list, optional):
            indices of unmasked channels. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, optional):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels. Defaults to True.
        mask_value (int, optional): Value to use for masking. Defaults to 0.
        seed_number (int, optional): Value to set for the random seed.

    Returns:
        Tensor: xb_mask, masked input, same shape as input Tensor: Mask tensor of shape [bs x c x n]
    """
    if seed_number:
        set_seed(seed_number)

    bs, nvars, L, D = xb.shape

    len_keep = int(L * (1 - mask_ratio))

    if channel_consistent_masking:
        noise = torch.rand(bs, 1, L, device=xb.device)  # noise in [0, 1], bs x 1 x  L
        noise = noise.repeat(1, nvars, 1)  # bs x nvars x L
    else:
        noise = torch.rand(bs, nvars, L, device=xb.device)  # noise in [0, 1], bs x nvars x L

    mask = torch.ones(bs, nvars, L, device=xb.device)  # mask: [bs x nvars x num_patch]
    mask[:, :, :len_keep] = 0

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=-1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=-1)  # ids_restore: [bs x nvars x L]

    mask = torch.gather(mask, dim=-1, index=ids_restore)
    mask = mask.unsqueeze(-1).repeat(1, 1, 1, D)  # mask: [bs x nvars x num_patches x patch_length]
    if unmasked_channel_indices is not None:
        mask[:, unmasked_channel_indices, :, :] = 0

    xb_mask = xb.masked_fill(mask.bool(), mask_value)
    return xb_mask, mask[..., 0]


def compute_num_patches(sequence_length, patch_length, stride):
    return (max(sequence_length, patch_length) - patch_length) // stride + 1


class Patchify(nn.Module):
    """
    Args:
    A class to patchify the time series sequence into different patches
        sequence_length (int, required): input sequence length patch_length (int, required): patch length stride (int,
        required): stride between patches
    Returns:
        z: output tensor data [bs x n_vars x num_patches x patch_length]
    """

    def __init__(
        self,
        sequence_length: int,
        patch_length: int,
        stride: int,
        padding: bool = False,  # TODO: use this to set whether we want to pad zeros to the sequence
    ):
        super().__init__()

        assert (
            sequence_length > patch_length
        ), f"Sequence length ({sequence_length}) has to be greater than the patch length ({patch_length})"

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.stride = stride

        # get the number of patches
        self.num_patches = compute_num_patches(sequence_length, patch_length, stride)
        new_sequence_length = patch_length + stride * (self.num_patches - 1)
        self.s_begin = sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Args:
            past_values (torch.Tensor, required): Input of shape [bs x sequence_length x n_vars]
        Returns:
            x: output tensor data [bs x n_vars x num_patches x patch_length]
        """
        sequence_length = past_values.shape[-2]
        assert (
            sequence_length == self.sequence_length
        ), f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."

        x = past_values[:, self.s_begin :, :]  # x: [bs x new_sequence_length x nvars]
        x = x.unfold(
            dimension=-2, size=self.patch_length, step=self.stride
        )  # x: [bs x num_patches x n_vars x patch_length]
        x = x.transpose(-2, -3).contiguous()  # xb: [bs x n_vars x num_patches x patch_length]
        return x


class PatchEmbeddings(nn.Module):
    """
    Args:
    A class to patchify the time series sequence into different patches
        sequence_length (int, required): input sequence length patch_length (int, required): patch length stride (int,
        required): stride between patches
    Returns:
        embeddings: output tensor data [bs x n_vars x num_patches x embed_dim]
    """

    def __init__(self, sequence_length: int, patch_length: int, stride: int, embed_dim: int):
        super().__init__()

        assert (
            sequence_length > patch_length
        ), f"Sequence length ({sequence_length}) has to be greater than the patch length ({patch_length})"

        # assert ((max(sequence_length, patch_length) - patch_length) % stride == 0), f"sequence length minus patch length has to be divisible to the stride"

        self.sequence_length = sequence_length
        self.patch_length = patch_length
        self.stride = stride
        self.embed_dim = embed_dim

        # get the number of patches
        self.num_patches = compute_num_patches(sequence_length, patch_length, stride)
        new_sequence_length = patch_length + stride * (self.num_patches - 1)
        self.s_begin = sequence_length - new_sequence_length

        # Embedding
        self.projection = nn.Conv1d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_length,
            stride=stride,
        )

    def forward(self, past_values: torch.Tensor):
        """
        Args:
            past_values (torch.Tensor, required): Input of shape [bs x sequence_length x n_vars]
        Returns:
            embeddings: output tensor data [bs x n_vars x num_patches x emb_dim]
        """
        bs, sequence_length, n_vars = past_values.shape
        assert (
            sequence_length == self.sequence_length
        ), f"Input sequence length ({sequence_length}) doesn't match the configuration sequence length ({self.sequence_length})."

        x = past_values[:, self.s_begin :, :]  # x: [bs x new_sequence_length x nvars]
        # convert past_values to shape [bs*n_vars x 1 x sequence_length ]
        x = x.transpose(1, 2).reshape(bs * n_vars, 1, -1).contiguous()
        # projection
        embeddings = self.projection(x)  # embeddings: [bs*n_vars x emb_dim x num_patches]
        # reshape
        embeddings = (
            embeddings.transpose(1, 2).view(bs, n_vars, -1, self.embed_dim).contiguous()
        )  # embeddings: [bs x n_vars x num_patches x emb_dim]
        # embeddings = embeddings.flatten(2).transpose(1, 2)
        return embeddings


class PatchMasking(nn.Module):
    """
    PatchMasking: Class to random or forcast masking.

    Args:
        mask_type (str, optional): Masking type. Allowed values are random, forecast. Defaults to random.
        mask_ratio (float, optional): Mask ratio.
        mask_patches (list, optional): List of patch lengths to mask in the end of the data.
        mask_patch_ratios (list, optional): List of weights to use for each patch length. For Ex.
        if patch_lengths is [5,4] and mix_ratio is [1,1], then equal weights to both patch lengths. Defaults to None.
        unmasked_channel_indices (list, optional):
            Control Variable channel indices. These channels will not be masked. Defaults to None.
        channel_consistent_masking (bool, optional):
            When true, masking will be same across all channels of a timeseries. Otherwise, masking positions will vary
            across channels. Defaults to True.
        mask_value (int, optional): Value to use for masking. Defaults to 0.
        seed_number (int, optional): Random seed, when None seed is not set. Defaults to None.
    """

    def __init__(
        self,
        mask_type: str = "random",
        mask_ratio=0.5,
        mask_patches: list = [2, 3],
        mask_patch_ratios: list = [1, 1],
        channel_consistent_masking: bool = False,
        unmasked_channel_indices: list = None,
        mask_value=0,
        seed_number: Optional[int] = None,
    ):
        # if seed_number:
        #     set_seed(seed_number)
        self.mask_ratio = mask_ratio
        self.channel_consistent_masking = channel_consistent_masking
        self.mask_type = mask_type
        self.mask_patches = mask_patches
        self.mask_patch_ratios = mask_patch_ratios
        self.unmasked_channel_indices = unmasked_channel_indices
        self.mask_value = mask_value
        if self.unmasked_channel_indices is not None:
            self.unmasked_channel_indices.sort()
        self.seed_number = seed_number

        super().__init__()

    def forward(self, x: torch.Tensor):
        """
        Input:
            x: patched input
                4D: [bs x n_vars x num_patches x patch_length]

        Output:
            x_mask: Masked patched input
                4D: [bs x n_vars x num_patches x patch_length]
            mask: bool tensor indicating True on masked points
                4D: [bs x n_vars x num_patch]
        """

        if self.mask_type == "random":
            x_mask, mask = random_masking(
                xb=x,
                mask_ratio=self.mask_ratio,
                unmasked_channel_indices=self.unmasked_channel_indices,
                channel_consistent_masking=self.channel_consistent_masking,
                mask_value=self.mask_value,
                seed_number=self.seed_number,
            )

        else:
            raise Exception("Invalid mask type")

        mask = mask.bool()  # mask: [bs x n_vars x num_patch]

        return x_mask, mask


class ChannelAttentionTSTEncoder(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.layers = nn.ModuleList([ChannelAttentionTSTEncoderLayer(config) for i in range(config.encoder_layers)])

    def forward(self, src: torch.Tensor, output_hidden_states: Optional[bool] = None):
        """
        src: tensor [bs x nvars x sequence_length x d_model] Return:
            Tensor [bs x nvars x sequence_length x d_model]
        """
        all_hidden_states = []
        for mod in self.layers:
            if output_hidden_states:
                src = mod(src)
                all_hidden_states.append(src)
        if output_hidden_states:
            return src, all_hidden_states
        return src, None


class ChannelAttentionTSTEncoderLayer(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        # Multi-Head attention
        self.self_attn = PatchTSTAttention(config)

        # Add & Norm of the sublayer 1
        self.dropout_path1 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
        else:
            self.norm_sublayer1 = nn.LayerNorm(config.d_model)

        # Add & Norm of the sublayer 2
        self.dropout_path2 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
        else:
            self.norm_sublayer2 = nn.LayerNorm(config.d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.encoder_ffn_dim, bias=config.bias),
            get_activation_fn(config.activation_function),
            nn.Dropout(config.ff_dropout) if config.ff_dropout > 0 else nn.Identity(),
            nn.Linear(config.encoder_ffn_dim, config.d_model, bias=config.bias),
        )

        # Add & Norm of sublayer 3
        self.dropout_path3 = nn.Dropout(config.dropout_path) if config.dropout_path > 0 else nn.Identity()
        if "batch" in config.norm.lower():
            self.norm_sublayer3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(config.d_model), Transpose(1, 2))
        else:
            self.norm_sublayer3 = nn.LayerNorm(config.d_model)

        self.pre_norm = config.pre_norm
        self.store_attn = config.store_attention

    def forward(self, src: torch.Tensor):
        """
        src: tensor [bs x nvars x sequence_length x d_model] Return:
            Tensor [bs x nvars x sequence_length x d_model]
        """
        bs, n_vars, sequence_length, d_model = src.shape

        # First sublayer: attention across time
        src = src.view(bs * n_vars, sequence_length, d_model)  # src: [(bs*nvars) x sequence_length x d_model]
        if self.pre_norm:
            ## Norm and Multi-Head attention and Add residual connection
            src = src + self.dropout_path1(
                self.self_attn(self.norm_sublayer1(src))
            )  # Add: residual connection with residual dropout
        else:
            ## Multi-Head attention and Add residual connection and Norm - Standard Transformer from BERT
            src = self.norm_sublayer1(
                src + self.dropout_path1(self.self_attn(src))
            )  # src: [(bs*nvars) x sequence_length x d_model]
        src = src.reshape(bs, n_vars, sequence_length, d_model)  # [bs x nvars x sequence_length x d_model]

        # second sublayer: attention across variable at any given time
        # [bs x nvars x sequence_length x d_model] -> [bs x sequence_length x nvars x d_model] -> [(bs*sequence_length) x nvars x d_model]
        src = (
            src.transpose(2, 1).contiguous().view(bs * sequence_length, n_vars, d_model)
        )  # [(bs*sequence_length) x nvars x d_model]
        if self.pre_norm:
            ## Norm and Multi-Head attention and Add residual connection
            src = src + self.dropout_path2(
                self.self_attn(self.norm_sublayer2(src))
            )  # Add: residual connection with residual dropout
        else:
            ## Multi-Head attention and Add residual connection and Norm - Standard Transformer from BERT
            src = self.norm_sublayer2(
                src + self.dropout_path2(self.self_attn(src))
            )  # src: [(bs*sequence_length) x nvars x d_model]
        src = (
            src.reshape(bs, sequence_length, n_vars, d_model).transpose(1, 2).contiguous()
        )  # src: [bs x nvars x sequence_length x d_model]

        # Third sublayer: mixing across hidden
        src = src.view(bs * n_vars, sequence_length, d_model)  # src: [(bs*nvars) x sequence_length x d_model]
        if self.pre_norm:
            ## Norm and Position-wise Feed-Forward and Add residual connection
            src = src + self.dropout_path3(
                self.ff(self.norm_sublayer3(src))
            )  # Add: residual connection with residual dropout
        else:
            ## Position-wise Feed-Forward and Add residual connection and Norm - Standard Transformer from BERT
            src = self.norm_sublayer3(
                src + self.dropout_path3(self.ff(src))
            )  # Add: residual connection with residual dropout
        src = src.reshape(bs, n_vars, sequence_length, d_model)  # [bs x nvars x sequence_length x d_model]

        return src


class PatchTSTPreTrainedModel(PreTrainedModel):
    config_class = PatchTSTConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize weights"""
        if self.config.use_cls_token:
            torch.nn.init.normal_(self.config.cls_token, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=self.config.init_std)
            module.bias_k.data.normal_(mean=0.0, std=self.config.init_std)
            module.bias_v.data.normal_(mean=0.0, std=self.config.init_std)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (ChannelAttentionPatchTSTEncoder)):
            module.gradient_checkpointing = value


class ChannelAttentionPatchTSTEncoder(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.n_vars = config.input_size
        self.num_patches = config.num_patches
        self.patch_length = config.patch_length
        self.d_model = config.d_model
        self.shared_embedding = config.shared_embedding
        self.use_cls_token = config.use_cls_token
        self.gradient_checkpointing = False

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not config.shared_embedding:
            self.w_p = nn.ModuleList()
            for _ in range(self.n_vars):
                self.w_p.append(nn.Linear(config.patch_length, config.d_model))
        else:
            self.w_p = nn.Linear(config.patch_length, config.d_model)
        # Positional encoding
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, config.d_model))
            self.w_pos = positional_encoding(
                config.positional_encoding,
                config.learn_pe,
                config.num_patches + 1,
                config.d_model,
            )
        else:
            self.w_pos = positional_encoding(
                config.positional_encoding,
                config.learn_pe,
                config.num_patches,
                config.d_model,
            )

        # Positional dropout
        self.dropout = nn.Dropout(config.positional_dropout) if config.positional_dropout > 0 else nn.Identity()

        # Encoder
        self.encoder = ChannelAttentionTSTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, past_values: torch.Tensor, output_hidden_states: Optional[bool] = None
    ) -> BaseModelOutputWithNoAttention:
        """
        past_values: tensor [bs x nvars x num_patches x patch_length] output_hidden_states (bool, optional): Boolean
        indicating if hidden states should be outtput return:
            tensor [bs x nvars x num_patches x d_model]
                or [bs x nvars x (num_patches+1) x d_model] if use cls_token
        """
        # bs, num_patches, n_vars, patch_length = x.shape
        bs, n_vars, num_patches, patch_length = past_values.shape

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.w_p[i](past_values[:, i, :, :])
                x_out.append(z)
            past_values = torch.stack(x_out, dim=1)
        else:
            past_values = self.w_p(past_values)  # x: [bs x nvars  x num_patches x d_model]

        if self.use_cls_token:
            past_values = self.dropout(past_values + self.w_pos[1:, :])  # x: [bs x nvars x num_patches x d_model]
            # append cls token
            cls_token = self.cls_token + self.w_pos[:1, :]  # cls_token: [1 x 1 x 1 x d_model]
            cls_tokens = cls_token.expand(past_values.shape[0], -1, -1)  # get the same copy for all the batch samples
            past_values = torch.cat((cls_tokens, past_values), dim=1)  # x: [bs x nvars x (num_patches+1) x d_model]
        else:
            past_values = self.dropout(past_values + self.w_pos)  # x: [bs x nvars x num_patches x d_model]

        # Encoder
        past_values, hidden_states = self.encoder(
            past_values, output_hidden_states
        )  # x: [bs x nvars x num_patches x d_model]
        # or [bs x nvars x (num_patches+1) x d_model] if use cls_token

        # return past_values, hidden_states
        return BaseModelOutputWithNoAttention(last_hidden_state=past_values, hidden_states=hidden_states)


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
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`):
            Past values of the time series, that serve as context in order to predict the future. The sequence size of
            this tensor must be larger than the `context_length` of the model, since the model will use the larger size
            to construct lag features, i.e. additional values from the past which are added in order to serve as "extra
            context".

            The `sequence_length` here is equal to `config.context_length` + `max(config.lags_sequence)`, which if no
            `lags_sequence` is configured, is equal to `config.context_length` + 7 (as by default, the largest
            look-back index in `config.lags_sequence` is 7). The property `_past_length` returns the actual length of
            the past.

            The `past_values` is what the Transformer encoder gets as input (with optional additional features, such as
            `static_categorical_features`, `static_real_features`, `past_time_features` and lags).

            Optionally, missing values need to be replaced with zeros and indicated via the `past_observed_mask`.

            For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
            variates in the time series per time step.
        past_time_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):
            Required time features, which the model internally will add to `past_values`. These could be things like
            "month of year", "day of the month", etc. encoded as vectors (for instance as Fourier features). These
            could also be so-called "age" features, which basically help the model know "at which point in life" a
            time-series is. Age features have small values for distant past time steps and increase monotonically the
            more we approach the current time step. Holiday features are also a good example of time features.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features. The Time Series Transformer only learns
            additional embeddings for `static_categorical_features`.

            Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
            must but known at prediction time.

            The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
        past_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected in
            `[0, 1]`:

            - 1 for values that are **observed**,
            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        static_categorical_features (`torch.LongTensor` of shape `(batch_size, number of static categorical features)`, *optional*):
            Optional static categorical features for which the model will learn an embedding, which it will add to the
            values of the time series.

            Static categorical features are features which have the same value for all time steps (static over time).

            A typical example of a static categorical feature is a time series ID.
        static_real_features (`torch.FloatTensor` of shape `(batch_size, number of static real features)`, *optional*):
            Optional static real features which the model will add to the values of the time series.

            Static real features are features which have the same value for all time steps (static over time).

            A typical example of a static real feature is promotion information.
        future_values (`torch.FloatTensor` of shape `(batch_size, prediction_length)` or `(batch_size, prediction_length, input_size)`, *optional*):
            Future values of the time series, that serve as labels for the model. The `future_values` is what the
            Transformer needs during training to learn to output, given the `past_values`.

            The sequence length here is equal to `prediction_length`.

            See the demo notebook and code snippets for details.

            Optionally, during training any missing values need to be replaced with zeros and indicated via the
            `future_observed_mask`.

            For multivariate time series, the `input_size` > 1 dimension is required and corresponds to the number of
            variates in the time series per time step.
        future_time_features (`torch.FloatTensor` of shape `(batch_size, prediction_length, num_features)`):
            Required time features for the prediction window, which the model internally will add to `future_values`.
            These could be things like "month of year", "day of the month", etc. encoded as vectors (for instance as
            Fourier features). These could also be so-called "age" features, which basically help the model know "at
            which point in life" a time-series is. Age features have small values for distant past time steps and
            increase monotonically the more we approach the current time step. Holiday features are also a good example
            of time features.

            These features serve as the "positional encodings" of the inputs. So contrary to a model like BERT, where
            the position encodings are learned from scratch internally as parameters of the model, the Time Series
            Transformer requires to provide additional time features. The Time Series Transformer only learns
            additional embeddings for `static_categorical_features`.

            Additional dynamic real covariates can be concatenated to this tensor, with the caveat that these features
            must but known at prediction time.

            The `num_features` here is equal to `config.`num_time_features` + `config.num_dynamic_real_features`.
        future_observed_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, input_size)`, *optional*):
            Boolean mask to indicate which `future_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:

            - 1 for values that are **observed**,
            - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

            This mask is used to filter out missing values for the final loss calculation.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on certain token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Mask to avoid performing attention on certain token indices. By default, a causal mask will be used, to
            make sure the model can only look at previous inputs in order to predict the future.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the decoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of `last_hidden_state`, `hidden_states` (*optional*) and `attentions` (*optional*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` (*optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
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
    "The bare PatchTST Model outputting raw hidden-states without any specific head.",
    PATCHTST_START_DOCSTRING,
)
class PatchTSTModelOutputWithNoAttention(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, num_channels, height, width)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        patched_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            patched input to the Transformer
        mask: (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches)`,*optional*)
            Bool masked tensor indicating which patches are masked
        revin_mean: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*)
            mean of the input data (batch_size, sequence_length, num_channels) over the sequence_length
        revin_std: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*)
            std of the input data (batch_size, sequence_length, num_channels) over the sequence_length
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    patched_input: torch.FloatTensor = None
    mask: torch.FloatTensor = None
    revin_mean: torch.FloatTensor = None
    revin_std: torch.FloatTensor = None


class RevIN(nn.Module):
    def __init__(self, start_dim=1, eps=1e-5, denorm_channels: list = None):
        """
        :param start_dim: it is 1 if [bs x seq_len x nvars], it is 3 is [bs x tsg1 x tsg2 x seq_len x n_vars]
        :denorm_channels if the denorm input shape has less number of channels, mention the channels in the denorm
        input here.
        """
        super(RevIN, self).__init__()
        self.stdev = None
        self.mean = None
        self.start_dim = start_dim
        self.denorm_channels = denorm_channels
        self.eps = eps

    def set_statistics(self, mean, stdev):
        # get statistics
        self.mean = mean
        self.stdev = stdev

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        elif mode == "transform":
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(self.start_dim, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        # denormalize the data
        if self.denorm_channels is None:
            x = x * self.stdev
            x = x + self.mean
        else:
            x = x * self.stdev[..., self.denorm_channels]
            x = x + self.mean[..., self.denorm_channels]

        return x


class PatchTSTModel(PatchTSTPreTrainedModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.use_revin = config.revin

        if self.use_revin:
            self.revin = RevIN()
        else:
            self.revin = nn.Identity()

        self.patching = Patchify(
            config.context_length,
            patch_length=config.patch_length,
            stride=config.stride,
        )
        self.mask_input = config.mask_input

        if self.mask_input:
            self.masking = PatchMasking(
                mask_type=config.mask_type,
                mask_ratio=config.mask_ratio,
                mask_patches=config.mask_patches,
                mask_patch_ratios=config.mask_patch_ratios,
                channel_consistent_masking=config.channel_consistent_masking,
                unmasked_channel_indices=config.unmasked_channel_indices,
                mask_value=config.mask_value,
                seed_number=config.seed_number,
            )
        else:
            self.masking = nn.Identity()
        self.encoder = ChannelAttentionPatchTSTEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        past_values = self.revin(past_values, mode="norm")  # x: tensor [bs x seq_len x in_channels]

        patched_values = self.patching(
            past_values
        )  # patched_values: [bs x n_vars x num_patches x patch_length] for pretrain
        if self.mask_input:
            masked_values, mask = self.masking(patched_values)
        else:
            masked_values, mask = self.masking(patched_values), None
        encoder_output = self.encoder(masked_values, output_hidden_states=output_hidden_states)
        return PatchTSTModelOutputWithNoAttention(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=encoder_output.hidden_states,
            patched_input=patched_values,
            mask=mask,
            revin_mean=self.revin.mean if self.use_revin else None,
            revin_stdev=self.revin.stdev if self.use_revin else None,
        )


class MaskPretrainHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.d_model, config.patch_length)
        self.use_cls_token = config.use_cls_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor [bs x nvars x num_patches x d_model]
                or [bs x nvars x (num_patches+1) x d_model] if use cls_token
        output: tensor [bs x nvars x num_patches x patch_length]
        """
        x = self.linear(self.dropout(x))  # [bs x nvars x num_patches x patch_length]
        if self.use_cls_token:
            x = x[:, :, 1:, :]  # remove the first cls token
        return x


class PatchTSTOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPredictiontion`].

    Args:
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


class PatchTSTForMaskPretraining(PatchTSTPreTrainedModel):
    # PatchTSTModel + Pretraining Head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        config.mask_input = True
        self.model = PatchTSTModel(config=config)
        self.head = MaskPretrainHead(config)
        self.loss = torch.nn.MSELoss(reduction="none")

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> PatchTSTOutput:
        """
        past_values (x): tensor [bs x sequence_length x n_vars ] future_values (y): labels
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # past_values: [bs x nvars x num_patches x d_model] or
        # [bs x nvars x (num_patches+1) x d_model] if use cls_token
        model_output = self.model(past_values, output_hidden_states=output_hidden_states)

        # model_output[0]: [bs x nvars x num_patches x patch_length] or
        # [bs x nvars x (num_patches+1) x patch_length] if use cls_token
        x_hat = self.head(model_output[0])

        # calculate masked_loss
        loss_val = self.loss(x_hat, model_output.patched_input)
        masked_loss = (loss_val.mean(dim=-1) * model_output.mask).sum() / (model_output.mask.sum() + 1e-10)

        return PatchTSTOutput(
            loss=masked_loss,
            prediction_output=x_hat,
            hidden_states=model_output.hidden_states,
        )


class PatchTSTForClassification(PatchTSTPreTrainedModel):
    # PatchTST model + classification head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.model = PatchTSTModel(config)
        self.head = ClassificationHead(config)
        self.loss = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, past_values, labels=None, output_hidden_states: Optional[bool] = None):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        model_output = self.model(past_values, output_hidden_states=output_hidden_states)
        y_hat = self.head(model_output[0])

        loss_val = None
        if labels is not None:
            loss_val = self.loss(y_hat, labels)
        return PatchTSTForClassificationOutput(
            loss=loss_val,
            prediction_logits=y_hat,
            hidden_states=model_output.hidden_states,
        )


class ClassificationHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(config.input_size * config.d_model, config.num_classes)

    def forward(self, x):
        """
        x: [bs x nvars x num_patches x d_model] or [bs x nvars x (num_patches+1) x d_model] if use cls_token output:
        [bs x n_classes]
        """
        if self.use_cls_token:
            x = x[:, :, 0, :]  # use the first output token, x: bs x nvars x d_model
        elif self.pooling == "mean":
            x = x.mean(dim=2)  # x: [bs x nvars x d_model]
        elif self.pooling == "max":
            x = x.max(dim=2)  # x: [bs x nvars x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")

        x = self.flatten(x)  # x: bs x nvars * d_model
        y = self.linear(self.dropout(x))  # y: bs x n_classes
        return y


class PatchTSTForClassificationOutput(ModelOutput):
    """
    Output type of [`PatchTSTForClassification`].

    Args:
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


class PredictionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.target_dimension = config.target_dimension
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling

        head_dim = config.input_size * config.d_model

        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(head_dim, config.prediction_length * config.target_dimension)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

    def forward(self, x):
        """
        x: [bs x nvars x num_patch x d_model]
            or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        output: [bs x pred_len x target_dimension]
        """
        batch_size = x.shape[0]
        if self.use_cls_token:
            x = x[:, :, 0, :]  # use the first output token, x: [bs x nvars x d_model]
        elif self.pooling == "mean":
            x = x.mean(dim=2)  # x: [bs x nvars x d_model]
        elif self.pooling == "max":
            x = x.max(dim=2)  # x: [bs x nvars x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")

        # flatten the input
        x = self.flatten(x)  # x: bs x (nvars * d_model)
        y = self.linear(self.dropout(x))  # y: bs x (pred_len * target_dimension)

        # reshape the data
        y = y.reshape(batch_size, -1, self.target_dimension)  # [bs x pred_len x target_dimension]
        return y


class PatchTSTForPrediction(PatchTSTPreTrainedModel):
    # PatchTST model + prediction head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.model = PatchTSTModel(config)
        self.head = PredictionHead(config)
        self.loss = nn.MSELoss(reduction="mean")

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        model_output = self.model(past_values, output_hidden_states=output_hidden_states)
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None
        if future_values is not None:
            loss_val = self.loss(y_hat, future_values)
        return PatchTSTOutput(
            loss=loss_val,
            prediction_output=y_hat,
            hidden_states=model_output.hidden_states,
        )


class PatchTSTForForecastingOutput(ModelOutput):
    """
    Output type of [`PatchTSTForPredictiontion`].

    Args:
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


class ForecastHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()

        self.individual = config.individual
        self.n_vars = config.input_size
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        head_dim = config.d_model if self.pooling else config.d_model * config.num_patches

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=2))
                self.linears.append(nn.Linear(head_dim, config.prediction_length))
                self.dropouts.append(nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity())
        else:
            self.flatten = nn.Flatten(start_dim=2)
            self.linear = nn.Linear(head_dim, config.prediction_length)
            self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        x: [bs x nvars x num_patches x d_model]
            or [bs x nvars x (num_patches+1) x d_model] if use cls_token
        output: [bs x forecast_len x nvars]
        """

        if self.use_cls_token:
            y = x[:, :, 0, :]  # y: [bs x nvars x d_model]
        else:
            if self.pooling == "mean":
                y = x.mean(dim=2)  # y: [bs x nvars x d_model]
            elif self.pooling == "max":
                y = x.max(dim=2)  # y: [bs x nvars x d_model]
            else:
                y = x  # y: [bs x nvars x num_patches x d_model]

        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](y[:, i, :])  # y: [bs x (d_model * num_patches)] or [bs x d_model)]
                z = self.linears[i](z)  # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x forecast_len]
        else:
            z = self.flatten(y)  # z: [bs x nvars x (d_model * num_patches)] or [bs x nvars x d_model)]
            z = self.dropout(z)
            x = self.linear(z)  # x: [bs x nvars x forecast_len]

        x = x.transpose(2, 1)  # [bs x forecast_len x nvars]

        return x


class PatchTSTForForecasting(PatchTSTPreTrainedModel):
    # PatchTST model + Forecasting head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.model = PatchTSTModel(config)
        self.head = ForecastHead(config)
        self.loss = nn.MSELoss(reduction="mean")
        self.use_revin = config.revin
        if self.use_revin:
            self.revin = RevIN()
        else:
            self.revin = nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor],
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        model_output = self.model(past_values, output_hidden_states=output_hidden_states)

        y_hat = self.head(model_output.last_hidden_state)

        if self.use_revin:
            self.revin.set_statistics(mean=model_output.revin_mean, stdev=model_output.revin_stdev)
            y_hat = self.revin(y_hat, mode="denorm")

        loss_val = None
        if future_values is not None:
            loss_val = self.loss(y_hat, future_values)
        return PatchTSTForForecastingOutput(
            loss=loss_val,
            forecast_outputs=y_hat,
            hidden_states=model_output.hidden_states,
        )


class RegressionHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.y_range = config.prediction_range
        self.use_cls_token = config.use_cls_token
        self.pooling = config.pooling
        # self.is_flatten = is_flatten

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        input_dim = config.input_size * config.d_model
        # if is_flatten: input_dim *= num_patch
        self.linear = nn.Linear(input_dim, config.target_dimension)

    def forward(self, past_values):
        """
        x: [bs x nvars x num_patch x d_model]
            or [bs x nvars x (num_patch+1) x d_model] if use cls_token
        output: [bs x output_dim]
        """
        if self.use_cls_token:
            past_values = past_values[:, :, 0, :]  # use the first output token, x: [bs x nvars x d_model]
        elif self.pooling == "mean":
            past_values = past_values.mean(dim=2)  # x: [bs x nvars x d_model]
        elif self.pooling == "max":
            past_values = past_values.max(dim=2)  # x: [bs x nvars x d_model]
        else:
            raise Exception(f"pooling operator {self.pooling} is not implemented yet")
        # flatten the input
        past_values = self.flatten(past_values)  # x: bs x nvars * d_model
        y = self.linear(self.dropout(past_values))  # y: bs x output_dim

        if self.y_range:
            y = torch.sigmoid(y) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

        return y


class PatchTSTForRegression(PatchTSTPreTrainedModel):
    # PatchTST model + Regression head
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)
        self.model = PatchTSTModel(config)
        self.head = RegressionHead(config)
        self.loss = nn.MSELoss(reduction="mean")

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        past_values: torch.Tensor,
        labels: Optional[torch.Tensor],
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        model_output = self.model(past_values, output_hidden_states=output_hidden_states)
        y_hat = self.head(model_output.last_hidden_state)

        loss_val = None
        if labels is not None:
            loss_val = self.loss(y_hat, labels)
        return PatchTSTOutput(
            loss=loss_val,
            prediction_output=y_hat,
            hidden_states=model_output.hidden_states,
        )
