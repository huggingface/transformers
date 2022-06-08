# coding=utf-8
# Copyright 2022 The HuggingFace Team The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch VQGAN model."""


from functools import partial
import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import SiLUActivation
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_vqgan import VQGANConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "vqgan-imagenet-f16-1024"
_CONFIG_FOR_DOC = "VQGANConfig"
_TOKENIZER_FOR_DOC = "VQGANTokenizer"

VQGAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "vqgan-imagenet-f16-1024",
    # See all VQGAN models at https://huggingface.co/models?filter=vqgan
]


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()

        self.with_conv = with_conv

        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, hidden_states):
        hidden_states = torch.nn.functional.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def forward(self, hidden_states):
        if self.with_conv:
            pad = (0,1,0,1)  # pad height and width dim
            hidden_states = torch.nn.functional.pad(hidden_states, pad, mode="constant", value=0)
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = torch.nn.functional.avg_pool2d(hidden_states, kernel_size=2, stride=2)
        return hidden_states


class ResnetBlock(nn.Module):
   
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        use_conv_shortcut: bool = False,
        temb_channels: int = 512,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_channels_ = self.in_channels if self.out_channels is None else self.out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.activation = SiLUActivation()

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels_,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if temb_channels:
            self.temb_proj = nn.Linear(temb_channels, self.out_channels_)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels_, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.conv2 = nn.Conv2d(
            self.out_channels_,
            self.out_channels_,
            kernel_size=3,
            stride=(1, 1),
            padding=1,
        )

        if self.in_channels != self.out_channels_:
            if use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    self.in_channels,
                    self.out_channels_,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    self.in_channels,
                    self.out_channels_,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, hidden_states, temb=None):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.activation(temb))[:, :, None, None]  # TODO: check shapes

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels_:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return hidden_states + residual


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.in_channels = in_channels
        conv = partial(
            nn.Conv2d, self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0
        )

        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.in_channels, eps=1e-6, affine=True)
        self.q, self.k, self.v = conv(), conv(), conv()
        self.proj_out = conv()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        # compute attentions
        batch, channels, height, width = query.shape
        query = query.reshape((batch, height * width, channels))
        key = key.reshape((batch, height * width, channels))
        
        attn_weights = torch.einsum("...qc,...kc->...qk", query, key)
        attn_weights = attn_weights * (int(channels) ** -0.5)
        attn_weights = nn.functional.softmax(attn_weights, dim=2)

        ## attend to values
        value = value.reshape((batch, height * width, channels))
        hidden_states = torch.einsum("...kc,...qk->...qc", value, attn_weights)
        hidden_states = hidden_states.reshape((batch, channels, height, width))

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class UpsamplingBlock(nn.Module):

    def __init__(self, config, curr_res: int, block_idx: int):
        super().__init__()

        self.config = config
        self.block_idx = block_idx
        self.curr_res = curr_res

        if self.block_idx == self.config.num_resolutions - 1:
            block_in = self.config.ch * self.config.ch_mult[-1]
        else:
            block_in = self.config.ch * self.config.ch_mult[self.block_idx + 1]

        block_out = self.config.ch * self.config.ch_mult[self.block_idx]
        self.temb_ch = 0

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks + 1):
            res_blocks.append(
                ResnetBlock(
                    block_in, block_out, temb_channels=self.temb_ch, dropout_prob=self.config.dropout
                )
            )
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.block = nn.ModuleList(res_blocks)
        self.attn = nn.ModuleList(attn_blocks)

        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in, self.config.resamp_with_conv)

    def forward(self, hidden_states, temb=None, deterministic: bool = True):
        for res_block in self.block:
            hidden_states = res_block(hidden_states, temb)
            for attn_block in self.attn:
                hidden_states = attn_block(hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class DownsamplingBlock(nn.Module):
    def __init__(self, config, curr_res: int, block_idx: int):
        super().__init__()

        self.config = config
        self.curr_res = curr_res
        self.block_idx = block_idx

        in_ch_mult = (1,) + tuple(self.config.ch_mult)
        block_in = self.config.ch * in_ch_mult[self.block_idx]
        block_out = self.config.ch * self.config.ch_mult[self.block_idx]
        self.temb_ch = 0

        res_blocks = nn.ModuleList()
        attn_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(
                ResnetBlock(
                    block_in, block_out, temb_channels=self.temb_ch, dropout_prob=self.config.dropout
                )
            )
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.block = res_blocks
        self.attn = attn_blocks

        self.downsample = None
        if self.block_idx != self.config.num_resolutions - 1:
            self.downsample = Downsample(block_in, self.config.resamp_with_conv)

    def forward(self, hidden_states, temb=None):
        for res_block in self.block:
            hidden_states = res_block(hidden_states, temb)
            for attn_block in self.attn:
                hidden_states = attn_block(hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class MidBlock(nn.Module):
    def __init__(self, config,  in_channels: int, temb_channels: int, dropout: float):
        super().__init__()
        
        self.config = config
        self.in_channels = in_channels
        self.temb_channels = temb_channels
        self.dropout = dropout
        
        self.block_1 = ResnetBlock(
            self.in_channels,
            self.in_channels,
            temb_channels=self.temb_channels,
            dropout_prob=self.dropout,
        )
        self.attn_1 = AttnBlock(self.in_channels)
        self.block_2 = ResnetBlock(
            self.in_channels,
            self.in_channels,
            temb_channels=self.temb_channels,
            dropout_prob=self.dropout,
        )

    def forward(self, hidden_states, temb=None):
        hidden_states = self.block_1(hidden_states, temb)
        hidden_states = self.attn_1(hidden_states)
        hidden_states = self.block_2(hidden_states, temb)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.temb_ch = 0

        # downsampling
        self.conv_in = nn.Conv2d(
            self.config.in_channels,
            self.config.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        curr_res = self.config.resolution
        downsample_blocks = []
        for i_level in range(self.config.num_resolutions):
            downsample_blocks.append(DownsamplingBlock(self.config, curr_res, block_idx=i_level))

            if i_level != self.config.num_resolutions - 1:
                curr_res = curr_res // 2
        self.down = nn.ModuleList(downsample_blocks)

        # middle
        mid_channels = self.config.ch * self.config.ch_mult[-1]
        self.mid = MidBlock(config, mid_channels, self.temb_ch, self.config.dropout)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=mid_channels, eps=1e-6, affine=True)
        self.activation = SiLUActivation()
        self.conv_out = nn.Conv2d(
            mid_channels,
            2 * self.config.z_channels if self.config.double_z else self.config.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )


    def forward(self, pixel_values):
        # timestep embedding
        temb = None

        # downsampling
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states, temb)

        # middle
        hidden_states = self.mid(hidden_states, temb)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    config: VQGANConfig
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.temb_ch = 0

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = self.config.ch * self.config.ch_mult[self.config.num_resolutions - 1]
        curr_res = self.config.resolution // 2 ** (self.config.num_resolutions - 1)
        self.z_shape = (1, self.config.z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(
            self.config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # middle
        self.mid = MidBlock(config, block_in, self.temb_ch, self.config.dropout)

        # upsampling
        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(UpsamplingBlock(self.config, curr_res, block_idx=i_level))
            if i_level != 0:
                curr_res = curr_res * 2
        self.up = nn.ModuleList(list(reversed(upsample_blocks)))  # reverse to get consistent order

        # end
        block_out = self.config.ch * self.config.ch_mult[0]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out, eps=1e-6, affine=True)
        self.activation = SiLUActivation()
        self.conv_out = nn.Conv2d(
            block_out,
            self.config.out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, hidden_states):
        # timestep embedding
        temb = None

        # z to block_in
        hidden_states = self.conv_in(hidden_states)

        # middle
        hidden_states = self.mid(hidden_states, temb)

        # upsampling
        for block in reversed(self.up):
            hidden_states = block(hidden_states, temb)

        # end
        if self.config.give_pre_end:
            return hidden_states

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________ Discretization bottleneck part of the VQ-VAE. Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(self.config.n_embed, self.config.embed_dim)  # TODO: init
        self.embedding.weight.data.uniform_(-1.0 / self.config.n_embed, 1.0 / self.config.n_embed)

    def forward(self, hidden_states):
        """
        Inputs the output of the encoder network z and maps it to a discrete one-hot vector that is the index of the
        closest embedding vector e_j z (continuous) -> z_q (discrete) z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states_flattended = hidden_states.reshape((-1, self.config.embed_dim))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        emb_weights = self.embedding.weight
        distance = (        
            torch.sum(hidden_states_flattended**2, dim=1, keepdims=True)
            + torch.sum(emb_weights**2, dim=1)
            - 2 * torch.matmul(hidden_states_flattended, emb_weights.T)
        )

        # get quantized latent vectors
        min_encoding_indices = torch.argmin(distance, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(hidden_states.shape)
        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute the codebook_loss (q_loss) outside the model
        # here we return the embeddings and indices
        return z_q, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        # indices are expected to be of shape (batch, num_tokens)
        # get quantized latent vectors
        batch, num_tokens = indices.shape
        z_q = self.embedding(indices)
        z_q = z_q.reshape(batch, -1, int(math.sqrt(num_tokens)), int(math.sqrt(num_tokens)))
        return z_q


class VQGANPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VQGANConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


class VQGANModel(VQGANPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantize = VectorQuantizer(self.config)
        self.quant_conv = nn.Conv2d(
            self.config.z_channels,
            self.config.embed_dim,
            kernel_size=1,
        )
        self.post_quant_conv = nn.Conv2d(
            self.config.embed_dim,
            self.config.z_channels,
            kernel_size=1,
        )

    def encode(self, pixel_values):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quant_states, indices = self.quantize(hidden_states)
        return quant_states, indices

    def decode(self, hidden_states):
        hidden_states = self.post_quant_conv(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def decode_code(self, code_b):
        hidden_states = self.quantize.get_codebook_entry(code_b)
        print(hidden_states.shape)
        hidden_states = self.decode(hidden_states)
        return hidden_states

    def forward(self, pixel_values):
        quant_states, indices = self.encode(pixel_values)
        # import ipdb; ipdb.set_trace()
        hidden_states = self.decode(quant_states)
        return hidden_states, indices

