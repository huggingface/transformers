# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
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

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...utils import (
    is_flash_attn_2_available,
    is_vision_available,
    logging,
)
from ..emu3.configuration_emu3 import (
    Emu3Config,
    Emu3VQVAEConfig,
)
from ..emu3.modeling_emu3 import (
    Emu3ForConditionalGeneration,
    Emu3VQVAE,
    Emu3VQVAEAttentionBlock,
    Emu3VQVAEDecoder,
    Emu3VQVAEDownBlock,
    Emu3VQVAEEncoder,
    Emu3VQVAEUpBlock,
)
from ..llama.modeling_llama import (
    LlamaModel,
)


if is_vision_available():
    pass


if is_flash_attn_2_available():
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


_CONFIG_FOR_DOC = "CosmosConfig"
_CHECKPOINT_FOR_DOC = "NVIDIA/Cosmos-4B-hf"

logger = logging.get_logger(__name__)


class CosmosVQVAEConfig(Emu3VQVAEConfig):
    def __init__(
        self,
        embed_dim: int = 6,
        base_channels: int = 128,
        attn_resolutions: List[int] = None,
        latent_channels: int = 16,
        temporal_downsample_factor: int = 8,
        patch_size: int = 4,
        levels: List[int] = [8, 8, 8, 5, 5, 5],
        dropout: float = 0.0,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)
        self.patch_size = patch_size
        self.levels = levels
        self.dropout = dropout


class CosmosConfig(Emu3Config):
    pass


class CosmosVQVAEVectorQuantizer(nn.Module):
    """
    A module for vector quantization using learned embedding vectors.

    This module implements the quantization process similar to the one described in
    the [Finite Scalar Quantization: VQ-VAE Made Simple paper](https://arxiv.org/abs/2309.15505). It quantizes continuous
    input vectors into discrete codebook vectors, which are learned during training.

    Adapted from: https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/
    vector_quantize_pytorch/finite_scalar_quantization.py. [Copyright (c) 2020 Phil Wang]
    """

    def __init__(self, config: CosmosVQVAEConfig):
        super().__init__()

        levels = config.levels

        self.codebook_dim = config.codebook_dim

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

        codebook_size = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(codebook_size))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    def forward(self, hidden_state: torch.Tensor):
        # shape [1, 6, 5, 40, 64]
        batch_size, temporal, channels, height, width = hidden_state.shape
        hidden_state = hidden_state.permute(0, 2, 3, 4, 1).contiguous()
        hidden_state_flattened = hidden_state.view(batch_size, -1, temporal).unsqueeze(-1)

        codes = self.quantize(hidden_state_flattened)
        indices = self.codes_to_indices(codes)
        indices = indices.view(batch_size, channels, height, width, -1)

        out = codes.flatten(2, 3)
        out = out.view(batch_size, channels, height, width, -1)
        out.permute(0, 4, 1, 2, 3)

        return indices, out

    def bound(self, hidden_state: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        # bound torch.Size([1, 12800, 1, 6])
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (hidden_state + shift).tanh() * half_l - offset

    def quantize(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.bound(hidden_state)
        quantized = hidden_state.round()
        quantized = hidden_state + (quantized - hidden_state).detach()
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def codes_to_indices(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # codes_to_indices torch.Size([1, 12800, 1, 6])
        half_width = self._levels // 2
        hidden_state = (hidden_state * half_width) + half_width
        hidden_state = hidden_state.float()
        return (hidden_state * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        # indices_to_codes torch.Size([64000])
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        half_width = self._levels // 2  # shape [64000, 6]
        codes = (codes_non_centered - half_width) / half_width
        return codes


class CosmosCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int],
        stride: int = 1,
        time_stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.time_pad = (kernel_size[0] - 1) + (1 - time_stride)
        self.padding = (padding,) * 4 + (0, 0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=(time_stride, stride, stride))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states_prev = hidden_states[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
        hidden_states = torch.cat([hidden_states_prev, hidden_states], dim=2)

        hidden_states = F.pad(hidden_states, self.padding)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class CosmosVQVAETemporalNorm(nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_channels=in_channels, num_groups=1, eps=1e-6, affine=True)

    def forward(self, hidden_states: torch.Tensor):
        # group time and batch dims, then ungroup back
        batch_size, temporal, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.view(-1, channels, height, width)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(batch_size, temporal, channels, height, width)
        return hidden_states


class CosmosVQVAEEncoderDownsample(nn.Module):
    def __init__(self, in_channels, temporal_down: bool = True):
        super().__init__()
        self.conv1 = CosmosCausalConv3d(
            in_channels, in_channels, kernel_size=(1, 3, 3), stride=2, time_stride=1, padding=0
        )
        self.conv2 = (
            CosmosCausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=2, padding=0)
            if temporal_down
            else nn.Identity()
        )
        self.conv3 = CosmosCausalConv3d(
            in_channels, in_channels, kernel_size=(1, 1, 1), stride=1, time_stride=1, padding=0
        )
        self.temporal_down = temporal_down

    def forward(self, hidden_states):
        # hybrid downsample spatially
        hidden_states = F.pad(hidden_states, pad=(0, 1, 0, 1, 0, 0), mode="constant", value=0)
        hidden_states_1 = self.conv1(hidden_states)
        hidden_states_2 = F.avg_pool3d(hidden_states, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        hidden_states = hidden_states_1 + hidden_states_2

        # hybrid downsample temporally
        if self.temporal_down:
            hidden_states = torch.cat([hidden_states[:, :, :1, ...], hidden_states], dim=2)
            hidden_states_1 = self.conv2(hidden_states)
            hidden_states_2 = F.avg_pool3d(hidden_states, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            hidden_states = hidden_states_1 + hidden_states_2

        # final 1x1x1 conv
        hidden_states = self.conv3(hidden_states)
        return hidden_states


class CosmosVQVAEEncoderUpsample(nn.Module):
    def __init__(self, in_channels, temporal_up: bool = True):
        super().__init__()
        self.conv1 = (
            CosmosCausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=1, padding=0)
            if temporal_up
            else nn.Identity()
        )
        self.conv2 = CosmosCausalConv3d(
            in_channels, in_channels, kernel_size=(1, 3, 3), stride=1, time_stride=1, padding=1
        )
        self.conv3 = CosmosCausalConv3d(
            in_channels, in_channels, kernel_size=(1, 1, 1), stride=1, time_stride=1, padding=0
        )
        self.temporal_up = temporal_up

    def forward(self, hidden_states):
        # hybrid upsample temporally
        if self.temporal_up:
            time_factor = int(hidden_states.shape[2] > 1)
            hidden_states = hidden_states.repeat_interleave((time_factor + 1), dim=2)
            hidden_states = hidden_states[..., time_factor:, :, :]
            hidden_states = self.conv1(hidden_states) + hidden_states

        # hybrid upsample spatially
        hidden_states = hidden_states.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        hidden_states = self.conv2(hidden_states) + hidden_states

        # final 1x1x1 conv
        hidden_states = self.conv3(hidden_states)
        return hidden_states


class CosmosPatch3D(nn.Module):
    """A 3D discrete wavelet transform for video data."""

    def __init__(self, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        wavelets = torch.tensor([0.7071067811865476, 0.7071067811865476])

        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer("_arange", torch.arange(2), persistent=False)
        self.register_buffer("wavelets", wavelets, persistent=False)
        self.register_buffer("patch_size_buffer", patch_size * torch.ones([1], dtype=torch.int32), persistent=False)
        for param in self.parameters():
            param.requires_grad = False

    def _dwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # Handles temporal axis.
        x = F.pad(x, pad=(max(0, n - 2), n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        # Handles spatial axes.
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            out = out / (2 * torch.sqrt(torch.tensor(2.0)))
        return out

    def forward(self, hidden_state: torch.Tensor):
        xi, xv = torch.split(hidden_state, [1, hidden_state.shape[2] - 1], dim=2)
        hidden_state = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)
        for _ in self.range:
            hidden_state = self._dwt(hidden_state, rescale=True)
        return hidden_state


class CosmosUnpatch3D(nn.Module):
    """A 3D inverse discrete wavelet transform for video wavelet decompositions."""

    def __init__(self, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        wavelets = torch.tensor([0.7071067811865476, 0.7071067811865476])

        self.register_buffer("wavelets", wavelets, persistent=False)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer("_arange", torch.arange(2), persistent=False)
        for param in self.parameters():
            param.requires_grad = False

    def _idwt(self, x, rescale=False):
        dtype = x.dtype
        h = self.wavelets

        g = x.shape[1] // 8  # split into 8 spatio-temporal filtered tesnors.
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hl = hl.to(dtype=dtype)
        hh = hh.to(dtype=dtype)

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(x, 8, dim=1)

        # Height height transposed convolutions.
        xll = F.conv_transpose3d(xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xll += F.conv_transpose3d(xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        xlh = F.conv_transpose3d(xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlh += F.conv_transpose3d(xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        xhl = F.conv_transpose3d(xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhl += F.conv_transpose3d(xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        xhh = F.conv_transpose3d(xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhh += F.conv_transpose3d(xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        # Handles width transposed convolutions.
        xl = F.conv_transpose3d(xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xl += F.conv_transpose3d(xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xh = F.conv_transpose3d(xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xh += F.conv_transpose3d(xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        # Handles time axis transposed convolutions.
        x = F.conv_transpose3d(xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        x += F.conv_transpose3d(xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        if rescale:
            x = x * (2 * torch.sqrt(torch.tensor(2.0)))
        return x

    def forward(self, x):
        for _ in self.range:
            x = self._idwt(x, rescale=True)
        x = x[:, :, self.patch_size - 1 :, ...]
        return x


# Copy from Emu3 fails because each layers init under condition aren't overwritten/skipped correctly
class CosmosVQVAEResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Sequential(
            CosmosCausalConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.conv2 = nn.Sequential(
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.dropout = torch.nn.Dropout(dropout)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = CosmosCausalConv3d(
                in_channels, out_channels, kernel_size=(1, 1, 1), stride=1, padding=0
            )

    def forward(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)

        return residual + hidden_states


class CosmosVQVAETemporalAttentionBlock(Emu3VQVAEAttentionBlock):
    pass


class CosmosVQVAEAttentionBlock(Emu3VQVAEAttentionBlock):
    pass


class CosmosVQVAEAttention(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()

        self.attn_1 = CosmosVQVAEAttentionBlock(config)
        self.attn_2 = CosmosVQVAETemporalAttentionBlock(config)
        self.attn_norm = CosmosVQVAETemporalNorm(in_channels)

    def forward(self, hidden_states: torch.Tensor):
        # Apply attn norm + attn in spatial dim
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = F.pad(hidden_states, pad=(1, 1, 1, 1, 0, 0), mode="constant", value=0.0)

        # b c t h w -> (b t) c h w
        batch_size, channels, temporal, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * temporal, channels, height * width).transpose(1, 2)
        hidden_states = self.attn_1(hidden_states)[0]
        hidden_states = hidden_states.reshape(batch_size, temporal, height, width, channels).permute(0, 4, 1, 2, 3)

        # Apply attn norm + attn in temporal dim
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self._replication_pad(hidden_states)

        # b c t h w -> (b h w) c t
        batch_size, channels, temporal, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 3, 4, 1, 2).contiguous()
        hidden_states = hidden_states.view(batch_size * height * width, channels, temporal).transpose(1, 2)
        hidden_states = self.attn_2(hidden_states)[0]
        hidden_states = hidden_states.reshape(batch_size, height, width, channels, temporal).permute(0, 3, 4, 1, 2)
        return hidden_states


class CosmosVQVAEMiddleBlock(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        self.block_1 = CosmosVQVAEResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            dropout=config.dropout,
        )
        self.attn = CosmosVQVAEAttention(config, in_channels)
        self.block_2 = CosmosVQVAEResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            dropout=config.dropout,
        )

    def forward(self, hidden_states: torch.FloatTensor):
        hidden_states = self.block_1(hidden_states)
        residual = hidden_states
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.block_2(hidden_states)
        return hidden_states


class CosmosVQVAEDownBlock(Emu3VQVAEDownBlock):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks

        base_channels = config.base_channels
        channel_multiplier = config.channel_multiplier
        self.num_temporal_downs = int(math.log2(config.temporal_downsample_factor)) - int(math.log2(config.patch_size))

        in_channel_multiplier = (1,) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    CosmosVQVAEResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if config.attn_resolutions is not None and i_level in config.attn_resolutions:
                    attn.append(CosmosVQVAEAttention(config, block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                temporal_down = i_level < self.num_temporal_downs
                down.downsample = CosmosVQVAEEncoderDownsample(block_in, temporal_down=temporal_down)

            self.down.append(down)


class CosmosVQVAEUpBlock(Emu3VQVAEUpBlock):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        self.num_temporal_ups = int(math.log2(config.temporal_downsample_factor)) - int(math.log2(config.patch_size))

        block_in = config.base_channels * config.channel_multiplier[-1]

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    CosmosVQVAEResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,  # DIFF HERE
                    )
                )
                block_in = block_out
                if config.attn_resolutions is not None and i_level in config.attn_resolutions:
                    attn.append(CosmosVQVAEAttention(config, block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                i_level_reverse = self.num_resolutions - i_level - 1
                temporal_up = 0 < i_level_reverse < self.num_temporal_ups + 1
                up.upsample = CosmosVQVAEEncoderUpsample(block_in, temporal_up=temporal_up)

            self.up.insert(0, up)


class CosmosVQVAEEncoder(Emu3VQVAEEncoder):
    def __init__(self, config):
        super().__init__(config)

        base_channels = config.base_channels
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier
        block_in = base_channels * channel_multiplier[-1]

        self.patch = CosmosPatch3D(config.patch_size)
        self.conv_in = nn.Sequential(
            CosmosCausalConv3d(in_channels, base_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(base_channels, base_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

        out_channels = 2 * latent_channels if double_latent else latent_channels
        self.norm_out = CosmosVQVAETemporalNorm(block_in)
        self.conv_out = nn.Sequential(
            CosmosCausalConv3d(block_in, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

        self.time_conv = nn.Identity()
        self.time_res_stack = nn.Identity()

    def forward(self, pixel_values: torch.LongTensor):
        pixel_values = self.patch(pixel_values)

        # downsampling & middle
        hidden_states = self.conv_in(pixel_values)
        hidden_states = self.down_block(hidden_states)
        hidden_states = self.middle_block(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class CosmosVQVAEDecoder(Emu3VQVAEDecoder):
    def __init__(self, config: CosmosVQVAEConfig):
        super().__init__(config)

        block_in = config.base_channels * config.channel_multiplier[-1]
        self.middle_block = CosmosVQVAEMiddleBlock(config, block_in)
        self.norm_out = CosmosVQVAETemporalNorm(block_in)
        self.unpatch = CosmosUnpatch3D(config.patch_size)
        self.conv_in = nn.Sequential(
            CosmosCausalConv3d(config.latent_channels, block_in, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(block_in, block_in, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.conv_out = nn.Sequential(
            CosmosCausalConv3d(block_in, config.out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CosmosCausalConv3d(config.out_channels, config.out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.conv_in(hidden_states)

        hidden_states = self.middle_block(hidden_states)
        hidden_states = self.up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        hidden_states = self.uppatch(hidden_states)

        return hidden_states


class CosmosVQVAE(Emu3VQVAE):
    def __init__(self, config: CosmosVQVAEConfig):
        super().__init__(config)

        self.config = config

        self.encoder = CosmosVQVAEEncoder(config)
        self.decoder = CosmosVQVAEDecoder(config)
        self.quantize = CosmosVQVAEVectorQuantizer(config)

        self.quant_conv = CosmosCausalConv3d(
            config.latent_channels, config.embed_dim, kernel_size=(1, 1, 1), padding=0
        )
        self.post_quant_conv = CosmosCausalConv3d(
            config.embed_dim, config.latent_channels, kernel_size=(1, 1, 1), padding=0
        )

        self.eval()  # VQ model is frozen and not implemented for training/tuning
        self.post_init()

    def encode(self, pixel_values: torch.Tensor):
        # b t c h w -> b c t h w
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)

        # b c t h w -> b t c h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        codes = self.quantize(hidden_states)
        return codes

    def decode(self, hidden_states: torch.Tensor):
        hidden_states = self.post_quant_conv(hidden_states)
        video = self.decoder(hidden_states)
        return video

    def forward(self, pixel_values):
        quant_info, quant_codes = self.encode(pixel_values)
        reconstructions = self.decode(quant_codes)
        return reconstructions, quant_info


class CosmosTextModel(LlamaModel):
    pass


class CosmosForConditionalGeneration(Emu3ForConditionalGeneration):
    pass


__all__ = [
    "CosmosForConditionalGeneration",
    "CosmosTextModel",
    "CosmosVQVAE",
    "CosmosConfig",
    "CosmosVQVAEConfig",
]
