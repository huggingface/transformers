# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
from functools import cached_property
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...generation import GenerationMixin
from ...modeling_outputs import (
    CausalLMOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.deprecation import deprecate_kwarg
from ..chameleon.modeling_chameleon import (
    ChameleonPreTrainedModel,
    ChameleonVQVAEEncoderConvDownsample,
)
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from ..siglip.modeling_siglip import SiglipAttention
from .configuration_emu3 import Emu3Config, Emu3TextConfig, Emu3VQVAEConfig


_CONFIG_FOR_DOC = "Emu3Config"
_CHECKPOINT_FOR_DOC = "BAAI/Emu3-Chat-hf"

logger = logging.get_logger(__name__)


# Has extra dropout which no other model in the library has
class Emu3DecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: Emu3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Emu3VQVAEVectorQuantizer(nn.Module):
    """
    A module for vector quantization using learned embedding vectors.

    This module implements the quantization process similar to te one described in
    the VQ-VAE (Vector Quantized Variational AutoEncoder) paper. It quantizes continuous
    input vectors into discrete codebook vectors, which are learned during training.
    Current implementation improves over previous ones by avoiding costly matrix multiplications
    and allowing for post-hoc remapping of indices.
    """

    def __init__(self, config: Emu3VQVAEConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / config.codebook_size, 1.0 / config.codebook_size)

    def forward(self, hidden_state: torch.Tensor):
        batch_size, temporal, channels, height, width = hidden_state.shape
        hidden_state = hidden_state.permute(0, 1, 3, 4, 2).contiguous()
        hidden_state_flattened = hidden_state.view(-1, channels)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        hidden_state_sum = torch.sum(hidden_state_flattened**2, dim=1, keepdim=True)
        embedding_sum = torch.sum(self.embedding.weight**2, dim=1)

        # "bd,dn->bn",
        distances = 2 * torch.matmul(hidden_state_flattened, self.embedding.weight.transpose(0, 1))
        distances = hidden_state_sum + embedding_sum - distances

        min_encoding_indices = torch.argmin(distances, dim=1)
        min_encoding_indices = min_encoding_indices.view(batch_size, temporal, height, width)
        return min_encoding_indices


class Emu3VQVAEEncoderConvDownsample(ChameleonVQVAEEncoderConvDownsample):
    pass


class Emu3VQVAEEncoderConvUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Emu3VQVAEConv3d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int],
        stride: Tuple[int],
    ):
        super().__init__()

        padding_sizes = [one_kernel - one_stride for one_kernel, one_stride in zip(kernel_size[1:], stride[1:])]
        self.padding = ()
        for pad_size in padding_sizes[::-1]:
            self.padding += (pad_size // 2 + pad_size % 2, pad_size // 2)
        self.padding += (2, 0)

        self.conv = nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = F.pad(hidden_states, self.padding)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Emu3VQVAESpatialNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_channels=out_channels,
            num_groups=32,
            eps=1e-6,
            affine=True,
        )

        self.conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_b = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, hidden_states: torch.Tensor, quant_states: torch.Tensor):
        quant_states = F.interpolate(quant_states, size=hidden_states.shape[-2:], mode="nearest")
        hidden_states = self.norm_layer(hidden_states)
        hidden_states = hidden_states * self.conv_y(quant_states) + self.conv_b(quant_states)
        return hidden_states


class Emu3VQVAETemporalUpsample(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ):
        super().__init__()
        self.conv = Emu3VQVAEConv3d(
            in_channel,
            out_channel,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, channels, temporal, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 1, 3, 4, 2).contiguous().view(batch_size, -1, temporal)
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = hidden_states.view(batch_size, channels, height, width, -1).permute(0, 1, 4, 2, 3).contiguous()
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Emu3VQVAETemporalDownsample(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
    ):
        super().__init__()
        self.conv = Emu3VQVAEConv3d(
            in_channel,
            out_channel,
            kernel_size=(4, 3, 3),
            stride=(2, 1, 1),
        )

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.conv(hidden_states)
        return hidden_states


class Emu3VQVAETemporalResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.BatchNorm3d(in_channels)
        self.conv1 = Emu3VQVAEConv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
        )
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.conv2 = Emu3VQVAEConv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)

        return residual + hidden_states


class Emu3VQVAEResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        quant_channels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.quant_channels = quant_channels

        if quant_channels is None:
            self.norm1 = nn.GroupNorm(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, num_groups=32, eps=1e-6, affine=True)
        else:
            self.norm1 = Emu3VQVAESpatialNorm(quant_channels, in_channels)
            self.norm2 = Emu3VQVAESpatialNorm(quant_channels, out_channels)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, hidden_states: torch.Tensor, quant_channels: Optional[torch.Tensor] = None):
        norm_args = () if self.quant_channels is None else (quant_channels,)

        residual = hidden_states
        hidden_states = self.norm1(hidden_states, *norm_args)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states, *norm_args)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)

        return residual + hidden_states


class Emu3VQVAEAttentionBlock(SiglipAttention):
    pass


class Emu3VQVAEGroupNorm(nn.GroupNorm):
    """
    Same as the torch GroupNorm with the only difference that this ones accepts
    an optional kwarg `quant_states` which is not used. This class makes it easier to
    use SpatialNorm or GroupNorm without conditionals
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input, quant_states=None):
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)


class Emu3VQVAEMiddleBlock(nn.Module):
    def __init__(self, config, in_channels, quant_channels=None):
        super().__init__()

        self.block_1 = Emu3VQVAEResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            quant_channels=quant_channels,
        )
        self.attn_1 = Emu3VQVAEAttentionBlock(config)
        if quant_channels is None:
            self.attn_norm = Emu3VQVAEGroupNorm(num_channels=in_channels, num_groups=32, eps=1e-6, affine=True)
        else:
            self.attn_norm = Emu3VQVAESpatialNorm(quant_channels, in_channels)

        self.block_2 = Emu3VQVAEResnetBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            quant_channels=quant_channels,
        )

    def forward(self, hidden_states: torch.FloatTensor, quant_states: torch.FloatTensor = None):
        hidden_states = self.block_1(hidden_states, quant_states)
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states, quant_states)
        batch_size, channels, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
        hidden_states = self.attn_1(hidden_states)[0]
        hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        hidden_states = residual + hidden_states
        hidden_states = self.block_2(hidden_states, quant_states)
        return hidden_states


class Emu3VQVAEDownBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        channel_multiplier = config.channel_multiplier

        in_channel_multiplier = (1,) + tuple(channel_multiplier)
        self.in_channel_multiplier = in_channel_multiplier
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            attn_norms = nn.ModuleList()
            block_in = base_channels * in_channel_multiplier[i_level]
            block_out = base_channels * channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Emu3VQVAEResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if config.attn_resolutions is not None and i_level in config.attn_resolutions:
                    attn.append(Emu3VQVAEAttentionBlock(config))
                    attn_norms.append(nn.GroupNorm(num_channels=block_in, num_groups=32, eps=1e-6, affine=True))

            down = nn.Module()
            down.block = block
            down.attn = attn
            down.attn_norms = attn_norms
            if i_level != self.num_resolutions - 1:
                down.downsample = Emu3VQVAEEncoderConvDownsample(block_in)
            self.down.append(down)

    def forward(self, hidden_states: torch.FloatTensor):
        for i_level, blocks in enumerate(self.down):
            for i_block in range(self.num_res_blocks):
                hidden_states = blocks.block[i_block](hidden_states)
                if len(blocks.attn) > 0:
                    residual = hidden_states
                    hidden_states = blocks.attn_norms[i_block](hidden_states)

                    batch_size, channels, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
                    hidden_states = blocks.attn[i_block](hidden_states)[0]

                    hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
                    hidden_states = residual + hidden_states

            if i_level != self.num_resolutions - 1:
                hidden_states = blocks.downsample(hidden_states)

        return hidden_states


class Emu3VQVAEUpBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks

        quant_channels = config.embed_dim
        block_in = config.base_channels * config.channel_multiplier[-1]

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            attn_norms = nn.ModuleList()
            block_out = config.base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Emu3VQVAEResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        quant_channels=quant_channels,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VQVAEAttentionBlock(config))
                    attn_norms.append(Emu3VQVAESpatialNorm(quant_channels, block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            up.attn_norms = attn_norms
            if i_level != 0:
                up.upsample = Emu3VQVAEEncoderConvUpsample(block_in)

            self.up.insert(0, up)

    def forward(self, hidden_states: torch.FloatTensor, quant_states: torch.FloatTensor):
        for i_level, blocks in enumerate(self.up[::-1]):
            for i_block in range(self.num_res_blocks + 1):
                hidden_states = blocks.block[i_block](hidden_states, quant_states)
                if len(blocks.attn) > 0:
                    residual = hidden_states
                    hidden_states = blocks.attn_norms[i_block](hidden_states, quant_states)

                    batch_size, channels, height, width = hidden_states.shape
                    hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
                    hidden_states = blocks.attn[i_block](hidden_states)[0]

                    hidden_states = hidden_states.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
                    hidden_states = residual + hidden_states
            if i_level != len(self.up) - 1:
                hidden_states = blocks.upsample(hidden_states)

        return hidden_states


class Emu3VQVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        base_channels = config.base_channels
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier
        out_channels = 2 * latent_channels if double_latent else latent_channels
        block_in = base_channels * channel_multiplier[-1]

        self.conv_in = torch.nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.down_block = Emu3VQVAEDownBlock(config)
        self.middle_block = Emu3VQVAEMiddleBlock(config, block_in)

        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        temporal_down_blocks = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.ModuleList()
        self.time_res_stack = nn.ModuleList()

        for i in range(temporal_down_blocks):
            conv = Emu3VQVAETemporalDownsample(out_channels, out_channels)
            self.time_conv.append(conv)

        for _ in range(config.num_res_blocks):
            time_res_conv = Emu3VQVAETemporalResnetBlock(
                in_channels=out_channels,
                out_channels=out_channels,
            )
            self.time_res_stack.append(time_res_conv)

    def forward(self, pixel_values: torch.LongTensor):
        temporal_dim = pixel_values.shape[1]
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])

        # downsampling & middle
        hidden_states = self.conv_in(pixel_values)
        hidden_states = self.down_block(hidden_states)
        hidden_states = self.middle_block(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        hidden_states = hidden_states.reshape(-1, temporal_dim, *hidden_states.shape[1:])
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        # temporal convs
        for conv in self.time_conv:
            hidden_states = conv(hidden_states)
            hidden_states *= torch.sigmoid(hidden_states)

        for layer in self.time_res_stack:
            hidden_states = layer(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        return hidden_states


class Emu3VQVAEDecoder(nn.Module):
    def __init__(self, config: Emu3VQVAEConfig):
        super().__init__()

        quant_channels = config.embed_dim
        block_in = config.base_channels * config.channel_multiplier[-1]
        self.time_res_stack = nn.ModuleList()
        for _ in range(config.num_res_blocks):
            time_res_conv = Emu3VQVAETemporalResnetBlock(
                in_channels=config.latent_channels, out_channels=config.latent_channels
            )
            self.time_res_stack.append(time_res_conv)

        temp_upsample_block_num = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.ModuleList()
        for i in range(temp_upsample_block_num):
            conv = Emu3VQVAETemporalUpsample(config.latent_channels, config.latent_channels)
            self.time_conv.append(conv)

        self.conv_in = nn.Conv2d(
            config.latent_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.middle_block = Emu3VQVAEMiddleBlock(config, block_in, quant_channels=quant_channels)
        self.up_block = Emu3VQVAEUpBlock(config)

        block_in = config.base_channels * config.channel_multiplier[0]
        self.norm_out = Emu3VQVAESpatialNorm(quant_channels, block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor, quant_states: torch.Tensor):
        hidden_quant_states = torch.cat((hidden_states, quant_states), dim=0)
        hidden_quant_states = hidden_quant_states.permute(0, 2, 1, 3, 4)

        # temporal convs
        for layer in self.time_res_stack:
            hidden_quant_states = layer(hidden_quant_states)

        for layer in self.time_conv:
            hidden_quant_states = layer(hidden_quant_states)
            hidden_quant_states *= torch.sigmoid(hidden_quant_states)

        hidden_quant_states = hidden_quant_states.permute(0, 2, 1, 3, 4)
        hidden_states, quant_states = torch.chunk(hidden_quant_states, 2, dim=0)
        hidden_states = hidden_states.reshape(-1, *hidden_states.shape[2:])
        quant_states = quant_states.reshape(-1, *quant_states.shape[2:])

        hidden_states = self.conv_in(hidden_states)

        # middle & upsampling
        hidden_states = self.middle_block(hidden_states, quant_states)
        hidden_states = self.up_block(hidden_states, quant_states)

        hidden_states = self.norm_out(hidden_states, quant_states)
        hidden_states *= torch.sigmoid(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


EMU3_VQ_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Emu3VQVAEConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    """The VQ-VAE model used in Emu3 for encoding/decoding images into discrete tokens.
    This model follows the "Make-a-scene: Scene-based text-to-image generation with human priors" paper from
    [ Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman](https://arxiv.org/abs/2203.13131).
    """,
    EMU3_VQ_START_DOCSTRING,
)
class Emu3VQVAE(PreTrainedModel):
    config_class = Emu3VQVAEConfig
    base_model_prefix = "emuvideovq"
    main_input_name = "pixel_values"
    _no_split_modules = [
        "Emu3VQVAETemporalResnetBlock",
        "Emu3VQVAEAttentionBlock",
        "Emu3VQVAEResnetBlock",
        "Emu3VQVAEVectorQuantizer",
    ]

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def __init__(self, config: Emu3VQVAEConfig):
        super().__init__(config)

        self.config = config

        self.encoder = Emu3VQVAEEncoder(config)
        self.decoder = Emu3VQVAEDecoder(config)
        self.quantize = Emu3VQVAEVectorQuantizer(config)
        self.vision_spatial_factor = 2 ** (len(config.channel_multiplier) - 1)

        self.quant_conv = Emu3VQVAEConv3d(
            config.latent_channels, config.embed_dim, kernel_size=(3, 1, 1), stride=(1, 1, 1)
        )
        self.post_quant_conv = Emu3VQVAEConv3d(
            config.embed_dim, config.latent_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1)
        )
        self.spatial_scale_factor = 2 ** (len(config.channel_multiplier) - 1)
        self.eval()  # Emu3's VQ model is frozen

        self.post_init()

    def encode(self, pixel_values: torch.Tensor, image_sizes: torch.Tensor):
        is_image = pixel_values.ndim == 4
        if is_image:
            temporal = self.config.temporal_downsample_factor
            batch_size, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.unsqueeze(1).repeat(1, temporal, 1, 1, 1)
        else:
            batch_size, temporal, channels, height, width = pixel_values.shape

        hidden_states = self.encoder(pixel_values)

        # b t c h w -> b c t h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = self.quant_conv(hidden_states)

        # b c t h w -> b t c h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        codes = self.quantize(hidden_states)

        image_tokens = codes.squeeze(1) if is_image else codes

        image_tokens = [
            single_image[: int(size[0] / self.vision_spatial_factor), : int(size[1] / self.vision_spatial_factor)]
            for single_image, size in zip(image_tokens, image_sizes)
        ]

        return image_tokens

    def decode(self, hidden_states: torch.Tensor):
        is_image = hidden_states.ndim == 3
        if is_image:
            hidden_states = hidden_states.unsqueeze(1)

        batch_size, temporal, height, width = hidden_states.shape
        quant = self.quantize.embedding(hidden_states.flatten())

        channels = quant.shape[-1]
        quant = quant.view(batch_size, temporal, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()
        post_quant = self.post_quant_conv(quant)

        quant = quant.permute(0, 2, 1, 3, 4)
        post_quant = post_quant.permute(0, 2, 1, 3, 4)

        video = self.decoder(post_quant, quant)
        video = video.reshape(
            batch_size,
            temporal * self.config.temporal_downsample_factor,
            self.config.out_channels,
            height * self.spatial_scale_factor,
            width * self.spatial_scale_factor,
        )
        return video[:, 0] if is_image else video


class Emu3ImageVocabularyMapping:
    """
    A class for mapping discrete image tokens from VQGAN to BPE tokens.
    """

    def __init__(self, vocab_map):
        self.vocab_map = vocab_map
        self.eol_token_id = vocab_map.get("<|extra_200|>")
        self.image_token_id = vocab_map.get("<image>")

    @cached_property
    def image_tokens(self):
        return sorted([val for name, val in self.vocab_map.items() if name.startswith("<|visual token")])

    @cached_property
    def image_tokens_str(self):
        return sorted([name for name, val in self.vocab_map.items() if name.startswith("<|visual token")])

    @cached_property
    def img2bpe(self):
        return {int(token[-8:-2]): self.vocab_map[token] for token in self.image_tokens_str}

    @cached_property
    def bpe2img(self):
        return {v: k for k, v in self.img2bpe.items()}

    @cached_property
    def bpe2img_mapping_tensor(self):
        mapping = torch.zeros(max(self.bpe2img.keys()) + 1, dtype=torch.int)
        for k, v in self.bpe2img.items():
            mapping[k] = v
        return mapping

    @cached_property
    def img2bpe_mapping_tensor(self):
        mapping = torch.zeros(max(self.img2bpe.keys()) + 1, dtype=torch.int)
        for k, v in self.img2bpe.items():
            mapping[k] = v
        return mapping

    def convert_img2bpe(self, img_batch: List[torch.Tensor]) -> torch.Tensor:
        device = img_batch.device
        eol_row = torch.ones((img_batch.shape[0], 1), dtype=torch.int) * self.eol_token_id
        img_tokens = self.img2bpe_mapping_tensor[img_batch.to("cpu")]
        img_tokens = torch.cat([img_tokens, eol_row], dim=-1)
        return img_tokens.to(device)

    def convert_bpe2img(self, img_batch: torch.Tensor) -> torch.Tensor:
        device = img_batch.device
        img_batch = img_batch[..., :-1]  # remove last row of EOL tokens
        img_tokens = self.bpe2img_mapping_tensor[img_batch.to("cpu")]
        return img_tokens.to(device)


class Emu3PreTrainedModel(ChameleonPreTrainedModel, Emu3VQVAE):
    _no_split_modules = [
        "Emu3DecoderLayer",
    ]
    _supports_flex_attn = True

    def _init_weights(self, module):
        std = self.config.get_text_config().initializer_range
        if isinstance(module, Emu3VQVAE):
            module.apply(module._init_weights)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


EMU3_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Has to be an instance of [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            The model will output the same cache type that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
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
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


EMU3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        pixel_values (`torch.FloatTensor` of shape `(batch_size, max_num_images, max_num_tiles, channels, image_size, image_size)):
            The tensors corresponding to the input images. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Emu3ImageProcessor.__call__`] for details ([]`Emu3Processor`] uses
            [`Emu3ImageProcessor`] for processing images).
        image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`):
                The sizes of the images in the batch, being (height, width) for each image. Image sizes can be obtained using
            [`AutoImageProcessor`]. See [`Emu3ImageProcessor.__call__`] for details ([]`Emu3Processor`] uses
            [`Emu3ImageProcessor`] for processing images).
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Has to be an instance of [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
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
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


class Emu3TextModel(LlamaModel, Emu3PreTrainedModel):
    def __init__(self, config: Emu3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Emu3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    @add_start_docstrings_to_model_forward(EMU3_TEXT_INPUTS_DOCSTRING)
    def forward(self, **super_kwargs):
        super().forward(**super_kwargs)


class Emu3ForCausalLM(LlamaForCausalLM, Emu3PreTrainedModel, GenerationMixin):
    config_class = Emu3TextConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Emu3TextModel(config)

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(EMU3_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="Emu3TextConfig")
    def forward(**super_kwargs):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import Emu3Processor, Emu3ForConditionalGeneration
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> model = Emu3ForCausalLM.from_pretrained("BAAI/Emu3-Chat-hf", torch_dtype=torch.bfloat16)
        >>> processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")

        >>> inputs = processor(text=["Can you write me a poem about winter."], return_tensors="pt").to(model.device)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        >>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        super().forward()


class Emu3ForConditionalGeneration(Emu3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["text_model.lm_head.weight"]
    _supports_static_cache = False  # `get_image_tokens()`, called when `pixel_values` is passed, is not compileable

    def __init__(self, config):
        super().__init__(config)
        self.text_model = Emu3ForCausalLM._from_config(config.text_config)
        self.vqmodel = Emu3VQVAE(config.vq_config)
        self.vocabulary_mapping = Emu3ImageVocabularyMapping(config.vocabulary_map)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_image_tokens(self, pixel_values: torch.FloatTensor, image_sizes: torch.LongTensor):
        """
        Tokenizes images into discrete tokens with VQGAN module. Converts
        obtained image tokens into BPE tokens and wraps with "boi" and "eoi"
        special tokens.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_sizes (`torch.LongTensor` of shape `(batch_size, 2)`):
                The sizes of the images in the batch, being (height, width) for each image.
        """
        image_tokens_list = self.vqmodel.encode(pixel_values, image_sizes)
        bpe_tokens_list = [self.vocabulary_mapping.convert_img2bpe(tokens).flatten() for tokens in image_tokens_list]
        bpe_tokens = torch.cat(bpe_tokens_list)
        return bpe_tokens

    @torch.no_grad
    def decode_image_tokens(self, image_tokens: torch.LongTensor, height: int, width: int):
        """
        Decodes generated image tokens from language model to continuous pixel values
        with VQGAN module via upsampling.

        Args:
            image_tokens (`torch.LongTensor` of shape `(batch_size, num_of_tokens)`):
                The tensors corresponding to the input images.
            height (`int`):
                Height of the generated image before upsampling.
            width (`int`):
                Width of the generated image before upsampling.
        """
        sequences = image_tokens[:, :-3].view(-1, height, width + 1)
        image_tokens = self.vocabulary_mapping.convert_bpe2img(sequences)
        image = self.vqmodel.decode(image_tokens)
        return image

    @add_start_docstrings_to_model_forward(EMU3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import Emu3Processor, Emu3ForConditionalGeneration
        >>> import torch
        >>> import requests
        >>> from PIL import Image

        >>> model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", torch_dtype=torch.bfloat16)
        >>> processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")

        >>> conversation = [
        ...     {
        ...     "role": "system",
        ...     "content": [
        ...         {"type": "text", "text": "You are a helpful assistant."},
        ...         ],
        ...     },
        ...     {
        ...     "role": "user",
        ...     "content": [
        ...         {"type": "image"},
        ...         {"type": "text", "text": "Please describe the image."},
        ...         ],
        ...     },
        ... ]

        >>> prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        >>> image = Image.open(requests.get("https://www.ilankelman.org/stopsigns/australia.jpg", stream=True).raw)

        >>> inputs = processor(images=[image], text=[prompt], return_tensors="pt").to(model.device, torch.bfloat16)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        >>> processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None:
            image_tokens = self.get_image_tokens(pixel_values, image_sizes)
            special_image_mask = input_ids == self.vocabulary_mapping.image_token_id
            image_tokens = image_tokens.to(input_ids.device, input_ids.dtype)
            input_ids = input_ids.masked_scatter(special_image_mask, image_tokens)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            use_cache=use_cache,
            **kwargs,
        )

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None

        return model_inputs


__all__ = [
    "Emu3ForConditionalGeneration",
    "Emu3ForCausalLM",
    "Emu3TextModel",
    "Emu3PreTrainedModel",
    "Emu3VQVAE",
]
