# coding=utf-8
# Copyright 2025 ByteDance and The HuggingFace Team. All rights reserved.
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


from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, can_return_tuple, is_torch_available
from ..janus.modeling_janus import (
    JanusVQVAEConvDownsample,
    JanusVQVAEConvUpsample,
    JanusVQVAEEncoder,
    JanusVQVAEMidBlock,
    JanusVQVAEResnetBlock,
)
from .configuration_bagel import BagelVQVAEConfig


if is_torch_available():
    import torch
    from torch import nn


@auto_docstring
class BagelPreTrainedModel(PreTrainedModel):
    pass


class BagelVQVAEResnetBlock(JanusVQVAEResnetBlock):
    pass


class BagelVQVAEMidBlock(JanusVQVAEMidBlock):
    pass


class BagelVQVAEConvDownsample(JanusVQVAEConvDownsample):
    pass


class BagelVQVAEConvUpsample(JanusVQVAEConvUpsample):
    pass


class BagelVQVAEEncoder(JanusVQVAEEncoder, nn.Module):
    def __init__(self, config: BagelVQVAEConfig):
        nn.Module().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        in_channels = config.in_channels
        double_latent = config.double_latent
        latent_channels = config.latent_channels
        channel_multiplier = config.channel_multiplier

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

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
                    BagelVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = BagelVQVAEConvDownsample(block_in)
            self.down.append(down)

        self.mid = BagelVQVAEMidBlock(config, block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * latent_channels if double_latent else latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )


class BagelVQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_resolutions = len(config.channel_multiplier)
        self.num_res_blocks = config.num_res_blocks
        base_channels = config.base_channels
        latent_channels = config.latent_channels
        out_channels = config.out_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = base_channels * config.channel_multiplier[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = nn.Conv2d(latent_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = BagelVQVAEMidBlock(config, block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_channels * config.channel_multiplier[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    BagelVQVAEResnetBlock(
                        config=config,
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = BagelVQVAEConvUpsample(block_in)
            self.up.insert(0, up)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        hidden_state = self.conv_in(hidden_state)

        # middle
        hidden_state = self.mid(hidden_state)

        # upsampling
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks + 1):
                hidden_state = self.up[i_level].block[i_block](hidden_state)
                if len(self.up[i_level].attn) > 0:
                    hidden_state = self.up[i_level].attn[i_block](hidden_state)
            if i_level != self.num_resolutions - 1:
                hidden_state = self.up[i_level].upsample(hidden_state)

        hidden_state = self.norm_out(hidden_state)
        hidden_state *= torch.sigmoid(hidden_state)
        hidden_state = self.conv_out(hidden_state)
        return hidden_state


class BagelVQVAEDiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


@auto_docstring(
    custom_intro="""
    The VQ-VAE model used in Bagel for encoding/decoding images into discrete tokens.
    This model follows the "Make-a-scene: Scene-based text-to-image generation with human priors" paper from
    [ Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman](https://arxiv.org/abs/2203.13131).
    """
)
class BagelVQVAE(BagelPreTrainedModel):
    config_class = BagelVQVAEConfig
    _no_split_modules = [
        "BagelVQVAEAttnBlock",
        "BagelVQVAEResnetBlock",
    ]
    main_input_name = "pixel_values"

    def __init__(self, config: BagelVQVAEConfig):
        super().__init__(config)

        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

        self.encoder = BagelVQVAEEncoder(config)
        self.decoder = BagelVQVAEDecoder(config)
        self.eval()  # Bagel's VQ model is frozen
        self.gradient_checkpointing = False

        # Initialize the VQVAE model.
        self.post_init()

    def encode(self, pixel_values: torch.LongTensor):
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.scale_factor * (hidden_states - self.shift_factor)
        return hidden_states

    def decode(self, image_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Decodes quantized token IDs into pixel values.
        Args:
            image_tokens (torch.LongTensor): Batch of token IDs.
        Returns:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                Pixel values decoded from the token IDs.
        """
        # if image_tokens.shape[1] != self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]:
        #     raise ValueError(
        #         f"Expected `image_tokens` to have shape `(batch_size, {self.quantize.quant_state_dims[0] * self.quantize.quant_state_dims[1]})`, "
        #         f"but got shape `{image_tokens.shape}`."
        #     )

        pixel_values = image_tokens / self.scale_factor + self.shift_factor
        pixel_values = self.decoder(pixel_values)
        return pixel_values

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        hidden_states = self.encode(pixel_values)
        hidden_states = self.decode(hidden_states)
        return hidden_states
