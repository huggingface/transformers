# coding=utf-8
# Copyright 2022 The Tamin Transformers authors and The HuggingFace Inc. team. All rights reserved.
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


import math
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import SiLUActivation
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging, replace_return_docstrings
from .configuration_vqgan import VQGANConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "CompVis/vqgan-imagenet-f16-1024"
_CONFIG_FOR_DOC = "VQGANConfig"

VQGAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "valhalla/vqgan_imagenet_f16_16384",  # TODO: upload this to CompVis org.
    # See all VQGAN models at https://huggingface.co/models?filter=vqgan
]


VQGAN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VQGANConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@dataclass
class VQGANQuantizerOutput(ModelOutput):
    """
    Args:
        quantized_states (`torch.FloatTensor` of shape `(batch_size, channels, latent_height, latent_width)`):
            The quantized states obtained by applying the `VectorQuantizer` to the output of `Encoder`.
        codebook_indices (`torch.LongTensor` of shape `(batch_size, latent_height*latent_width)`):
            The indices in the codebook (embedding) matrix for the `quantized_states`.
        codebook_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            The codebook loss to optimize.
    """

    quantized_states: torch.FloatTensor = None
    codebook_indices: torch.LongTensor = None
    codebook_loss: Optional[torch.FloatTensor] = None


@dataclass
class VQGANModelOutput(ModelOutput):
    """
    Args:
        reconstructed_pixel_values (`torch.FloatTensor` of shape `(batch_size, channels, height, width)`):
            The reconstructed pixel values.
        codebook_indices (`torch.LongTensor` of shape `(batch_size, latent_height*latent_width)`):
            The indices in the codebook (embedding) matrix for the `quantized_states`.
        quantized_states (`torch.FloatTensor` of shape `(batch_size, channels, latent_height, latent_width)`):
            The quantized states obtained by applying the `VectorQuantizer` to the output of `Encoder`.
        codebook_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            The codebook loss to optimize.
    """

    reconstructed_pixel_values: torch.FloatTensor = None
    quantized_states: torch.FloatTensor = None
    codebook_indices: torch.LongTensor = None
    codebook_loss: Optional[torch.FloatTensor] = None


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
            pad = (0, 1, 0, 1)  # pad height and width dim
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

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv1(hidden_states)

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
        conv = partial(nn.Conv2d, self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0)

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
        query = query.reshape((batch, channels, height * width))
        query = query.permute(0, 2, 1)  # (b, hw, c)
        key = key.reshape((batch, channels, height * width))

        attn_weights = torch.bmm(query, key)  # b,hw,hw
        attn_weights = attn_weights * (int(channels) ** -0.5)
        attn_weights = nn.functional.softmax(attn_weights, dim=2)

        # attend to values
        value = value.reshape((batch, channels, height * width))
        attn_weights = attn_weights.permute(0, 2, 1)
        hidden_states = torch.bmm(value, attn_weights)
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
            block_in = self.config.hidden_channels * self.config.channel_mult[-1]
        else:
            block_in = self.config.hidden_channels * self.config.channel_mult[self.block_idx + 1]

        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = []
        attn_blocks = []
        for _ in range(self.config.num_res_blocks + 1):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout_prob=self.config.dropout))
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.block = nn.ModuleList(res_blocks)
        self.attn = nn.ModuleList(attn_blocks)

        self.upsample = None
        if self.block_idx != 0:
            self.upsample = Upsample(block_in, self.config.resample_with_conv)

    def forward(self, hidden_states):
        for i, res_block in enumerate(self.block):
            hidden_states = res_block(hidden_states)
            if len(self.attn) > 1:
                hidden_states = self.attn[i](hidden_states)

        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states


class DownsamplingBlock(nn.Module):
    def __init__(self, config, curr_res: int, block_idx: int):
        super().__init__()

        self.config = config
        self.curr_res = curr_res
        self.block_idx = block_idx

        in_channel_mult = (1,) + tuple(self.config.channel_mult)
        block_in = self.config.hidden_channels * in_channel_mult[self.block_idx]
        block_out = self.config.hidden_channels * self.config.channel_mult[self.block_idx]

        res_blocks = nn.ModuleList()
        attn_blocks = nn.ModuleList()
        for _ in range(self.config.num_res_blocks):
            res_blocks.append(ResnetBlock(block_in, block_out, dropout_prob=self.config.dropout))
            block_in = block_out
            if self.curr_res in self.config.attn_resolutions:
                attn_blocks.append(AttnBlock(block_in))

        self.block = res_blocks
        self.attn = attn_blocks

        self.downsample = None
        if self.block_idx != self.config.num_resolutions - 1:
            self.downsample = Downsample(block_in, self.config.resample_with_conv)

    def forward(self, hidden_states):
        for i, res_block in enumerate(self.block):
            hidden_states = res_block(hidden_states)
            if len(self.attn) > 1:
                hidden_states = self.attn[i](hidden_states)

        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)

        return hidden_states


class MidBlock(nn.Module):
    def __init__(self, config, in_channels: int, dropout: float):
        super().__init__()

        self.config = config
        self.in_channels = in_channels
        self.dropout = dropout

        self.block_1 = ResnetBlock(
            self.in_channels,
            self.in_channels,
            dropout_prob=self.dropout,
        )
        self.attn_1 = AttnBlock(self.in_channels)
        self.block_2 = ResnetBlock(
            self.in_channels,
            self.in_channels,
            dropout_prob=self.dropout,
        )

    def forward(self, hidden_states):
        hidden_states = self.block_1(hidden_states)
        hidden_states = self.attn_1(hidden_states)
        hidden_states = self.block_2(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # downsampling
        self.conv_in = nn.Conv2d(
            self.config.num_channels,
            self.config.hidden_channels,
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
        mid_channels = self.config.hidden_channels * self.config.channel_mult[-1]
        self.mid = MidBlock(config, mid_channels, self.config.dropout)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=mid_channels, eps=1e-6, affine=True)
        self.activation = SiLUActivation()
        self.conv_out = nn.Conv2d(
            mid_channels,
            self.config.z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, pixel_values):
        # downsampling
        hidden_states = self.conv_in(pixel_values)
        for block in self.down:
            hidden_states = block(hidden_states)

        # middle
        hidden_states = self.mid(hidden_states)

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

        # compute in_channel_mult, block_in and curr_res at lowest res
        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
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
        self.mid = MidBlock(config, block_in, self.config.dropout)

        # upsampling
        upsample_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            upsample_blocks.append(UpsamplingBlock(self.config, curr_res, block_idx=i_level))
            if i_level != 0:
                curr_res = curr_res * 2
        self.up = nn.ModuleList(list(reversed(upsample_blocks)))  # reverse to get consistent order

        # end
        block_out = self.config.hidden_channels * self.config.channel_mult[0]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_out, eps=1e-6, affine=True)
        self.activation = SiLUActivation()
        self.conv_out = nn.Conv2d(
            block_out,
            self.config.num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, hidden_states):
        # z to block_in
        hidden_states = self.conv_in(hidden_states)

        # middle
        hidden_states = self.mid(hidden_states)

        # upsampling
        for block in reversed(self.up):
            hidden_states = block(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    Discretization bottleneck part of the VQ-VAE.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        r"""
        Args:
            num_embeddings: number of vectors in the quantized space.
            embedding_dim: dimensionality of the tensors in the quantized space.
                Inputs to the modules must be in this format as well.
            commitment_cost: scalar which controls the weighting of the loss terms
                (see equation 4 in the paper https://arxiv.org/abs/1711.00937 - this variable is Beta).
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, hidden_states, return_loss=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete one-hot vector that is the index of the
        closest embedding vector e_j z (continuous) -> z_q (discrete) z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        hidden_states = hidden_states.permute(0, 2, 3, 1).contiguous()
        hidden_states_flattended = hidden_states.reshape((-1, self.embedding_dim))

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        emb_weights = self.embedding.weight
        distance = (
            torch.sum(hidden_states_flattended**2, dim=1, keepdims=True)
            + torch.sum(emb_weights**2, dim=1)
            - 2 * torch.matmul(hidden_states_flattended, emb_weights.T)
        )

        min_encoding_indices = torch.argmin(distance, axis=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.num_embeddings).to(hidden_states)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(hidden_states.shape)

        # reshape to (batch, num_tokens)
        min_encoding_indices = min_encoding_indices.reshape(hidden_states.shape[0], -1)

        # compute loss for embedding
        loss = None
        if return_loss:
            loss = torch.mean((z_q.detach() - hidden_states) ** 2) + self.commitment_cost * torch.mean(
                (z_q - hidden_states.detach()) ** 2
            )
            # preserve gradients
            z_q = hidden_states + (z_q - hidden_states).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, min_encoding_indices, loss

    def get_codebook_entry(self, indices):
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
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)


@add_start_docstrings(VQGAN_START_DOCSTRING)
class VQGANModel(VQGANPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantize = VectorQuantizer(
            self.config.num_embeddings, self.config.quantized_embed_dim, self.config.commitment_cost
        )
        self.quant_conv = nn.Conv2d(
            self.config.z_channels,
            self.config.quantized_embed_dim,
            kernel_size=1,
        )
        self.post_quant_conv = nn.Conv2d(
            self.config.quantized_embed_dim,
            self.config.z_channels,
            kernel_size=1,
        )

    @replace_return_docstrings(output_type=VQGANQuantizerOutput, config_class=_CONFIG_FOR_DOC)
    def encode(self, pixel_values, return_loss=False, return_dict=True):
        r"""
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained
                using [`VQGANFeatureExtractor`]. See [`VQGANFeatureExtractor.__call__`] for details.
            return_loss (`bool`, *optional*):
                Whether or not to return the codebook loss.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import VQGANFeatureExtractor, VQGANModel

        >>> model = VQGANModel.from_pretrained("CompVis/vqgan-imagenet-f16-1024")
        >>> feature_extractor = VQGANFeatureExtractor.from_pretrained("CompVis/vqgan-imagenet-f16-1024")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = feature_extractor(image, return_tensors="pt")

        >>> outputs = model.encode(inputs["pixel_values"])
        >>> quantized_states = outputs.quantized_states  # the quantized latent vectors
        >>> codebook_indices = outputs.codebook_indices  # indices of the closest codebook vectors
        >>> codebook_loss = outputs.codebook_loss  # the codebook loss
        ```
        """
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states, return_loss)

        if return_dict:
            return VQGANQuantizerOutput(
                quantized_states=quantized_states,
                codebook_indices=codebook_indices,
                codebook_loss=None,
            )
        output = (quantized_states, codebook_indices)
        if return_loss:
            output = output + (codebook_loss,)
        return output

    def decode(self, quantized_states):
        r"""
        Args:
            quantized_states (`torch.FloatTensor` of shape `(batch_size, channels, latent_height, latent_width)`):
            The quantized states obtained by applying the `VectorQuantizer` to the output of `Encoder`.

        Returns:
            reconstructed_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                The reconstructed image.

        Example:
        ```python
        >>> from transformers import VQGANModel

        >>> model = VQGANModel.from_pretrained("CompVis/vqgan-imagenet-f16-1024")

        >>> quantized_states = torch.randn(
        ...     1, model.config.z_channels, model.config.latent_size, model.config.latent_size
        ... )
        >>> reconstructed_pixel_values = model.decode(quantized_states)
        ```
        """
        hidden_states = self.post_quant_conv(quantized_states)
        reconstructed_pixel_values = self.decoder(hidden_states)
        return reconstructed_pixel_values

    def decode_code(self, codebook_indices):
        r"""
        Args:
        Reconstruct the image from the codebook indices.
            codebook_indices (`torch.LongTensor` of shape `(batch_size, num_tokens)`):
                The indices of the codebook vectors. Here `num_tokens=latent_height*latent_width`.
        Returns:
            reconstructed_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                The reconstructed image.

        Example:
        ```python
        >>> from transformers import VQGANModel

        >>> model = VQGANModel.from_pretrained("CompVis/vqgan-imagenet-f16-1024")

        >>> codebook_indices = torch.randint(
        ...     0, model.config.num_embeddings, (1, model.config.latent_size * model.config.latent_size)
        ... )
        >>> reconstructed_pixel_values = model.decode_code(codebook_indices)
        ```
        """
        quantized_states = self.quantize.get_codebook_entry(codebook_indices)
        reconstructed_pixel_values = self.decode(quantized_states)
        return reconstructed_pixel_values

    @replace_return_docstrings(output_type=VQGANModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values, return_loss=False, return_dict=True):
        r"""
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained
                using [`VQGANFeatureExtractor`]. See [`VQGANFeatureExtractor.__call__`] for details.
            return_loss (`bool`, *optional*):
                Whether or not to return the codebook loss.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import VQGANFeatureExtractor, VQGANModel

        >>> model = VQGANModel.from_pretrained("CompVis/vqgan-imagenet-f16-1024")
        >>> feature_extractor = VQGANFeatureExtractor.from_pretrained("CompVis/vqgan-imagenet-f16-1024")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = feature_extractor(image, return_tensors="pt")

        >>> output = model(**inputs)
        >>> reconstructed_pixel_values = output.reconstructed_pixel_values  # this is the reconstructed image
        >>> codebook_loss = output.codebook_loss  # this is the codebook loss to be optimized

        >>> # convert the tensor to PIL image
        >>> reconstructed_pixel_values = torch.clamp(reconstructed_pixel_values.detach(), -1.0, 1.0)
        >>> reconstructed_pixel_values = (reconstructed_pixel_values + 1.0) / 2.0
        >>> reconstructed_pixel_values = reconstructed_pixel_values.transpose(1, 2, 0).cpu().numpy()
        >>> reconstructed_pixel_values = (reconstructed_pixel_values * 255.0).astype(np.uint8)
        >>> reconstructed_image = Image.fromarray(reconstructed_pixel_values)
        ```
        """
        hidden_states = self.encoder(pixel_values)
        hidden_states = self.quant_conv(hidden_states)
        quantized_states, codebook_indices, codebook_loss = self.quantize(hidden_states, return_loss)
        reconstructed_pixel_values = self.decode(quantized_states)

        if return_dict:
            return VQGANModelOutput(
                reconstructed_pixel_values=reconstructed_pixel_values,
                quantized_states=quantized_states,
                codebook_indices=codebook_indices,
                codebook_loss=codebook_loss,
            )

        outputs = (reconstructed_pixel_values, quantized_states, codebook_indices)
        if return_loss:
            outputs = outputs + (codebook_loss,)
        return outputs
