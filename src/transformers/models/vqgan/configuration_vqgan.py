# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" VQGAN model configuration"""

from typing import Tuple

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

VQGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "valhalla/vqgan_imagenet_f16_16384": (  # TODO: upload this to CompVis org.
        "https://huggingface.co/valhalla/vqgan_imagenet_f16_16384/resolve/main/config.json"
    ),
    # See all VQGAN models at https://huggingface.co/models?filter=vqgan
}


class VQGANConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VQGANModel`]. It is used to instantiate an VQGAN
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the VQGAN
    [valhalla/vqgan_imagenet_f16_16384](https://huggingface.co/valhalla/vqgan_imagenet_f16_16384) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        resolution (`int`, *optional*, defaults to 256):
            The resolution of the input image.
        num_channels (`int`, *optional*, defaults to 3):
            The number of channels of the input image.
        hidden_channels (`int`, *optional*, defaults to 128):
            The number of channels of the hidden representation.
        channel_mult (`tuple`, *optional*, defaults to (1, 1, 2, 2, 4)):
            The channel multipliers for the hidden representation.
        num_res_blocks (`int`, *optional*, defaults to 2):
            The number of residual blocks.
        attn_resolutions (`tuple`, *optional*, defaults to (16,)):
            The resolutions of the attention heads.
        z_channels (`int`, *optional*, defaults to 256):
            The number of channels of the quantized (latent) representation.
        num_embeddings (`int`, *optional*, defaults to 1024):
            The number of embedding vectors in the quantized (latent) space.
        quantized_embed_dim (`int`, *optional*, defaults to 256):
            The dimension of the quantized (latent) embedding vectors.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability.
        resample_with_conv (`bool`, *optional*, defaults to True):
            Whether to use convolutional upsampling/downsampling.
        commitment_cost (`float`, *optional*, defaults to 0.25):
            Scalar which controls the weighting of the loss terms in the codebook loss.
    """

    def __init__(
        self,
        resolution: int = 256,
        num_channels: int = 3,
        hidden_channels: int = 128,
        channel_mult: Tuple = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: int = (16,),
        z_channels: int = 256,
        num_embeddings: int = 1024,
        quantized_embed_dim: int = 256,
        dropout: float = 0.0,
        resample_with_conv: bool = True,
        commitment_cost: float = 0.25,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.z_channels = z_channels
        self.channel_mult = list(channel_mult)
        self.attn_resolutions = list(attn_resolutions)
        self.num_embeddings = num_embeddings
        self.quantized_embed_dim = quantized_embed_dim
        self.dropout = dropout
        self.resample_with_conv = resample_with_conv
        self.commitment_cost = commitment_cost

    @property
    def num_resolutions(self):
        return len(self.channel_mult)

    @property
    def reduction_factor(self):
        return 2 ** (self.num_resolutions - 1)

    @property
    def latent_size(self):
        return self.resolution // self.reduction_factor
