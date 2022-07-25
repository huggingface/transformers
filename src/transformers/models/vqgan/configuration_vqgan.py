# coding=utf-8
# Copyright 2022 The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
    "vqgan-imagenet-f16-1024": "https://huggingface.co/vqgan-imagenet-f16-1024/resolve/main/config.json",
    # See all VQGAN models at https://huggingface.co/models?filter=vqgan
}


class VQGANConfig(PretrainedConfig):
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
        self.resamp_with_conv = resample_with_conv
        self.num_resolutions = len(channel_mult)
