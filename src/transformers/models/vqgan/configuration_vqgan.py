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
        ch: int = 128,
        out_ch: int = 3,
        in_channels: int = 3,
        num_res_blocks: int = 2,
        resolution: int = 256,
        z_channels: int = 256,
        ch_mult: Tuple = (1, 1, 2, 2, 4),
        attn_resolutions: int = (16,),
        n_embed: int = 1024,
        embed_dim: int = 256,
        dropout: float = 0.0,
        double_z: bool = False,
        resamp_with_conv: bool = True,
        give_pre_end: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ch = ch
        self.out_ch = out_ch
        self.in_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.z_channels = z_channels
        self.ch_mult = list(ch_mult)
        self.attn_resolutions = list(attn_resolutions)
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.double_z = double_z
        self.resamp_with_conv = resamp_with_conv
        self.give_pre_end = give_pre_end
        self.num_resolutions = len(ch_mult)
