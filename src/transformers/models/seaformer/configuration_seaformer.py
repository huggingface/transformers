# coding=utf-8
# Copyright 2023 NVIDIA and The HuggingFace Inc. team. All rights reserved.
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
""" Seaformer model configuration"""

import warnings
from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SEAFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "seaformer-large": "https://huggingface.co/seaformer-large/resolve/main/config.json",
}


class SeaformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SeaformerModel`]. It is used to instantiate an
    Seaformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Seaformer
    [nvidia/seaformer-b0-finetuned-ade-512-512](https://huggingface.co/nvidia/seaformer-b0-finetuned-ade-512-512)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_encoder_blocks (`int`, *optional*, defaults to 3):
            The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
        depths (`List[int]`, *optional*, defaults to `[3, 3, 3]`):
            The number of layers in each encoder block.
        num_labels (`int`, *optional*, defaults to 150):
            Number of classes in output
        channels (`List[int]`, *optional*, defaults to `[32, 64, 128, 192, 256, 320]`):
            Number of input channels in each StackedMV2Block
        cfgs (`List[List[List[int]]]`, *optional*, defaults to `[
                [   [3, 3, 32, 1],  
                    [3, 4, 64, 2], 
                    [3, 4, 64, 1]],  
                [
                    [5, 4, 128, 2],  
                    [5, 4, 128, 1]],  
                [
                    [3, 4, 192, 2],  
                    [3, 4, 192, 1]],
                [
                    [5, 4, 256, 2]],  
                [
                    [3, 6, 320, 2]]
            ]`):
            Input parameters [kernel_size, expand_ratio, out_channels, stride]
            for all Inverted Residual blocks within each StackedMV2Block
        emb_dims (`List[int]`, *optional*, defaults to `[192, 256, 320]`): 
            Dimension of Seaformer Attention block
        key_dims (`List[int]`, *optional*, defaults to `[16, 20, 24]`):
            Dimension into which key and query will be projected
        attn_ratios (`int`, *optional*, defaults to 2):
            Ratio of dimension of value to query
        in_channels (`List[int]`, *optional*, defaults to `[128, 192, 256, 320]`):
            Input channels in fusion block
        in_index (`List[int]`, *optional*, defaults to `[0, 1, 2, 3]`):
            Indexes required by decoder head from hidden_states 
        decoder_channels (`int`, *optional*, defaults to 192):
            Dimension of last fusion block output which will be fed to decoder head
        embed_dims (`List[int]`, *optional*, defaults to `[128, 160, 192]`):
            Embedding dimension of Fusion block
        is_depthwise (`bool`, *optional*, defaults to True):
            Flag if set True will perform depthwise convolution
        hidden_sizes (`List[int]`, *optional*, defaults to `[128]`):
            Dimension of each of the encoder blocks.
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[2, 4, 6]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    Example:

    ```python
    >>> from transformers import SeaformerModel, SeaformerConfig

    >>> # Initializing a Seaformer nvidia/seaformer-b0-finetuned-ade-512-512 style configuration
    >>> configuration = SeaformerConfig()

    >>> # Initializing a model from the nvidia/seaformer-b0-finetuned-ade-512-512 style configuration
    >>> model = SeaformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "seaformer"

    def __init__(
        self,
        num_channels=3,
        num_encoder_blocks=3,
        depths=[3, 3, 3],
        num_labels = 150,
        channels = [32, 64, 128, 192, 256, 320],
        cfgs = [
                [   [3, 3, 32, 1],  
                    [3, 4, 64, 2], 
                    [3, 4, 64, 1]],  
                [
                    [5, 4, 128, 2],  
                    [5, 4, 128, 1]],  
                [
                    [3, 4, 192, 2],  
                    [3, 4, 192, 1]],
                [
                    [5, 4, 256, 2]],  
                [
                    [3, 6, 320, 2]]
            ],
        drop_path_rate = 0.1,
        emb_dims = [192, 256, 320],
        key_dims = [16, 20, 24],
        num_attention_heads=8,
        mlp_ratios=[2,4,6],
        attn_ratios = 2,
        act_layer = None,
        in_channels = [128, 192, 256, 320],
        in_index = [0, 1, 2, 3],
        decoder_channels = 192,
        embed_dims = [128, 160, 192],
        is_depthwise = True,
        align_corners = False,
        semantic_loss_ignore_index=255,
        hidden_sizes = [128],
        **kwargs
    ):
        super().__init__(**kwargs)

        if "reshape_last_stage" in kwargs and kwargs["reshape_last_stage"] is False:
            warnings.warn(
                "Reshape_last_stage is set to False in this config. This argument is deprecated and will soon be"
                " removed, as the behaviour will default to that of reshape_last_stage = True.",
                FutureWarning,
            )

        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.channels = channels
        self.cfgs = cfgs
        self.drop_path_rate = drop_path_rate
        self.emb_dims = emb_dims
        self.key_dims = key_dims
        self.num_attention_heads = num_attention_heads
        self.mlp_ratios = mlp_ratios
        self.attn_ratios = attn_ratios
        self.act_layer = act_layer
        self.in_channels = in_channels
        self.in_index = in_index
        self.embed_dims = embed_dims
        self.decoder_channels = decoder_channels
        self.is_depthwise = is_depthwise
        self.align_corners = align_corners
        self.num_labels = num_labels
        self.hidden_sizes = hidden_sizes
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.reshape_last_stage = kwargs.get("reshape_last_stage", True)
