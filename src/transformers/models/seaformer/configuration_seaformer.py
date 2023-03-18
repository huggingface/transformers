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
from ...onnx import OnnxConfig
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
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
        depths (`List[int]`, *optional*, defaults to [2, 2, 2, 2]):
            The number of layers in each encoder block.
        sr_ratios (`List[int]`, *optional*, defaults to [8, 4, 2, 1]):
            Sequence reduction ratios in each encoder block.
        hidden_sizes (`List[int]`, *optional*, defaults to [32, 64, 160, 256]):
            Dimension of each of the encoder blocks.
        patch_sizes (`List[int]`, *optional*, defaults to [7, 3, 3, 3]):
            Patch size before each encoder block.
        strides (`List[int]`, *optional*, defaults to [4, 2, 2, 2]):
            Stride before each encoder block.
        num_attention_heads (`List[int]`, *optional*, defaults to [1, 2, 5, 8]):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to [4, 4, 4, 4]):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability before the classification head.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        decoder_hidden_size (`int`, *optional*, defaults to 256):
            The dimension of the all-MLP decode head.
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
        num_heads=8,
        mlp_ratios=[2,4,6],
        attn_ratios = 2,
        act_layer = None,
        in_channels = [128, 192, 256, 320],
        in_index = [0, 1, 2, 3],
        decoder_channels = 192,
        dropout_ratio = 0.1,
        embed_dims = [128, 160, 192],
        is_dw = True,
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
        self.num_heads = num_heads
        self.mlp_ratios = mlp_ratios
        self.attn_ratios = attn_ratios
        self.act_layer = act_layer
        self.in_channels = in_channels
        self.in_index = in_index
        self.dropout_ratio = dropout_ratio
        self.embed_dims = embed_dims
        self.decoder_channels = decoder_channels
        self.is_dw = is_dw
        self.align_corners = align_corners
        self.num_labels = num_labels
        self.hidden_sizes = hidden_sizes
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.reshape_last_stage = kwargs.get("reshape_last_stage", True)
        
class SeaformerOnnxConfig(OnnxConfig):
    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    @property
    def default_onnx_opset(self) -> int:
        return 12
