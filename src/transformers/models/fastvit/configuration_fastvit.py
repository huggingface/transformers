# coding=utf-8
# Copyright 2023 Apple and The HuggingFace Inc. team. All rights reserved.
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
""" FastViT model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

FASTVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "apple/fastvit-t8": "https://huggingface.co/apple/fastvit-t8/resolve/main/config.json",
    "apple/fastvit-t12": "https://huggingface.co/apple/fastvit-t12/resolve/main/config.json",
    "apple/fastvit-s12": "https://huggingface.co/apple/fastvit-s12/resolve/main/config.json",
    "apple/fastvit-sa12": "https://huggingface.co/apple/fastvit-sa12/resolve/main/config.json",
    "apple/fastvit-sa24": "https://huggingface.co/apple/fastvit-sa24/resolve/main/config.json",
    "apple/fastvit-sa36": "https://huggingface.co/apple/fastvit-sa36/resolve/main/config.json",
    "apple/fastvit-ma36": "https://huggingface.co/apple/fastvit-ma36/resolve/main/config.json",
    # See all FastViT models at https://huggingface.co/models?filter=fastvit
}


class FastViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FastViTModel`]. It is used to instantiate an
    FastViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the FastViT
    [JorgeAV/fastvit_t8](https://huggingface.co/JorgeAV/fastvit_t8) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        image_size (`int`, *optional*, defaults to 256):
            The size (resolution) of each image.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        depths (`List[int]`, *optional*, defaults to `[2, 2, 4, 2]`):
            The number of Token Mixer blocks in each FastViTLayer Block.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_sizes (`List[int]`, *optional*, defaults to `[48, 96, 192, 384]`):
            Dimensionality of the encoder layers in each FastViTLayer Block.
        pos_embeds (`List[Bool]`, *optional*):
            Whether to add a Conditional Positional Encoding in each FastViTLayer Block. `"RepCPE"` is the option to
            put CPE in a specific Layer.
        token_mixers (`List[str]` *optional*, defaults to `['repmixer', 'repmixer', 'repmixer', 'repmixer']`):
            Whether to use RepMixer block or Attention block per each FastViTLayer Block. `"repmixer"` and
            `"attention"` are supported.
        mlp_ratio (`float`, *optional*, defaults to 3.0):
            The ratio of the number of channels in the output of the MLP to the number of channels in the input.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries, keys and values.
        inference (`bool`, *optional*, defaults to `False`):
            Whether to delete batchnorms and residual connections for much faster inference (more info in the paper)

    Example:
    ```python
    >>> from transformers import FastViTConfig, FastViTModel

    >>> # Initializing a fastvit-t8 style configuration
    >>> configuration = FastViTConfig()

    >>> # Initializing a model (with random weights) from the fastvit-t8 style configuration
    >>> model = FastViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "fastvit"

    def __init__(
        self,
        image_size=256,
        num_channels=3,
        depths=[2, 2, 4, 2],
        num_attention_heads=32,
        hidden_sizes=[48, 96, 192, 384],
        pos_embeds=None,
        token_mixers=["repmixer", "repmixer", "repmixer", "repmixer"],
        mlp_ratio=3.0,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        qkv_bias=False,
        inference=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depths = depths
        self.hidden_sizes = hidden_sizes
        self.mlp_ratio = mlp_ratio
        self.pos_embeds = pos_embeds
        self.token_mixers = token_mixers
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.num_attention_heads = num_attention_heads
        self.inference = inference


class FastViTOnnxConfig(OnnxConfig):
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
