# coding=utf-8
# Copyright 2023 Google AI and The HuggingFace Inc. team. All rights reserved.
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
""" SegGPT model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SEGGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "BAAI/SegGPT": "https://huggingface.co/BAAI/SegGPT/resolve/main/config.json",
}


class SegGPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SegGPTModel`]. It is used to instantiate an SegGPT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the SegGPT
    [google/seggpt-base-patch16-224](https://huggingface.co/google/seggpt-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to `(896, 448)`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        mlp_ratio (`<fill_type>`, *optional*, defaults to 4.0): <fill_docstring>
        drop_path_rate (`<fill_type>`, *optional*, defaults to 0.1): <fill_docstring>
        pretrain_img_size (`<fill_type>`, *optional*, defaults to 224): <fill_docstring>
        decoder_hidden_size (`<fill_type>`, *optional*, defaults to 64): <fill_docstring>
        use_rel_pos (`<fill_type>`, *optional*, defaults to `True`): <fill_docstring>
        merge_index (`<fill_type>`, *optional*, defaults to 2): <fill_docstring>
        encoder_output_indicies (`<fill_type>`, *optional*, defaults to `[5, 11, 17, 23]`): <fill_docstring>
        beta (`<fill_type>`, *optional*, defaults to 0.01): <fill_docstring>

    Example:

    ```python
    >>> from transformers import SegGPTConfig, SegGPTModel

    >>> # Initializing a SegGPT seggpt-base-patch16-224 style configuration
    >>> configuration = SegGPTConfig()

    >>> # Initializing a model (with random weights) from the seggpt-base-patch16-224 style configuration
    >>> model = SegGPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "seggpt"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=(896, 448),
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        pretrain_img_size=224,
        decoder_hidden_size=64,
        use_rel_pos=True,
        merge_index=2,
        encoder_output_indicies=[5, 11, 17, 23],
        beta=0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.pretrain_img_size = pretrain_img_size
        self.decoder_hidden_size = decoder_hidden_size
        self.use_rel_pos = use_rel_pos
        if merge_index > min(encoder_output_indicies):
            raise ValueError(
                f"Merge index must be less than the minimum encoder output index, but got {merge_index=} and {encoder_output_indicies=}"
            )
        self.merge_index = merge_index
        self.encoder_output_indicies = encoder_output_indicies


class SegGPTOnnxConfig(OnnxConfig):
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
