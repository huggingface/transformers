# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""VJEPA 2 model configuration"""

from ...configuration_utils import PretrainedConfig


class VJEPA2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VJEPA2Model`]. It is used to instantiate an
    VJEPA2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VJEPA2
    [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        crop_size (`int`, *optional*, defaults to 256):
            Input resolution of the model
        frames_per_clip (`int`, *optional*, defaults to 64):
            The number of frames the model has been pretrained with. Does not impact inference.
        tubelet_size (`int`, *optional*, defaults to 2):
            The number of temporal frames used for a single rastor, check paper for more information.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimensionality of the encoder layers
        in_chans (`int`, *optional*, defaults to 3):
            The number of input channels
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Encoder
        num_hidden_layers (`int`, *optional*, defaults to 24):
            The number of hidden layers
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of the hidden size of the MLPs used in Encoder relative to the `hidden_size`.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for attentions.
            The dropout probability for all fully connected layers.
        hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for attentions.
        num_pooler_layers (`int`, *optional*, defaults to 3):
            The number of self-attention layers in the pooler.
        pred_hidden_size (`int`, *optional*, defaults to 384):
            Dimensionality of the predictor layers
        pred_num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Predictor
        pred_num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Predictor
        pred_num_mask_tokens (`int`, *optional*, defaults to 10):
            Define the number of mask tokens to use in the Predictor
        pred_zero_init_mask_tokens (`bool`, *optional*, defaults to `True`):
            Initialize the mask tokens in the predictor with 0.
        pred_mlp_ratio (`float`, *optional*, defaults to 4.0):
            Ratio of the hidden size of the MLPs used in Predictor relative to the `pred_hidden_size`.

    Example:

    ```python
    >>> from transformers import VJEPA2Config, VJEPA2Model

    >>> # Initializing a VJEPA2 vjepa2-vitl-fpc64-256 style configuration
    >>> configuration = VJEPA2Config()

    >>> # Initializing a model (with random weights) from the vjepa2-vitl-fpc64-256  style configuration
    >>> model = VJEPA2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vjepa2"

    def __init__(
        self,
        patch_size=16,
        crop_size=256,
        frames_per_clip=64,
        tubelet_size=2,
        hidden_size=1024,
        in_chans=3,
        num_attention_heads=16,
        num_hidden_layers=24,
        drop_path_rate=0.0,
        mlp_ratio=4.0,
        layer_norm_eps=1e-6,
        qkv_bias=True,
        attention_probs_dropout_prob=0.0,
        hidden_act="gelu",
        initializer_range=0.02,
        attention_dropout=0.0,
        num_pooler_layers=3,
        # predictor params
        pred_hidden_size=384,
        pred_num_attention_heads=12,
        pred_num_hidden_layers=12,
        pred_num_mask_tokens=10,
        pred_zero_init_mask_tokens=True,
        pred_mlp_ratio=4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.crop_size = crop_size
        self.frames_per_clip = frames_per_clip
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.drop_path_rate = drop_path_rate
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.image_size = crop_size
        self.attention_dropout = attention_dropout
        self.num_pooler_layers = num_pooler_layers
        # predictor params
        self.pred_hidden_size = pred_hidden_size
        self.pred_num_attention_heads = pred_num_attention_heads
        self.pred_num_hidden_layers = pred_num_hidden_layers
        self.pred_num_mask_tokens = pred_num_mask_tokens
        self.pred_zero_init_mask_tokens = pred_zero_init_mask_tokens
        self.pred_mlp_ratio = pred_mlp_ratio


__all__ = ["VJEPA2Config"]
