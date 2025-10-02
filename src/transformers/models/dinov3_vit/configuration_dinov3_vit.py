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
"""DINOv3 model configuration"""

from typing import Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)


class DINOv3ViTConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DINOv3Model`]. It is used to instantiate an
    DINOv3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the DINOv3
    [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_size (`int`, *optional*, defaults to 384):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 1536):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        rope_theta (`float`, *optional*, defaults to 100.0):
            The base period of the RoPE embeddings.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        query_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the query projection.
        key_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the key projection.
        value_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the value projection.
        proj_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the output projection.
        mlp_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the MLP layers.
        layerscale_value (`float`, *optional*, defaults to 1.0):
            Initial value to use for layer scale.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate per sample (when applied in the main path of residual layers).
        use_gated_mlp (`bool`, *optional*, defaults to `False`):
            Whether to use the SwiGLU feedforward neural network.
        num_register_tokens (`int`, *optional*, defaults to 0):
            The number of register tokens.
        pos_embed_shift (`float`, *optional*):
            Amount to randomly shift position embedding coordinates in [-shift, shift],
            applied only in training mode if not `None`.
        pos_embed_jitter (`float`, *optional*):
            Amount to randomly jitter position embedding coordinates in log-uniform value in [1/jitter, jitter],
            applied only in training mode if not `None`.
        pos_embed_rescale (`float`, *optional*, defaults to 2.0):
            Amount to randomly rescale position embedding coordinates in log-uniform value in [1/rescale, rescale],
            applied only in training mode if not `None`.
             out_features (`List[str]`, *optional*):
            If used as backbone, list of features to output. Can be any of `"stem"`, `"stage1"`, `"stage2"`, etc.
            (depending on how many stages the model has). Will default to the last stage if unset.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc.
            (depending on how many stages the model has). Will default to the last stage if unset.
        apply_layernorm (`bool`, *optional*, defaults to `True`):
            Whether to apply layer normalization to the feature maps when used as backbone.
        reshape_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to reshape the hidden states to spatial dimensions when used as backbone.

    Example:

    ```python
    >>> from transformers import DINOv3ViTConfig, DINOv3ViTModel

    >>> # Initializing a DINOv3 ViT-small style configuration
    >>> config = DINOv3ViTConfig()

    >>> # Initializing a model (with random weights) from the config
    >>> model = DINOv3ViTModel(config)

    >>> # Accessing the model config
    >>> config = model.config
    ```"""

    model_type = "dinov3_vit"

    def __init__(
        self,
        patch_size: int = 16,
        hidden_size: int = 384,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 6,
        hidden_act: str = "gelu",
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        rope_theta: float = 100.0,
        image_size: int = 224,
        num_channels: int = 3,
        query_bias: bool = True,
        key_bias: bool = False,
        value_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        layerscale_value: float = 1.0,
        drop_path_rate: float = 0.0,
        use_gated_mlp: bool = False,
        num_register_tokens: int = 0,
        # train augs
        pos_embed_shift: Optional[float] = None,
        pos_embed_jitter: Optional[float] = None,
        pos_embed_rescale: Optional[float] = 2.0,
        out_features: Optional[list[str]] = None,
        out_indices: Optional[list[int]] = None,
        apply_layernorm: bool = True,
        reshape_hidden_states: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.layerscale_value = layerscale_value
        self.drop_path_rate = drop_path_rate
        self.use_gated_mlp = use_gated_mlp
        self.rope_theta = rope_theta
        self.query_bias = query_bias
        self.key_bias = key_bias
        self.value_bias = value_bias
        self.proj_bias = proj_bias
        self.mlp_bias = mlp_bias
        self.num_register_tokens = num_register_tokens

        # train augs
        self.pos_embed_shift = pos_embed_shift
        self.pos_embed_jitter = pos_embed_jitter
        self.pos_embed_rescale = pos_embed_rescale
        # Initialize backbone-specific configuration
        self.apply_layernorm = apply_layernorm
        self.reshape_hidden_states = reshape_hidden_states

        # Initialize backbone stage names
        stage_names = ["stem"] + [f"stage{i}" for i in range(1, num_hidden_layers + 1)]
        self.stage_names = stage_names

        # Initialize backbone features/indices
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=stage_names
        )


__all__ = ["DINOv3ViTConfig"]
