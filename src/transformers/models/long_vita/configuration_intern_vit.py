# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
from typing import Union

from ...utils.configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)


class InternVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`InternVisionModel`]. It is used to
    instantiate a vision encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            Number of color channels in the input images (e.g., 3 for RGB).
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add a bias to the queries and values in the self-attention layers.
        hidden_size (`int`, *optional*, defaults to 3200):
            Dimensionality of the encoder layers and the pooler layer.
        num_attention_heads (`int`, *optional*, defaults to 25):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 12800):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        qk_normalization (`bool`, *optional*, defaults to `True`):
            Whether to normalize the queries and keys in the self-attention layers.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        use_flash_attn (`bool`, *optional*, defaults to `True`):
            Whether to use flash attention mechanism.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            Dropout rate for stochastic depth.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float`, *optional*, defaults to 0.1):
            A factor for layer scale.
    """

    model_type = 'intern_vit_6b'

    def __init__(
            self,
            num_channels=3,
            patch_size=14,
            image_size=224,
            qkv_bias=False,
            hidden_size=3200,
            num_attention_heads=25,
            intermediate_size=12800,
            qk_normalization=True,
            num_hidden_layers=48,
            use_flash_attn=True,
            hidden_act='gelu',
            norm_type='rms_norm',
            layer_norm_eps=1e-6,
            dropout=0.0,
            drop_path_rate=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=0.1,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.norm_type = norm_type
        self.qkv_bias = qkv_bias
        self.qk_normalization = qk_normalization
        self.use_flash_attn = use_flash_attn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'vision_config' in config_dict:
            config_dict = config_dict['vision_config']

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            )

        return cls.from_dict(config_dict, **kwargs)
