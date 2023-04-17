# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" SAM model configuration"""

import copy
import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SAM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/sam-vit-h": "https://huggingface.co/facebook/sam-vit-h/resolve/main/config.json",
}


class SamPromptEncoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamVisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM
    [facebook/sam-vit-h](https://huggingface.co/facebook/sam-vit-h) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            Dimensionality of the encoder layers and the pooler layer.

    """

    def __init__(
        self,
        hidden_size=256,
        input_image_size=1024,
        patch_size=16,
        mask_input_channels=16,
        num_point_embeddings=4,
        hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.input_image_size = input_image_size
        self.patch_size = patch_size
        self.image_embedding_size = input_image_size // patch_size
        self.mask_input_channels = mask_input_channels
        self.num_point_embeddings = num_point_embeddings
        self.hidden_act = hidden_act


class SamMaskDecoderConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamVisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM
    [facebook/sam-vit-h](https://huggingface.co/facebook/sam-vit-h) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            Dimensionality of the encoder layers and the pooler layer.

    """

    def __init__(
        self,
        hidden_size=256,
        hidden_act="relu",
        mlp_dim=2048,
        num_hidden_layers=2,
        num_attention_heads=8,
        attention_downsample_rate=2,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.mlp_dim = mlp_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_downsample_rate = attention_downsample_rate
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim
        self.layer_norm_eps = layer_norm_eps


class SamVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SamVisionModel`]. It is used to instantiate a SAM
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    defaults will yield a similar configuration to that of the SAM
    [facebook/sam-vit-h](https://huggingface.co/facebook/sam-vit-h) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 39):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported. layer_norm_eps (`float`, *optional*, defaults
            to 1e-5): The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries and values in the self-attention layers.

    Example:

    ```python
    >>> from transformers import SamVisionConfig, SamVisionModel

    >>> # Initializing a SamVisionConfig with facebook/sam-vit-h style configuration
    >>> configuration = SamVisionConfig()

    >>> # Initializing a SamVisionModel (with random weights) from the facebook/sam-vit-h style configuration
    >>> model = SamVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sam_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=6144,
        projection_dim=512,
        output_channels=256,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=1024,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=1e-10,
        initializer_factor=1.0,
        qkv_bias=True,
        mlp_ratio=4.0,
        use_abs_pos=True,
        use_rel_pos=True,
        rel_pos_zero_init=False,
        window_size=14,
        global_attn_indexes=[2, 5, 8, 11],
        num_pos_feats=128,
        mlp_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.dropout = dropout
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
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos
        self.output_channels = output_channels
        self.mlp_ratio = mlp_ratio
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes
        self.num_pos_feats = num_pos_feats
        self.mlp_dim = int(hidden_size * mlp_ratio) if mlp_dim is None else mlp_dim

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SamConfig
        if config_dict.get("model_type") == "sam":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class SamConfig(PretrainedConfig):
    r"""
    [`SamConfig`] is the configuration class to store the configuration of a [`SamForConditionalGeneration`]. It is
    used to instantiate a SAM model according to the specified arguments, defining the vision model, Q-Former model and
    language model configs. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the SAM [facebook/sam-vit-h](https://huggingface.co/facebook/sam-vit-h) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`SamVisionConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     SamVisionConfig,
    ...     OPTConfig,
    ...     SamConfig,
    ...     SamForConditionalGeneration,
    ... )

    >>> # Initializing a SamConfig with facebook/sam-vit-h style configuration
    >>> configuration = SamConfig()

    >>> # Initializing a SamForConditionalGeneration (with random weights) from the facebook/sam-vit-h style configuration
    >>> model = SamForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a SamConfig from a SamVisionConfig and any PretrainedConfig

    >>> # Initializing SAM vision, SAM Q-Former and language model configurations
    >>> vision_config = SamVisionConfig()
    >>> text_config = OPTConfig()

    >>> config = SamConfig.from_text_vision_configs(vision_config, text_config)
    ```"""

    model_type = "sam"
    is_composition = True

    def __init__(self, vision_config={}, prompt_encoder_config={}, mask_decoder_config={}, **kwargs):
        super().__init__(**kwargs)

        if isinstance(vision_config, SamVisionConfig):
            vision_config = vision_config.to_dict()
        if isinstance(prompt_encoder_config, SamPromptEncoderConfig):
            prompt_encoder_config = prompt_encoder_config.to_dict()
        if isinstance(mask_decoder_config, SamMaskDecoderConfig):
            mask_decoder_config = mask_decoder_config.to_dict()

        self.vision_config = SamVisionConfig(**vision_config)
        self.prompt_encoder_config = SamPromptEncoderConfig(**prompt_encoder_config)
        self.mask_decoder_config = SamMaskDecoderConfig(**mask_decoder_config)

        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["prompt_encoder_config"] = self.prompt_encoder_config.to_dict()
        output["mask_decoder_config"] = self.mask_decoder_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
