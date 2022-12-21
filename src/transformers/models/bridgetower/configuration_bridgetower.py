# coding=utf-8
# Copyright 2022 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BridgeTower model configuration"""

import copy
import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "BridgeTower/bridgetower-base": "https://huggingface.co/BridgeTower/bridgetower-base/blob/main/config.json",
    "BridgeTower/bridgetower-base-itm-mlm": (
        "https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm/blob/main/config.json"
    ),
}


class BridgeTowerVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the vision configuration of a [`BridgeTowerModel`]. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the bridgetower-base
    [BridegTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        output_resolution (`int`, *optional*, defaults to 288):
            The final size (resolution) of each image.
        input_resolution_before (`int`, *optional*, defaults to 224):
            The size (resolution) of each input image.
        stop_gradient (`bool`, *optional*, defaults to `False`):
            Whether to stop gradient for training.
        embed_dim (`int`, *optional*, defaults to 512):
            Dimension of embeddings in vit model.
        width (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in vit model.
        vit_layernorm_init_from_vit (`bool`, *optional*, defaults to `False`):
            Whether to init vit LayerNorm from vit.
        vit_layernorm_shared (`bool`, *optional*, defaults to `True`):
            Whether vit's LayerNorm layers are shared.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch in vit.
        vit_remove_last (`bool`, *optional*, defaults to `False`):
            Whether to remove vit's last layer.
        transformer_width (`int`, *optional*, defaults to 512):
            Width of vit's transformer.

    Example:

    ```python
    >>> from transformers import BridgeTowerVisionConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration for the vision model
    >>> configuration = BridgeTowerVisionConfig()

    >>> # Accessing the configuration
    >>> configuration
    ```"""
    model_type = "bridgetower_vision_model"

    def __init__(
        self,
        embed_dim=512,
        input_resolution=224,
        width=768,
        layers=12,
        patch_size=16,
        transformer_width=512,
        output_resolution=288,
        stop_gradient=False,
        vit_layernorm_shared=True,
        vit_remove_last=False,
        vit_layernorm_init_from_vit=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.width = width
        self.layers = layers
        self.patch_size = patch_size
        self.transformer_width = transformer_width
        self.output_resolution = output_resolution
        self.stop_gradient = stop_gradient
        self.vit_layernorm_shared = vit_layernorm_shared
        self.vit_remove_last = vit_remove_last
        self.vit_layernorm_init_from_vit = vit_layernorm_init_from_vit

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "bridgetower":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class BridgeTowerTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the text configuration of a [`BridgeTowerModel`]. The default values here
    are copied from RoBERTa. Instantiating a configuration with the defaults will yield a similar configuration to that
    of the bridgetower-base [BridegTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the text part of the model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`BridgeTowerModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids`.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        is_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.

    Example:

    ```python
    >>> from transformers import BridgeTowerTextConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration for the text model
    >>> configuration = BridgeTowerTextConfig()

    >>> # Accessing the configuration
    >>> configuration
    ```"""
    model_type = "bridgetower_text_model"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "bridgetower":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class BridgeTowerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BridgeTowerModel`]. It is used to instantiate a
    BridgeTower model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the bridgetower-base
    [BridegTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        cross_modal_transform_shared (`bool`, *optional*, defaults to `True`):
            Whether cross modal transformer layers are shared.
        drop_rate (`float`, *optional*, defaults to 0.1):
            Drop out probability.
        head_hidden_scale (`int`, *optional*, defaults to 2):
            Scale of hidden layers head.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        image_size (`int`, *optional*, defaults to 288):
            The size (resolution) of each image.
        input_image_embed_size (`int`, *optional*, defaults to 768):
            Embedding size of the input image.
        input_text_embed_size (`int`, *optional*, defaults to 768):
            Embedding size of the input text.
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether this is an encoder/decoder model
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        link_tower_shared (`bool`, *optional*, defaults to `False`):
            Whether the bride/link tower layers are shared.
        link_tower_type (`str`, *optional*, defaults to `"add"`):
            Type of the bridge/link layer.
        max_text_len (`int`, *optional*, defaults to 50):
            Maximum text length.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of MLP hidden dim to embedding dim.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BridgeTowerTextConfig`].
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether embedding weights are tied with the decoder
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BridgeTowerVisionConfig`].
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the text part of the model. Defines the number of different tokens that can be
            represented by the `inputs_ids` passed when calling [`BridgeTowerModel`].

    Example:

    ```python
    >>> from transformers import BridgeTowerModel, BridgeTowerConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration
    >>> configuration = BridgeTowerConfig()

    >>> # Initializing a model from the BridgeTower/bridgetower-base style configuration
    >>> model = BridgeTowerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "bridgetower"

    def __init__(
        self,
        cross_modal_transform_shared=True,
        drop_rate=0.1,
        head_hidden_scale=2,
        hidden_act="gelu",
        hidden_size=768,
        image_size=288,
        input_image_embed_size=768,
        input_text_embed_size=768,
        is_encoder_decoder=False,
        layer_norm_eps=1e-05,
        link_tower_shared=False,
        link_tower_type="add",
        max_text_len=50,
        mlp_ratio=4,
        num_attention_heads=12,
        num_hidden_layers=6,
        text_config=None,
        tie_word_embeddings=False,
        vision_config=None,
        vocab_size=50265,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cross_modal_transform_shared = cross_modal_transform_shared
        self.drop_rate = drop_rate
        self.head_hidden_scale = head_hidden_scale
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.input_image_embed_size = input_image_embed_size
        self.input_text_embed_size = input_text_embed_size
        self.is_encoder_decoder = is_encoder_decoder
        self.layer_norm_eps = layer_norm_eps
        self.link_tower_shared = link_tower_shared
        self.link_tower_type = link_tower_type
        self.max_text_len = max_text_len
        self.mlp_ratio = mlp_ratio
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size

        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        if text_config_dict is not None:
            text_config = text_config_dict
        if vision_config_dict is not None:
            vision_config = vision_config_dict

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the BridgeTowerTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the BridgeTowerVisionConfig with default values.")

        self.text_config = BridgeTowerTextConfig(**text_config)
        self.vision_config = BridgeTowerTextConfig(**vision_config)

    @classmethod
    def from_text_vision_configs(
        cls, text_config: BridgeTowerTextConfig, vision_config: BridgeTowerVisionConfig, **kwargs
    ):
        r"""
        Instantiate a [`BridgeTowerConfig`] (or a derived class) from BridgeTower text model configuration. Returns:
            [`BridgeTowerConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
