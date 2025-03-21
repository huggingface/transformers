#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/siglip2/modular_siglip2.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_siglip2.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)


class Siglip2TextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Siglip2TextModel`]. It is used to instantiate a
    Siglip2 text encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the text encoder of the Siglip2
    [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Siglip2 text model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`Siglip2Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        max_position_embeddings (`int`, *optional*, defaults to 64):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the padding token in the vocabulary.
        bos_token_id (`int`, *optional*, defaults to 49406):
            The id of the beginning-of-sequence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 49407):
            The id of the end-of-sequence token in the vocabulary.
        projection_size (`int`, *optional*, defaults to `hidden_size`):
            The size of the projection head.

    Example:

    ```python
    >>> from transformers import Siglip2TextConfig, Siglip2TextModel

    >>> # Initializing a Siglip2TextConfig with google/siglip2-base-patch16-224 style configuration
    >>> configuration = Siglip2TextConfig()

    >>> # Initializing a Siglip2TextModel (with random weights) from the google/siglip2-base-patch16-224 style configuration
    >>> model = Siglip2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip2_text_model"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=64,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        # This differs from `CLIPTokenizer`'s default and from openai/siglip2
        # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        projection_size=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.projection_size = (
            projection_size if projection_size is not None else hidden_size
        )


class Siglip2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Siglip2VisionModel`]. It is used to instantiate a
    Siglip2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Siglip2
    [google/siglip2-base-patch16-naflex](https://huggingface.co/google/siglip2-base-patch16-naflex) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            Number of channels in the input images.
        num_patches (`int`, *optional*, defaults to 256):
            The number of patches in the image with the size of (`patch_size`, `patch_size`).
            The image is resized to fill maximum of this number of patches, and to preserve
            the aspect ratio. In case the resulted number of patches is lower, the image is
            padded in "patch" dimension.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` `"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    Example:

    ```python
    >>> from transformers import Siglip2VisionConfig, Siglip2VisionModel

    >>> # Initializing a Siglip2VisionConfig with google/siglip2-base-patch16-naflex style configuration
    >>> configuration = Siglip2VisionConfig()

    >>> # Initializing a Siglip2VisionModel (with random weights) from the google/siglip2-base-patch16-naflex style configuration
    >>> model = Siglip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip2_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        num_patches=256,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.num_patches = num_patches


class Siglip2Config(PretrainedConfig):
    r"""
    [`Siglip2Config`] is the configuration class to store the configuration of a [`Siglip2Model`]. It is used to
    instantiate a Siglip2 model according to the specified arguments, defining the text model and vision model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Siglip2
    [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Siglip2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Siglip2VisionConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Siglip2Config, Siglip2Model

    >>> # Initializing a Siglip2Config with google/siglip2-base-patch16-224 style configuration
    >>> configuration = Siglip2Config()

    >>> # Initializing a Siglip2Model (with random weights) from the google/siglip2-base-patch16-224 style configuration
    >>> model = Siglip2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Siglip2Config from a Siglip2TextConfig and a Siglip2VisionConfig
    >>> from transformers import Siglip2TextConfig, Siglip2VisionConfig

    >>> # Initializing a Siglip2Text and Siglip2Vision configuration
    >>> config_text = Siglip2TextConfig()
    >>> config_vision = Siglip2VisionConfig()

    >>> config = Siglip2Config.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "siglip2"
    sub_configs = {
        "text_config": Siglip2TextConfig,
        "vision_config": Siglip2VisionConfig,
    }

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info(
                "`text_config` is `None`. Initializing the `Siglip2TextConfig` with default values."
            )

        if vision_config is None:
            vision_config = {}
            logger.info(
                "`vision_config` is `None`. initializing the `Siglip2VisionConfig` with default values."
            )

        self.text_config = Siglip2TextConfig(**text_config)
        self.vision_config = Siglip2VisionConfig(**vision_config)

        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: Siglip2TextConfig,
        vision_config: Siglip2VisionConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Siglip2Config`] (or a derived class) from siglip2 text model configuration and siglip2 vision
        model configuration.

        Returns:
            [`Siglip2Config`]: An instance of a configuration object
        """

        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )


__all__ = ["Siglip2Config", "Siglip2TextConfig", "Siglip2VisionConfig"]
