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
""" CogVLM model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

COGVLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "THUDM/cogvlm-chat-hf": "https://huggingface.co/THUDM/cogvlm-chat-hf/resolve/main/config.json",
}


class CogVLMVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CogVLMVisionModel`]. It is used to instantiate a
    CogVLM vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the CogVLM
    [THUDM/cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf) architecture.

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
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used for layernorm layers.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import CogVLMVisionConfig, CogVLMVisionModel

    >>> # Initializing a CogVLMVisionConfig with THUDM/cogvlm-chat-hf style configuration
    >>> configuration = CogVLMVisionConfig()

    >>> # Initializing a CogVLMVisionModel (with random weights) from the THUDM/cogvlm-chat-hf style configuration
    >>> model = CogVLMVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cogvlm_vision_model"

    def __init__(
        self,
        hidden_size=1408,
        intermediate_size=6144,
        num_hidden_layers=39,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        initializer_range=1e-10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CogVLMConfig
        if config_dict.get("model_type") == "cogvlm":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CogVLMConfig(PretrainedConfig):
    r"""
    [`CogVLMConfig`] is the configuration class to store the configuration of a [`CogVLMForCausalLM`]. It is
    used to instantiate a CogVLM model according to the specified arguments, defining the vision model
    and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the CogVLM [THUDM/cogvlm-chat-hf](https://huggingface.co/THUDM/cogvlm-chat-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import CogVLMConfig, CogVLMForCausalLM

    >>> # Initializing a CogVLMConfig with THUDM/cogvlm-chat-hf style configuration
    >>> configuration = CogVLMConfig()

    >>> # Initializing a CogVLMForCausalLM (with random weights) from the THUDM/cogvlm-chat-hf style configuration
    >>> model = CogVLMForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "cogvlm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the CogVLMVisionConfig with default values.")

        self.vision_config = CogVLMVisionConfig(**vision_config)
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
