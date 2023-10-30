# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
""" LLaVA model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging

from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)


LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shauray/Llava-Llama-2-7B-hf": "https://huggingface.co/shauray/Llava-Llama-2-7B-hf/resolve/main/config.json",
}

class LlavaVisionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptModel`]. It is used to instantiate a Mpt model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        d_model (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        mm_hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations for vision model.
        n_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        vocab_size (`int`, *optional*, defaults to 50282):
            Vocabulary size of the Mpt model. Defines the maximum number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`MptModel`]. Check [this
            discussion](https://huggingface.co/bigscience/mpt/discussions/120#633d28389addb8530b406c2a) on how the
            `vocab_size` has been defined.
        projector (`str`, *optional*, defaults to `"Linear"`): Checks if the model in v1.5 or v1.0

    Example:

    """

    model_type = "llava_vision"
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        d_model: int = 4096,
        mm_hidden_size: int = 1024,
        n_heads: int = 32,
        n_layers: int = 32,
        use_cache: bool = True,
        vocab_size: int = 50282,
        projector: str = "Linear",
        hidden_size=1024,
        intermediate_size=4096,
        projection_dim=768,
        proj_hidden_size = 4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=336,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        self.mm_hidden_size = mm_hidden_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.projector = projector
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
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
        self.proj_hidden_size = proj_hidden_size
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from InstructBlipConfig
        if config_dict.get("model_type") == "llava":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class LlavaConfig(PretrainedConfig):
    r"""
    [`LlavaConfig`] is the configuration class to store the configuration of a [`LlavaForCausalLM`]. It is used to
    instantiate a Llava model according to the specified arguments, defining the llama model and a llava model configs.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    [Salesforce/instruct-blip-flan-t5](https://huggingface.co/Salesforce/instruct-blip-flan-t5) architecture. objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`LlavaTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`LlavaVisionConfig`].
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers.models.llava.configuration_llava import LlavaTextConfig
    >>> from transformers import LlavaVisionConfig, LlavaConfig, LlavaForCausalLM

    >>> # Initializing a LlavaConfig with shauray/Llava-Llama-2-7B-hf style configuration
    >>> configuration = LlavaConfig()

    >>> # Initializing a LlavaForCausalLM (with random weights) from the shauray/Llava-Llama-2-7B-hf style configuration
    >>> model = LlavaForCausalLM(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # Initializing Llava, Llama configurations
    >>> text_config = LlavaTextConfig()
    >>> vision_config = LlavaVisionConfig()

    >>> config = LlavaConfig.from_llava_configs(
    ...     vision_config,
    ...     text_config,
    ... )
    ```"""

    model_type = "llava"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        
        if text_config is None:
            text_config = {}
            logger.info("vision_config is None. Initializing the LlavaVisionConfig with default values.")
            
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the LlavaVisionConfig with default values.")
            
        text_model_type = text_config["model_type"] if "model_type" in text_config else "llama"

        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)
        self.vision_config = LlavaVisionConfig(**vision_config)

        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_llava_configs(
        cls,
        text_config: PretrainedConfig,
        vision_config: LlavaVisionConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`LlavaConfig`] (or a derived class) from a Llama and a Llava model,

        Returns:
            [`LlavaConfig`]: An instance of a configuration object
        """

        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )


