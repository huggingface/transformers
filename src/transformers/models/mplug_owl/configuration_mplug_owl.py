# coding=utf-8
# Copyright 2022 x-plug and The HuggingFace Inc. team. All rights reserved.
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
""" MplugOwl model configuration"""
import copy
import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

MPLUG_OWL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "MAGAer13/mplug-owl-llama-7b": "https://huggingface.co/MAGAer13/mplug-owl-llama-7b/resolve/main/config.json",
    # See all MplugOwl models at https://huggingface.co/models?filter=mplug_owl
}


class MplugOwlVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MplugOwlVisionModel`]. It is used to instantiate
    a mPLUG-Owl vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the mPLUG-Owl
    [x-plug/x_plug-llama-7b](https://huggingface.co/x-plug/x_plug-llama-7b) architecture. Configuration objects inherit
    from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.

     Args:
         hidden_size (`int`, *optional*, defaults to 768):
             Dimensionality of the encoder layers and the pooler layer.
         intermediate_size (`int`, *optional*, defaults to 3072):
             Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
         num_hidden_layers (`int`, *optional*, defaults to 12):
             Number of hidden layers in the Transformer encoder.
         num_attention_heads (`int`, *optional*, defaults to 12):
             Number of attention heads for each attention layer in the Transformer encoder.
         image_size (`int`, *optional*, defaults to 224):
             The size (resolution) of each image.
         patch_size (`int`, *optional*, defaults to 32):
             The size (resolution) of each patch.
         hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
             The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
             `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
         layer_norm_eps (`float`, *optional*, defaults to 1e-5):
             The epsilon used by the layer normalization layers.
         attention_dropout (`float`, *optional*, defaults to 0.0):
             The dropout ratio for the attention probabilities.
         initializer_range (`float`, *optional*, defaults to 0.02):
             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
         initializer_factor (`float`, *optional*, defaults to 1):
             A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
             testing).
    """

    model_type = "mplug_owl_vision_model"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        projection_dim=768,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        use_flash_attn=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        self.use_flash_attn = use_flash_attn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from MplugOwlConfig
        if config_dict.get("model_type") == "mplug-owl":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MplugOwlVisualAbstractorConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MplugOwlVisionAbstractor`].
    """
    model_type = "mplug_owl_visual_abstract"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=4096,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        encoder_hidden_size=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the visual_abstractor config dict if we are loading from MplugOwlConfig
        if config_dict.get("model_type") == "mplug-owl":
            config_dict = config_dict["abstractor_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MplugOwlConfig(PretrainedConfig):
    r"""
    [`MplugOwlConfig`] is the configuration class to store the configuration of a [`MplugOwlForConditionalGeneration`].
    It is used to instantiate a mPLUG-Owl model according to the specified arguments, defining the vision model, visual
    abstractor and language model configs. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the mPLUG-Owl [x-plug/x_plug-llama-7b](https://huggingface.co/x-plug/x_plug-llama-7b)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MplugOwlVisionConfig`].
        visual_abstractor_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`MplugOwlVisualAbstractorConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    """
    model_type = "mplug-owl"
    is_composition = True

    def __init__(
        self, vision_config=None, visual_abstractor_config=None, text_config=None, num_query_tokens=64, **kwargs
    ):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = MplugOwlVisionConfig().to_dict()
            logger.info("vision_config is None.")

        if visual_abstractor_config is None:
            visual_abstractor_config = {}
            logger.info("abstractor_config is None. ")

        if text_config is None:
            # we use LLAMA 7b by default
            from ..llama.configuration_llama import LlamaConfig

            text_config = LlamaConfig(pad_token_id=2).to_dict()
            logger.info("text_config is None.")

        self.vision_config = MplugOwlVisionConfig(**vision_config)
        self.visual_abstractor_config = MplugOwlVisualAbstractorConfig(**visual_abstractor_config)
        # self.visual_abstractor_config.layer_norm_eps = 1e-6
        text_model_type = text_config["model_type"] if "model_type" in text_config else "llama"
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder

        self.num_query_tokens = num_query_tokens
        # self.visual_abstractor_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

        for attr in dir(self.text_config):
            if not hasattr(self, attr):
                setattr(self, attr, getattr(self.text_config, attr))

    @classmethod
    def from_vision_visual_abstractor_text_configs(
        cls,
        vision_config: MplugOwlVisionConfig,
        visual_abstractor_config: MplugOwlVisualAbstractorConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MplugOwlConfig`] (or a derived class) from a mPLUG-Owl vision model, Q-Former and language
        model configurations.

        Returns:
            [`MplugOwlConfig`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            visual_abstractor_config=visual_abstractor_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["visual_abstractor_config"] = self.visual_abstractor_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
