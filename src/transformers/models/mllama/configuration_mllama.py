# coding=utf-8
# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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
"""Mllama model configuration"""

import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


def compute_intermediate_size(n, ffn_dim_multiplier=1.3, multiple_of=4096):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


class MllamaCrossAttentionVisionConfig(PretrainedConfig):
    # TODO fix config docstring
    r"""
    This is the configuration class to store the configuration of a [`MllamaForConditionalGeneration`]. It is used to instantiate an
    Mllama model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-9B.

    e.g. [mllama-hf/mllama-9b](https://huggingface.co/mllama-hf/mllama-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
        The index of the layer to select the vision feature.

    Example:

    ```python
    >>> from transformers import MllamaForConditionalGeneration, MllamaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Mllama mllama-1.5-7b style configuration
    >>> configuration = MllamaConfig(vision_config, text_config)

    >>> # Initializing a model from the mllama-1.5-7b style configuration
    >>> model = MllamaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama"

    def __init__(
        self,
        n_heads=16,
        vision_chunk_size=560,
        vision_max_num_chunks=4,
        patch_size=14,
        projection_dim=8192,
        vision_input_dim=1280,
        return_intermediate="3,7,15,23,30",
        global_vision_layers=8,
        max_num_tiles=4,  # same as vision max num chunks?
        norm_eps=1.0e-5,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.vision_chunk_size = vision_chunk_size
        self.vision_max_num_chunks = vision_max_num_chunks
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.vision_input_dim = vision_input_dim
        self.return_intermediate = return_intermediate
        self.global_vision_layers = global_vision_layers
        self.max_num_tiles = max_num_tiles
        self.norm_eps = norm_eps
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mllama":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MllamaCrossAttentionTextConfig(PretrainedConfig):
    # TODO fix config docstring
    r"""
    This is the configuration class to store the configuration of a [`MllamaForConditionalGeneration`]. It is used to instantiate an
    Mllama model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-9B.

    e.g. [mllama-hf/mllama-9b](https://huggingface.co/mllama-hf/mllama-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
        The index of the layer to select the vision feature.

    Example:

    ```python
    >>> from transformers import MllamaForConditionalGeneration, MllamaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Mllama mllama-1.5-7b style configuration
    >>> configuration = MllamaConfig(vision_config, text_config)

    >>> # Initializing a model from the mllama-1.5-7b style configuration
    >>> model = MllamaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama"

    def __init__(
        self,
        vocab_size=128256,
        n_layers=80,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=512,
        ffn_dim_multiplier=1.3,
        norm_eps=1.0e-5,
        rope_theta=500000,
        use_scaled_rope=True,
        vision_num_cross_attention_layers=20,  # TODO comon
        multiple_of=4096,  # TODO common
        vision_input_dim=1280,  # TODO common
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dim = dim

        # FOR STATIC CACHE, TEMP SOLUTION UNTIL STANDARDIZING VAR NAMES
        self.hidden_size = dim
        self.num_attention_heads = n_heads
        self.num_key_value_heads = n_kv_heads
        self.num_hidden_layers = n_layers

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.vision_num_cross_attention_layers = vision_num_cross_attention_layers
        self.rope_theta = rope_theta
        self.use_scaled_rope = use_scaled_rope
        self.norm_eps = norm_eps
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.intermediate_size = compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of)
        self.vision_input_dim = vision_input_dim
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") == "mllama":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class MllamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MllamaForConditionalGeneration`]. It is used to instantiate an
    Mllama model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Mllama-9B.

    e.g. [mllama-hf/mllama-9b](https://huggingface.co/mllama-hf/mllama-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
        The index of the layer to select the vision feature.

    Example:

    ```python
    >>> from transformers import MllamaForConditionalGeneration, MllamaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    >>> # Initializing a Mllama mllama-1.5-7b style configuration
    >>> configuration = MllamaConfig(vision_config, text_config)

    >>> # Initializing a model from the mllama-1.5-7b style configuration
    >>> model = MllamaForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mllama"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        **kwargs,
    ):
        if vision_config is None:
            self.vision_config = MllamaCrossAttentionVisionConfig()
            logger.info("vision_config is None, using default mllama vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = MllamaCrossAttentionVisionConfig(**vision_config)
        elif isinstance(vision_config, MllamaCrossAttentionVisionConfig):
            self.vision_config = vision_config

        if text_config is None:
            self.text_config = MllamaCrossAttentionTextConfig()
            logger.info("text_config is None, using default mllama text config")
        elif isinstance(text_config, dict):
            self.text_config = MllamaCrossAttentionTextConfig(**text_config)
        elif isinstance(text_config, MllamaCrossAttentionTextConfig):
            self.text_config = text_config

        super().__init__(**kwargs)
