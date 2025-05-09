# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextCausalLMOutputWithPast,
    LlavaNextPreTrainedModel,
)
from ...configuration_utils import PretrainedConfig
from ...utils import (
    is_torchdynamo_compiling,
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)

class MagmaVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MagmaVisionModel`]. It is used to instantiate a
    Magma vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the vision encoder of the Magma
    [microsoft/Magma-8B](https://huggingface.co/microsoft/Magma-8B) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        img_anyres_strategy (`str`, *optional*, defaults to `"crop"`):
            The strategy to use for any-resolution cropping.
        img_size (`int`, *optional*, defaults to 512):
            The size (resolution) of each image.
        max_num_crops (`int`, *optional*, defaults to 4):
            The maximum number of crops to use.
        mm_hidden_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        mm_projector_type (`str`, *optional*, defaults to `"mlp2x_gelu"`):
            The type of projector to use.
        mm_use_im_patch_token (`bool`, *optional*, defaults to `False`):
            Whether to use the image patch token.
        mm_use_im_start_end (`bool`, *optional*, defaults to `False`):
            Whether to use the image start and end tokens.
        vision_backbone (`str`, *optional*, defaults to `"convnextxxlarge"`):
            The vision backbone to use.
        vision_feature_layer (`str`, *optional*, defaults to `"clip_vis_dense"`):
            The vision feature layer to use.

    Example:

    ```python
    >>> from transformers import MagmaVisionConfig

    >>> # Initializing a MagmaVisionConfig with microsoft/Magma-8B style configuration
    >>> configuration = MagmaVisionConfig()
    ```"""

    model_type = "magma_vision_tower"
    base_config_key = "vision_config"

    def __init__(
        self,
        img_anyres_strategy="crop",
        img_size=512,
        max_num_crops=4,
        mm_hidden_size=3072,
        mm_projector_type="mlp2x_gelu",
        mm_use_im_patch_token=False,
        mm_use_im_start_end=False,
        vision_backbone="convnextxxlarge",
        vision_feature_layer="clip_vis_dense",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.img_anyres_strategy = img_anyres_strategy
        self.max_num_crops = max_num_crops
        self.img_size = img_size
        self.mm_hidden_size = mm_hidden_size
        self.mm_projector_type = mm_projector_type
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_use_im_start_end = mm_use_im_start_end
        self.vision_backbone = vision_backbone
        self.vision_feature_layer = vision_feature_layer
        
class MagmaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MagmaModel`]. It is used to instantiate an Magma
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Magma-8B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `SiglipVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `Qwen2Config`):
            The config object or dictionary of the text backbone.
        image_token_id (`int`, *optional*, defaults to 128257):
            The image token index to encode the image prompt.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.

    ```python
    >>> from transformers import MagmaModel, MagmaConfig

    >>> # Initializing a Magma-8B style configuration
    >>> configuration = MagmaConfig()

    >>> # Initializing a model from the Magma-8B style configuration
    >>> model = MagmaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "magma"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        
        self.image_token_id = image_token_id
        
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig(**vision_config)
        elif vision_config is None:
            vision_config = MagmaVisionConfig()

        self.vision_config = vision_config
        
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


@dataclass
class MagmaCausalLMOutputWithPast(LlavaNextCausalLMOutputWithPast):
    """
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor`  of size `(batch_size * num_frames, num_videos, sequence_length, hidden_size)`.
        video_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """
    pass

class MagmaPreTrainedModel(LlavaNextPreTrainedModel):
    _no_split_modules = ["MagmaImageTower"]


__all__ = ["MagmaConfig", "MagmaPreTrainedModel"]
