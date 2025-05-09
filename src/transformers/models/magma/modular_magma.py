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
        image_token_index (`int`, *optional*, defaults to 128257):
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
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        
        if isinstance(vision_config, dict):
            vision_config = PretrainedConfig(**vision_config)
        self.vision_config = vision_config
        
        self.image_token_index = image_token_index

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        
        if text_config is not None:
            # copy all variables in text_config to self
            for key, value in text_config.__dict__.items():
                if not key.startswith("_") and not key.startswith("__"):
                    setattr(self, key, value)
            self.text_config = text_config
        else:
            self.text_config = None

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
