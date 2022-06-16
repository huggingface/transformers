# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" CLIP model configuration"""

import copy
import os
from typing import Union

from ...configuration_utils import PretrainedConfig
from ..auto.configuration_auto import AutoConfig
from ..clip.configuration_clip import CLIPVisionConfig
from ...utils import logging


logger = logging.get_logger(__name__)

OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/owlvit-clip32": "config.json",
    # See all Owl-ViT models at https://huggingface.co/models?filter=owl-vit
}

class OwlViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        vision_config_dict (`dict`):
            Dictionary of configuration options that defines vison model config.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
    Example:
    ```python
    >>> from transformers import OwlViTModel, OwlViTConfig
    >>> # Initializing a OwlViT owlvit-clip32 style configuration
    >>> configuration = OwlViTConfig()

    >>> # Initializing a model from the owlvit-clip32 style configuration
    >>> model = OwlViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "owlvit"

    def __init__(
        self,
        box_bias="both",
        merge_class_token:"mul-ln",
        normalize=True,
        image_size=768,
        projection_dim=512,
        max_query_length=16,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.projection_dim = projection_dim
        self.max_query_length = max_query_length
        self.box_bias = box_bias
        self.normalize = normalize
        self.merge_class_token = merge_class_token

        if "vision_config" not in kwargs:
            raise ValueError("`vision_config` can not be `None`.")


        vision_config = kwargs.pop("vision_config")
        body_config = kwargs.pop("body_config")
        vision_model_type = vision_config.pop("model_type")
        self.vision_config = CLIPVisionConfig(**vision_config)


    @classmethod
    def from_vision_body_configs(cls, vision_config: PretrainedConfig, body_config: PretrainedConfig, **kwargs):
        r"""
        """

        return cls(vision_config=vision_config.to_dict(), body_config=body_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["body_config"] = self.body_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
© 2022 GitHub, Inc.
Terms
Privacy



