# coding=utf-8
# Copyright 2024 Meta AI and The HuggingFace Inc. team. All rights reserved.
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
"""Moshigeneration configuration"""

import copy
from typing import Dict

from ...generation.configuration_utils import GenerationConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class MoshiGenerationConfig(GenerationConfig):
    model_type = "moshi"
    is_composition = True

    # TODO (joao): nested from_dict

    def __init__(
        self,
        depth_decoder_config: Dict = None,
        **kwargs,
    ):
        """Class that holds a generation configuration for [`MoshiForConditionalGeneration`].

        The [`MoshiForConditionalGeneration`] model needs to encapsulates two generation config, the main one, and one for its depth decoder.

        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the
        documentation from [`GenerationConfig`] for more information.

        Args:
            depth_decoder_config (`Dict`, *optional*):
                Depth decoder generation configuration.
            # TODO(YL): kwargs

        """
        super.__init__(**kwargs)
        if depth_decoder_config is None:
            depth_decoder_config = {}
            logger.info("depth_decoder_config is None. initializing the semantic model with default values.")

        self.depth_decoder_config = GenerationConfig(**depth_decoder_config)

    @classmethod
    def from_depth_decoder_config(
        cls,
        depth_decoder_config: GenerationConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`MoshiGenerationConfig`] (or a derived class) from Moshi depth decoder generation configuration.

        Returns:
            [`MoshiGenerationConfig`]: An instance of a configuration object
        """
        return cls(
            depth_decoder_config=depth_decoder_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        output["depth_decoder_config"] = self.depth_decoder_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output
