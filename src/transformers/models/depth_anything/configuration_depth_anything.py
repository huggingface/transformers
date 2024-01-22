# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" DepthAnything model configuration"""

import copy

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tiktok/depth-anything-small": "https://huggingface.co/tiktok/depth-anything-small/resolve/main/config.json",
}


class DepthAnythingConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DepthAnythingModel`]. It is used to instantiate an DepthAnything
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DepthAnything
    [tiktok/depth-anything-small](https://huggingface.co/tiktok/depth-anything-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        initializer_range (`<fill_type>`, *optional*, defaults to 0.02): <fill_docstring>
        readout_type (`str`, *optional*, defaults to `"project"`):
            The readout type to use when processing the readout token (CLS token) of the intermediate hidden states of
            the ViT backbone. Can be one of [`"ignore"`, `"add"`, `"project"`].

            - "ignore" simply ignores the CLS token.
            - "add" passes the information from the CLS token to all other tokens by adding the representations.
            - "project" passes information to the other tokens by concatenating the readout to all other tokens before
              projecting the
            representation to the original feature dimension D using a linear layer followed by a GELU non-linearity.
        reassemble_factors (`List[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`List[str]`, *optional*, defaults to `[96, 192, 384, 768]`):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 256):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the heads.
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the backbone model. Only used in case `is_hybrid` is `True` or in case you want to
            leverage the [`AutoBackbone`] API.

    Example:

    ```python
    >>> from transformers import DepthAnythingModel, DepthAnythingConfig

    >>> # Initializing a DepthAnything depth_anything-large style configuration
    >>> configuration = DepthAnythingConfig()

    >>> # Initializing a model from the depth_anything-large style configuration
    >>> model = DepthAnythingModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "depth_anything"

    def __init__(
        self,
        initializer_range=0.02,
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[96, 192, 384, 768],
        fusion_hidden_size=256,
        head_in_index=-1,
        backbone_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        self.backbone_config = backbone_config
        self.initializer_range = initializer_range
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)

        if output["backbone_config"] is not None:
            output["backbone_config"] = self.backbone_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output
