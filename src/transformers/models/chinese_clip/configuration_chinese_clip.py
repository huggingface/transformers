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
""" Chinese-CLIP model configuration"""

import copy
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional


if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    from ...utils import TensorType

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
from ..bert.configuration_bert import BertConfig
from ..clip.configuration_clip import CLIPVisionConfig


logger = logging.get_logger(__name__)

CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "OFA-Sys/chinese-clip-vit-base-patch16": (
        "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/config.json"
    ),
}


class ChineseCLIPConfig(PretrainedConfig):
    r"""
    [`ChineseCLIPConfig`] is the configuration class to store the configuration of a [`ChineseCLIPModel`]. It is used
    to instantiate Chinese-CLIP model according to the specified arguments, defining the text model and vision model
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the
    Chinese-CLIP [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BertConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`CLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original ChineseCLIP
            implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ChineseCLIPConfig, ChineseCLIPModel

    >>> # Initializing a ChineseCLIPConfig with OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> configuration = ChineseCLIPConfig()

    >>> # Initializing a ChineseCLIPModel (with random weights) from the OFA-Sys/chinese-clip-vit-base-patch16 style configuration
    >>> model = ChineseCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ChineseCLIPConfig from a BertConfig and a CLIPVisionConfig

    >>> # Initializing a BertConfig and CLIPVisionConfig configuration
    >>> config_text = BertConfig()
    >>> config_vision = CLIPVisionConfig()

    >>> config = ChineseCLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""

    model_type = "chinese-clip"
    is_composition = True

    def __init__(
        self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs
    ):
        super().__init__(**kwargs)

        # If `_config_dict` exist, we use them for the backward compatibility.
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        if text_config_dict is not None:
            text_config = text_config_dict
        if vision_config_dict is not None:
            vision_config = vision_config_dict

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the BertConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the CLIPVisionConfig with default values.")

        self.text_config = BertConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: BertConfig, vision_config: CLIPVisionConfig, **kwargs):
        r"""
        Instantiate a [`ChineseCLIPConfig`] (or a derived class) from Chinese-CLIP text model configuration and
        Chinese-CLIP vision model configuration. Returns:
            [`ChineseCLIPConfig`]: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class ChineseCLIPOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("logits_per_image", {0: "batch"}),
                ("logits_per_text", {0: "batch"}),
                ("text_embeds", {0: "batch"}),
                ("image_embeds", {0: "batch"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:

        text_input_dict = super().generate_dummy_inputs(
            processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
        )
        image_input_dict = super().generate_dummy_inputs(
            processor.feature_extractor, batch_size=batch_size, framework=framework
        )
        return {**text_input_dict, **image_input_dict}

    @property
    def default_onnx_opset(self) -> int:
        return 14
