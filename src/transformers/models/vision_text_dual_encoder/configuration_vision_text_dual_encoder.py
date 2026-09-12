# Copyright The HuggingFace Inc. team. All rights reserved.
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
"""VisionTextDualEncoder model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring
@strict
class VisionTextDualEncoderConfig(PreTrainedConfig):
    r"""
    Examples:

    ```python
    >>> from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig, VisionTextDualEncoderModel

    >>> # Initializing a BERT and ViT configuration
    >>> config = VisionTextDualEncoderConfig(projection_dim=512)

    >>> # Initializing a BERT and ViT model (with random weights)
    >>> model = VisionTextDualEncoderModel(config=config)

    >>> # Accessing the model configuration
    >>> config_vision = model.config.vision_config
    >>> config_text = model.config.text_config

    >>> # Saving the model, including its configuration
    >>> model.save_pretrained("vit-bert")

    >>> # loading model and config from pretrained folder
    >>> vision_text_config = VisionTextDualEncoderConfig.from_pretrained("vit-bert")
    >>> model = VisionTextDualEncoderModel.from_pretrained("vit-bert", config=vision_text_config)
    ```"""

    model_type = "vision-text-dual-encoder"
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="bert"),
        "vision_config": SubConfigSpec(config_class=AutoConfig, model_type="vit"),
    }

    text_config: PreTrainedConfig | dict | None = None
    vision_config: PreTrainedConfig | dict | None = None
    projection_dim: int = 512
    logit_scale_init_value: int | float = 2.6592

    @classmethod
    def from_vision_text_configs(cls, vision_config: PreTrainedConfig, text_config: PreTrainedConfig, **kwargs):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.

        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """

        return cls(vision_config=vision_config.to_dict(), text_config=text_config.to_dict(), **kwargs)


__all__ = ["VisionTextDualEncoderConfig"]
