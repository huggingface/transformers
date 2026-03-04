# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
"""GraniteDoclingHybrid model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..granitemoehybrid.configuration_granitemoehybrid import GraniteMoeHybridConfig
from ..idefics3.configuration_idefics3 import Idefics3VisionConfig


logger = logging.get_logger(__name__)


class GraniteDoclingHybridVisionConfig(Idefics3VisionConfig):
    pass


class GraniteDoclingHybridGraniteMoeHybridConfig(GraniteMoeHybridConfig):
    pass


class GraniteDoclingHybridConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GraniteDoclingHybridModel`]. It is used to instantiate a
    GraniteDoclingHybrid model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Idefics3 model architecture,
    but with a GraniteMoeHybrid text model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cache the key/value pairs of the attention mechanism. Only
            relevant if `config.is_decoder=True`.
        image_token_id (`int`, *optional*, defaults to 128257):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`GraniteDoclingHybridVisionConfig` or `dict`, *optional*, defaults to `GraniteDoclingHybridVisionConfig`):
            Custom vision config or dict for the vision tower
        text_config (`PretrainedConfig` or `dict`, *optional*, defaults to `GraniteMoeHybridConfig`):
            Custom text config or dict for the text model
        scale_factor (`int`, *optional*, defaults to 2):
            The scale factor for the image encoder.
        pad_token_id (`int`, *optional*, defaults to 128002):
            The id of the padding token.

    Example:
    ```python
    >>> from transformers import GraniteDoclingHybridModel, GraniteDoclingHybridConfig
    >>> # Initializing configuration
    >>> configuration = GraniteDoclingHybridConfig()
    >>> # Initializing a model from the configuration
    >>> model = GraniteDoclingHybridModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "granite_docling_hybrid"
    sub_configs = {"text_config": CONFIG_MAPPING, "vision_config": GraniteDoclingHybridVisionConfig}

    def __init__(
        self,
        use_cache=True,
        image_token_id=128257,
        tie_word_embeddings=False,
        vision_config=None,
        text_config=None,
        scale_factor=2,
        pad_token_id=128_002,
        **kwargs,
    ):
        self.image_token_id = image_token_id
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings

        if vision_config is None:
            self.vision_config = GraniteDoclingHybridVisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = GraniteDoclingHybridVisionConfig(**vision_config)
        elif isinstance(vision_config, GraniteDoclingHybridVisionConfig):
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config.get("model_type", "granitemoehybrid")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            logger.info("text_config is None, using default GraniteMoeHybrid text config")
            text_config = CONFIG_MAPPING["granitemoehybrid"]()

        self.text_config = text_config
        self.scale_factor = scale_factor

        super().__init__(**kwargs, pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings)


__all__ = ["GraniteDoclingHybridConfig"]
