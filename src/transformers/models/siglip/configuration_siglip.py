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
"""Siglip model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/siglip-base-patch16-224")
@strict
class SiglipTextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import SiglipTextConfig, SiglipTextModel

    >>> # Initializing a SiglipTextConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipTextConfig()

    >>> # Initializing a SiglipTextModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip_text_model"
    base_config_key = "text_config"

    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 64
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    # This differs from `CLIPTokenizer`'s default and from openai/siglip
    # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
    pad_token_id: int | None = 1
    bos_token_id: int | None = 49406
    eos_token_id: int | list[int] | None = 49407
    projection_size: int | None = None

    def __post_init__(self, **kwargs):
        self.projection_size = self.projection_size if self.projection_size is not None else self.hidden_size
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="google/siglip-base-patch16-224")
@strict
class SiglipVisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import SiglipVisionConfig, SiglipVisionModel

    >>> # Initializing a SiglipVisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipVisionConfig()

    >>> # Initializing a SiglipVisionModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "siglip_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0


@auto_docstring(checkpoint="google/siglip-base-patch16-224")
@strict
class SiglipConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import SiglipConfig, SiglipModel

    >>> # Initializing a SiglipConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = SiglipConfig()

    >>> # Initializing a SiglipModel (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = SiglipModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a SiglipConfig from a SiglipTextConfig and a SiglipVisionConfig
    >>> from transformers import SiglipTextConfig, SiglipVisionConfig

    >>> # Initializing a SiglipText and SiglipVision configuration
    >>> config_text = SiglipTextConfig()
    >>> config_vision = SiglipVisionConfig()

    >>> config = SiglipConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "siglip"
    sub_configs = {"text_config": SiglipTextConfig, "vision_config": SiglipVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = SiglipTextConfig()
            logger.info("`text_config` is `None`. Initializing the `SiglipTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = SiglipTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = SiglipVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `SiglipVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = SiglipVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["SiglipConfig", "SiglipTextConfig", "SiglipVisionConfig"]
