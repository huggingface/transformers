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
"""OWL-ViT model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/owlvit-base-patch16")
@strict
class OwlViTTextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import OwlViTTextConfig, OwlViTTextModel

    >>> # Initializing a OwlViTTextModel with google/owlvit-base-patch32 style configuration
    >>> configuration = OwlViTTextConfig()

    >>> # Initializing a OwlViTTextConfig from the google/owlvit-base-patch32 style configuration
    >>> model = OwlViTTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "owlvit_text_model"
    base_config_key = "text_config"

    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 16
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0
    pad_token_id: int | None = 0
    bos_token_id: int | None = 49406
    eos_token_id: int | list[int] | None = 49407


@auto_docstring(checkpoint="google/owlvit-base-patch16")
@strict
class OwlViTVisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import OwlViTVisionConfig, OwlViTVisionModel

    >>> # Initializing a OwlViTVisionModel with google/owlvit-base-patch32 style configuration
    >>> configuration = OwlViTVisionConfig()

    >>> # Initializing a OwlViTVisionModel model from the google/owlvit-base-patch32 style configuration
    >>> model = OwlViTVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "owlvit_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 768
    patch_size: int | list[int] | tuple[int, int] = 32
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0


@auto_docstring(checkpoint="google/owlvit-base-patch16")
@strict
class OwlViTConfig(PreTrainedConfig):
    model_type = "owlvit"
    sub_configs = {"text_config": OwlViTTextConfig, "vision_config": OwlViTVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    return_dict: bool = True
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = OwlViTTextConfig()
            logger.info("`text_config` is `None`. initializing the `OwlViTTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = OwlViTTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = OwlViTVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `OwlViTVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = OwlViTVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["OwlViTConfig", "OwlViTTextConfig", "OwlViTVisionConfig"]
