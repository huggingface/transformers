# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""OWLv2 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@strict
@auto_docstring(checkpoint="google/owlv2-base-patch16")
# Copied from transformers.models.owlvit.configuration_owlvit.OwlViTTextConfig with OwlViT->Owlv2, owlvit-base-patch32->owlv2-base-patch16, owlvit->owlv2, OWL-ViT->OWLv2
class Owlv2TextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Owlv2TextConfig, Owlv2TextModel

    >>> # Initializing a Owlv2TextModel with google/owlv2-base-patch16 style configuration
    >>> configuration = Owlv2TextConfig()

    >>> # Initializing a Owlv2TextConfig from the google/owlv2-base-patch16 style configuration
    >>> model = Owlv2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "owlv2_text_model"
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


@strict
@auto_docstring(checkpoint="google/owlv2-base-patch16")
# Copied from transformers.models.owlvit.configuration_owlvit.OwlViTVisionConfig with OwlViT->Owlv2, owlvit-base-patch32->owlv2-base-patch16, owlvit->owlv2, OWL-ViT->OWLv2, 32->16
class Owlv2VisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Owlv2VisionConfig, Owlv2VisionModel

    >>> # Initializing a Owlv2VisionModel with google/owlv2-base-patch16 style configuration
    >>> configuration = Owlv2VisionConfig()

    >>> # Initializing a Owlv2VisionModel model from the google/owlv2-base-patch16 style configuration
    >>> model = Owlv2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "owlv2_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 768
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    initializer_factor: float = 1.0


@strict
@auto_docstring(checkpoint="google/owlv2-base-patch16")
# Copied from transformers.models.owlvit.configuration_owlvit.OwlViTConfig with OwlViT->Owlv2, owlvit-base-patch32->owlv2-base-patch16, owlvit->owlv2, OWL-ViT->OWLv2
class Owlv2Config(PreTrainedConfig):
    model_type = "owlv2"
    sub_configs = {"text_config": Owlv2TextConfig, "vision_config": Owlv2VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592
    return_dict: bool = True
    initializer_factor: float = 1.0

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Owlv2TextConfig()
            logger.info("`text_config` is `None`. initializing the `Owlv2TextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = Owlv2TextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Owlv2VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Owlv2VisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Owlv2VisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["Owlv2Config", "Owlv2TextConfig", "Owlv2VisionConfig"]
