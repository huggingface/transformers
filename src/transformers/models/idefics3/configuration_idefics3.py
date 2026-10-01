# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Idefics3 model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="HuggingFaceM4/Idefics3-8B-Llama3")
@strict
class Idefics3VisionConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer
    >>> from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig

    >>> # Initializing a Idefics3VisionConfig with google/siglip-base-patch16-224 style configuration
    >>> configuration = Idefics3VisionConfig()

    >>> # Initializing a Idefics3VisionTransformer (with random weights) from the google/siglip-base-patch16-224 style configuration
    >>> model = Idefics3VisionTransformer(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics3_vision"
    base_config_key = "vision_config"

    hidden_size: int = 1152
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 32
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="HuggingFaceM4/Idefics3-8B-Llama3")
@strict
class Idefics3Config(PreTrainedConfig):
    r"""
    scale_factor (`int`, *optional*, defaults to 2):
        The scale factor for the image encoder.

    Example:
    ```python
    >>> from transformers import Idefics3Model, Idefics3Config
    >>> # Initializing configuration
    >>> configuration = Idefics3Config()
    >>> # Initializing a model from the configuration
    >>> model = Idefics3Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "idefics3"
    sub_configs = {"text_config": AutoConfig, "vision_config": Idefics3VisionConfig}

    use_cache: bool = True
    image_token_id: int = 128257
    tie_word_embeddings: bool = False
    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    scale_factor: int = 2
    pad_token_id: int | None = 128_002

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = Idefics3VisionConfig()
            logger.info("vision_config is None, using default vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Idefics3VisionConfig(**self.vision_config)

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "llama")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            logger.info("text_config is None, using default Llama text config")
            self.text_config = CONFIG_MAPPING["llama"](
                rms_norm_eps=1e-5,
                pad_token_id=self.pad_token_id,
            )

        super().__post_init__(**kwargs)


__all__ = ["Idefics3Config", "Idefics3VisionConfig"]
