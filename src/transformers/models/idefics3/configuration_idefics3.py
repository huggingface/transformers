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

from ...configuration_utils import PreTrainedConfig, SubConfigSpec
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


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
    sub_configs_defaults = {
        "text_config": SubConfigSpec(config_class=AutoConfig, model_type="llama", init_kwargs={"rms_norm_eps": 1e-5}),
        "vision_config": SubConfigSpec(config_class=Idefics3VisionConfig),
    }

    use_cache: bool = True
    image_token_id: int = 128257
    tie_word_embeddings: bool = False
    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    scale_factor: int = 2

    def __post_init__(self, **kwargs):
        self._pad_token_id = kwargs.pop("pad_token_id", 128_002)
        super().__post_init__(**kwargs)

    @property
    def pad_token_id(self):
        logger.warning_once(
            "`self.pad_token_id` is deprecated and might not reflect the actual PAD used by model. "
            "Access with `self.text_config.pad_token_id` to get the correct token ID, `self.pad_token_id` "
            "will be removed in v5.22."
        )
        return self._pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value):
        logger.warning_once(
            "`self.pad_token_id` is deprecated and might not reflect the actual PAD used by model. "
            "Access with `self.text_config.pad_token_id` to get the correct token ID, `self.pad_token_id` "
            "will be removed in v5.22."
        )
        self._pad_token_id = value


__all__ = ["Idefics3Config", "Idefics3VisionConfig"]
