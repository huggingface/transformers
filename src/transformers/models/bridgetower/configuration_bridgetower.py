# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BridgeTower model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="BridgeTower/bridgetower-base")
@strict
class BridgeTowerVisionConfig(PreTrainedConfig):
    r"""
    stop_gradient (`bool`, *optional*, defaults to `False`):
        Whether to stop gradient for training.
    share_layernorm (`bool`, *optional*, defaults to `True`):
        Whether LayerNorm layers are shared.
    remove_last_layer (`bool`, *optional*, defaults to `False`):
        Whether to remove the last layer from the vision encoder.

    Example:

    ```python
    >>> from transformers import BridgeTowerVisionConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration for the vision model
    >>> configuration = BridgeTowerVisionConfig()

    >>> # Accessing the configuration
    >>> configuration
    ```"""

    model_type = "bridgetower_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    image_size: int | list[int] | tuple[int, int] = 288
    initializer_factor: float | int = 1
    layer_norm_eps: float = 1e-05
    stop_gradient: bool = False
    share_layernorm: bool = True
    remove_last_layer: bool = False


@auto_docstring(checkpoint="BridgeTower/bridgetower-base")
@strict
class BridgeTowerTextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import BridgeTowerTextConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration for the text model
    >>> configuration = BridgeTowerTextConfig()

    >>> # Accessing the configuration
    >>> configuration
    ```"""

    model_type = "bridgetower_text_model"
    base_config_key = "text_config"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    initializer_factor: float | int = 1
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    layer_norm_eps: float = 1e-05
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    use_cache: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False


@auto_docstring(checkpoint="BridgeTower/bridgetower-base")
@strict
class BridgeTowerConfig(PreTrainedConfig):
    r"""
    share_cross_modal_transformer_layers (`bool`, *optional*, defaults to `True`):
        Whether cross modal transformer layers are shared.
    share_link_tower_layers (`bool`, *optional*, defaults to `False`):
        Whether the bride/link tower layers are shared.
    link_tower_type (`str`, *optional*, defaults to `"add"`):
        Type of the bridge/link layer.
    init_layernorm_from_vision_encoder (`bool`, *optional*, defaults to `False`):
        Whether to init LayerNorm from the vision encoder.

    Example:

    ```python
    >>> from transformers import BridgeTowerModel, BridgeTowerConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration
    >>> configuration = BridgeTowerConfig()

    >>> # Initializing a model from the BridgeTower/bridgetower-base style configuration
    >>> model = BridgeTowerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bridgetower"
    sub_configs = {"text_config": BridgeTowerTextConfig, "vision_config": BridgeTowerVisionConfig}

    share_cross_modal_transformer_layers: bool = True
    hidden_act: str = "gelu"
    hidden_size: int = 768
    initializer_factor: float | int = 1
    layer_norm_eps: float = 1e-05
    share_link_tower_layers: bool = False
    link_tower_type: str = "add"
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    tie_word_embeddings: bool = False
    init_layernorm_from_vision_encoder: bool = False
    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        # TODO: remove this once the Hub files are updated.
        _ = kwargs.pop("text_config_dict", None)
        _ = kwargs.pop("vision_config_dict", None)

        if self.text_config is None:
            self.text_config = BridgeTowerTextConfig()
            logger.info("`text_config` is `None`. initializing the `BridgeTowerTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = BridgeTowerTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = BridgeTowerVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `BridgeTowerVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = BridgeTowerVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["BridgeTowerConfig", "BridgeTowerTextConfig", "BridgeTowerVisionConfig"]
