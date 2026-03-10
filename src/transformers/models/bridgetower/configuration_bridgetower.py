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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="BridgeTower/bridgetower-base")
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

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_channels=3,
        patch_size=16,
        image_size=288,
        initializer_factor=1,
        layer_norm_eps=1e-05,
        stop_gradient=False,
        share_layernorm=True,
        remove_last_layer=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.stop_gradient = stop_gradient
        self.share_layernorm = share_layernorm
        self.remove_last_layer = remove_last_layer


@auto_docstring(checkpoint="BridgeTower/bridgetower-base")
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

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        initializer_factor=1,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-05,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        use_cache=True,
        is_decoder=False,
        add_cross_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_factor = initializer_factor
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


@auto_docstring(checkpoint="BridgeTower/bridgetower-base")
class BridgeTowerConfig(PreTrainedConfig):
    r"""
    share_cross_modal_transformer_layers (`bool`, *optional*, defaults to `True`):
        Whether cross modal transformer layers are shared.
    share_link_tower_layers (`bool`, *optional*, defaults to `False`):
        Whether the bride/link tower layers are shared.
    init_layernorm_from_vision_encoder (`bool`, *optional*, defaults to `False`):
        Whether to init LayerNorm from the vision encoder.
    link_tower_type (`str`, *optional*, defaults to `"add"`):
        Type of the bridge/link layer.

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

    def __init__(
        self,
        share_cross_modal_transformer_layers=True,
        hidden_act="gelu",
        hidden_size=768,
        initializer_factor=1,
        layer_norm_eps=1e-05,
        share_link_tower_layers=False,
        link_tower_type="add",
        num_attention_heads=12,
        num_hidden_layers=6,
        tie_word_embeddings=False,
        init_layernorm_from_vision_encoder=False,
        text_config=None,
        vision_config=None,
        **kwargs,
    ):
        # TODO: remove this once the Hub files are updated.
        _ = kwargs.pop("text_config_dict", None)
        _ = kwargs.pop("vision_config_dict", None)

        self.share_cross_modal_transformer_layers = share_cross_modal_transformer_layers
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.share_link_tower_layers = share_link_tower_layers
        self.link_tower_type = link_tower_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.tie_word_embeddings = tie_word_embeddings
        self.init_layernorm_from_vision_encoder = init_layernorm_from_vision_encoder

        if text_config is None:
            text_config = BridgeTowerTextConfig()
            logger.info("`text_config` is `None`. initializing the `BridgeTowerTextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config = BridgeTowerTextConfig(**text_config)

        if vision_config is None:
            vision_config = BridgeTowerVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `BridgeTowerVisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = BridgeTowerVisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["BridgeTowerConfig", "BridgeTowerTextConfig", "BridgeTowerVisionConfig"]
