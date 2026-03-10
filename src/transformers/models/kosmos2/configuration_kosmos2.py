# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""KOSMOS-2 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/kosmos-2-patch14-224")
class Kosmos2TextConfig(PreTrainedConfig):
    r"""
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    ```"""

    model_type = "kosmos_2_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }

    def __init__(
        self,
        vocab_size=65037,
        max_position_embeddings=2048,
        embed_dim=2048,
        layers=24,
        ffn_dim=8192,
        attention_heads=32,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        layerdrop=0.0,
        layer_norm_eps=1e-5,
        init_std=0.02,
        scale_embedding=True,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        add_cross_attention=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.add_cross_attention = add_cross_attention

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = embed_dim
        self.layers = layers
        self.ffn_dim = ffn_dim
        self.attention_heads = attention_heads
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.layerdrop = layerdrop
        self.layer_norm_eps = layer_norm_eps
        self.init_std = init_std
        self.scale_embedding = scale_embedding
        self.use_cache = use_cache


@auto_docstring(checkpoint="microsoft/kosmos-2-patch14-224")
class Kosmos2VisionConfig(PreTrainedConfig):
    model_type = "kosmos_2_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_channels=3,
        image_size=224,
        patch_size=14,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


@auto_docstring(checkpoint="microsoft/kosmos-2-patch14-224")
class Kosmos2Config(PreTrainedConfig):
    r"""
    latent_query_num (`int`, *optional*, defaults to 64):
        The number of latent query tokens that represent the image features used in the text decoder component.

    Example:

    ```python
    >>> from transformers import Kosmos2Config, Kosmos2Model

    >>> # Initializing a Kosmos-2 kosmos-2-patch14-224 style configuration
    >>> configuration = Kosmos2Config()

    >>> # Initializing a model (with random weights) from the kosmos-2-patch14-224 style configuration
    >>> model = Kosmos2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kosmos-2"
    sub_configs = {"text_config": Kosmos2TextConfig, "vision_config": Kosmos2VisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        latent_query_num=64,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Kosmos2TextConfig()
            logger.info("`text_config` is `None`. initializing the `Kosmos2TextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config = Kosmos2TextConfig(**text_config)

        if vision_config is None:
            vision_config = Kosmos2VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Kosmos2VisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = Kosmos2VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.latent_query_num = latent_query_num
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Kosmos2Config"]
