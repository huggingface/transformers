# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""KOSMOS-2.5 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
class Kosmos2_5TextConfig(PreTrainedConfig):
    r"""
    activation_dropout (`float`, *optional*, defaults to 0.0):
        The dropout ratio for activations inside the fully connected layer.
    ```"""

    model_type = "kosmos_2_5_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }

    def __init__(
        self,
        vocab_size=108481,
        max_position_embeddings=4096,
        embed_dim=1536,
        layers=24,
        ffn_dim=6144,
        attention_heads=16,
        activation_function="gelu",
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        layer_norm_eps=1e-5,
        init_std=0.02,
        scale_embedding=True,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

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


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
class Kosmos2_5VisionConfig(PreTrainedConfig):
    r"""
    dense_act_fn (`str` or `function`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
    max_num_patches (`int`, *optional*, defaults to 4096):
        Maximum sequence length (here number of patches) supported by the model.
    patch_embed_hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the input patch_embedding layer in the Transformer encoder.

    Example:

    ```python
    >>> from transformers import Kosmos2_5VisionConfig, Kosmos2_5VisionModel

    >>> # Initializing a Kosmos2_5VisionConfig with microsoft/kosmos-2.5 style configuration
    >>> configuration = Kosmos2_5VisionConfig()

    >>> # Initializing a Kosmos2_5VisionModel (with random weights) from the microsoft/kosmos-2.5 style configuration
    >>> model = Kosmos2_5VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "kosmos_2_5_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        hidden_size=1536,
        patch_embed_hidden_size=768,
        intermediate_size=3968,
        head_dim=64,
        num_hidden_layers=18,
        num_attention_heads=24,
        dense_act_fn="gelu_new",
        layer_norm_eps=1e-6,
        dropout_rate=0.0,
        attention_dropout=0.0,
        max_num_patches=4096,
        initializer_factor=1.0,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.patch_embed_hidden_size = patch_embed_hidden_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.dense_act_fn = dense_act_fn
        self.max_num_patches = max_num_patches
        self.head_dim = head_dim
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
class Kosmos2_5Config(PreTrainedConfig):
    r"""
    latent_query_num (`int`, *optional*, defaults to 2048):
        The number of latent query tokens that represent the image features used in the text decoder component.
    """

    model_type = "kosmos-2.5"
    sub_configs = {"text_config": Kosmos2_5TextConfig, "vision_config": Kosmos2_5VisionConfig}

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        latent_query_num=2048,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if text_config is None:
            text_config = Kosmos2_5TextConfig()
            logger.info("`text_config` is `None`. initializing the `Kosmos2_5TextConfig` with default values.")
        elif isinstance(text_config, dict):
            text_config = Kosmos2_5TextConfig(**text_config)

        if vision_config is None:
            vision_config = Kosmos2_5VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Kosmos2_5VisionConfig` with default values.")
        elif isinstance(vision_config, dict):
            vision_config = Kosmos2_5VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.latent_query_num = latent_query_num
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = ["Kosmos2_5Config"]
