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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
@strict
class Kosmos2_5TextConfig(PreTrainedConfig):
    model_type = "kosmos_2_5_text_model"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "embed_dim",
        "num_hidden_layers": "layers",
    }

    vocab_size: int = 108481
    max_position_embeddings: int = 4096
    embed_dim: int = 1536
    layers: int = 24
    ffn_dim: int = 6144
    attention_heads: int = 16
    activation_function: str = "gelu"
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    layerdrop: float | int = 0.0
    layer_norm_eps: float = 1e-5
    init_std: float = 0.02
    scale_embedding: bool = True
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
@strict
class Kosmos2_5VisionConfig(PreTrainedConfig):
    r"""
    patch_embed_hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the input patch_embedding layer in the Transformer encoder.
    dense_act_fn (`str` or `function`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
    max_num_patches (`int`, *optional*, defaults to 4096):
        Maximum sequence length (here number of patches) supported by the model.

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

    hidden_size: int = 1536
    patch_embed_hidden_size: int = 768
    intermediate_size: int = 3968
    head_dim: int = 64
    num_hidden_layers: int = 18
    num_attention_heads: int = 24
    dense_act_fn: str = "gelu_new"
    layer_norm_eps: float = 1e-6
    dropout_rate: float = 0.0
    attention_dropout: float | int = 0.0
    max_num_patches: int = 4096
    initializer_factor: float = 1.0
    initializer_range: float = 0.02


@auto_docstring(checkpoint="microsoft/kosmos-2.5")
@strict
class Kosmos2_5Config(PreTrainedConfig):
    r"""
    latent_query_num (`int`, *optional*, defaults to 2048):
        The number of latent query tokens that represent the image features used in the text decoder component.
    """

    model_type = "kosmos-2.5"
    sub_configs = {"text_config": Kosmos2_5TextConfig, "vision_config": Kosmos2_5VisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    latent_query_num: int = 2048
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Kosmos2_5TextConfig()
            logger.info("`text_config` is `None`. initializing the `Kosmos2_5TextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config = Kosmos2_5TextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Kosmos2_5VisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Kosmos2_5VisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Kosmos2_5VisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)


__all__ = ["Kosmos2_5Config"]
