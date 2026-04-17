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
"""Pix2Struct model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/pix2struct-base")
@strict
class Pix2StructTextConfig(PreTrainedConfig):
    r"""
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance of the longer sequences for the bucket separation.
    dense_act_fn (`Union[Callable, str]`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string).

    Example:

    ```python
    >>> from transformers import Pix2StructTextConfig, Pix2StructTextModel

    >>> # Initializing a Pix2StructTextConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructTextConfig()

    >>> # Initializing a Pix2StructTextModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pix2struct_text_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "decoder_attention_heads": "num_heads",
        "encoder_attention_heads": "num_heads",
        "encoder_layers": "num_layers",
        "decoder_layers": "num_layers",
    }

    vocab_size: int = 50244
    hidden_size: int = 768
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 12
    num_heads: int = 12
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_factor: float = 1.0
    dense_act_fn: str = "gelu_new"
    decoder_start_token_id: int = 0
    use_cache: bool = False
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = None
    tie_word_embeddings: bool = False
    is_decoder: bool = True
    add_cross_attention: bool = False


@auto_docstring(checkpoint="google/pix2struct-base")
@strict
class Pix2StructVisionConfig(PreTrainedConfig):
    r"""
    patch_embed_hidden_size (`int`, *optional*, defaults to 768):
        Dimensionality of the input patch_embedding layer in the Transformer encoder.
    d_ff (`int`, *optional*, defaults to 2048):
        Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    d_kv (`int`, *optional*, defaults to 64):
        Dimensionality of the key, query, value projections per attention head.
    The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
        `"relu"`, `"selu"` and `"gelu_new"` `"gelu"` are supported.
    dense_act_fn (`Union[Callable, str]`, *optional*, defaults to `"gelu_new"`):
        The non-linear activation function (function or string).
    seq_len (`int`, *optional*, defaults to 4096):
        Maximum sequence length (here number of patches) supported by the model.
    relative_attention_num_buckets (`int`, *optional*, defaults to 32):
        The number of buckets to use for each attention layer.
    relative_attention_max_distance (`int`, *optional*, defaults to 128):
        The maximum distance (in tokens) to use for each attention layer.

    Example:

    ```python
    >>> from transformers import Pix2StructVisionConfig, Pix2StructVisionModel

    >>> # Initializing a Pix2StructVisionConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructVisionConfig()

    >>> # Initializing a Pix2StructVisionModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pix2struct_vision_model"

    hidden_size: int = 768
    patch_embed_hidden_size: int = 768
    d_ff: int = 2048
    d_kv: int = 64
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    dense_act_fn: str = "gelu_new"
    layer_norm_eps: float = 1e-6
    dropout_rate: float | int = 0.0
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    initializer_factor: float = 1.0
    seq_len: int = 4096
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128


@auto_docstring(checkpoint="google/pix2struct-base")
@strict
class Pix2StructConfig(PreTrainedConfig):
    r"""
    is_vqa (`bool`, *optional*, defaults to `False`):
        Whether the model has been fine-tuned for VQA or not.

    Example:

    ```python
    >>> from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration

    >>> # Initializing a Pix2StructConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructConfig()

    >>> # Initializing a Pix2StructForConditionalGeneration (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Pix2StructConfig from a Pix2StructTextConfig and a Pix2StructVisionConfig

    >>> # Initializing a Pix2Struct text and Pix2Struct vision configuration
    >>> config_text = Pix2StructTextConfig()
    >>> config_vision = Pix2StructVisionConfig()

    >>> config = Pix2StructConfig(text_config=config_text, vision_config=config_vision)
    ```"""

    model_type = "pix2struct"
    sub_configs = {"text_config": Pix2StructTextConfig, "vision_config": Pix2StructVisionConfig}

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    initializer_factor: float = 1.0
    initializer_range: float = 0.02
    is_vqa: bool = False
    tie_word_embeddings: bool = False
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Pix2StructTextConfig(
                is_encoder_decoder=self.is_encoder_decoder,
                tie_word_embeddings=self.tie_word_embeddings,
            )
            logger.info("`text_config` is `None`. initializing the `Pix2StructTextConfig` with default values.")
        elif isinstance(self.text_config, dict):
            self.text_config["is_encoder_decoder"] = self.is_encoder_decoder
            self.text_config["tie_word_embeddings"] = self.tie_word_embeddings
            self.text_config = Pix2StructTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Pix2StructVisionConfig()
            logger.info("`vision_config` is `None`. initializing the `Pix2StructVisionConfig` with default values.")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Pix2StructVisionConfig(**self.vision_config)

        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        self.pad_token_id = self.text_config.pad_token_id
        self.eos_token_id = self.text_config.eos_token_id

        self.text_config.initializer_range = self.initializer_range
        self.vision_config.initializer_range = self.initializer_range

        super().__post_init__(**kwargs)


__all__ = ["Pix2StructConfig", "Pix2StructTextConfig", "Pix2StructVisionConfig"]
