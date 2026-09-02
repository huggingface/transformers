# Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. All rights reserved.
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
"""LED model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="allenai/led-base-16384")
@strict
class LEDConfig(PreTrainedConfig):
    r"""
    max_encoder_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that the encoder might ever be used with.
    max_decoder_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that the decoder might ever be used with.
    attention_window (`int` or `list[int]`, *optional*, defaults to 512):
        Size of an attention window around each token. If an `int`, use the same size for all layers. To specify a
        different window size for each layer, use a `list[int]` where `len(attention_window) == num_hidden_layers`.

    Example:

    ```python
    >>> from transformers import LEDModel, LEDConfig

    >>> # Initializing a LED allenai/led-base-16384 style configuration
    >>> configuration = LEDConfig()

    >>> # Initializing a model from the allenai/led-base-16384 style configuration
    >>> model = LEDModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "led"
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
        "initializer_range": "init_std",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 50265
    max_encoder_position_embeddings: int = 16384
    max_decoder_position_embeddings: int = 1024
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int = 2
    classifier_dropout: float | int = 0.0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    attention_window: list[int] | int = 512
    tie_word_embeddings: bool = True


__all__ = ["LEDConfig"]
