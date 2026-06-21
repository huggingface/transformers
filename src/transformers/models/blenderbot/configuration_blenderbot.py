# Copyright 2021 The Facebook, Inc. and The HuggingFace Inc. team. All rights reserved.
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
"""Blenderbot model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/blenderbot-3B")
@strict
class BlenderbotConfig(PreTrainedConfig):
    r"""
    encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 3):
        Number of ngrams to not be repeated in the encoder.

    Example:

    ```python
    >>> from transformers import BlenderbotConfig, BlenderbotModel

    >>> # Initializing a Blenderbot facebook/blenderbot-3B style configuration
    >>> configuration = BlenderbotConfig()

    >>> # Initializing a model (with random weights) from the facebook/blenderbot-3B style configuration
    >>> model = BlenderbotModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "blenderbot"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 8008
    max_position_embeddings: int = 128
    encoder_layers: int = 2
    encoder_ffn_dim: int = 10240
    encoder_attention_heads: int = 32
    decoder_layers: int = 24
    decoder_ffn_dim: int = 10240
    decoder_attention_heads: int = 32
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 2560
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int = 1
    scale_embedding: bool = False
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    encoder_no_repeat_ngram_size: int = 3
    forced_eos_token_id: int | list[int] | None = 2
    is_decoder: bool = False
    tie_word_embeddings: bool = True


__all__ = ["BlenderbotConfig"]
