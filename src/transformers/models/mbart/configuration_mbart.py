# Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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
"""MBART model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/mbart-large-cc25")
@strict
class MBartConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import MBartConfig, MBartModel

    >>> # Initializing a MBART facebook/mbart-large-cc25 style configuration
    >>> configuration = MBartConfig()

    >>> # Initializing a model (with random weights) from the facebook/mbart-large-cc25 style configuration
    >>> model = MBartModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 50265
    max_position_embeddings: int = 1024
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
    classifier_dropout: float | int = 0.0
    scale_embedding: int = False
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    decoder_start_token_id: int | None = None
    forced_eos_token_id: int | list[int] | None = 2
    is_decoder: bool = False
    tie_word_embeddings: bool = True


__all__ = ["MBartConfig"]
