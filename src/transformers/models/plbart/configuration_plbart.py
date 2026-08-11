# Copyright 2022, UCLA NLP, The Facebook AI Research Team and The HuggingFace Inc. team. All rights reserved.
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
"""PLBART model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="uclanlp/plbart-base")
@strict
class PLBartConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import PLBartConfig, PLBartModel

    >>> # Initializing a PLBART uclanlp/plbart-base style configuration
    >>> configuration = PLBartConfig()

    >>> # Initializing a model (with random weights) from the uclanlp/plbart-base style configuration
    >>> model = PLBartModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "plbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "initializer_range": "init_std",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 50005
    max_position_embeddings: int = 1024
    encoder_layers: int = 6
    encoder_ffn_dim: int = 3072
    encoder_attention_heads: int = 12
    decoder_layers: int = 6
    decoder_ffn_dim: int = 3072
    decoder_attention_heads: int = 12
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 768
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    classifier_dropout: float | int = 0.0
    scale_embedding: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    forced_eos_token_id: int | list[int] | None = 2
    is_decoder: bool = False
    tie_word_embeddings: bool = True


__all__ = ["PLBartConfig"]
