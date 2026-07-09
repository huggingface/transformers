# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
"""BART model configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/bart-large")
@strict
class BartConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import BartConfig, BartModel

    >>> # Initializing a BART facebook/bart-large style configuration
    >>> configuration = BartConfig()

    >>> # Initializing a model (with random weights) from the facebook/bart-large style configuration
    >>> model = BartModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 50265
    max_position_embeddings: int = 1024
    encoder_layers: int | None = 12
    encoder_ffn_dim: int | None = 4096
    encoder_attention_heads: int | None = 16
    decoder_layers: int | None = 12
    decoder_ffn_dim: int | None = 4096
    decoder_attention_heads: int | None = 16
    encoder_layerdrop: float | None = 0.0
    decoder_layerdrop: float | None = 0.0
    activation_function: str | None = "gelu"
    d_model: int | None = 1024
    dropout: float | int | None = 0.1
    attention_dropout: float | int | None = 0.0
    activation_dropout: float | int | None = 0.0
    init_std: float | None = 0.02
    classifier_dropout: float | int | None = 0.0
    scale_embedding: bool | None = False
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    is_encoder_decoder: bool | None = True
    decoder_start_token_id: int | None = 2
    forced_eos_token_id: int | list[int] | None = 2
    is_decoder: bool | None = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        # Set the default `num_labels` only if `id2label` is not
        # yet set, i.e. user didn't pass `id2label/lable2id` in kwargs
        if self.id2label is None:
            self.num_labels = kwargs.pop("num_labels", 3)

        super().__post_init__(**kwargs)


__all__ = ["BartConfig"]
