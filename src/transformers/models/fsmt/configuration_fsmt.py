# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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
"""FSMT configuration"""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/wmt19-en-ru")
@strict
class FSMTConfig(PreTrainedConfig):
    r"""
    langs (`list[str]`):
        A list with source language and target_language (e.g., ['en', 'ru']).
    src_vocab_size (`int`):
        Vocabulary size of the encoder. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed to the forward method in the encoder.
    tgt_vocab_size (`int`):
        Vocabulary size of the decoder. Defines the number of different tokens that can be represented by the
        `inputs_ids` passed to the forward method in the decoder.
    max_length (`int`, *optional*, defaults to 200):
        Maximum length to generate.
    num_beams (`int`, *optional*, defaults to 5):
        Number of beams for beam search that will be used by default in the `generate` method of the model. 1 means
        no beam search.
    length_penalty (`float`, *optional*, defaults to 1):
        Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
        the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
        likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
        `length_penalty` < 0.0 encourages shorter sequences.
    early_stopping (`bool`, *optional*, defaults to `False`):
        Flag that will be used by default in the `generate` method of the model. Whether to stop the beam search
        when at least `num_beams` sentences are finished per batch or not.

    Examples:

    ```python
    >>> from transformers import FSMTConfig, FSMTModel

    >>> # Initializing a FSMT facebook/wmt19-en-ru style configuration
    >>> config = FSMTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FSMTModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fsmt"
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "vocab_size": "tgt_vocab_size",
        "num_hidden_layers": "encoder_layers",
    }

    langs: list[str] | tuple[str, ...] = ("en", "de")
    src_vocab_size: int = 42024
    tgt_vocab_size: int = 42024
    activation_function: str = "relu"
    d_model: int = 1024
    max_length: int = 200
    max_position_embeddings: int = 1024
    encoder_ffn_dim: int = 4096
    encoder_layers: int = 12
    encoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.0
    decoder_ffn_dim: int = 4096
    decoder_layers: int = 12
    decoder_attention_heads: int = 16
    decoder_layerdrop: float | int = 0.0
    attention_dropout: float | int = 0.0
    dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int | None = 2
    is_encoder_decoder: bool = True
    scale_embedding: bool = True
    tie_word_embeddings: bool = False
    num_beams: int = 5
    length_penalty: float = 1.0
    early_stopping: bool = False
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    forced_eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs):
        kwargs.pop("decoder", None)  # delete unused kwargs
        super().__post_init__(**kwargs)


__all__ = ["FSMTConfig"]
