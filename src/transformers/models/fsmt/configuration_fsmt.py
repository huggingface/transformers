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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class DecoderConfig(PreTrainedConfig):
    model_type = "fsmt_decoder"

    def __init__(self, vocab_size=0, bos_token_id=0, is_encoder_decoder=True, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.is_encoder_decoder = is_encoder_decoder


@auto_docstring(checkpoint="facebook/wmt19-en-ru")
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
    max_length (`int`, *optional*, defaults to 200):
        Maximum length to generate.

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
    }

    # update the defaults from config file
    def __init__(
        self,
        langs=["en", "de"],
        src_vocab_size=42024,
        tgt_vocab_size=42024,
        activation_function="relu",
        d_model=1024,
        max_length=200,
        max_position_embeddings=1024,
        encoder_ffn_dim=4096,
        encoder_layers=12,
        encoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_ffn_dim=4096,
        decoder_layers=12,
        decoder_attention_heads=16,
        decoder_layerdrop=0.0,
        attention_dropout=0.0,
        dropout=0.1,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        is_encoder_decoder=True,
        scale_embedding=True,
        tie_word_embeddings=False,
        num_beams=5,
        length_penalty=1.0,
        early_stopping=False,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        **common_kwargs,
    ):
        self.langs = langs
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim

        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.init_std = init_std  # Normal(0, this parameter)
        self.activation_function = activation_function

        common_kwargs.pop("decoder", None)  # delete unused kwargs
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        # 3 Types of Dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **common_kwargs,
        )


__all__ = ["FSMTConfig"]
