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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="allenai/led-base-16384")
class LEDConfig(PreTrainedConfig):
    r"""
    attention_window (`int` or `list[int]`, *optional*, defaults to 512):
        Size of an attention window around each token. If an `int`, use the same size for all layers. To specify a
        different window size for each layer, use a `list[int]` where `len(attention_window) == num_hidden_layers`.
    max_encoder_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that the encoder might ever be used with.
    max_decoder_position_embeddings (`int`, *optional*, defaults to 16384):
        The maximum sequence length that the decoder might ever be used with.

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
    }

    def __init__(
        self,
        vocab_size=50265,
        max_encoder_position_embeddings=16384,
        max_decoder_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        classifier_dropout=0.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        attention_window: list[int] | int = 512,
        tie_word_embeddings: bool | None = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_encoder_position_embeddings = max_encoder_position_embeddings
        self.max_decoder_position_embeddings = max_decoder_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.attention_window = attention_window

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


__all__ = ["LEDConfig"]
