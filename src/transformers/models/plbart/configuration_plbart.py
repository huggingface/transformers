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

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="pixtral-hf/pixtral-9b")
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
    }

    def __init__(
        self,
        vocab_size=50005,
        max_position_embeddings=1024,
        encoder_layers=6,
        encoder_ffn_dim=3072,
        encoder_attention_heads=12,
        decoder_layers=6,
        decoder_ffn_dim=3072,
        decoder_attention_heads=12,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=768,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        is_decoder=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.is_decoder = is_decoder
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
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
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.forced_eos_token_id = forced_eos_token_id
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


__all__ = ["PLBartConfig"]
