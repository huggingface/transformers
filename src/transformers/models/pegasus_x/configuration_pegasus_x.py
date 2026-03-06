# Copyright 2022, Google and The HuggingFace Inc. team. All rights reserved.
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
"""PEGASUS-X model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/pegasus-x-large")
class PegasusXConfig(PreTrainedConfig):
    r"""
    num_global_tokens (`int`, *optional*, defaults to 128):
        Number of global tokens to use for the encoder
    block_size (`int`, *optional*, defaults to 512):
        Block size for encoder local attention. Sequence length should be an exact multiple of block size.
        block_size must be a multiple of 2 if stagger_local_block is True
    stagger_local_blocks (`bool`, *optional*, defaults to `True`):
        Whether to stagger every other local attention by half a block

    Example:

    ```python
    >>> from transformers import PegasusXConfig, PegasusXModel

    >>> # Initializing a PEGASUS google/pegasus-x-large style configuration
    >>> configuration = PegasusXConfig()

    >>> # Initializing a model (with random weights) from the google/pegasus-x-large style configuration
    >>> model = PegasusXModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pegasus_x"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    def __init__(
        self,
        vocab_size=96103,
        max_position_embeddings=16384,
        encoder_layers=16,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=16,
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
        decoder_start_token_id=0,
        scale_embedding=True,
        pad_token_id=0,
        eos_token_id=1,
        forced_eos_token_id=1,
        num_global_tokens=32,
        block_size=512,
        stagger_local_blocks=True,
        tie_word_embeddings=True,
        **kwargs,
    ):
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
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        self.num_global_tokens = num_global_tokens
        self.block_size = block_size
        self.stagger_local_blocks = stagger_local_blocks
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )


__all__ = ["PegasusXConfig"]
