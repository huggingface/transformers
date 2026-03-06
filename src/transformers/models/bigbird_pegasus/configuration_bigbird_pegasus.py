# Copyright Google Research and The HuggingFace Inc. team. All rights reserved.
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
"""BigBirdPegasus model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/bigbird-pegasus-large-arxiv")
class BigBirdPegasusConfig(PreTrainedConfig):
    r"""
    attention_type (`str`, *optional*, defaults to `"block_sparse"`):
        Whether to use block sparse attention (with n complexity) as introduced in paper or original attention
        layer (with n^2 complexity). Possible values are `"original_full"` and `"block_sparse"`.
    use_bias (`bool`, *optional*, defaults to `True`):
        Whether to use bias in query, key, value.
    rescale_embeddings (`bool`, *optional*, defaults to `False`):
        Whether to rescale embeddings with (hidden_size ** 0.5).
    block_size (`int`, *optional*, defaults to 64):
        Size of each block. Useful only when `attention_type == "block_sparse"`.
    num_random_blocks (`int`, *optional*, defaults to 3):
        Each query is going to attend these many number of random blocks. Useful only when `attention_type ==
        "block_sparse"`.

    Example:

    ```python
    >>> from transformers import BigBirdPegasusConfig, BigBirdPegasusModel

    >>> # Initializing a BigBirdPegasus bigbird-pegasus-base style configuration
    >>> configuration = BigBirdPegasusConfig()

    >>> # Initializing a model (with random weights) from the bigbird-pegasus-base style configuration
    >>> model = BigBirdPegasusModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bigbird_pegasus"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
    }

    def __init__(
        self,
        vocab_size=96103,
        max_position_embeddings=4096,
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
        activation_function="gelu_new",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        decoder_start_token_id=2,
        classifier_dropout=0.0,
        scale_embedding=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        attention_type="block_sparse",  # only for encoder
        block_size=64,
        num_random_blocks=3,
        use_bias=False,
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

        # extra config
        self.attention_type = attention_type
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.use_bias = use_bias

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.decoder_start_token_id = decoder_start_token_id
        super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)


__all__ = ["BigBirdPegasusConfig"]
