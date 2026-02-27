# Copyright Google AI and The HuggingFace Inc. team. All rights reserved.
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
"""CANINE model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/canine-s")
class CanineConfig(PreTrainedConfig):
    r"""
    downsampling_rate (`int`, *optional*, defaults to 4):
        The rate at which to downsample the original character sequence length before applying the deep Transformer
        encoder.
    upsampling_kernel_size (`int`, *optional*, defaults to 4):
        The kernel size (i.e. the number of characters in each window) of the convolutional projection layer when
        projecting back from `hidden_size`*2 to `hidden_size`.
    num_hash_functions (`int`, *optional*, defaults to 8):
        The number of hash functions to use. Each hash function has its own embedding matrix.
    num_hash_buckets (`int`, *optional*, defaults to 16384):
        The number of hash buckets to use.
    local_transformer_stride (`int`, *optional*, defaults to 128):
        The stride of the local attention of the first shallow Transformer encoder. Defaults to 128 for good
        TPU/XLA memory alignment.

    Example:

    ```python
    >>> from transformers import CanineConfig, CanineModel

    >>> # Initializing a CANINE google/canine-s style configuration
    >>> configuration = CanineConfig()

    >>> # Initializing a model (with random weights) from the google/canine-s style configuration
    >>> model = CanineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "canine"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=16384,
        type_vocab_size=16,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        bos_token_id=0xE000,
        eos_token_id=0xE001,
        downsampling_rate=4,
        upsampling_kernel_size=4,
        num_hash_functions=8,
        num_hash_buckets=16384,
        local_transformer_stride=128,  # Good TPU/XLA memory alignment.
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps

        # Character config:
        self.downsampling_rate = downsampling_rate
        self.upsampling_kernel_size = upsampling_kernel_size
        self.num_hash_functions = num_hash_functions
        self.num_hash_buckets = num_hash_buckets
        self.local_transformer_stride = local_transformer_stride


__all__ = ["CanineConfig"]
