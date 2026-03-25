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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/canine-s")
@strict
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

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 16384
    type_vocab_size: int = 16
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = 0xE000
    eos_token_id: int | list[int] | None = 0xE001
    downsampling_rate: int = 4
    upsampling_kernel_size: int = 4
    num_hash_functions: int = 8
    num_hash_buckets: int = 16384
    local_transformer_stride: int = 128  # Good TPU/XLA memory alignment


__all__ = ["CanineConfig"]
