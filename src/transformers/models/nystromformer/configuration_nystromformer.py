# Copyright 2022 UW-Madison and The HuggingFace Inc. team. All rights reserved.
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
"""Nystromformer model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="uw-madison/nystromformer-512")
class NystromformerConfig(PreTrainedConfig):
    r"""
    segment_means_seq_len (`int`, *optional*, defaults to 64):
        Sequence length used in segment-means.
    num_landmarks (`int`, *optional*, defaults to 64):
        The number of landmark (or Nystrom) points to use in Nystrom approximation of the softmax self-attention
        matrix.
    conv_kernel_size (`int`, *optional*, defaults to 65):
        The kernel size of depthwise convolution used in Nystrom approximation.
    inv_coeff_init_option (`bool`, *optional*, defaults to `False`):
        Whether or not to use exact coefficient computation for the initial values for the iterative method of
        calculating the Moore-Penrose inverse of a matrix.

    Example:

    ```python
    >>> from transformers import NystromformerModel, NystromformerConfig

    >>> # Initializing a Nystromformer uw-madison/nystromformer-512 style configuration
    >>> configuration = NystromformerConfig()

    >>> # Initializing a model from the uw-madison/nystromformer-512 style configuration
    >>> model = NystromformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "nystromformer"

    def __init__(
        self,
        vocab_size=30000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=510,
        type_vocab_size=2,
        segment_means_seq_len=64,
        num_landmarks=64,
        conv_kernel_size=65,
        inv_coeff_init_option=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        add_cross_attention=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.add_cross_attention = add_cross_attention
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
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
        self.segment_means_seq_len = segment_means_seq_len
        self.num_landmarks = num_landmarks
        self.conv_kernel_size = conv_kernel_size
        self.inv_coeff_init_option = inv_coeff_init_option
        self.layer_norm_eps = layer_norm_eps
        super().__init__(**kwargs)


__all__ = ["NystromformerConfig"]
