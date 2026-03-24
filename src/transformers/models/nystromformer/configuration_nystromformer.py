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

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="uw-madison/nystromformer-512")
@strict
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

    vocab_size: int = 30000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu_new"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 510
    type_vocab_size: int = 2
    segment_means_seq_len: int = 64
    num_landmarks: int = 64
    conv_kernel_size: int = 65
    inv_coeff_init_option: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["NystromformerConfig"]
