# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# All rights reserved.
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
""" H3 configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

H3_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "stanford/H3-125m": "https://huggingface.co/stanford/H3-125m/resolve/main/config.json",
}


class H3Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of an [`H3Model`]. It is used to instantiate a GPT-2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the H3 [h3](https://huggingface.co/h3) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 50264):
            Vocabulary size of the H3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`H3Model`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. If not set, will set it to 4 times the `hidden_size`.
        residual_dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embedding_dropout (`int`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        fused_mlp (`bool`, *optional*, defaults to `False`):
            Whether to leverage a fused MLP.
        fused_dropout_add_ln (`bool`, *optional*, defaults to `False`):
            Whether to leverage a fused version of dropout + add + layer_norm.
        residual_in_fp32 (`bool`, *optional*, defaults to `False`):
            Whether apply the residual in floating point 32.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example:

    ```python
    >>> from transformers import H3Config, H3Model

    >>> # Initializing a H3 configuration
    >>> configuration = H3Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = H3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "h3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50264,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=16,
        n_inner=None,
        residual_dropout=0.1,
        embedding_dropout=0.1,
        fused_mlp=False,
        fused_dropout_add_ln=False,
        residual_in_fp32=False,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_inner = n_inner
        self.residual_dropout = residual_dropout
        self.embedding_dropout = embedding_dropout
        self.fused_mlp = fused_mlp
        self.fused_dropout_add_ln = fused_dropout_add_ln
        self.residual_in_fp32 = residual_in_fp32
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
