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
    This is the configuration class to store the configuration of an [`H3Model`]. It is used to instantiate a H3 model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the H3
    [stanford/H3-125m](https://huggingface.co/stanford/H3-125m) architecture.

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
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. If not set, will set it to 4 times the `hidden_size`.
        residual_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the residual connections inside each block.
        embedding_dropout (`int`, *optional*, defaults to 0.1):
            The dropout probability for the embeddings.
        residual_in_fp32 (`bool`, *optional*, defaults to `False`):
            Whether apply the residual in floating point 32.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rescale_prenorm_residual (`bool`, *optional*, defaults to `True`):
            Whether to (...).
        glu_act (`bool`, *optional*, defaults to `False`):
            Whether to (...).
        ssm_mode (`str`, *optional*, defaults to `diag`):
            SSM mode to use.
        ssm_measure (`str`, *optional*, defaults to `diag-lin`):
            SSM measure to use.

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
        num_attention_heads=12,
        n_inner=None,
        residual_dropout=0.0,
        embedding_dropout=0.1,
        residual_in_fp32=False,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        glu_act=False,
        ssm_mode="diag",
        ssm_measure="diag-lin",
        attn_layer_idx=[6],
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
        self.residual_in_fp32 = residual_in_fp32
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.glu_act = glu_act
        self.ssm_mode = ssm_mode
        self.ssm_measure = ssm_measure
        self.attn_layer_idx = attn_layer_idx
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
