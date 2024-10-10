# coding=utf-8
# Copyright 2024 GLM & ZhipuAI team. All rights reserved.
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

"""GLM model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class GLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GLMModel`]. It is used to instantiate a GLM
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer decoder.
        vocab_size (`int`, *optional*, defaults to 151552):
            Vocabulary size of the Phi-3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GLMModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 13696):
            Dimension of the MLP representations.
        kv_channels (`int`, *optional*, defaults to 128):
            Defines the number of channels for the key and value tensors.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the hidden layer.
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after computing the attention scores.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon value used for the RMSNorm.
        add_qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to the query, key, value tensors.
            Whether to use multi query attention or not.
        multi_query_attention (`bool`, *optional*, defaults to `False`):
            Whether to use multi query attention or not.
        multi_query_group_num (`int`, *optional*, defaults to 2):
            The number of groups in the multi query attention
        rope_theta (`float`, *optional*, defaults to 1.0):
            The base period of the RoPE embeddings.
        apply_query_key_layer_scaling (`bool`, *optional*, defaults to `True`):
            Whether to apply layer scaling to query and key.
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to use fp32 for softmax in attention.
            Whether to use fp32 for residual connection.
        fp32_residual_connection (`bool`, *optional*, defaults to `False`):
            Whether to use fp32 for residual connection.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`. Whether to tie weight embeddings or not.
    Example:

    ```python
    >>> from transformers import GLMModel, GLMConfig
    >>> configuration = GLMConfig.from_pretrained("THUDM/glm-4-9b-chat")
    >>> model = GLMModel(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "glm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=4096,
        intermediate_size=13696,
        kv_channels=128,
        num_hidden_layers=40,
        num_attention_heads=32,
        max_position_embeddings=131072,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        add_qkv_bias=True,
        multi_query_attention=False,
        multi_query_group_num=2,
        rope_theta=1.0,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.kv_channels = kv_channels

        self.add_qkv_bias = add_qkv_bias
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.rope_theta = rope_theta
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.use_cache = use_cache

        super().__init__(**kwargs)
