# coding=utf-8
# Copyright 2022 THUDM and The HuggingFace Inc. team. All rights reserved.
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
""" ChatGLM model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CHATGLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "THUDM/chatglm3-6b": "https://huggingface.co/THUDM/chatglm3-6b/resolve/main/config.json",
    "THUDM/chatglm3-6b-32k": "https://huggingface.co/THUDM/chatglm3-6b-32k/resolve/main/config.json",
    "THUDM/chatglm3-6b-base": "https://huggingface.co/THUDM/chatglm3-6b-base/resolve/main/config.json",
    # See all ChatGLM models at https://huggingface.co/models?filter=chatglm
}


class ChatGLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ChatGLMModel`].
    It is used to instantiate an ChatGLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the ChatGLM3-6B [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) architecture.
    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.
    Args:
        num_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        padded_vocab_size (`int`, *optional*, defaults to 65024):
            The final vocabulary size.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers and the pooler layer.
        ffn_hidden_size (`int`, *optional*, defaults to 13696):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        kv_channels (`int`, *optional*, defaults to 128):
            Number of KV channels.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        seq_length (`int`, *optional*, defaults to 8192):
            Length of the sequence.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate in the hidden layers.
        classifier_dropout (`float`, *optional*):
            The dropout rate in the classification layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout rate in the attention layers.
        layernorm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        rmsnorm (`bool`, *optional*, defaults to `True`):
            Whether to use RMSNorm.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            Whether to apply residual connection on post layernorm. If set to `False`, residual connections are added to hidden states.
        post_layer_norm (`bool`, *optional*, defaults to `True`):
            Whether to add a final layernom to the transformer.
        add_bias_linear (`bool`, *optional*, defaults to `False`):
            Whether to add bias to linear layers (including QKV).
        add_qkv_bias (`bool`, *optional*, defaults to `False`):
            Whether to add bias to qkv linear layers.
        multi_query_attention (`bool`, *optional*, defaults to `True`):
            Whether to use multi query attention.
        multi_query_group_num (`int`, *optional*, defaults to 2):
            Whether to use grouped query attention.
        apply_query_key_layer_scaling (`bool`, *optional*, defaults to `True`):
            Whether to apply query key layer scaling.
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to convert activations to Float32 for softmax. Always `true` if `apply_query_key_layer_scaling` is true.
        fp32_residual_connection (`bool`, *optional*, defaults to `False`):
            If true, convert embeddings to float32.
        quantization_bit (`int`, *optional*, defaults to 0):
            Specifies quantization bits.
        pre_seq_len (`int`, *optional*):
            Length of the prefix sequence to be encoded using `PrefixEncoder`.
        prefix_projection (`bool`, *optional*, defaults to `False`):
            Whether to use a 2-layer MLP to encode the prefix in `PrefixEncoder`.
        original_rope (`bool`, *optional*, defaults to `True`):
            Whether to use original RoPE. Note: ChatGLM uses paritial RoPE even it is true.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to enable KV cache.

        test_force_initializer_range (`float`, *optional*):
            (Test ONLY) Force initializes the model weights in the given range.

    ```python
    >>> from transformers import ChatGLMModel, ChatGLMConfig

    >>> # Initializing a ChatGLM3-6B THUDM/ChatGLM3-6B style configuration
    >>> configuration = ChatGLMConfig()

    >>> model = ChatGLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "chatglm"

    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=8192,
        hidden_dropout=0.0,
        classifier_dropout=None,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        multi_query_attention=True,
        multi_query_group_num=2,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        original_rope=True,
        use_cache=True,
        test_force_initializer_range=None,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        self.original_rope = original_rope
        self.use_cache = use_cache

        # Parameters which are test-use only
        self.test_force_initializer_range = test_force_initializer_range  # only used in unittesting

        super().__init__(**kwargs)
