# coding=utf-8
# Copyright 2024 JetMoE AI and the HuggingFace Inc. team. All rights reserved.
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
""" JetMoE model configuration"""

import json

import torch.nn.init as init

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


from ..deprecated._archive_maps import JETMOE_PRETRAINED_CONFIG_ARCHIVE_MAP  # noqa: F401, E402


class JetMoEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JetMoEModel`]. It is used to instantiate an
    JetMoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the JetMoE-7B-v0.1 or JetMoE-7B-Instruct-v0.1.

    [jetmoe/jetmoe-8b](https://huggingface.co/jetmoe/jetmoe-8b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the JetMoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JetMoEModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14336):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `8`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to `4096*32`):
            The maximum sequence length that this model might ever be used with. JetMoE's sliding window attention
            allows sequence of up to 4096*32 tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention window size. If not specified, will default to `4096`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import JetMoEModel, JetMoEConfig

    >>> # Initializing a JetMoE 7B style configuration
    >>> configuration = JetMoEConfig()

    >>> # Initializing a model from the JetMoE 7B style configuration
    >>> model = JetMoEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "jetmoe"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50295,
        hidden_size=1024,
        num_layers=24,
        num_attention_heads=16,
        kv_channels = 128,
        ffn_hidden_size=2048,
        max_position_embeddings=4096,
        rotary_percent=1.0,
        activation_function="silu",
        glu=True,
        moe_num_experts=8,
        moe_top_k=2,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        bias=True,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        initializer_range=0.01,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels
        self.ffn_hidden_size = ffn_hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rotary_percent = rotary_percent
        self.activation_function = activation_function
        self.glu = glu
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.init_method = init.xavier_uniform_
        self.output_layer_init_method = init.xavier_uniform_
        self.bias = bias
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )

    def to_dict(self):
        """Returns a dictionary representation of the config, excluding non-serializable attributes."""
        return {k: v for k, v in self.__dict__.items() if k not in ['init_method', 'output_layer_init_method', 'torch_dtype', '_pre_quantization_dtype', 'quantization_config']}

    def to_json_string(self, use_diff=False):
        """Serializes this instance to a JSON string, excluding non-serializable attributes.
        
        Args:
            use_diff (bool): Whether to use differences with the default config. This argument is
                             accepted for compatibility with the transformers library but is not
                             used in this custom implementation.
        """
        config_dict = self.to_dict()  # Assuming you have a to_dict method as shown earlier
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
