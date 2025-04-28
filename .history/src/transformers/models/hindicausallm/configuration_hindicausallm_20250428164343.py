# coding=utf-8
# Copyright 2024 ConvaiInnovations and The HuggingFace Inc. team. All rights reserved.
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
"""HindiCausalLM model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

HINDICAUSALLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "convaiinnovations/hindi-causal-lm": "https://huggingface.co/convaiinnovations/hindi-causal-lm/resolve/main/config.json",
}


class HindiCausalLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`HindiCausalLMModel`]. It is used to instantiate a
    HindiCausalLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [convaiinnovations/hindi-causal-lm](https://huggingface.co/convaiinnovations/hindi-causal-lm) model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 16000):
            Vocabulary size of the HindiCausalLM model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`HindiCausalLMModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `4`.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the RMS normalization layers. Changed from layer_norm_eps for consistency with Llama/Gemma.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of sequence token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of sequence token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie input and output embeddings.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        cache_implementation (`str`, *optional*, defaults to `"dynamic"`):
             The cache structure implementation to use if `use_cache=True`. Must be one of `"static"`, `"dynamic"`,
             or `"hybrid"`. Defaults to `"dynamic"`.

    Example:

    ```python
    >>> from transformers import HindiCausalLMConfig, HindiCausalLMModel

    >>> # Initializing a HindiCausalLM style configuration
    >>> configuration = HindiCausalLMConfig()

    >>> # Initializing a model from the configuration
    >>> model = HindiCausalLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "hindicausallm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=16000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=3072,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        rms_norm_eps=1e-5, # Renamed from layer_norm_eps
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        attention_bias=False, # Added attention_bias
        attention_dropout=0.0,
        cache_implementation="dynamic", # Added cache_implementation
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # For GQA
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.cache_implementation = cache_implementation

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )