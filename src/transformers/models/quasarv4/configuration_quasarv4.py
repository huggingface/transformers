# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""QuasarV4 model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

QUASARV4_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "silx-ai/QuasarV4-600M-Transformer": "https://huggingface.co/silx-ai/QuasarV4-600M-Transformer/resolve/main/config.json",
}


class QuasarV4Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`QuasarV4Model`]. It is used to instantiate a
    QuasarV4 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the QuasarV4
    [silx-ai/QuasarV4-600M-Transformer](https://huggingface.co/silx-ai/QuasarV4-600M-Transformer) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the QuasarV4 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`QuasarV4Model`].
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group of `num_attention_heads/num_key_value_heads`
            heads in the original checkpoint is merged into a single key/value head in the GQA checkpoint. For more details checkout
            [this paper](https://arxiv.org/pdf/2305.13245.pdf).
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of the attention heads.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 40960):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Linear scaling is described in the paper [Scaling Transformer to 1M tokens
            and beyond with RoPE](https://arxiv.org/abs/2312.06657). Dynamic scaling is described in the paper
            [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071).
            Both scaling strategies can be configured by setting `rope_scaling` to a dictionary with the following keys:
            - `type` (`str`) -- The scaling strategy to use. Can be either `"linear"` or `"dynamic"`.
            - `factor` (`float`) -- The scaling factor to use. For linear scaling, this factor is directly multiplied to
              the positions before calculating the RoPE embeddings. For dynamic scaling, the window length is scaled by
              this factor.
            Additional keys that need to be set for dynamic scaling:
            - `original_max_position_embeddings` (`int`) -- The original maximum context length the model was trained
              with. E.g. for Llama 2 this would be 4096.
            - `attn_factor` (`float`, defaults to 1.0) -- The factor to scale the attention logits by. This is useful to
              re-scale the attention logits to the same magnitude as during training.
            - `beta_fast` (`float`, defaults to 32.0) -- The beta factor for short context lengths.
            - `beta_slow` (`float`, defaults to 1.0) -- The beta factor for long context lengths.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to None (no sliding window).
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention for the model. If not specified, will default to `False`.
        token_temperature (`Dict`, *optional*):
            Dictionary containing the configuration for the token temperature mechanism. If not specified, will default to None.
        output_adaptation (`Dict`, *optional*):
            Dictionary containing the configuration for the output adaptation layers. If not specified, will default to None.
        dense_residual_connections (`Dict`, *optional*):
            Dictionary containing the configuration for the dense residual connections. If not specified, will default to None.
        temperature_aggregation (`Dict`, *optional*):
            Dictionary containing the configuration for the temperature aggregation layers. If not specified, will default to None.

    Example:

    ```python
    >>> from transformers import QuasarV4Config, QuasarV4Model

    >>> # Initializing a QuasarV4 configuration
    >>> configuration = QuasarV4Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = QuasarV4Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "quasarv4"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=40960,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=None,
        use_sliding_window=False,
        bos_token_id=151643,
        eos_token_id=151645,
        token_temperature={
            "enabled": True,
            "temperature_dim": None,  # Will be set to hidden_size // 4 if None
            "num_temperature_layers": 4,
            "position_dependent_scaling": True,
            "context_aware_scaling": True,
        },
        output_adaptation={"enabled": True, "adaptation_factor": 0.1},
        dense_residual_connections={"enabled": True, "connection_factor": 0.05},
        temperature_aggregation={
            "enabled": True,
            "aggregation_layers": 5,
            "global_scaling_factor": 0.05,
        },
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window

        # QuasarV4 specific config
        if token_temperature["temperature_dim"] is None:
            token_temperature["temperature_dim"] = hidden_size // 4
        self.token_temperature = token_temperature
        self.output_adaptation = output_adaptation
        self.dense_residual_connections = dense_residual_connections
        self.temperature_aggregation = temperature_aggregation

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
