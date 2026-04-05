# coding=utf-8
# Copyright 2025 Xiaomi and The HuggingFace Inc. team. All rights reserved.
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
"""MiMo-V2 Flash model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)


class MiMoV2FlashConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MiMoV2FlashModel`]. It is used to instantiate an
    MiMo-V2-Flash model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the MiMo-V2-Flash [XiaomiMiMo/MiMo-V2-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 152576):
            Vocabulary size of the MiMo-V2-Flash model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MiMoV2FlashModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 16384):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        head_dim (`int`, *optional*, defaults to 192):
            The attention head dimension for Q and K.
        v_head_dim (`int`, *optional*, defaults to 128):
            The attention head dimension for V. This is specific to the MiMo-V2 architecture.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        rope_parameters (`dict`, *optional*):
            A dictionary containing parameters for the Rotary Position Embedding (RoPE). Expected keys are `rope_theta`
            (defaults to 5000000.0) and `partial_rotary_factor` (defaults to 0.334).
        sliding_window (`int`, *optional*, defaults to 128):
            Sliding window attention window size.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        layer_types (`List[str]`, *optional*):
            List of strings defining the type of attention for each layer (e.g., "full_attention" or "sliding_attention").
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            The number of active experts per token.
        n_routed_experts (`int`, *optional*, defaults to 256):
            The total number of routed experts.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            The intermediate size of each expert.
        scoring_func (`str`, *optional*, defaults to `"sigmoid"`):
            The scoring function used for the MoE router.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the top-k probabilities in the MoE router.

    """

    model_type = "mimo_v2_flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=152576,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=48,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=192,
        v_head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        rope_parameters=None,
        sliding_window=128,
        attention_dropout=0.0,
        layer_types=None,
        num_experts_per_tok=8,
        n_routed_experts=256,
        moe_intermediate_size=2048,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = v_head_dim

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout

        # RoPE Parameters
        if rope_parameters is None:
            self.rope_parameters = {
                "rope_theta": 5000000.0,
                "partial_rotary_factor": 0.334,
            }
        else:
            self.rope_parameters = rope_parameters

        # Layer Types
        self.layer_types = layer_types

        # MoE Config
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob

        super().__init__(**kwargs)
