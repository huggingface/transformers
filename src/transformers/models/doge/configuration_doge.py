# coding=utf-8
# Copyright 2024 Jingze Shi and the HuggingFace Inc. team.    All rights reserved.
#
# This code is based on the Wonderful Matrices paper implementation.
#
#     https://arxiv.org/abs/2407.16958
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
"""PyTorch Doge model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


class DogeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DogeModel`]. It is used to instantiate an Doge
    model according to the specified arguments, defining the model architecture like [LoserCheems/doge-tiny-test](https://huggingface.co/LoserCheems/doge-tiny-test)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32768):
            Vocabulary size of the Doge model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`DogeModel`]
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 16):
            Number of hidden layers in the Transformer decoder.
        hidden_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in the hidden layers.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability for each sequence transformation and state transformation module.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 16384):
            The maximum sequence length that this model might ever be used with.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_inner_values (`int`, *optional*, defaults to 8):
            Number of inner values for each attention layer in the Transformer decoder.
        cross_domain_intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the Cross Domain representations for the Cross Domain Mixture of Experts.
        private_expert_intermediate_size (`int`, *optional*, defaults to 1024):
            Dimension of the Private Expert representations for the Cross Domain Mixture of Experts.
        num_cdmmoe_experts (`int`, *optional*, defaults to 4096):
            Number of Private Experts for the Cross Domain Mixture of Experts.
        num_cdmmoe_heads (`int`, *optional*, defaults to 1):
            Number of heads of Private Experts for the Cross Domain Mixture of Experts.
        num_cdmmoe_experts_per_head (`int`, *optional*, defaults to 2):
            Number of Private Experts per head for the Cross Domain Mixture of Experts.
    """

    model_type = "doge"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=1024,
        num_hidden_layers=16,
        hidden_bias=False,
        hidden_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=16384,
        rope_theta=10000.0,
        rope_scaling=None,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        num_attention_heads=16,
        num_inner_values=8,
        cross_domain_intermediate_size=4096,
        private_expert_intermediate_size=1024,
        num_cdmmoe_experts=4096,
        num_cdmmoe_heads=1,
        num_cdmmoe_experts_per_head=2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_bias = hidden_bias
        self.hidden_dropout = hidden_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_inner_values = num_inner_values
        self.cross_domain_intermediate_size = cross_domain_intermediate_size
        self.private_expert_intermediate_size = private_expert_intermediate_size
        self.num_cdmmoe_experts = num_cdmmoe_experts
        self.num_cdmmoe_heads = num_cdmmoe_heads
        self.num_cdmmoe_experts_per_head = num_cdmmoe_experts_per_head

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
