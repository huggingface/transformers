# coding=utf-8
# Copyright 2024 JetMoe AI and the HuggingFace Inc. team. All rights reserved.
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
"""JetMoe model configuration"""

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters, rope_config_validation, standardize_rope_params
from ...utils import logging


logger = logging.get_logger(__name__)


class JetMoeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`JetMoeModel`]. It is used to instantiate a
    JetMoe model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a configuration of the JetMoe-4B.

    [jetmoe/jetmoe-8b](https://huggingface.co/jetmoe/jetmoe-8b)

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the JetMoe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`JetMoeModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each key and value in the Transformer encoder.
        kv_channels (`int`, *optional*, defaults to 128):
            Defines the number of channels for the key and value tensors.
        intermediate_size (`int`, *optional*, defaults to 5632):
            Dimension of the MLP representations.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with. JetMoe's attention allows sequence of
            up to 4096 tokens.
        activation_function (`string`, *optional*, defaults to `"silu"`):
            Defines the activation function for MLP experts.
        num_local_experts (`int`, *optional*, defaults to 8):
            Defines the number of experts in the MoE and MoA.
        num_experts_per_tok (`int, *optional*, defaults to 2):
            The number of experts to route per-token and for MoE and MoA.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss.
        aux_loss_coef (`float`, *optional*, defaults to 0.01):
            The coefficient for the auxiliary loss.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionaty should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.01):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import JetMoeModel, JetMoeConfig

    >>> # Initializing a JetMoe 4B style configuration
    >>> configuration = JetMoeConfig()

    >>> # Initializing a model from the JetMoe 4B style configuration
    >>> model = JetMoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "jetmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"head_dim": "kv_channels"}

    def __init__(
        self,
        vocab_size: Optional[int] = 32000,
        hidden_size: Optional[int] = 2048,
        num_hidden_layers: Optional[int] = 12,
        num_key_value_heads: Optional[int] = 16,
        kv_channels: Optional[int] = 128,
        intermediate_size: Optional[int] = 5632,
        max_position_embeddings: Optional[int] = 4096,
        activation_function: Optional[str] = "silu",
        num_local_experts: Optional[int] = 8,
        num_experts_per_tok: Optional[int] = 2,
        output_router_logits: Optional[bool] = False,
        aux_loss_coef: Optional[float] = 0.01,
        use_cache: Optional[bool] = True,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        tie_word_embeddings: Optional[bool] = True,
        rope_parameters: Optional[RopeParameters | dict[RopeParameters]] = None,
        rms_norm_eps: Optional[int] = 1e-6,
        initializer_range: Optional[float] = 0.01,
        attention_dropout: Optional[float] = 0.0,
        **kwargs,
    ):
        if num_experts_per_tok > num_local_experts:
            raise ValueError("`num_experts_per_tok` must be less than or equal to `num_local_experts`")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_key_value_heads * num_experts_per_tok
        self.num_key_value_heads = num_key_value_heads
        self.kv_channels = kv_channels
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.activation_function = activation_function
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.rms_norm_eps = rms_norm_eps
        # Try to set `rope_scaling` if available, otherwise use `rope_parameters`
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or rope_parameters

        # Validate the correctness of rotary position embeddings parameters
        rope_theta = kwargs.get("rope_theta", 10000.0)
        standardize_rope_params(self, rope_theta=rope_theta)
        rope_config_validation(self)

        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )


__all__ = ["JetMoeConfig"]
