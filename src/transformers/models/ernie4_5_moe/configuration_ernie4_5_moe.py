# Copyright (c) 2025 Baidu, Inc. and HuggingFace Inc. team. All Rights Reserved.
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
"""Ernie 4.5 MoE model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import logging


logger = logging.get_logger(__name__)


class Ernie4_5_MoeConfig(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ernie4_5_MoeModel`]. It is used to instantiate a
    Ernie 4.5 MoE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of [baidu/ERNIE-4.5-21B-A3B-PT](https://huggingface.co/baidu/ERNIE-4.5-21B-A3B-PT).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 103424):
            Vocabulary size of the Ernie 4.5 MoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Ernie4_5_MoeModel`]
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        hidden_size (`int`, *optional*, defaults to 2560):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 12288):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in any of the projections including mlp and attention for example.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Intermediate size of the routed expert.
        moe_k (`int`, *optional*, defaults to 6):
            Number of selected experts.
        moe_num_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        moe_num_shared_experts (`int`, *optional*, defaults to 2):
            The number of experts that are shared for all MoE forwards.
        moe_layer_start_index (`int`, *optional*, defaults to 1):
            The first index at which MoE layers start to appear.
        moe_layer_end_index (`int`, *optional*, defaults to -1):
            The last possible index for a MoE layer.
        moe_layer_interval (`int`, *optional*, defaults to 1):
            The intervals between MoE layers to appear.
        moe_norm_min (`float`, *optional*, defaults to 1e-12):
            Minimum division value during routing normalization.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether or not the router logits should be returned by the model. Enabling this will also
            allow the model to output the auxiliary loss, including load balancing loss and router z-loss.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.

    ```python
    >>> from transformers import Ernie4_5_MoeModel, Ernie4_5_MoEConfig

    >>> # Initializing a Ernie4_5_MoE style configuration
    >>> configuration = Ernie4_5_MoEConfig()

    >>> # Initializing a model from the ERNIE-4.5-21B-A3B style configuration
    >>> model = Ernie4_5_MoeModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "ernie4_5_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_experts": "moe_num_experts", "num_experts_per_tok": "moe_k"}
    default_theta = 500000.0

    # Default tensor parallel plan for base model `Ernie4_5_MoE`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 103424,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        hidden_size: int | None = 2560,
        intermediate_size: int | None = 12288,
        num_hidden_layers: int | None = 28,
        num_attention_heads: int | None = 20,
        num_key_value_heads: int | None = 4,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 131072,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        use_bias: int | None = False,
        moe_intermediate_size: int | None = 1536,
        moe_k: int | None = 6,
        moe_num_experts: int | None = 64,
        moe_num_shared_experts: int | None = 2,
        moe_layer_start_index: int | None = 1,
        moe_layer_end_index: int | None = -1,
        moe_layer_interval: int | None = 1,
        moe_norm_min: int | None = 1e-12,
        output_router_logits: bool | None = False,
        router_aux_loss_coef: float | None = 0.001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.use_bias = use_bias

        # MoE arguments
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_k = moe_k
        self.moe_num_experts = moe_num_experts
        self.moe_num_shared_experts = moe_num_shared_experts
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_layer_end_index = self.num_hidden_layers - 1 if moe_layer_end_index == -1 else moe_layer_end_index
        self.moe_layer_interval = moe_layer_interval
        self.moe_norm_min = moe_norm_min
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.rope_parameters = rope_parameters

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Ernie4_5_MoeConfig"]
