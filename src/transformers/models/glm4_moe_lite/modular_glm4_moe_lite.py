# Copyright 2025 the HuggingFace Team. All rights reserved.
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


import torch.nn as nn

from ...configuration_utils import PreTrainedConfig, layer_type_validation
from ...modeling_rope_utils import RopeParameters, RotaryEmbeddingConfigMixin
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer,
    Glm4MoeForCausalLM,
    Glm4MoeMLP,
    Glm4MoeModel,
    Glm4MoeMoE,
    Glm4MoeNaiveMoe,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
    Glm4MoeRotaryEmbedding,
    Glm4MoeTopkRouter,
)


class Glm4MoeLiteConfig(PreTrainedConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`Glm4MoeLiteModel`]. It is used to instantiate an DeepSeek
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DeepSeek-V3.
    e.g. [bzantium/tiny-deepseek-v3](https://huggingface.co/bzantium/tiny-deepseek-v3)
    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 154880):
            Vocabulary size of the Deep model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4MoeLiteModel`]
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10240):
            Dimension of the MLP representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Dimension of the MoE representations.
        num_hidden_layers (`int`, *optional*, defaults to 47):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 20):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 20):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details, check out [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to
            `num_attention_heads`.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 64):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 1.8):
            Scaling factor or routed experts.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the LoRA matrices for key and value projections.
        q_lora_rank (`int`, *optional*, defaults to 768):
            Rank of the LoRA matrices for query projections.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of the query/key heads that use rotary position embeddings.
        v_head_dim (`int`, *optional*, defaults to 256):
            Dimension of the value heads.
        qk_nope_head_dim (`int`, *optional*, defaults to 192):
            Dimension of the query/key heads that don't use rotary position embeddings.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to 4):
            Number of selected experts, None means dense model.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 202752):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 0):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 1):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings.
        mlp_layer_types (`list`, *optional*):
            MLP (Moe vs Dense) pattern for each layer.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Glm4MoeLiteModel, Glm4MoeLiteConfig

    >>> # Initializing a Deepseek-V3 style configuration
    >>> configuration = Glm4MoeLiteConfig()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4_moe_lite"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 154880,
        hidden_size: int | None = 2048,
        intermediate_size: int | None = 10240,
        moe_intermediate_size: int | None = 1536,
        num_hidden_layers: int | None = 47,
        num_attention_heads: int | None = 20,
        num_key_value_heads: int | None = 20,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 64,
        routed_scaling_factor: float | None = 1.8,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 768,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 256,
        qk_nope_head_dim: int | None = 192,
        n_group: int | None = 1,
        topk_group: int | None = 1,
        num_experts_per_tok: int | None = 4,
        norm_topk_prob: bool | None = True,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 202752,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-5,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | None = 1,
        pretraining_tp: int | None = 1,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        rope_interleave: bool | None = True,
        mlp_layer_types=None,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

        # Default to MoE from the second layer and on
        self.mlp_layer_types = mlp_layer_types
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
        layer_type_validation(self.mlp_layer_types, self.num_hidden_layers, attention=False)

        self.moe_intermediate_size = moe_intermediate_size
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


class Glm4MoeLiteRotaryEmbedding(Glm4MoeRotaryEmbedding):
    pass


class Glm4MoeLiteAttention(DeepseekV3Attention):
    pass


class Glm4MoeLiteMLP(Glm4MoeMLP):
    pass


class Glm4MoeLiteTopkRouter(Glm4MoeTopkRouter):
    pass


class Glm4MoeLiteRMSNorm(Glm4MoeRMSNorm):
    pass


class Glm4MoeLiteNaiveMoe(Glm4MoeNaiveMoe):
    pass


class Glm4MoeLiteMoE(Glm4MoeMoE):
    pass


class Glm4MoeLiteDecoderLayer(Glm4MoeDecoderLayer, nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = Glm4MoeLiteAttention(config, layer_idx)

        if config.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = Glm4MoeLiteMoE(config)
        else:
            self.mlp = Glm4MoeLiteMLP(config)

        self.input_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeLiteRMSNorm(config.hidden_size, config.rms_norm_eps)


class Glm4MoeLitePreTrainedModel(Glm4MoePreTrainedModel):
    pass


class Glm4MoeLiteModel(Glm4MoeModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.47.*"]


class Glm4MoeLiteForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "Glm4MoeLiteConfig",
    "Glm4MoeLitePreTrainedModel",
    "Glm4MoeLiteModel",
    "Glm4MoeLiteForCausalLM",
]
