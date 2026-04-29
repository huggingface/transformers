# Copyright 2025 SK Telecom and the HuggingFace Inc. team. All rights reserved.
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
"""A.X-K2 model configuration"""

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


AXK2_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


@auto_docstring(checkpoint="skt/A.X-K2")
class AXK2Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`AXK2Model`]. It is used to instantiate an A.X-K2
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the A.X-K2.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 163840):
            Vocabulary size of the A.X-K2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`AXK2Model`].
        hidden_size (`int`, *optional*, defaults to 7168):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 18432):
            Dimension of the MLP representations for dense layers.
        moe_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimension of the MLP representations for each MoE expert.
        num_hidden_layers (`int`, *optional*, defaults to 61):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*, defaults to 64):
            Number of key-value heads for Grouped Query Attention. If `num_key_value_heads=num_attention_heads`,
            the model uses Multi Head Attention (MHA). If `num_key_value_heads=1`, it uses Multi Query
            Attention (MQA); otherwise GQA is used.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts. `None` means dense model.
        n_routed_experts (`int`, *optional*, defaults to 256):
            Number of routed experts. `None` means dense model.
        routed_scaling_factor (`float`, *optional*, defaults to 2.5):
            Scaling factor for routed expert outputs.
        kv_lora_rank (`int`, *optional*, defaults to 512):
            Rank of the low-rank key-value joint compression in Multi-head Latent Attention (MLA).
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Rank of the low-rank query compression in Multi-head Latent Attention (MLA).
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Dimension of each attention head for the rotary part of query and key.
        v_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head for value.
        qk_nope_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head for the non-rotary part of query and key.
        n_group (`int`, *optional*, defaults to 8):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 4):
            Number of selected groups for each token (ensuring selected experts are only within
            `topk_group` groups).
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts per token. `None` means dense model.
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of dense layers in shallow layers (embed->dense->moe->moe...->lm_head).
        moe_layer_freq (`int`, *optional*, defaults to 1):
            The frequency of the MoE layer: one expert layer for every `moe_layer_freq - 1` dense layers.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the weights of the routed experts.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions. Only relevant if
            `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 163691):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 163691):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings.
        rope_parameters ([`RopeParameters`], *optional*):
            Configuration for Rotary Position Embeddings (RoPE).
        rope_interleave (`bool`, *optional*, defaults to `True`):
            Whether to interleave the rotary position embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import AXK2Model, AXK2Config

    >>> # Initializing a A.X-K1 style configuration
    >>> configuration = AXK2Config()

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "AXK2"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
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
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int | None = 163840,
        hidden_size: int | None = 7168,
        intermediate_size: int | None = 18432,
        moe_intermediate_size: int | None = 2048,
        num_hidden_layers: int | None = 61,
        num_attention_heads: int | None = 64,
        num_key_value_heads: int | None = 64,
        n_shared_experts: int | None = 1,
        n_routed_experts: int | None = 256,
        routed_scaling_factor: float | None = 2.5,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        qk_nope_head_dim: int | None = 128,
        n_group: int | None = 8,
        topk_group: int | None = 4,
        num_experts_per_tok: int | None = 8,
        first_k_dense_replace: int | None = 1,
        moe_layer_freq: int | None = 1,
        norm_topk_prob: bool | None = True,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 4096,
        initializer_range: float | None = 0.02,
        rms_norm_eps: int | None = 1e-6,
        use_cache: bool | None = True,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 163691,
        eos_token_id: int | None = 163691,
        pretraining_tp: int | None = 1,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        rope_interleave: bool | None = True,
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings

        # Multi-head Latent Attention (MLA)
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = qk_rope_head_dim

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        # Mixture of Experts (MoE)
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.norm_topk_prob = norm_topk_prob

        # RoPE
        self.rope_parameters = rope_parameters
        self.rope_interleave = rope_interleave

        # General
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation: set | None = None, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))
        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)

        for key in ["beta_fast", "beta_slow", "factor"]:
            if key in self.rope_parameters:
                self.rope_parameters[key] = float(self.rope_parameters[key])
        return kwargs


__all__ = ["AXK2Config"]
