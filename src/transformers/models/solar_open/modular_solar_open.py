# Copyright 2026 Upstage and HuggingFace Inc. team.
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
"""PyTorch SolarOpen model."""

from ...modeling_rope_utils import RopeParameters
from ...utils import logging
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeForCausalLM,
    Glm4MoeModel,
    Glm4MoeMoE,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
)
from ..llama.modeling_llama import LlamaDecoderLayer


logger = logging.get_logger(__name__)


class SolarOpenConfig(Glm4MoeConfig):
    r"""
    This is the configuration class to store the configuration of a [`SolarOpenModel`]. It is used to instantiate a
    SolarOpen model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Instantiating a configuration defaults will yield a similar configuration to that of
    [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B) architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 196608):
            Vocabulary size of the SolarOpen model.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        moe_intermediate_size (`int`, *optional*, defaults to 1280):
            Intermediate size of the routed expert.
        num_hidden_layers (`int`, *optional*, defaults to 48):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key_value heads for Grouped Query Attention.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 128):
            Number of routed experts.
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 131072):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts per token.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for routed experts.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token.
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*):
            End of stream token id.
        pad_token_id (`int`, *optional*):
            Padding token id.
    """

    model_type = "solar_open"
    default_theta = 1_000_000.0

    # Default tensor parallel plan for base model `SolarOpenModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "local_rowwise",
        "layers.*.mlp.experts.down_proj": "local_rowwise",
        "layers.*.mlp.experts": "gather",
    }

    def __init__(
        self,
        vocab_size: int = 196608,
        hidden_size: int = 4096,
        moe_intermediate_size: int = 1280,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        n_shared_experts: int = 1,
        n_routed_experts: int = 128,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: int = 1e-5,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: RopeParameters | None = None,
        num_experts_per_tok: int = 8,
        routed_scaling_factor: float = 1.0,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        # Default partial_rotary_factor to 1.0 (instead of 0.5 in Glm4MoeConfig).
        # `setdefault` ensures this value is not overridden by subsequent calls.
        # This workaround is required due to modular inheritance limitations.
        kwargs.setdefault("partial_rotary_factor", 1.0)
        self.head_dim = head_dim

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            moe_hidden_size=moe_intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            n_shared_experts=n_shared_experts,
            n_routed_experts=n_routed_experts,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_parameters=rope_parameters,
            num_experts_per_tok=num_experts_per_tok,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        del self.intermediate_size
        del self.first_k_dense_replace
        del self.attention_bias
        del self.attention_dropout
        del self.use_qk_norm

    def convert_rope_params_to_dict(self, ignore_keys_at_rope_validation: set | None = None, **kwargs):
        default_rope_params = RopeParameters(
            rope_type="yarn",
            factor=2.0,
            original_max_position_embeddings=65_536,
        )

        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params

        # Standardize and validate the correctness of rotary position embeddings parameters
        self.rope_parameters.setdefault("rope_theta", kwargs.pop("rope_theta", self.default_theta))

        if "partial_rotary_factor" in kwargs:
            self.rope_parameters.setdefault("partial_rotary_factor", kwargs["partial_rotary_factor"])
            ignore_keys_at_rope_validation = {"partial_rotary_factor"}

        self.standardize_rope_params()
        self.validate_rope(ignore_keys=ignore_keys_at_rope_validation)
        return kwargs


class SolarOpenDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: SolarOpenConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.mlp = SolarOpenMoE(config)


class SolarOpenMoE(Glm4MoeMoE):
    pass


class SolarOpenAttention(Glm4MoeAttention):
    def __init__(self, config: SolarOpenConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.head_dim = config.head_dim


class SolarOpenRMSNorm(Glm4MoeRMSNorm):
    pass


class SolarOpenPreTrainedModel(Glm4MoePreTrainedModel):
    pass


class SolarOpenModel(Glm4MoeModel):
    _keys_to_ignore_on_load_unexpected = []


class SolarOpenForCausalLM(Glm4MoeForCausalLM):
    pass


__all__ = [
    "SolarOpenConfig",
    "SolarOpenPreTrainedModel",
    "SolarOpenModel",
    "SolarOpenForCausalLM",
]
