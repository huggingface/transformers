# coding=utf-8
# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_rope_utils import rope_config_validation
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeAttention,
    Glm4MoeDecoderLayer,
    Glm4MoeMLP,
    Glm4MoeMoE,
    Glm4MoeRMSNorm,
    Glm4MoeTopkRouter,
    eager_attention_forward,
)
from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vVisionConfig
from ..glm4v.modeling_glm4v import (
    Glm4vCausalLMOutputWithPast,
    Glm4vForConditionalGeneration,
    Glm4VisionMlp,
    Glm4vModel,
    Glm4vModelOutputWithPast,
    Glm4vPreTrainedModel,
    Glm4vTextModel,
    Glm4vTextRotaryEmbedding,
    Glm4vVisionAttention,
    Glm4vVisionBlock,
    Glm4vVisionEmbeddings,
    Glm4vVisionModel,
    Glm4vVisionPatchEmbed,
    Glm4vVisionPatchMerger,
    Glm4vVisionRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)


logger = logging.get_logger(__name__)


class Glm4v_moeVisionConfig(Glm4vVisionConfig):
    pass


class Glm4v_moeTextConfig(Glm4MoeConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4v_moeModel`]. It is used to instantiate a
    GLM-4.5V model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-4.5V [THUDM/GLM-4.5V](https://huggingface.co/THUDM/GLM-4.5V).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151424):
            Vocabulary size of the Glm4v_moe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4v_moeModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10944):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 46):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 96):
            Number of attention heads for each attention layer in the Transformer encoder.
        partial_rotary_factor (`float`, *optional*, defaults to 0.5): The factor of the partial rotary position.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
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
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 1000000.0):
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
        attention_bias (`bool`, defaults to `True`, *optional*, defaults to `True`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        moe_intermediate_size (`int`, *optional*, defaults to 1408):
            Intermediate size of the routed expert.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            number of experts per token.
        n_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        n_routed_experts (`int`, *optional*, defaults to 128):
            Number of routed experts.
        routed_scaling_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor or routed experts.
        n_group (`int`, *optional*, defaults to 1):
            Number of groups for routed experts.
        topk_group (`int`, *optional*, defaults to 1):
            Number of selected groups for each token(for each token, ensuring the selected experts is only within `topk_group` groups).
        first_k_dense_replace (`int`, *optional*, defaults to 1):
            Number of dense layers in shallow layers(embed->dense->dense->...->dense->moe->moe...->lm_head).
                                                                    \--k dense layers--/
        norm_topk_prob (`bool`, *optional*, defaults to `True`):
            Whether to normalize the topk probabilities.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether to use query-key normalization in the attention.

    ```python
    >>> from transformers import Glm4v_moeTextModel, Glm4v_moeConfig

    >>> # Initializing a GLM-4.5V style configuration
    >>> configuration = Glm4v_moeConfig()

    >>> # Initializing a model from the GLM-4.5V style configuration
    >>> model = Glm4v_moeTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4v_moe_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Glm4v_moe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_rep",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_rep",  # we need to replicate here due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151424,
        hidden_size=4096,
        intermediate_size=10944,
        num_hidden_layers=46,
        num_attention_heads=96,
        partial_rotary_factor=0.5,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=True,
        attention_dropout=0.0,
        moe_intermediate_size=1408,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_routed_experts=128,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        use_qk_norm=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.partial_rotary_factor = partial_rotary_factor

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self, ignore_keys={"mrope_section"})

        # MoE arguments
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Glm4v_moeConfig(Glm4vConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4v_moeModel`]. It is used to instantiate a
    GLM-4.5V model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-4.5V [zai_org/GLM-4.5V](https://huggingface.co/zai_org/GLM-4.5V).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Glm4v_moeTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Glm4v_moeVisionConfig`):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151363):
            The image token index to encode the image prompt.
        video_token_id (`int`, *optional*, defaults to 151364):
            The video token index to encode the image prompt.
        image_start_token_id (`int`, *optional*, defaults to 151339):
            The image start token index to encode the start of image.
        image_end_token_id (`int`, *optional*, defaults to 151340):
            The image end token index to encode the end of image.
        video_start_token_id (`int`, *optional*, defaults to 151341):
            The video start token index to encode the start of video.
        video_end_token_id (`int`, *optional*, defaults to 151342):
            The video end token index to encode the end of video.

    ```python
    >>> from transformers import Glm4v_moeForConditionalGeneration, Glm4v_moeConfig

    >>> # Initializing a GLM-4.5V style configuration
    >>> configuration = Glm4v_moeConfig()

    >>> # Initializing a model from the GLM-4.5V style configuration
    >>> model = Glm4v_moeForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=151363,
        video_token_id=151364,
        image_start_token_id=151339,
        image_end_token_id=151340,
        video_start_token_id=151341,
        video_end_token_id=151342,
        **kwargs,
    ):
        super().__init__()


class Glm4v_moeRMSNorm(Glm4MoeRMSNorm):
    pass


class Glm4_moeVisionMlp(Glm4VisionMlp):
    pass


class Glm4v_moeVisionPatchEmbed(Glm4vVisionPatchEmbed):
    pass


class Glm4v_moeVisionRotaryEmbedding(Glm4vVisionRotaryEmbedding):
    pass


class Glm4v_moeVisionPatchMerger(Glm4vVisionPatchMerger):
    pass


class Glm4v_moeVisionEmbeddings(Glm4vVisionEmbeddings):
    pass


class Glm4v_moeVisionAttention(Glm4vVisionAttention):
    pass


class Glm4v_moeVisionBlock(Glm4vVisionBlock):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = Glm4v_moeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = Glm4v_moeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Glm4v_moeVisionAttention(config)
        self.mlp = Glm4_moeVisionMlp(config, bias=False)


class Glm4v_moePreTrainedModel(Glm4vPreTrainedModel):
    pass


class Glm4v_moeVisionModel(Glm4vVisionModel):
    pass


class Glm4v_moeTextRotaryEmbedding(Glm4vTextRotaryEmbedding):
    pass


def rotate_half_llm(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Glm4v_moeTextAttention(Glm4MoeAttention):
    def __init__(self, config: Glm4v_moeTextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        if self.use_qk_norm:  # main diff from Llama
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(  # diff with Llama
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class Glm4v_moeTextTopkRouter(Glm4MoeTopkRouter, nn.Module):
    def __init__(self, config: Glm4v_moeTextConfig):
        super().__init__(config)


class Glm4v_moeTextMoE(Glm4MoeMoE):
    def __init__(self, config: Glm4v_moeTextConfig):
        super().__init__(config)
        self.config = config
        self.experts = nn.ModuleList(
            [
                Glm4v_moeTextMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = Glm4v_moeTextTopkRouter(config)
        self.shared_experts = Glm4v_moeTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )


class Glm4v_moeTextMLP(Glm4MoeMLP):
    pass


class Glm4v_moeTextDecoderLayer(Glm4MoeDecoderLayer, nn.Module):
    def __init__(self, config: Glm4v_moeTextConfig, layer_idx: int):
        nn.Module.__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Glm4v_moeTextAttention(config=config, layer_idx=layer_idx)

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = Glm4v_moeTextMoE(config)
        else:
            self.mlp = Glm4v_moeTextMLP(config)

        self.input_layernorm = Glm4v_moeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Glm4v_moeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Glm4v_moeModelOutputWithPast(Glm4vModelOutputWithPast):
    pass


class Glm4v_moeTextModel(Glm4vTextModel):
    pass


class Glm4v_moeModel(Glm4vModel):
    pass


class Glm4v_moeCausalLMOutputWithPast(Glm4vCausalLMOutputWithPast):
    pass


class Glm4v_moeForConditionalGeneration(Glm4vForConditionalGeneration):
    pass


__all__ = [
    "Glm4v_moeConfig",
    "Glm4v_moeTextConfig",
    "Glm4v_moeForConditionalGeneration",
    "Glm4v_moeModel",
    "Glm4v_moePreTrainedModel",
    "Glm4v_moeTextModel",
]
