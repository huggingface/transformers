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
from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn as nn

from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_rope_utils import RopeParameters, RotaryEmbeddingConfigMixin
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import OutputRecorder, check_model_inputs
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3NaiveMoe
from ..glm4.modeling_glm4 import Glm4Attention
from ..glm4_moe.configuration_glm4_moe import Glm4MoeConfig
from ..glm4_moe.modeling_glm4_moe import (
    Glm4MoeDecoderLayer,
    Glm4MoeMLP,
    Glm4MoeMoE,
    Glm4MoePreTrainedModel,
    Glm4MoeRMSNorm,
    Glm4MoeTopkRouter,
    eager_attention_forward,
)
from ..glm4v.configuration_glm4v import Glm4vConfig, Glm4vVisionConfig
from ..glm4v.modeling_glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vTextModel,
    Glm4vTextRotaryEmbedding,
    Glm4vVisionModel,
    rotate_half,
)
from ..qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeCausalLMOutputWithPast,
    Qwen3VLMoeModelOutputWithPast,
    load_balancing_loss_func,
)


logger = logging.get_logger(__name__)


class Glm4vMoeVisionConfig(Glm4vVisionConfig):
    pass


class Glm4vMoeRMSNorm(Glm4MoeRMSNorm):
    pass


class Glm4vMoeTextConfig(Glm4MoeConfig, RotaryEmbeddingConfigMixin):
    r"""
    This is the configuration class to store the configuration of a [`Glm4vMoeModel`]. It is used to instantiate a
    GLM-4.5V model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-4.5V [zai-org/GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151424):
            Vocabulary size of the Glm4vMoe model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Glm4vMoeModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 10944):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 46):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 96):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 65536):
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
        rope_parameters (`RopeParameters`, *optional*):
            Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain
            a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE
            with longer `max_position_embeddings`.
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
        router_aux_loss_coef (`float`, *optional*, defaults to 0.0001):
            The aux loss factor for the loss.
    ```python
    >>> from transformers import Glm4vMoeTextModel, Glm4vMoeConfig

    >>> # Initializing a GLM-4.5V style configuration
    >>> configuration = Glm4vMoeConfig()

    >>> # Initializing a model from the GLM-4.5V style configuration
    >>> model = Glm4vMoeTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "glm4v_moe_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `Glm4vMoe`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
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
        vocab_size: Optional[int] = 151424,
        hidden_size: Optional[int] = 4096,
        intermediate_size: Optional[int] = 10944,
        num_hidden_layers: Optional[int] = 46,
        num_attention_heads: Optional[int] = 96,
        num_key_value_heads: Optional[int] = 8,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 65536,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[int] = 1e-5,
        use_cache: Optional[bool] = True,
        tie_word_embeddings: Optional[bool] = False,
        rope_parameters: Optional[RopeParameters | dict[str, RopeParameters]] = None,
        attention_bias: Optional[bool] = True,
        attention_dropout: Optional[float] = 0.0,
        moe_intermediate_size: Optional[int] = 1408,
        num_experts_per_tok: Optional[int] = 8,
        n_shared_experts: Optional[int] = 1,
        n_routed_experts: Optional[int] = 128,
        routed_scaling_factor: Optional[float] = 1.0,
        n_group: Optional[int] = 1,
        topk_group: Optional[int] = 1,
        first_k_dense_replace: Optional[int] = 1,
        norm_topk_prob: Optional[bool] = True,
        router_aux_loss_coef: Optional[float] = 0.0001,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC

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
        self.router_aux_loss_coef = router_aux_loss_coef
        PreTrainedConfig.__init__(
            self, tie_word_embeddings=tie_word_embeddings, ignore_keys_at_rope_validation={"mrope"}, **kwargs
        )


class Glm4vMoeConfig(Glm4vConfig):
    r"""
    This is the configuration class to store the configuration of a [`Glm4vMoeModel`]. It is used to instantiate a
    GLM-4.5V model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of
    GLM-4.5V [zai-org/GLM-4.5V](https://huggingface.co/zai-org/GLM-4.5V).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.


    Args:
        text_config (`Union[PreTrainedConfig, dict]`, *optional*, defaults to `Glm4vMoeTextConfig`):
            The config object or dictionary of the text backbone.
        vision_config (`Union[PreTrainedConfig, dict]`,  *optional*, defaults to `Glm4vMoeVisionConfig`):
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
    >>> from transformers import Glm4vMoeForConditionalGeneration, Glm4vMoeConfig

    >>> # Initializing a GLM-4.5V style configuration
    >>> configuration = Glm4vMoeConfig()

    >>> # Initializing a model from the GLM-4.5V style configuration
    >>> model = Glm4vMoeForConditionalGeneration(configuration)

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


class Glm4vMoeModelOutputWithPast(Qwen3VLMoeModelOutputWithPast):
    pass


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class Glm4vMoeTextRotaryEmbedding(Glm4vTextRotaryEmbedding):
    def __init__(self, config: Glm4vMoeTextConfig, device=None, layer_type=None):
        super().__init__(config, device=device, layer_type=layer_type)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[Glm4vMoeTextConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


class Glm4vMoeTextAttention(Glm4Attention):
    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.rope_parameters = config.rope_parameters

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(  # diff with Llama
            query_states, key_states, cos, sin, self.rope_parameters["mrope_section"]
        )

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

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
        return attn_output, attn_weights


class Glm4vMoeTextTopkRouter(Glm4MoeTopkRouter, nn.Module):
    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__(config)


class Glm4vMoeTextNaiveMoe(DeepseekV3NaiveMoe):
    pass


class Glm4vMoeTextMoE(Glm4MoeMoE):
    def __init__(self, config: Glm4vMoeTextConfig):
        super().__init__(config)
        self.config = config
        self.experts = Glm4vMoeTextNaiveMoe(config)
        self.gate = Glm4vMoeTextTopkRouter(config)
        self.shared_experts = Glm4vMoeTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )


class Glm4vMoeTextMLP(Glm4MoeMLP):
    pass


class Glm4vMoeTextDecoderLayer(Glm4MoeDecoderLayer):
    def __init__(self, config: Glm4vMoeTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)


class Glm4vMoePreTrainedModel(Glm4MoePreTrainedModel):
    config: Glm4vMoeConfig
    base_model_prefix = "model"
    input_modalities = ("text", "image", "video")
    _no_split_modules = ["Glm4vMoeTextDecoderLayer", "Glm4vMoeVisionBlock"]
    _skip_keys_device_placement = "past_key_values"

    _can_record_outputs = {
        "hidden_states": Glm4vMoeTextDecoderLayer,
        "attentions": Glm4vMoeTextAttention,
        "router_logits": OutputRecorder(nn.Linear, layer_name="mlp.gate", index=0),
    }


class Glm4vMoeCausalLMOutputWithPast(Qwen3VLMoeCausalLMOutputWithPast):
    pass


@auto_docstring
class Glm4vMoeVisionModel(Glm4vVisionModel):
    pass


@auto_docstring
class Glm4vMoeTextModel(Glm4vTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # NOTE: we need to pass text position ids for packing. Qwen2-VL uses 3D positions
        # where each dim indicates visual spatial positions for temporal/height/width grids.
        # There are two scenarios when FA2-like packed masking might be activated.
        # 1. User specifically passed packed `position_ids` and no attention mask.
        #    In this case we expect the useer to create correct position ids for all 3 grids
        #    and prepend text-only position ids to it. The final tensor will be [4, bs, seq-len]
        # 2. User runs forward with no attention mask and no position ids. In this case, position ids
        #    are prepared by the model (`get_rope_index`) as `[4, bs, seq-len]` tensor. Text-only positions are
        #    prepended by us when creating positions so that the mask is constructed correctly. NOTE: failing to pass
        #    text-only positions will cause incorrect mask construction, do not change `prepare_input_for_generation`
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            # If inputs are not packed (usual 3D positions), do not prepare mask from position_ids
            text_position_ids = None

        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        # Create the masks
        causal_mask = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Glm4vMoeForConditionalGeneration(Glm4vForConditionalGeneration):
    @auto_docstring
    @check_model_inputs()
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Glm4vMoeCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(
                    loss.device
                )  # make sure to reside in the same device

        return Glm4vMoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


__all__ = [
    "Glm4vMoeConfig",
    "Glm4vMoeTextConfig",
    "Glm4vMoeVisionConfig",
    "Glm4vMoeForConditionalGeneration",
    "Glm4vMoeModel",  # noqa: F822
    "Glm4vMoePreTrainedModel",
    "Glm4vMoeTextModel",
    "Glm4vMoeVisionModel",
]
