# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...configuration_utils import PreTrainedConfig
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from ...utils.generic import is_flash_attention_requested
from ...utils.output_capturing import OutputRecorder
from ..auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from ..deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4Model
from ..exaone4_5.modeling_exaone4_5 import Exaone4_5_Model
from ..glm5_next.configuration_glm5_next import Glm5NextConfig
from ..glm5_next.modeling_glm5_next import (
    Glm5NextExperts,
    Glm5NextForgetGate,
    Glm5NextLinearAttention,
    Glm5NextMLP,
    Glm5NextMoE,
    Glm5NextPreTrainedModel,
    Glm5NextRMSNorm,
    Glm5NextRMSNormGated,
    Glm5NextTopkRouter,
)
from ..glm46v.modeling_glm46v import Glm46VForConditionalGeneration
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
from ..llama.modeling_llama import eager_attention_forward
from ..mixtral.modeling_mixtral import load_balancing_loss_func


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="zai-org/GLM-5-Next-VL")
@strict
class Glm5NextVLTextConfig(Glm5NextConfig):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of routed expert groups.
    mlp_layer_types (`list[str]`, *optional*):
        Per-layer feed-forward schedule. Values are `"dense"` or `"sparse"`.
    layer_types (`list[str]`, *optional*):
        Per-layer attention cache schedule. Values are `"linear_attention"` for
        KDA layers and `"deepseek_sparse_attention"` for MLA layers.
    swiglu_limit (`float`, *optional*, defaults to 10.0):
        Clamp limit applied to SwiGLU gate/up projections.
    linear_head_dim (`int`, *optional*, defaults to 128):
        Dimension of each head in linear attention.
    linear_num_heads (`int`, *optional*, defaults to 64):
        Number of heads used in linear attention layers.
    linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
        Kernel size of the convolution used in linear attention layers.
    linear_lower_bound (`float`, *optional*, defaults to -5.0):
        Whether the forget gate has a lower bound to apply to the decay.
    hc_mult (`int`, *optional*, defaults to 4):
        Number of MHC residual streams.
    hc_eps (`float`, *optional*, defaults to 1e-6):
        Numerical floor used by MHC Sinkhorn normalization.
    hc_sinkhorn_iters (`int`, *optional*, defaults to 20):
        Number of Sinkhorn iterations used by MHC routing.
    """

    model_type = "glm5_next_vl_text"
    base_config_key = "text_config"

    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    head_dim: int = 0
    max_position_embeddings: int = 4096  # TODO: check this value on relase
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    moe_intermediate_size: int = 2048
    num_experts_per_tok: int = 8
    n_routed_experts: int = 288
    swiglu_limit: float | None = 10.0
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_lower_bound: float | None = -5.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    eos_token_id: int | list[int] | None = [154820, 154827, 154829]

    # TODO: add when we have an indexer trained
    index_head_dim = AttributeError()
    index_n_heads = AttributeError()
    index_topk = AttributeError()
    index_kpool = AttributeError()
    index_kpool_always_select_tail = AttributeError()
    indexer_types = AttributeError()

    # TODO: will rope be added?
    rope_parameters = AttributeError()

    # TODO: Let it be reinherited after indexer
    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (
                self.num_hidden_layers - 3
            )

        if self.layer_types is None:
            kda_layers = [idx for idx in range(self.num_hidden_layers) if idx % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "deepseek_sparse_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]

        # Convert dict to attributes (if given)
        linear_attn_dict = kwargs.pop("linear_attn_config", None)
        if linear_attn_dict is not None:
            self.linear_head_dim = linear_attn_dict.get("head_dim", self.linear_head_dim)
            self.linear_num_heads = linear_attn_dict.get("num_heads", self.linear_num_heads)
            self.linear_conv_kernel_dim = linear_attn_dict.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
            self.linear_lower_bound = linear_attn_dict.get("lower_bound", self.linear_lower_bound)

            # Additional lower bound logic as per original dict
            if linear_attn_dict.get("safe_gate", False) and self.linear_lower_bound is None:
                self.linear_lower_bound = -5.0

        # NOTE: this forces an intentional override as we have the convention of head_dim being the RoPE based dim
        kwargs.pop("head_dim", None)
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        PreTrainedConfig.__post_init__(self, **kwargs)

    # TODO: Readd along indexer
    def validate_architecture(self):
        PreTrainedConfig.validate_architecture(self)


@auto_docstring(checkpoint="zai-org/GLM-5-Next-VL")
@strict
class Glm5NextVLConfig(PreTrainedConfig):
    r"""
    image_token_id (`int`, *optional*, defaults to 154854):
        The image token index to encode the image prompt.
    video_token_id (`int`, *optional*, defaults to 154855):
        The video token index to encode the video prompt.
    image_start_token_id (`int`, *optional*, defaults to 154830):
        The image start token index to encode the start of image.
    image_end_token_id (`int`, *optional*, defaults to 154831):
        The image end token index to encode the end of image.
    video_start_token_id (`int`, *optional*, defaults to 154832):
        The video start token index to encode the start of video.
    video_end_token_id (`int`, *optional*, defaults to 154833):
        The video end token index to encode the end of video.

    ```python
    >>> from transformers import Glm5NextVLConfig

    >>> # Initializing a GLM-5-Next-VL style configuration
    >>> configuration = Glm5NextVLConfig()
    ```"""

    model_type = "glm5_next_vl"
    sub_configs = {"vision_config": AutoConfig, "text_config": Glm5NextVLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            # Flat (text-only) GLM-5-Next checkpoints store the text fields at the
            # top level; forward them so `text_config` is populated for BC.
            self.text_config = self.sub_configs["text_config"](**kwargs)

        super().__post_init__(**kwargs)


class Glm5NextVLTextRMSNorm(Glm5NextRMSNorm):
    pass


class Glm5NextVLTextMLP(Glm5NextMLP):
    pass


class Glm5NextVLTextExperts(Glm5NextExperts):
    pass


class Glm5NextVLTextTopkRouter(Glm5NextTopkRouter):
    pass


class Glm5NextVLTextMoE(Glm5NextMoE):
    def __init__(self, config: Glm5NextVLConfig):
        super().__init__(config)
        self.experts = Glm5NextVLTextExperts(config)
        self.gate = Glm5NextVLTextTopkRouter(config)
        self.shared_experts = Glm5NextVLTextMLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )


class Glm5NextVLTextHyperConnection(DeepseekV4HyperConnection):
    pass


class Glm5NextVLTextHyperHead(nn.Module):
    """Final GLM-5-Next HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


class Glm5NextVLTextForgetGate(Glm5NextForgetGate):
    pass


class Glm5NextVLTextRMSNormGated(Glm5NextRMSNormGated):
    pass


class Glm5NextVLTextLinearAttention(Glm5NextLinearAttention):
    def __init__(
        self,
        config: Glm5NextVLConfig,
        layer_idx: int,
    ):
        super().__init__(config, layer_idx)
        self.forget_gate = Glm5NextVLTextForgetGate(config)
        self.o_norm = Glm5NextVLTextRMSNormGated(self.head_dim, eps=self.layer_norm_epsilon)



# TODO: add indexer if trained
class Glm5NextVLTextAttention(DeepseekV2Attention):
    def __init__(self, config: Glm5NextVLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.q_a_layernorm = Glm5NextVLTextRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = Glm5NextVLTextRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        # LoRA based path is guaranteed based on the config validation
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(key_shape).transpose(1, 2)
        key_states, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # Flash attention head_dim padding
        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

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

        if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Glm5NextVLTextDecoderLayer(GlmMoeDsaDecoderLayer):
    def __init__(self, config: Glm5NextVLTextConfig, layer_idx: int):
        self.block_type = config.layer_types[layer_idx]

        super().__init__(config, layer_idx)
        self.self_attn = (
            Glm5NextVLTextLinearAttention(config, layer_idx)
            if self.block_type == "linear_attention"
            else Glm5NextVLTextAttention(config, layer_idx)
        )

        self.attn_hc = Glm5NextVLTextHyperConnection(config)
        self.ffn_hc = Glm5NextVLTextHyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, None]:
        dtype = hidden_states.dtype

        residual = hidden_states
        post, comb, hidden_states = self.attn_hc(hidden_states)
        # Self attn
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        residual = hidden_states
        post, comb, hidden_states = self.ffn_hc(hidden_states)
        # Feed forward
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = post.to(dtype).unsqueeze(-1) * hidden_states.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

        return hidden_states


@auto_docstring
class Glm5NextVLPreTrainedModel(Glm5NextPreTrainedModel):
    config: Glm5NextVLConfig
    _no_split_modules = ["Glm5NextVLTextDecoderLayer"]

    _can_record_outputs = {
        "attentions": Glm5NextVLTextAttention,
        "hidden_states": Glm5NextVLTextDecoderLayer,
        "router_logits": OutputRecorder(Glm5NextVLTextTopkRouter, index=0),  # noqa: F821
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Glm5NextVLTextForgetGate):
            # Following FLA initialization
            # NOTE: This is incredibly important so keep it this way at all costs
            if module.safe_gate_lower_bound is not None:
                init.zeros_(module.A_log)
            else:
                init.copy_(
                    module.A_log,
                    init.uniform_(module.A_log, a=1.0, b=16.0).log(),
                )

            init.uniform_(
                module.dt_bias,
                a=math.log(1e-3),
                b=math.log(1e-1),
            )
            dt = module.dt_bias.exp().clamp_min(1e-4)

            # (stable) inverse softplus
            init.copy_(
                module.dt_bias,
                dt + torch.log(-torch.expm1(-dt)),
            )
        elif isinstance(module, Glm5NextVLTextRMSNormGated):
            init.ones_(module.weight)
        elif isinstance(module, Glm5NextVLTextHyperConnection):
            init.normal_(module.fn, mean=0.0, std=0.02)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, Glm5NextVLTextExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Glm5NextVLTextTopkRouter):
            init.zeros_(module.e_score_correction_bias)
            init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)


@auto_docstring
class Glm5NextVLTextModel(DeepseekV4Model, Glm5NextVLPreTrainedModel):
    config: Glm5NextVLTextConfig

    def __init__(self, config):
        super().__init__(self, config)
        del self.rotary_emb

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)

        # TODO: masks change based on the indexer or not
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        # Key change: NoPE
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=position_ids,
                position_embeddings=None,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(self.hc_head(hidden_states))
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class Glm5NextVLModel(Exaone4_5_Model, Glm5NextVLPreTrainedModel):
    config: Glm5NextVLConfig
    _no_split_modules = AttributeError()

    def __init__(self, config):
        super().__init__(config)
        self.visual = AutoModel._from_config(config.vision_config)
        self.language_model = Glm5NextVLTextModel._from_config(config.text_config)
        del self.rope_deltas

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        # Same as in `Glm46V`
        # reshape video_grid_thw -> [b, 3] -> [1, h, w] * frames
        t = video_grid_thw[:, 0]
        hw = video_grid_thw[:, 1:]
        # repeat each (h,w) row `t` times
        flattened_hw = torch.repeat_interleave(hw, t, dim=0)
        prefix_ones = video_grid_thw.new_ones(flattened_hw.shape[0], 1)
        flattened_video_grid_thw = torch.cat([prefix_ones, flattened_hw], dim=1)

        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        vision_outputs = self.visual(pixel_values_videos, grid_thw=flattened_video_grid_thw, **kwargs)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output, split_sizes)
        return vision_outputs

    # TODO: `get_placeholder_mask` is broken for simultaneous img + vid
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            # GLM 5 Next VL special_video_mask is special_image_mask
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                n_image_tokens * inputs_embeds.shape[-1] == image_features.numel(),
                f"Image features and image tokens do not match, tokens: {n_image_tokens}, features: {image_features.shape[0]}",
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                n_video_tokens * inputs_embeds.shape[-1] == video_features.numel(),
                f"Video features and video tokens do not match, tokens: {n_video_tokens}, features: {video_features.shape[0]}",
            )
        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        super().forward(**super_kwargs)


class Glm5NextVLForConditionalGeneration(Glm46VForConditionalGeneration, Glm5NextVLPreTrainedModel):
    """
    Main Glm5NextVL conditional generation class.
    """

    def __init__(self, config):
        super().__init__(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:

        ```python
        >>> from transformers import AutoProcessor, Glm5NextVLForConditionalGeneration
        >>> import torch

        >>> model = Glm5NextVLForConditionalGeneration.from_pretrained("zai-org/GLM-5-Next-VL")
        >>> processor = AutoProcessor.from_pretrained("zai-org/GLM-5-Next-VL")

        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
        ...             {"type": "text", "text": "Describe the image."},
        ...         ],
        ...     }
        ... ]
        >>> inputs = processor.apply_chat_template(
        ...     messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ... )
        >>> inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        >>> generated_ids = model.generate(**inputs, max_new_tokens=64)
        ```
        """

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

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
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        # Force recomputation of 2D-RoPE and ignore rope_deltas
        model_inputs["position_ids"] = None

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        raise AttributeError()


__all__ = [
    "Glm5NextVLConfig",
    "Glm5NextVLTextConfig",
    "Glm5NextVLPreTrainedModel",
    "Glm5NextVLTextModel",
    "Glm5NextVLModel",
    "Glm5NextVLForConditionalGeneration",
]
