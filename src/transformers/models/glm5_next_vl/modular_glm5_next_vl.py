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

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_recurrent_attention_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ...utils.generic import is_flash_attention_requested
from ...utils.output_capturing import OutputRecorder
from ..auto.modeling_auto import AutoModel
from ..deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention
from ..deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection, DeepseekV4Model
from ..exaone4_5.modeling_exaone4_5 import Exaone4_5_ForConditionalGeneration, Exaone4_5_Model
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
from ..glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer
from ..llama.modeling_llama import eager_attention_forward
from .configuration_glm5_next_vl import Glm5NextVLConfig, Glm5NextVLTextConfig


logger = logging.get_logger(__name__)


# =============================================================================
# RMSNorm(Gated), MLP, MoE, RoPE
# =============================================================================


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


# =============================================================================
# MHC (Manifold-Constrained Hyper-Connection) helpers
# =============================================================================


class Glm5NextVLTextHyperConnection(DeepseekV4HyperConnection):
    pass


class Glm5NextVLTextHyperHead(nn.Module):
    """Final GLM-5-Next HC-stream collapse. Unlike DeepSeek-V4, this is an unweighted mean."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)


# =============================================================================
# KDA Linear Attention
# =============================================================================


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


# =============================================================================
# MLA (Multi-head Latent Attention) with optional DSA indexer scaffold
# =============================================================================


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


# =============================================================================
# Decoder Layer
# =============================================================================


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


# =============================================================================
# PreTrainedModel, Model, CausalLM
# =============================================================================


@auto_docstring
class Glm5NextVLPreTrainedModel(Glm5NextPreTrainedModel):
    config: Glm5NextVLConfig
    _no_split_modules = ["Glm5NextTextDecoderLayer"]

    _can_record_outputs = {
        "attentions": Glm5NextVLTextAttention,
        "hidden_states": Glm5NextVLTextDecoderLayer,
        "router_logits": OutputRecorder(Glm5NextVLTextTopkRouter, index=0),  # noqa: F821
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)

        if isinstance(module, Glm5NextVLTextForgetGate):
            nn.init.normal_(module.A_log, mean=0.0, std=0.02)
            nn.init.zeros_(module.dt_bias)
        elif isinstance(module, Glm5NextVLTextLinearAttention):
            nn.init.ones_(module.o_norm.weight)
        elif isinstance(module, Glm5NextVLTextHyperConnection):
            nn.init.normal_(module.fn, mean=0.0, std=0.02)
            nn.init.zeros_(module.base)
            nn.init.ones_(module.scale)
        elif isinstance(module, Glm5NextVLTextExperts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


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


# =============================================================================
# Composite VLM (vision tower + text model)
# =============================================================================


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
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        # TODO: let the processor return a flattened grid instead --> fully inherit here
        t = video_grid_thw[:, 0]
        hw = video_grid_thw[:, 1:]
        flattened_hw = torch.repeat_interleave(hw, t, dim=0)
        prefix_ones = video_grid_thw.new_ones(flattened_hw.shape[0], 1)
        flattened_video_grid_thw = torch.cat([prefix_ones, flattened_hw], dim=1)
        vision_outputs = self.visual(pixel_values_videos, grid_thw=flattened_video_grid_thw, **kwargs)
        split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        vision_outputs.pooler_output = torch.split(vision_outputs.pooler_output, split_sizes)
        return vision_outputs

    @can_return_tuple
    @auto_docstring
    def forward(self, **super_kwargs):
        super().forward(**super_kwargs)


class Glm5NextVLForConditionalGeneration(Exaone4_5_ForConditionalGeneration, Glm5NextVLPreTrainedModel):
    """
    Main Glm5NextVL conditional generation class.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model = Glm5NextVLModel(config)


__all__ = [
    "Glm5NextVLPreTrainedModel",
    "Glm5NextVLTextModel",
    "Glm5NextVLModel",
    "Glm5NextVLForConditionalGeneration",
]
