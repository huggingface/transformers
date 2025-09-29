# coding=utf-8
# Copyright 2025 Meituan and the HuggingFace Inc. team. All rights reserved.
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
import torch.nn.functional as F
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    DeepseekV3PreTrainedModel,
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3TopkRouter,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)


logger = logging.get_logger(__name__)


class LongcatFlashRMSNorm(DeepseekV3RMSNorm):
    pass


class LongcatFlashRotaryEmbedding(DeepseekV3RotaryEmbedding):
    pass


# TODO remap config key ffn_hidden_size -> intermediate_size
class LongcatFlashMLP(DeepseekV3MLP):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.intermediate_size = config.ffn_hidden_size if intermediate_size is None else intermediate_size


# TODO remap config key moe_topk -> num_experts_per_tok
class LongcatFlashTopkRouter(DeepseekV3TopkRouter):
    def __init__(self, config):
        super().__init__(config)
        del self.n_group
        del self.topk_group
        del self.weight
        del self.norm_topk_prob

        self.top_k = config.moe_topk
        self.n_routed_experts = config.n_routed_experts + (config.zero_expert_num or 0)
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        self.router_bias = getattr(config, "router_bias", False)
        self.classifier = nn.Linear(config.hidden_size, self.n_routed_experts, bias=self.router_bias)

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.classifier.weight.type(torch.float32))
        scores = router_logits.softmax(dim=-1)
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights


# remap config key expert_ffn_hidden_size -> moe_intermediate_size
class LongcatFlashMoE(DeepseekV3MoE):
    """
    A mixed expert module containing zero compute (identity) experts.
    """

    def __init__(self, config):
        self.intermediate_size = config.expert_ffn_hidden_size
        super().__init__(config)
        del self.gate
        del self.shared_experts

        self.experts = nn.ModuleList(
            [LongcatFlashMLP(config, intermediate_size=self.intermediate_size) for _ in range(config.n_routed_experts)]
            + [nn.Identity() for _ in range(config.zero_expert_num)]
        )

        self.router = LongcatFlashTopkRouter(config)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.router(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        return hidden_states


class LongcatFlashMLA(DeepseekV3Attention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.mla_scale_q_lora = (config.hidden_size / self.q_lora_rank) ** 0.5
        self.mla_scale_kv_lora = (config.hidden_size / self.kv_lora_rank) ** 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        # we always do a lora for queries as well
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(k_pass)

        # apply LoRA scaling
        q_pass = q_pass * self.mla_scale_q_lora
        q_rot = q_rot * self.mla_scale_q_lora
        k_pass = k_pass * self.mla_scale_kv_lora

        k_pass = self.kv_b_proj(k_pass).view(key_shape).transpose(1, 2)
        k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

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

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LongcatFlashDecoderLayer(GradientCheckpointingLayer):
    """
    LongCat decoder layer with dual-sublayer + shortcut MoE architecture.

    Each logical layer contains:
    - 2 attention sublayers (with layer indices: layer_idx*2, layer_idx*2+1)
    - 2 MLP sublayers
    - 1 shortcut MoE connection
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.mlp = LongcatFlashMoE(config)

        self.self_attn = nn.ModuleList([LongcatFlashMLA(config=config, layer_idx=layer_idx * 2 + i) for i in [0, 1]])
        self.mlps = nn.ModuleList([LongcatFlashMLP(config) for _ in [0, 1]])
        self.input_layernorm = nn.ModuleList(
            [LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in [0, 1]]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in [0, 1]]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm[0](hidden_states)

        hidden_states, _ = self.self_attn[0](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm[0](hidden_states)

        shortcut_mlp_output = self.mlp(hidden_states)
        hidden_states = self.mlps[0](hidden_states)
        hidden_states = residual + hidden_states

        # shortcut connection after second sublayer
        residual = hidden_states
        hidden_states = self.input_layernorm[1](hidden_states)

        hidden_states, _ = self.self_attn[1](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm[1](hidden_states)

        hidden_states = self.mlps[1](hidden_states)
        hidden_states = residual + hidden_states + shortcut_mlp_output

        return hidden_states


class LongcatFlashPreTrainedModel(DeepseekV3PreTrainedModel):
    _can_record_outputs = {
        "hidden_states": LongcatFlashDecoderLayer,
        "attentions": LongcatFlashMLA,
    }

    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, LongcatFlashTopkRouter):
            module.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


class LongcatFlashModel(DeepseekV3Model):
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LongcatFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        # Each layer above has 2 sublayers, config hack to have a correct cache (to avoid a checkpoint change)
        self.head_dim = config.head_dim  # For CI happiness (we didn't convert so head_dim is not directly used)

        self.config.num_hidden_layers = 2 * config.num_layers
        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LongcatFlashRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class LongcatFlashForCausalLM(DeepseekV3ForCausalLM):
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LongcatFlashModel(config)


__all__ = ["LongcatFlashPreTrainedModel", "LongcatFlashModel", "LongcatFlashForCausalLM"]
