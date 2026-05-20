# Copyright 2026 Cohere Inc. HuggingFace Inc. team. All rights reserved.
#
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

from ... import initialization as init
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.output_capturing import OutputRecorder
from ..cohere2.modeling_cohere2 import (
    Cohere2Attention,
    Cohere2DecoderLayer,
    Cohere2ForCausalLM,
    Cohere2LayerNorm,
    Cohere2MLP,
    Cohere2Model,
    Cohere2PreTrainedModel,
    Cohere2RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from ..llama.modeling_llama import LlamaRMSNorm
from ..mixtral.modeling_mixtral import MixtralExperts
from .configuration_cohere2_moe import Cohere2MoeConfig


class Cohere2MoeRMSNorm(LlamaRMSNorm):
    pass


class Cohere2MoeLayerNorm(Cohere2LayerNorm):
    pass


class Cohere2MoeMLP(Cohere2MLP):
    def __init__(self, config: Cohere2MoeConfig, intermediate_size=None):
        super().__init__(config)
        if intermediate_size is not None:
            self.intermediate_size = intermediate_size
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class Cohere2MoeExperts(MixtralExperts):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts


class Cohere2MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.expert_selection_fn = config.expert_selection_fn
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = F.linear(hidden_states, self.weight)
        router_scores, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.expert_selection_fn == "softmax":
            router_scores = F.softmax(router_scores, dim=1, dtype=torch.float)
        elif self.expert_selection_fn == "sigmoid":
            router_scores = F.sigmoid(router_scores)
            if self.norm_topk_prob:
                router_scores = router_scores / torch.sum(router_scores, dim=-1, keepdims=True)
        else:
            raise ValueError("`expert_selection_fn` can only be `softmax` or `sigmoid`")

        # we cast back to the input dtype if it was upcasted in softmax
        router_scores = router_scores.to(hidden_states.dtype)

        return router_logits, router_scores, selected_experts


class Cohere2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_shared_experts = config.num_shared_experts
        self.shared_expert_combination_strategy = config.shared_expert_combination_strategy

        self.gate = Cohere2MoeTopKRouter(config)
        self.experts = Cohere2MoeExperts(config)
        if self.num_shared_experts > 0:
            self.shared_experts = Cohere2MoeMLP(
                config,
                intermediate_size=config.intermediate_size * config.num_shared_experts,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        _, router_scores, selected_experts = self.gate(hidden_states_flat)
        final_hidden_states = self.experts(hidden_states_flat, selected_experts, router_scores)

        if self.num_shared_experts > 0:
            shared_expert_output = self.shared_experts(hidden_states_flat)
            if self.shared_expert_combination_strategy == "sum":
                final_hidden_states = final_hidden_states + shared_expert_output
            elif self.shared_expert_combination_strategy == "average":
                final_hidden_states = (final_hidden_states + shared_expert_output) / 2
            else:
                raise ValueError("`shared_expert_combination_strategy` can only be `sum` or `average`")

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Cohere2MoeAttention(Cohere2Attention):
    def __init__(self, config: Cohere2MoeConfig, layer_idx: int | None = None):
        super().__init__(config, layer_idx)
        self.force_rope = (
            self.layer_idx < config.first_k_dense_replace and config.prefix_dense_sliding_window_pattern == 1
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.sliding_window is not None or self.force_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Cohere2MoeDecoderLayer(Cohere2DecoderLayer):
    def __init__(self, config: Cohere2MoeConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = (
            Cohere2MoeRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            if config.rms_norm_eps is not None
            else Cohere2MoeLayerNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        )
        self.mlp = (
            Cohere2MoeMLP(config, config.prefix_dense_intermediate_size)
            if layer_idx < config.first_k_dense_replace
            else Cohere2MoeSparseMoeBlock(config)
        )


class Cohere2MoeRotaryEmbedding(Cohere2RotaryEmbedding):
    pass


class Cohere2MoePreTrainedModel(Cohere2PreTrainedModel):
    _can_record_outputs = {
        "hidden_states": Cohere2MoeDecoderLayer,
        "attentions": Cohere2MoeAttention,
        "router_logits": OutputRecorder(Cohere2MoeTopKRouter, index=0),
    }

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = self.config.initializer_range
        if isinstance(module, Cohere2MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, Cohere2MoeTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)


class Cohere2MoeModel(Cohere2Model):
    def __init__(self, config: Cohere2MoeConfig):
        super().__init__()
        self.norm = (
            Cohere2MoeRMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
            if config.rms_norm_eps is not None
            else Cohere2MoeLayerNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        )

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(  # only diff with Cohere2 is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Cohere2MoeForCausalLM(Cohere2ForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits * self.logit_scale

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return MoeCausalLMOutputWithPast(  # only diff with Cohere2 is the output type, we need MoE
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["Cohere2MoeForCausalLM", "Cohere2MoeModel", "Cohere2MoePreTrainedModel"]
