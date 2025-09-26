# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import MoeModelOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.deprecation import deprecate_kwarg
from ...utils.import_utils import is_causal_conv1d_available
from ..lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2HybridConvCache,
    Lfm2ShortConv,
    Lfm2DecoderLayer,
)
from ..llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..mixtral.modeling_mixtral import MixtralForCausalLM, MixtralModel
from .configuration_lfm2_moe import Lfm2MoeConfig


if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_fn, causal_conv1d_update = None, None


kernel_modules = (causal_conv1d_fn, causal_conv1d_update)
is_fast_path_available = all(kernel_modules)


logger = logging.get_logger(__name__)


class Lfm2MoeRMSNorm(LlamaRMSNorm):
    pass


class Lfm2MoeRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class Lfm2MoeMLP(nn.Module):
    def __init__(
        self,
        config: Lfm2MoeConfig,
        intermediate_size: int = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob

        if config.use_expert_bias:
            self.register_buffer("expert_bias", torch.zeros(self.num_experts, dtype=torch.float32))
        else:
            self.register_buffer("expert_bias", None)

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Lfm2MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = router_logits.sigmoid()

        if self.expert_bias is not None:
            scores_for_routing = routing_weights + self.expert_bias
            _, selected_experts = torch.topk(scores_for_routing, k=self.top_k, dim=-1)
            routing_weights = torch.gather(routing_weights, dim=1, index=selected_experts).type_as(router_logits)
        else:
            routing_weights, selected_experts = torch.topk(routing_weights, k=self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)

        if self.routed_scaling_factor:
            routing_weights = routing_weights * self.routed_scaling_factor

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.to(hidden_states.dtype).reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class Lfm2MoeHybridConvCache(Lfm2HybridConvCache):
    pass


class Lfm2MoeAttention(Lfm2Attention):
    pass


class Lfm2MoeShortConv(Lfm2ShortConv):
    pass


class Lfm2MoeDecoderLayer(Lfm2DecoderLayer):
    def __init__(self, config: Lfm2MoeConfig, layer_idx: int):
        super().__init__()
        self.feed_forward = Lfm2MoeMLP(
            config, intermediate_size=config.intermediate_size
        ) if layer_idx < config.num_dense_layers else Lfm2MoeSparseMoeBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.is_attention_layer:
            hidden_states, _ = self.self_attn(
                hidden_states=self.operator_norm(hidden_states),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        else:
            hidden_states = self.conv(
                hidden_states=self.operator_norm(hidden_states),
                past_key_values=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )

        hidden_states = hidden_states + residual
        ff_out = self.feed_forward(self.ffn_norm(hidden_states))
        if isinstance(ff_out, tuple):
            ff_out, _ = ff_out

        hidden_states = hidden_states + ff_out

        return hidden_states


class Lfm2MoePreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = False


class Lfm2MoeModel(MixtralModel):
    def __init__(self, config: Lfm2MoeConfig):
        super().__init__(config)
        self.pos_emb = Lfm2MoeRotaryEmbedding(config)
        self.embedding_norm = Lfm2MoeRMSNorm(config.hidden_size, eps=config.norm_eps)
        del self.norm
        del self.rotary_emb

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Lfm2MoeHybridConvCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            batch_size = inputs_embeds.shape[0]
            past_key_values = Lfm2MoeHybridConvCache(
                config=self.config, max_batch_size=batch_size, dtype=self.dtype, device=self.device
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
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
        position_embeddings = self.pos_emb(hidden_states, position_ids)

        # decoder layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.embedding_norm(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Lfm2MoeForCausalLM(MixtralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Lfm2MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok


__all__ = ["Lfm2MoeForCausalLM", "Lfm2MoeModel", "Lfm2MoePreTrainedModel"]
