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

import torch
import torch.nn.functional as F
from torch import nn

from ... import initialization as init
from ...masking_utils import create_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, logging
from ...utils.import_utils import is_causal_conv1d_available
from ..lfm2.modeling_lfm2 import (
    Lfm2Attention,
    Lfm2DecoderLayer,
    Lfm2HybridConvCache,
    Lfm2MLP,
    Lfm2RotaryEmbedding,
    Lfm2ShortConv,
)
from ..llama.modeling_llama import LlamaForCausalLM, LlamaPreTrainedModel, LlamaRMSNorm
from ..mixtral.modeling_mixtral import MixtralModel
from ..qwen2_moe.modeling_qwen2_moe import Qwen2MoeExperts
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


class Lfm2MoeRotaryEmbedding(Lfm2RotaryEmbedding):
    pass


class Lfm2MoeMLP(Lfm2MLP):
    def __init__(self, config: Lfm2MoeConfig, intermediate_size: int | None = None):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)


class Lfm2MoeExperts(Qwen2MoeExperts):
    def __init__(self, config):
        super().__init__(config)
        self.act_fn = F.silu


class Lfm2MoeSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.use_expert_bias = config.use_expert_bias

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Lfm2MoeExperts(config)
        if self.use_expert_bias:
            self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))

    def route_tokens_to_experts(self, router_logits):
        routing_weights = router_logits.sigmoid()
        if self.use_expert_bias:
            scores_for_routing = routing_weights + self.expert_bias
            _, selected_experts = torch.topk(scores_for_routing, k=self.top_k, dim=-1)
            routing_weights = torch.gather(routing_weights, dim=1, index=selected_experts).type_as(router_logits)
        else:
            routing_weights, selected_experts = torch.topk(routing_weights, k=self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-6)
        routing_weights = routing_weights * self.routed_scaling_factor
        return selected_experts, routing_weights

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_reshaped)
        selected_experts, routing_weights = self.route_tokens_to_experts(router_logits)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Lfm2MoeHybridConvCache(Lfm2HybridConvCache):
    pass


class Lfm2MoeAttention(Lfm2Attention):
    pass


class Lfm2MoeShortConv(Lfm2ShortConv):
    pass


class Lfm2MoeDecoderLayer(Lfm2DecoderLayer):
    def __init__(self, config: Lfm2MoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.feed_forward = (
            Lfm2MoeMLP(config, intermediate_size=config.intermediate_size)
            if layer_idx < config.num_dense_layers
            else Lfm2MoeSparseMoeBlock(config)
        )


class Lfm2MoePreTrainedModel(LlamaPreTrainedModel):
    _can_compile_fullgraph = False  # uses a non-compilable custom cache class Lfm2MoeHybridConvCache

    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        if isinstance(module, Lfm2MoeExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Lfm2MoeSparseMoeBlock):
            if module.use_expert_bias:
                init.zeros_(module.expert_bias)


class Lfm2MoeModel(MixtralModel):
    def __init__(self, config: Lfm2MoeConfig):
        super().__init__(config)
        self.pos_emb = Lfm2MoeRotaryEmbedding(config)
        self.embedding_norm = Lfm2MoeRMSNorm(config.hidden_size, eps=config.norm_eps)
        del self.norm
        del self.rotary_emb

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Lfm2MoeHybridConvCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
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
        # Skip masking for decoding stage. We check shape here to be compile-friendly
        linear_attention = attention_mask if inputs_embeds.shape[1] != 1 else None

        hidden_states = inputs_embeds
        position_embeddings = self.pos_emb(hidden_states, position_ids=position_ids)

        # decoder layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_mask = causal_mask if decoder_layer.is_attention_layer else linear_attention
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
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


class Lfm2MoeForCausalLM(LlamaForCausalLM):
    pass


__all__ = ["Lfm2MoeForCausalLM", "Lfm2MoeModel", "Lfm2MoePreTrainedModel"]
