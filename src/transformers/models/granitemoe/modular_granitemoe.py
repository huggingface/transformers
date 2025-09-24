# coding=utf-8
# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
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
from typing import Optional

import torch
from torch import nn
from typing_extensions import Unpack

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_outputs import MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, TransformersKwargs
from ...utils.generic import check_model_inputs
from ..granite.modeling_granite import GraniteRMSNorm, GraniteRotaryEmbedding
from ..jetmoe.modeling_jetmoe import JetMoeParallelExperts, JetMoeTopKGating
from ..llama.modeling_llama import LlamaAttention
from ..mixtral.modeling_mixtral import MixtralDecoderLayer, MixtralForCausalLM
from .configuration_granitemoe import GraniteMoeConfig


class GraniteMoeRMSNorm(GraniteRMSNorm):
    pass


class GraniteMoeRotaryEmbedding(GraniteRotaryEmbedding):
    pass


class GraniteMoeParallelExperts(JetMoeParallelExperts):
    pass


class GraniteMoeTopKGating(JetMoeTopKGating):
    pass


class GraniteMoeMoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        config:
            Configuration object with model hyperparameters.
    """

    def __init__(self, config: GraniteMoeConfig):
        super().__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.intermediate_size
        self.activation = ACT2FN[config.hidden_act]
        self.input_linear = GraniteMoeParallelExperts(config.num_local_experts, self.input_size, self.hidden_size * 2)
        self.output_linear = GraniteMoeParallelExperts(config.num_local_experts, self.hidden_size, self.input_size)

        self.router = GraniteMoeTopKGating(
            input_size=self.input_size,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
        )

    def forward(self, layer_input):
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)
        _, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)

        expert_inputs = layer_input[batch_index]
        hidden_states = self.input_linear(expert_inputs, expert_size)
        chunked_hidden_states = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
        expert_outputs = self.output_linear(hidden_states, expert_size)

        expert_outputs = expert_outputs * batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        layer_output = zeros.index_add(0, batch_index, expert_outputs)
        layer_output = layer_output.view(bsz, length, self.input_size)
        return layer_output, router_logits


class GraniteMoeAttention(LlamaAttention):
    pass


class GraniteMoeDecoderLayer(MixtralDecoderLayer):
    def __init__(self, config: GraniteMoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GraniteMoeAttention(config=config, layer_idx=layer_idx)
        self.block_sparse_moe = GraniteMoeMoE(config)
        self.input_layernorm = GraniteMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GraniteMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.residual_multiplier = config.residual_multiplier  # Only diff with mixtral!

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states * self.residual_multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


@auto_docstring
class GraniteMoePreTrainedModel(PreTrainedModel):
    config: GraniteMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GraniteMoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GraniteMoeParallelExperts):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)


@auto_docstring
class GraniteMoeModel(GraniteMoePreTrainedModel):
    def __init__(self, config: GraniteMoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GraniteMoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GraniteMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.embedding_multiplier = config.embedding_multiplier  # only diff with Mixtral
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.position_embedding_type = config.position_embedding_type
        self.rotary_emb = GraniteMoeRotaryEmbedding(config) if self.position_embedding_type == "rope" else None

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GraniteMoeForCausalLM(MixtralForCausalLM):
    pass


__all__ = ["GraniteMoeForCausalLM", "GraniteMoeModel", "GraniteMoePreTrainedModel"]
