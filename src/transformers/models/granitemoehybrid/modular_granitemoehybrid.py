# coding=utf-8
# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
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
from typing import Optional, Union

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import check_model_inputs
from ..bamba.configuration_bamba import BambaConfig
from ..bamba.modeling_bamba import BambaMixer, BambaRMSNormGated, HybridMambaAttentionDynamicCache
from ..gemma2.modeling_gemma2 import Gemma2RotaryEmbedding
from ..granitemoeshared.modeling_granitemoeshared import (
    GraniteFlashAttentionKwargs,
    GraniteMoeSharedAttention,
    GraniteMoeSharedDecoderLayer,
    GraniteMoeSharedForCausalLM,
    GraniteMoeSharedMLP,
    GraniteMoeSharedModel,
    GraniteMoeSharedMoE,
    GraniteMoeSharedPreTrainedModel,
    eager_attention_forward,
)
from .configuration_granitemoehybrid import GraniteMoeHybridConfig


logger = logging.get_logger(__name__)


class GraniteMoeHybridAttention(GraniteMoeSharedAttention):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)

    def forward(  # FIME: @ARTHUR this forward is also classic: attention nope
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
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


class GraniteMoeHybridMambaLayer(BambaMixer):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int):
        super().__init__(BambaConfig(config), layer_idx)


class GraniteMoeHybridRMSNormGated(BambaRMSNormGated):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)


class GraniteMoeHybridMLP(GraniteMoeSharedMLP):
    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__(config)


class GraniteMoeHybridRotaryEmbedding(Gemma2RotaryEmbedding):
    pass


class GraniteMoeHybridMoE(GraniteMoeSharedMoE):
    pass


class GraniteMoeHybridDecoderLayer(GraniteMoeSharedDecoderLayer):
    def __init__(self, config: GraniteMoeHybridConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.shared_mlp = GraniteMoeHybridMLP(config)
        # Either attention or mamba will be initialized, depending on the layer type.
        self.self_attn = None
        self.mamba = None

        if config.layers_block_type[layer_idx] == "mamba":
            self.mamba = GraniteMoeHybridMambaLayer(config, layer_idx)
        else:
            self.self_attn = GraniteMoeHybridAttention(config, layer_idx)
        self.layer_type = config.layers_block_type[layer_idx]

        # Allow non-MoE (dense)
        self.block_sparse_moe = GraniteMoeHybridMoE(config) if config.num_local_experts > 0 else None

        # Accept 0 experts: skip MoE if num_local_experts == 0
        self.has_experts = getattr(config, "num_local_experts", 0) > 0

    @auto_docstring
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.mamba is not None:
            hidden_states = self.mamba(
                hidden_states=hidden_states,
                cache_position=cache_position,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states * self.residual_multiplier
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.has_experts:
            moe_hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
        else:
            hidden_states = self.shared_mlp(hidden_states)

        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


class GraniteMoeHybridPreTrainedModel(GraniteMoeSharedPreTrainedModel):
    config: GraniteMoeHybridConfig
    _no_split_modules = ["GraniteMoeHybridDecoderLayer"]
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, GraniteMoeHybridMambaLayer):
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.log(torch.arange(1, module.num_heads + 1)))
            init.ones_(module.D)
        elif isinstance(module, GraniteMoeHybridRMSNormGated):
            init.ones_(module.weight)


class GraniteMoeHybridModel(GraniteMoeSharedModel):
    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteMoeHybridDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.embedding_multiplier = config.embedding_multiplier

    @auto_docstring
    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds * self.embedding_multiplier

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            self.config,
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values,
        )
        mamba_mask = self._update_mamba_mask(attention_mask, cache_position)

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            # Depending on the layer type we opt for 2D base attention mask (Mamba) or 4D causal mask (Attention)
            layer_mask = mamba_mask if decoder_layer.layer_type == "mamba" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_mamba_mask(self, attention_mask, cache_position):
        """
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        mamba_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            mamba_mask = None
        return mamba_mask


class GraniteMoeHybridForCausalLM(GraniteMoeSharedForCausalLM):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: GraniteMoeHybridConfig):
        super().__init__(config)
        self.model = GraniteMoeHybridModel(config)
        # Initialize weights and apply final processing
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # Overwritten -- has a unique cache type, `HybridMambaAttentionDynamicCache`

        empty_past_kv = past_key_values is None

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        if not empty_past_kv:
            if (
                inputs_embeds is not None  # Exception 1
                or cache_position[-1] >= input_ids.shape[1]  # Exception 3
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        elif use_cache:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}  # `contiguous()` needed for compilation use cases

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )

        # Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        return model_inputs


__all__ = ["GraniteMoeHybridForCausalLM", "GraniteMoeHybridModel", "GraniteMoeHybridPreTrainedModel"]
