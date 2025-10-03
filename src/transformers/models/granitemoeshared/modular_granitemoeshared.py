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
from typing import Optional, TypedDict

import torch
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...processing_utils import Unpack
from ...utils import logging
from ..granitemoe.modeling_granitemoe import (
    GraniteMoeDecoderLayer,
    GraniteMoeForCausalLM,
    GraniteMoeModel,
    GraniteMoePreTrainedModel,
)
from .configuration_granitemoeshared import GraniteMoeSharedConfig


logger = logging.get_logger(__name__)


class GraniteFlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for advanced Flash Attention, causal-conv1d, and mamba_ssm kernel usage.
    Use cases include padding-free training and fewer `torch.compile` graph breaks.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`)
            Gets cumulative sequence length for key state.
        max_length_q (`int`):
            Maximum sequence length for query state.
        max_length_k (`int`):
            Maximum sequence length for key state.
        seq_idx (`torch.IntTensor):
            Index of each packed sequence.
    """

    cu_seq_lens_q: torch.LongTensor
    cu_seq_lens_k: torch.LongTensor
    max_length_q: int
    max_length_k: int
    seq_idx: torch.IntTensor


class GraniteMoeSharedMLP(nn.Module):
    """
    MLP layer for shared experts

    Args:
        config:
            Configuration object with model hyperparameters.
    """

    def __init__(self, config: GraniteMoeSharedConfig):
        super().__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.shared_intermediate_size
        self.activation = ACT2FN[config.hidden_act]
        self.input_linear = nn.Linear(self.input_size, self.hidden_size * 2, bias=False)
        self.output_linear = nn.Linear(self.hidden_size, self.input_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_linear(hidden_states)
        chunked_hidden_states = hidden_states.chunk(2, dim=-1)
        hidden_states = self.activation(chunked_hidden_states[0]) * chunked_hidden_states[1]
        hidden_states = self.output_linear(hidden_states)
        return hidden_states


class GraniteMoeSharedDecoderLayer(GraniteMoeDecoderLayer):
    def __init__(self, config: GraniteMoeSharedConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.shared_mlp = None if config.shared_intermediate_size == 0 else GraniteMoeSharedMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[GraniteFlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_hidden_states = self.block_sparse_moe(hidden_states)

        if self.shared_mlp is None:
            hidden_states = moe_hidden_states
        else:
            hidden_states = moe_hidden_states + self.shared_mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


class GraniteMoeSharedPreTrainedModel(GraniteMoePreTrainedModel):
    config: GraniteMoeSharedConfig
    _no_split_modules = ["GraniteMoeSharedDecoderLayer"]


class GraniteMoeSharedModel(GraniteMoeModel):
    def __init__(self, config: GraniteMoeSharedConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [GraniteMoeSharedDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )


class GraniteMoeSharedForCausalLM(GraniteMoeForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GraniteMoeSharedConfig):
        super().__init__(config)
        self.model = GraniteMoeSharedModel(config)
        # Initialize weights and apply final processing
        self.post_init()


__all__ = ["GraniteMoeSharedForCausalLM", "GraniteMoeSharedModel", "GraniteMoeSharedPreTrainedModel"]
