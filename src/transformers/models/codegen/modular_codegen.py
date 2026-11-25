# Copyright 2022 Salesforce authors, The EleutherAI, and HuggingFace Teams. All rights reserved.
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
"""PyTorch CodeGen model."""

from collections.abc import Callable

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..gptj.modeling_gptj import (
    GPTJMLP,
    GPTJAttention,
    GPTJBlock,
    GPTJForCausalLM,
    GPTJModel,
    GPTJPreTrainedModel,
    GPTJRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


class CodeGenRotaryEmbedding(GPTJRotaryEmbedding):
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        # The original implementation applies rotary embeddings in fp32
        sincos = self.embed_positions[position_ids]
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        return sin, cos


class CodeGenAttention(GPTJAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        del self.q_proj
        del self.k_proj
        del self.v_proj
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        # The fused projection is interleaved across the 4 model parallel shards CodeGen was trained with
        self.mp_num = 4

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        cache: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        local_dim = self.embed_dim // self.mp_num
        qkv = self.qkv_proj(hidden_states).view(*input_shape, -1, 3 * local_dim)
        query, value, key = qkv.split(local_dim, dim=-1)
        query = query.reshape(hidden_shape)
        key = key.reshape(hidden_shape)
        value = value.reshape(hidden_shape).transpose(1, 2)

        sin, cos = position_embeddings
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        query = query.transpose(1, 2).to(value.dtype)
        key = key.transpose(1, 2).to(value.dtype)

        if cache is not None:
            key, value = cache.update(key, value, self.layer_idx, kwargs=kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class CodeGenMLP(GPTJMLP):
    pass


class CodeGenBlock(GPTJBlock):
    pass


class CodeGenPreTrainedModel(GPTJPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"attn.causal_mask"]


class CodeGenModel(GPTJModel):
    pass


class CodeGenForCausalLM(GPTJForCausalLM):
    pass


__all__ = ["CodeGenForCausalLM", "CodeGenModel", "CodeGenPreTrainedModel"]
