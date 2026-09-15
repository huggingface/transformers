# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""ZGCM: mixed sliding/full attention with partial RoPE and attention output gates."""

from collections.abc import Callable

import torch
from torch import nn

from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ..qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
    Qwen3PreTrainedModel,
    eager_attention_forward,
    rotate_half,
)
from .configuration_zgcm import ZgcmConfig


class ZgcmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # The checkpoint implementation multiplies the learned scale in float32.
        states = hidden_states.float()
        states = states * torch.rsqrt(states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (states * self.weight.float()).to(hidden_states.dtype)


class ZgcmMLP(Qwen3MLP):
    pass


class ZgcmRotaryEmbedding(nn.Module):
    def __init__(self, config: ZgcmConfig):
        super().__init__()
        dim = int(config.head_dim * config.partial_rotary_factor)
        self.dim = dim - dim % 2
        self.rope_theta = config.rope_theta

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Keep frequencies and rotations in float32, including under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            inv_freq = 1.0 / (
                self.rope_theta ** (torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32) / self.dim)
            )
            freqs = position_ids.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    dim = cos.shape[-1]
    q_rot, k_rot = q[..., :dim].float(), k[..., :dim].float()
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return (torch.cat((q_rot.to(q.dtype), q[..., dim:]), dim=-1), torch.cat((k_rot.to(k.dtype), k[..., dim:]), dim=-1))


class ZgcmAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ZgcmConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = ZgcmRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = ZgcmRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.g_proj = (
            nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
            if config.attention_gate_layers[layer_idx]
            else None
        )
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
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
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        if self.g_proj is not None:
            gate = torch.sigmoid(self.g_proj(hidden_states).float()).to(attn_output.dtype)
            attn_output = attn_output * gate
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class ZgcmDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ZgcmConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = ZgcmAttention(config=config, layer_idx=layer_idx)

        self.mlp = ZgcmMLP(config)
        self.post_attention_layernorm = ZgcmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = ZgcmRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ZgcmPreTrainedModel(Qwen3PreTrainedModel):
    _supports_flex_attn = False
    _can_compile_fullgraph = False


class ZgcmModel(Qwen3Model):
    pass


class ZgcmForCausalLM(Qwen3ForCausalLM):
    def forward(self, **super_kwargs: Unpack[TransformersKwargs]):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, ZgcmForCausalLM
        >>> model = ZgcmForCausalLM.from_pretrained("zgcagi/ZGCM-1-7B")
        >>> tokenizer = AutoTokenizer.from_pretrained("zgcagi/ZGCM-1-7B")
        >>> inputs = tokenizer("Hello", return_tensors="pt")
        >>> output = model.generate(**inputs, max_new_tokens=20)
        ```
        """
        return super().forward(**super_kwargs)


__all__ = ["ZgcmPreTrainedModel", "ZgcmModel", "ZgcmForCausalLM"]
