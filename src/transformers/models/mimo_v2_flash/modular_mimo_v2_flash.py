# coding=utf-8
# Copyright 2025 Xiaomi and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch MiMo-V2-Flash model."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..llama.modeling_llama import LlamaRMSNorm as MiMoV2FlashRMSNorm
from .configuration_mimo_v2_flash import MiMoV2FlashConfig

logger = logging.get_logger(__name__)


# Helper functions for Rotary Embeddings
def rotate_half(x):
    """Rotate half of the hidden dims off the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MiMoV2FlashRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_heads, seq_len, head_dim]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 for high precision in rotary calculation
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MiMoV2FlashAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified for MiMo-V2:
    - Differing head dimensions for Q/K vs V.
    - Partial Rotary Embeddings.
    """

    def __init__(self, config: MiMoV2FlashConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True

        # RoPE parameters
        rope_params = config.rope_parameters
        self.rope_theta = rope_params.get("rope_theta", 5000000.0)
        self.partial_factor = rope_params.get("partial_rotary_factor", 0.334)

        # Ensure rotary_dim is even
        self.rotary_dim = int(self.qk_head_dim * self.partial_factor)
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1

        # Projection sizes
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim
        self.o_size = (
            self.num_heads * self.v_head_dim
        )  # Output projects from v's dim back to hidden

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.qk_head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.qk_head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = MiMoV2FlashRotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Projection
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.qk_head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.v_head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(value_states, position_ids)

        q_rot, q_pass = (
            query_states[..., : self.rotary_dim],
            query_states[..., self.rotary_dim :],
        )
        k_rot, k_pass = (
            key_states[..., : self.rotary_dim],
            key_states[..., self.rotary_dim :],
        )

        # Apply RoPE to the first part
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        # Concatenate back
        query_states = torch.cat((q_rot, q_pass), dim=-1)
        key_states = torch.cat((k_rot, k_pass), dim=-1)

        # KV Cache Update
        if past_key_values is not None:
            # sin and cos are gathered, no need to re-compute
            # cache is (key_states, value_states)
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos}
            )

        # Repeat K/V for GQA (if needed) to match num_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention Logic (Scaled Dot Product)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.qk_head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, past_key_values


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MiMoV2FlashMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            intermediate_size or config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            intermediate_size or config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size or config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiMoV2FlashMoE(nn.Module):
    """
    Router using Sigmoid -> TopK -> Normalization logic.
    """

    def __init__(self, config):
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                MiMoV2FlashMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.top_k = config.num_experts_per_tok

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))

        logits = self.router(hidden_states)
        scores = torch.sigmoid(logits)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        # Normalize weights so they can sum to 1
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20

        final_hidden_states = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.experts):
            token_indices, topk_indices = torch.where(indices == i)
            if token_indices.shape[0] > 0:
                expert_output = expert(hidden_states[token_indices])
                final_hidden_states.index_add_(
                    0,
                    token_indices,
                    expert_output * weights[token_indices, topk_indices].unsqueeze(-1),
                )

        return final_hidden_states.view(original_shape)


class MiMoV2FlashDecoderLayer(nn.Module):
    def __init__(self, config: MiMoV2FlashConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiMoV2FlashAttention(config=config, layer_idx=layer_idx)

        # Decide if this layer is MoE or Dense
        # Logic: If moe_layer_freq exists, we check if layer_idx fits the pattern.
        # config.moe_layer_freq is a list of 0 (Dense) and 1 (MoE)
        is_moe = (
            config.moe_layer_freq is not None and config.moe_layer_freq[layer_idx] == 1
        )
        self.mlp = MiMoV2FlashMoE(config) if is_moe else MiMoV2FlashMLP(config)

        self.input_layernorm = MiMoV2FlashRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiMoV2FlashRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Handling layer_types strings
        if config.layer_types is not None and layer_idx < len(config.layer_types):
            self.use_sliding_window = (
                config.layer_types[layer_idx] == "sliding_attention"
            )
        else:
            self.use_sliding_window = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Normalization before attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states

        # Normalization before MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP (Dense or MoE)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


class MiMoV2FlashPreTrainedModel(PreTrainedModel):

    config_class = MiMoV2FlashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiMoV2FlashDecoderLayer"]
    _supports_sdpa = True

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MiMoV2FlashModel(MiMoV2FlashPreTrainedModel):

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                MiMoV2FlashDecoderLayer(config, i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiMoV2FlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if return_dict is None:
            return_dict = True

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        batch_size, seq_length = hidden_states.shape[:2]

        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is not None:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_length
            )

        next_cache = [] if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values[i] if past_key_values else None,
                output_attentions,
                use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache.append(layer_outputs[2])

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class MiMoV2FlashForCausalLM(MiMoV2FlashPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = MiMoV2FlashModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if return_dict is None:
            return_dict = True

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[..., :-1, :].reshape(-1, self.config.vocab_size),
                labels[..., 1:].reshape(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
