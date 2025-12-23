# coding=utf-8
# Copyright 2024 Xiaomi and The HuggingFace Inc. team. All rights reserved.
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

from transformers.configuration_utils import PretrainedConfig

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
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


class MiMoV2FlashPreTrainedModel(PreTrainedModel):
    config_class = MiMoV2FlashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MiMoV2FlashDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


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
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.v_head_dim = config.v_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # MiMo specific: Partial RoPE
        # We only rotate the first 'partial_rotary_factor' of the head_dim
        # Must be even fo RoPE rotation
        self.rotary_dim = int(self.head_dim * config.partial_rotary_factor)
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1

        # Projection sizes
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim
        self.o_size = (
            self.num_heads * self.v_head_dim
        )  # Output projects from v's dim back to hidden

        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.k_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.v_size, bias=False)
        self.o_proj = nn.Linear(self.o_size, self.hidden_size, bias=False)

        self._init_rope()

    def _init_rope(self):
        # We use the custom rotary embedding we defined earlier
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
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for Multi-Head
        # Q: [bsz, heads, seq, head_dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # K: [bsz, kv_heads, seq, head_dim]
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        # V: [bsz, kv_heads, seq, v_head_dim]
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.v_head_dim
        ).transpose(1, 2)

        # Partial Rotary Embeddings
        # We split Q and K into "rotate" and "pass" parts
        # rotate: [..., :rotary_dim], pass: [..., rotary_dim:]
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
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to DynamicCache if used
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat K/V for GQA (if needed) to match num_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention Logic (Scaled Dot Product)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # Handle standard mask (4D)
            attn_weights = attn_weights + attention_mask

        # Softmax & Dropout
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )

        # Output Projection
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # [bsz, heads, seq, v_dim] -> [bsz, seq, heads, v_dim] -> [bsz, seq, hidden]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.o_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values


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
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiMoV2FlashMoERouter(nn.Module):
    """
    Router using Sigmoid -> TopK -> Normalization logic.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.gate = nn.Linear(config.hidden_size, self.n_routed_experts, bias=False)
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [bs, seq_len, n_routed_experts]
        logits = self.gate(hidden_states)

        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = torch.softmax(logits, dim=-1)

        # Select Top-K
        # topk_weights: [bs, seq_len, k]
        # topk_indices: [bs, seq_len, k]
        topk_weights, topk_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1, sorted=False
        )

        if self.norm_topk_prob:
            # Normalize the selected weights so they sum to 1
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights = topk_weights / denominator

        return topk_weights, topk_indices


class MiMoV2FlashMoE(nn.Module):
    """
    Mixture of Experts Layer.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.router = MiMoV2FlashMoERouter(config)

        # Initialize Experts
        # Experts are usually smaller than the main MLP
        self.experts = nn.ModuleList(
            [
                MiMoV2FlashMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.n_routed_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Routing
        # weights: [batch * seq, k], indices: [batch * seq, k]
        router_weights, router_indices = self.router(hidden_states)

        final_hidden_states = torch.zeros_like(hidden_states)

        # Loop over experts
        for expert_idx, expert_layer in enumerate(self.experts):
            # Find tokens that selected this expert
            # router_indices is [total_tokens, k]
            token_indices, topk_idx = torch.where(router_indices == expert_idx)

            if token_indices.shape[0] > 0:
                # Gather inputs
                expert_input = hidden_states[token_indices]

                # Forward pass
                expert_output = expert_layer(expert_input)

                # Weight by probability
                # router_weights[token_indices, topk_idx] gives the weight for this expert for these tokens
                weight = router_weights[token_indices, topk_idx].unsqueeze(-1)

                # Add to output
                # We use index_add_ or scatter_add_
                final_hidden_states.index_add_(0, token_indices, expert_output * weight)

        return final_hidden_states.view(batch_size, seq_len, hidden_dim)


class MiMoV2FlashRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MiMoV2FlashRMSNorm is equivalent to T5LayerNorm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MiMoV2FlashDecoderLayer(nn.Module):
    def __init__(self, config: MiMoV2FlashConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiMoV2FlashAttention(config=config, layer_idx=layer_idx)

        # Decide if this layer is MoE or Dense
        # Logic: If moe_layer_freq exists, we check if layer_idx fits the pattern.
        # config.moe_layer_freq is a list of 0 (Dense) and 1 (MoE)
        if config.moe_layer_freq is not None and layer_idx < len(config.moe_layer_freq):
            is_moe = config.moe_layer_freq[layer_idx] == 1
        else:
            # Fallback (e.g., is list is missing, default to MoE or Dense depending on config)
            # Based on config.json, most layers are MoE (1), first layer is Dense (0).
            is_moe = False  # Conservative default

        if is_moe:
            self.mlp = MiMoV2FlashMoE(config)
        else:
            self.mlp = MiMoV2FlashMLP(config)

        self.input_layernorm = MiMoV2FlashRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = MiMoV2FlashRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Hybrid Attention Logic setup
        # config.hybrid_layer_pattern is a list of 0 (full) and 1 (SWA)
        if config.hybrid_layer_pattern is not None and layer_idx < len(
            config.hybrid_layer_pattern
        ):
            self.use_sliding_window = config.hybrid_layer_pattern[layer_idx] == 1
        else:
            self.use_sliding_window = False

        self.sliding_window = config.sliding_window if self.use_sliding_window else None

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

        # Attention
        # If sliding window is active, we might need to modify the mask here or inside attention.
        # Ideally, the mask passed in `attention_mask` handles SWA if generated correctly by the model class.
        # But for eager implementation, we pass the flag or rely on the causal mask generation.

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


class MiMoV2FlashModel(MiMoV2FlashPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MiMoV2FlashDecoderLayer`]
    """

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                MiMoV2FlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiMoV2FlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Retrieve inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Legacy past_key_values handling
        past_key_values_length = 0
        if past_key_values is not None:
            # Taking the length of the first cache's key
            # past_key_values[0][0] shape: [bs, num_heads, seq_len, head_dim]
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # Create 4D causal attention mask
        # Note: In production, use _prepare_4d_causal_attention_mask from modeling_attn_mask_utils
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(
                    past_key_values[i] if past_key_values is not None else None
                ),
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
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

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, (tuple, list)):
                # Standard tuple cache
                past_length = past_key_values[0][0].shape[2]
            else:
                # DynamicCache
                past_length = past_key_values.get_seq_length()

            # Some generation methods already pass only the last token
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
) -> Optional[torch.Tensor]:
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_length)`
    """
    batch_size, query_length = input_shape
    key_length = past_key_values_length + query_length

    # Create causal mask
    # Make a lower triangular matrix
    if attention_mask is None:
        attention_mask = torch.ones(batch_size, key_length, device=inputs_embeds.device)

    # Convert to 4D
    # [batch_size, key_length] -> [batch_size, 1, query_length, key_length]
    if attention_mask.dim() == 2:
        # Create causal mask
        causal_mask = torch.triu(
            torch.full(
                (query_length, key_length), float("-inf"), device=inputs_embeds.device
            ),
            diagonal=past_key_values_length + 1,
        )
        # Apply attention mask
        attention_mask = attention_mask[:, None, None, :].to(inputs_embeds.dtype)
        attention_mask = attention_mask * (causal_mask == 0).to(attention_mask.dtype)
        attention_mask = attention_mask + causal_mask

    return attention_mask
