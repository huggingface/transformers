# coding=utf-8
# Copyright 2025 The HuggingFace Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_evo2 import Evo2Config


logger = logging.get_logger(__name__)


# =========================
# Norm + Rotary helpers
# =========================


class Evo2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # standard RMSNorm
        norm = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return (self.weight * x).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    Simple rotary embedding (RoPE) implementation.
    We keep this minimal; you can later swap for the shared one from another model.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        return torch.cos(emb).to(dtype), torch.sin(emb).to(dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [b, s, h, d]
    cos/sin: [1, s, 1, d]
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[..., ::2]
    sin = sin[..., ::2]
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    x_rot = torch.stack([x1_rot, x2_rot], dim=-1)
    x_rot = x_rot.flatten(-2)
    return x_rot


# =========================
# Attention block
# =========================


class Evo2Attention(nn.Module):
    def __init__(self, config: Evo2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_proj_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_proj_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_proj_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.mha_out_proj_bias)

        self.rotary_emb = RotaryEmbedding(self.head_dim, base=config.rotary_emb_base)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # [b, s, h, d]
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim)

        cos, sin = self.rotary_emb(seq_len, hidden_states.device, hidden_states.dtype)
        cos = cos[None, :, None, :]  # [1, s, 1, d]
        sin = sin[None, :, None, :]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_key_value = (k, v) if use_cache else None

        # [b, h, s, d]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # attention_mask expected [b, 1, 1, s_k];  add additive mask
            attn_weights = attn_weights + attention_mask

        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)  # [b, h, s, d]

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, present_key_value


# =========================
# Hyena-ish block (simplified)
# =========================


class Evo2HyenaBlock(nn.Module):
    """
    Simplified Hyena-style block.

    This is NOT the full HyenaCascade from Vortex. It’s a placeholder:
    - depthwise conv over time
    - small MLP

    You can later replace this with a faithful StripedHyena2 port.
    """

    def __init__(self, config: Evo2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.short_filter_length = config.short_filter_length

        self.dw_conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.short_filter_length,
            padding=self.short_filter_length // 2,
            groups=self.hidden_size,
            bias=config.short_filter_bias,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, config.inner_mlp_size),
            nn.GELU(),  # matches mlp_activation default
            nn.Linear(config.inner_mlp_size, self.hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [b, s, h] -> [b, h, s] for conv
        x = hidden_states.transpose(1, 2)
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        return x


# =========================
# Evo2Block
# =========================


class Evo2Block(nn.Module):
    def __init__(self, config: Evo2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm1 = Evo2RMSNorm(config.hidden_size, eps=config.eps)
        self.norm2 = Evo2RMSNorm(config.hidden_size, eps=config.eps)

        if layer_idx in config.attn_layer_idxs:
            self.block_type = "attn"
            self.attn = Evo2Attention(config)
            self.hyena = None
        else:
            self.block_type = "hyena"
            self.attn = None
            self.hyena = Evo2HyenaBlock(config)

        # Simple MLP for the second residual (you can adjust to ParallelGatedMLP later)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.inner_mlp_size),
            nn.GELU(),
            nn.Linear(config.inner_mlp_size, config.hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)

        present_key_value = None

        if self.block_type == "attn":
            attn_output, present_key_value = self.attn(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            hidden_states = residual + attn_output
        else:
            hyena_out = self.hyena(hidden_states)
            hidden_states = residual + hyena_out

        # Second norm + MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


# =========================
# Base model
# =========================


class Evo2PreTrainedModel(PreTrainedModel):
    config_class = Evo2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Evo2Block"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class Evo2Model(Evo2PreTrainedModel):
    """
    Decoder-only Evo2 backbone: embeddings + stack of Evo2Blocks.
    """

    def __init__(self, config: Evo2Config):
        super().__init__(config)

        self.padding_idx = 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)

        self.layers = nn.ModuleList(
            [Evo2Block(config, layer_idx=i) for i in range(config.num_layers)]
        )

        self.final_norm = Evo2RMSNorm(config.hidden_size, eps=config.eps) if config.final_norm else None

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutputWithPast:
        if output_attentions:
            logger.warning_once("Evo2Model does not currently return attentions.")

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.size()

        # Build causal attention mask if not provided (2D mask with 1 for non-padded tokens)
        if attention_mask is not None:
            # [b, s] -> [b, 1, 1, s] additive mask
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        all_hidden_states = [] if output_hidden_states else None
        next_past_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            past_kv = past_key_values[idx] if past_key_values is not None else None

            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                next_past_key_values.append(present_kv)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        if not return_dict:
            outputs = (hidden_states, next_past_key_values)
            if output_hidden_states:
                outputs = (hidden_states, next_past_key_values, all_hidden_states)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_past_key_values,
            hidden_states=all_hidden_states,
        )


# =========================
# Causal LM head
# =========================


class Evo2ForCausalLM(Evo2PreTrainedModel):
    """
    Evo2 language model with a LM head on top of Evo2Model.
    """

    def __init__(self, config: Evo2Config):
        super().__init__(config)
        self.model = Evo2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.tie_weights()

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.model.embed_tokens = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head, self.get_input_embeddings())

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Standard decoder-only prepare_inputs_for_generation:
        # if we have past_key_values, only feed the last token.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # shift for causal LM
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits, outputs.past_key_values)
            if output_hidden_states:
                output = (logits, outputs.past_key_values, outputs.hidden_states)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
