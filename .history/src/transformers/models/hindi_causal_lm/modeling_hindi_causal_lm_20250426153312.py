# coding=utf-8
# Copyright 2025 ConvAI Innovations and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from .configuration_hindi_causal_lm import HindiCausalLMConfig


_CHECKPOINT_FOR_DOC = "convaiinnovations/hindi-foundational-model-base"
_CONFIG_FOR_DOC = "HindiCausalLMConfig"
HINDI_CAUSAL_LM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "convaiinnovations/hindi-foundational-model-base",
]


def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid_inp = torch.einsum("i,j->ij", torch.arange(num_pos, dtype=torch.float), inv_freq)
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class CausalSelfAttention(nn.Module):
    """Causal self-attention layer."""

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = config.hidden_size

        self.q = nn.Linear(config.hidden_size, self.all_head_size)
        self.k = nn.Linear(config.hidden_size, self.all_head_size)
        self.v = nn.Linear(config.hidden_size, self.all_head_size)
        self.proj = nn.Sequential(
            nn.Linear(self.all_head_size, config.hidden_size),
            nn.Dropout(config.attention_probs_dropout_prob),
        )
        mask = torch.triu(
            torch.ones(config.max_position_embeddings, config.max_position_embeddings) * float("-inf"),
            diagonal=1,
        )
        self.register_buffer("mask", mask)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()
        x = x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        b, t, _ = x.size()
        q = self._split_heads(self.q(x))
        k = self._split_heads(self.k(x))
        v = self._split_heads(self.v(x))

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + self.mask[:t, :t]
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(b, t, -1)
        out = self.proj(context)
        return (out, probs) if output_attentions else (out,)


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        hid = config.intermediate_size
        if config.hidden_act == "swiglu":
            self.ff = nn.Sequential(
                nn.Linear(config.hidden_size, hid * 2),
                SwiGLU(),
                nn.Linear(hid, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob),
            )
        else:
            act = ACT2FN[config.hidden_act]
            self.ff = nn.Sequential(
                nn.Linear(config.hidden_size, hid),
                act,
                nn.Linear(hid, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob),
            )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        h = x + self.attn(self.ln1(x), attention_mask, use_cache, output_attentions)[0]
        return h + self.ff(self.ln2(h))


class HindiCausalLMPreTrainedModel(PreTrainedModel):
    config_class = HindiCausalLMConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(0.0, self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


class HindiCausalLMModel(HindiCausalLMPreTrainedModel):
    """Transformer decoder."""

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
        position_ids: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ):
        b, t = input_ids.size()
        device = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(t, device=device).unsqueeze(0).expand(b, t)

        x = self.wte(input_ids) + self.wpe(position_ids)
        x = self.drop(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = mask.to(dtype=x.dtype) * -10000.0
        else:
            mask = None

        for block in self.h:
            x = block(x, mask, use_cache, output_attentions)
        x = self.ln_f(x)

        if not return_dict:
            return (x,)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=None)


class HindiCausalLMForCausalLM(HindiCausalLMPreTrainedModel):
    """Language modeling head."""

    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attention\.masked_bias", r"lm_head.weight"]

    def __init__(self, config: HindiCausalLMConfig):
        super().__init__(config)
        self.transformer = HindiCausalLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        return_dict: bool = None,
    ):
        outputs = self.transformer(input_ids, attention_mask=attention_mask, use_cache=use_cache)
        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            return (logits,) if loss is None else (loss, logits)

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=None, hidden_states=None, attentions=None
        )


# Alias for AutoModel fallback
HindiCausalLMHeadModel = HindiCausalLMForCausalLM
