# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.

import torch
import torch.nn as nn
from ...modeling_utils import PreTrainedModel
from .configuration_circuit_gpt import CircuitGptConfig


class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features, sparsity=0.0, bias=True):
        super().__init__(in_features, out_features, bias)
        self.sparsity = sparsity

    def forward(self, input):
        if self.sparsity <= 0:
            return super().forward(input)

        w = self.weight
        k = int(w.numel() * (1 - self.sparsity))
        if k < w.numel():
            topk_values, _ = torch.topk(torch.abs(w.flatten()), k)
            threshold = topk_values[-1]
            mask = (torch.abs(w) >= threshold).to(w.dtype)
            w = w * mask

        return nn.functional.linear(input, w, self.bias)


class CircuitGptPreTrainedModel(PreTrainedModel):
    config_class = CircuitGptConfig
    base_model_prefix = "transformer"


class CircuitGptMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = SparseLinear(config.n_embd, 4 * config.n_embd, sparsity=config.sparsity)
        self.c_proj = SparseLinear(4 * config.n_embd, config.n_embd, sparsity=config.sparsity)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CircuitGptAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = SparseLinear(config.n_embd, 3 * config.n_embd, sparsity=config.sparsity)
        self.c_proj = SparseLinear(config.n_embd, config.n_embd, sparsity=config.sparsity)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Scaled Dot-Product Attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class CircuitGptBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CircuitGptAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = CircuitGptMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CircuitGptModel(CircuitGptPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(1024, config.n_embd)
        self.h = nn.ModuleList([CircuitGptBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.post_init()

    def forward(self, input_ids):
        device = input_ids.device
        t = input_ids.size(1)
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x = self.wte(input_ids) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return x


class CircuitGptForCausalLM(CircuitGptPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CircuitGptModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
