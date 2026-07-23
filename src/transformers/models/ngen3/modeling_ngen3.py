# src/transformers/models/ngen3/modeling_ngen3.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from .configuration_ngen3 import NGen3Config

# -----------------------------------------------------------------------------
# ROTARY POSITIONAL EMBEDDINGS
# -----------------------------------------------------------------------------
def apply_rotary_pos_emb(q, k):
    """
    Applies rotary positional embeddings to queries and keys.
    q, k: Tensors of shape (B, n_head, T, head_dim)
    """
    T = q.size(-2)
    dim = q.size(-1)
    device = q.device
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    positions = torch.arange(T, device=device).float()
    sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
    sin = torch.sin(sinusoid_inp).unsqueeze(0).unsqueeze(0)
    cos = torch.cos(sinusoid_inp).unsqueeze(0).unsqueeze(0)
    q1, q2 = q[..., :dim//2], q[..., dim//2:]
    k1, k2 = k[..., :dim//2], k[..., dim//2:]
    q_rot = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1)
    k_rot = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1)
    return q_rot, k_rot

# -----------------------------------------------------------------------------
# GPT Architecture Components
# -----------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # No absolute positional embeddings; using RoPE.
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Optional instruct extra projection.
        if config.instruct:
            self.instruct_dense = nn.Linear(config.n_embd, config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def forward(self, inputs, return_hidden=False, instruct_mode=False):
        x, targets = inputs  # x: (B, T)
        x = self.tok_emb(x)
        x = self.drop(x)
        hidden_states = [] if return_hidden else None
        for block in self.blocks:
            x = block(x)
            if return_hidden:
                hidden_states.append(x)
        x = self.ln_f(x)
        if self.config.instruct and instruct_mode:
            x = self.instruct_dense(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_hidden:
            return logits, loss, hidden_states
        else:
            return logits, loss

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if config.use_moe:
            self.mlp = MoEMLP(config)
        else:
            self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 2)
        self.fc2 = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x_proj = self.fc1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# --- Optional MoE Components ---
class ExpertMLP(nn.Module):
    def __init__(self, config):
        super(ExpertMLP, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 2)
        self.fc2 = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x_proj = self.fc1(x)
        x1, x2 = x_proj.chunk(2, dim=-1)
        x = F.silu(x1) * x2
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MoEMLP(nn.Module):
    def __init__(self, config):
        super(MoEMLP, self).__init__()
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([ExpertMLP(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.n_embd, self.num_experts)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        gate_scores = self.gate(x)  # (B, T, num_experts)
        gate_probs = F.softmax(gate_scores, dim=-1)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=2)  # (B, T, num_experts, n_embd)
        gate_probs = gate_probs.unsqueeze(-1)  # (B, T, num_experts, 1)
        output = torch.sum(gate_probs * expert_outputs, dim=2)
        output = self.dropout(output)
        return output

# -----------------------------------------------------------------------------
# HF Model Wrapper
# -----------------------------------------------------------------------------
class NGen3ForCasualLM(PreTrainedModel):
    config_class = NGen3Config
    base_model_prefix = "ngen3"

    def __init__(self, config):
        super().__init__(config)
        self.ngen3 = GPT(config)
        self.init_weights()
    
    def forward(self, input_ids, labels=None, instruct_mode=None):
        if instruct_mode is None:
            instruct_mode = self.config.instruct
        logits, loss = self.ngen3((input_ids, labels), instruct_mode=instruct_mode)
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)
