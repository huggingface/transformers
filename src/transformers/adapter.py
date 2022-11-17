import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

class PrefixTuning(nn.Module):
    """Layer-wise prefix for encoder or decoder."""

    def __init__(self, config, num_heads, embed_dim):
        super().__init__()
        self.prefix_length = config.prefix_length
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(p=config.prefix_dropout)
        if 'prefix-tuning' in config.efficient_methods:
            self.method = 'prefix-tuning'
            self.prefix_embedding = nn.Embedding(config.prefix_length, embed_dim)
            self.prefix_trans = nn.Sequential(
                nn.Linear(embed_dim, config.prefix_mid_dim),
                nn.ReLU(),
                nn.Linear(config.prefix_mid_dim, 2 * embed_dim),
            )
        else:
            self.method = 'p-tuning-v2'
            self.prefix_embedding = nn.Embedding(config.prefix_length, 2 * embed_dim)

    def forward(self, key_states: torch.Tensor, value_states: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = key_states.size(0)
        device = key_states.device
        
        if self.method == 'prefix-tuning':
            prefix = self.prefix_trans(self.prefix_embedding.weight) # p_l, 2*e
        else: # 'p-tuning-v2'
            prefix = self.prefix_embedding.weight # p_l, 2*e
        prefix = prefix.view(self.prefix_length, 2, self.num_heads, self.head_dim) # p_l, 2, h_n, h_e
        prefix = self.dropout(prefix)
        prefix = prefix.permute([1, 2, 0, 3]) # 2, h_n, p_l, h_e

        if key_states.dim() == 4:
            key_states = torch.cat([prefix[0].expand(bsz, -1, -1, -1), key_states], dim=-2) # 2, h_n, p_l+t_l, h_e
            value_states = torch.cat([prefix[1].expand(bsz, -1, -1, -1), value_states], dim=-2)
        elif key_states.dim() == 5:
            key_states = torch.cat([prefix[0].expand(bsz, key_states.size(1), -1, -1, -1), key_states], dim=-2) # 2, beam_num, h_n, p_l+t_l, h_e
            value_states = torch.cat([prefix[1].expand(bsz, value_states.size(1), -1, -1, -1), value_states], dim=-2)
        if attention_mask is not None:
            prompt_mask = torch.zeros(bsz, 1, attention_mask.size(2), prefix.size(2)).to(device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=(-1))
        return key_states, value_states, attention_mask

class LoRALinear(nn.Linear):

    def __init__(self, in_features, out_features, config, bias):
        super().__init__(in_features, out_features, bias=bias)
        self.r = config.lora_r
        self.alpha = config.lora_alpha
        self.scaling = self.alpha / self.r
        self.A = nn.Linear(in_features, self.r, bias=False)
        self.B = nn.Linear(self.r, out_features, bias=False)
        self.dropout = nn.Dropout(p=config.lora_dropout)
        self.A.weight.data.normal_(std=0.02)
        self.B.weight.data.zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias) + self.B(self.A(self.dropout(input))) * self.scaling

class Adapter(nn.Module):

    def __init__(self, embed_dim, config):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, config.adapter_mid_dim),
            nn.ReLU(),
            nn.Linear(config.adapter_mid_dim, embed_dim),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.adapter(input)