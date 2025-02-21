import math

import torch


####### ROPE PARAMETERS
partial_rotary_factor=1.0
hidden_size: int = 256
num_attention_heads: int = 8
rope_theta: float = 10000

base = rope_theta
head_dim = hidden_size // num_attention_heads
dim = int(head_dim * partial_rotary_factor)

attention_factor = 1.0  # Unused in this type of RoPE

# Compute the inverse frequencies
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
##################

seq_len = 10
position_ids = torch.arange(seq_len).reshape(1, seq_len)

############### ROPE CALCULATION
inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
position_ids_expanded = position_ids[:, None, :].float()
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos_llama = emb.cos()
sin_llama = emb.sin()

cos_llama = cos_llama * attention_factor
sin_llama = sin_llama * attention_factor
##################




max_shape = (256, 256)
d_model = 256
################ ELOFTR ROPE

i_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(-1)  # [H, 1]
j_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(-1)  # [W, 1]

div_term = torch.exp(torch.arange(0, d_model // 4, 1).float() * (-math.log(10000.0) / (d_model // 4)))
# div_term = 1.0 / (base ** (torch.arange(0, dim, 1, dtype=torch.int64).float().to(device) / dim))

div_term = div_term[None, None, :]  # [1, 1, C//4]
sin_eloftr = torch.zeros(*max_shape, d_model // 2, dtype=torch.float32)
cos_eloftr = torch.zeros(*max_shape, d_model // 2, dtype=torch.float32)
sin_eloftr[:, :, 0::2] = i_position * div_term
sin_eloftr[:, :, 1::2] = j_position * div_term
cos_eloftr[:, :, 0::2] = i_position * div_term
cos_eloftr[:, :, 1::2] = j_position * div_term
# sin_eloftr[:, :, 0::2] = i_position
# sin_eloftr[:, :, 1::2] = j_position
# cos_eloftr[:, :, 0::2] = i_position
# cos_eloftr[:, :, 1::2] = j_position

sin_eloftr = torch.sin(sin_eloftr)
cos_eloftr = torch.cos(cos_eloftr)

sin_eloftr = sin_eloftr.repeat_interleave(2, dim=-1).unsqueeze(0)
cos_eloftr = cos_eloftr.repeat_interleave(2, dim=-1).unsqueeze(0)

################



######## NEW ELOFTR ROPE PARAMETERS
partial_rotary_factor=1.0
hidden_size: int = 256
num_attention_heads: int = 8
rope_theta: float = 10000

base = rope_theta
head_dim = hidden_size // 4
dim = int(head_dim * partial_rotary_factor)

attention_factor = 1.0  # Unused in this type of RoPE

# Compute the inverse frequencies
eloftr_inv_freq = 1.0 / (base ** (torch.arange(0, dim, 1, dtype=torch.int64).float() / dim))
###########################

shape = (256, 256)
x = torch.rand(1, 256, 256, 64)

############## ELOFTR COMPUTATION
_, h, w, _ = x.shape

i_position_ids = torch.ones(h, w).cumsum(0).float().unsqueeze(-1)
j_position_ids = torch.ones(h, w).cumsum(1).float().unsqueeze(-1)
# Core RoPE block
inv_freq_expanded = eloftr_inv_freq[None, None, :].float().expand(x.shape[0], 1, -1)
# Force float32 (see https://github.com/huggingface/transformers/pull/29285)
device_type = x.device.type
device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
with torch.autocast(device_type=device_type, enabled=False):
    emb = torch.zeros(h, w, hidden_size // 2)
    emb[:, :, 0::2] = i_position_ids * inv_freq_expanded
    emb[:, :, 1::2] = j_position_ids * inv_freq_expanded

sin_new_eloftr = emb.sin()
cos_new_eloftr = emb.cos()
sin_new_eloftr = sin_new_eloftr.repeat_interleave(2, dim=-1).unsqueeze(0)
cos_new_eloftr = cos_new_eloftr.repeat_interleave(2, dim=-1).unsqueeze(0)
# Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
cos_new_eloftr = cos_new_eloftr * attention_factor
sin_new_eloftr = sin_new_eloftr * attention_factor
#####################

print(cos_llama)
print(sin_llama)

print(cos_eloftr)
print(sin_eloftr)

print(cos_new_eloftr)
print(sin_new_eloftr)

print(torch.allclose(sin_eloftr, sin_new_eloftr, rtol=1e-4, atol=1e-4))
