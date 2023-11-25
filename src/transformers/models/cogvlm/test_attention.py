import torch
import xformers.ops as xops


torch.manual_seed(0)

queries = torch.randn(1, 1226, 16, 112).to("cuda")
keys = torch.randn(1, 1226, 16, 112).to("cuda")
values = torch.randn(1, 1226, 16, 112).to("cuda")

scale = 112**-0.5

xops_outputs = xops.memory_efficient_attention(queries, keys, values, scale=scale)

# xops_outputs = F.scaled_dot_product_attention(queries, keys, values, scale=scale)


def equiv_fn(q, k, v, attn_bias=None):
    scale = 1.0 / q.shape[-1] ** 0.5
    q = q * scale
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    attn = attn @ v
    return attn.transpose(1, 2)


# reshape hidden_states
# attn = self.reshape_batch_dim_to_heads(attn)

outputs = equiv_fn(queries, keys, values, attn_bias=None)

print("Shape of xops_outputs:", xops_outputs.shape)
print("Shape of outputs:", outputs.shape)

print("First values of xops_outputs:", xops_outputs[0, 0, :3, :3])
print("First values of outputs:", outputs[0, 0, :3, :3])

assert torch.allclose(xops_outputs, outputs, atol=1e-5)
