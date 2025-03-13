# From https://github.com/lucidrains/conformer.git
import torch
from torch import nn, einsum
import torch.nn.functional as F

# helper functions

def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        
    def forward(self, x):
        x = x.permute(self.dims)
        return x


# helper classes

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in, bias=False)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module

class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class PreNormAttn(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context_size, **kwargs):
        x = self.norm(x)
        return self.fn(x, context_size, **kwargs)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        context_size = 200,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context_size):
        device, h, max_pos_emb = x.device, self.heads, self.max_pos_emb
        bs, n, d = x.shape
        assert(context_size > 0 and context_size <= max_pos_emb)

        nb = n // context_size
        nr = n % context_size
        if nr > 0:
            y = torch.zeros(x.shape[0], context_size-nr, x.shape[2], device=device)
            x = torch.cat((x,y), dim=1)
            nb += 1

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))
        q, k, v = map(lambda t: t.reshape(bs, nb, context_size, h, -1).transpose(2, 3), (q, k, v))
        dots = einsum('b m h i d, b m h j d -> b m h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(context_size, device = device)
        dist = seq.view(-1, 1) - seq.view(1, -1)
        dist = torch.clamp(dist,-context_size, context_size) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b m h c d, c r d -> b m h c r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if nr > 0:
            mask = torch.ones(context_size, context_size, device=device)
            mask[:nr,:nr] = 0
            mask_value = -torch.finfo(dots.dtype).max
            dots[:,-1,:].masked_fill_(mask.bool(), mask_value)

        attn = dots.softmax(dim = -1)

        out = einsum('b m h i j, b m h j d -> b m h i d', attn, v)
        out = out.transpose(2, 3).reshape(bs, x.shape[1], -1)
        out = self.to_out(out[:,:n,:])
        return self.dropout(out)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Permute(dims=(0, 2, 1)),
            nn.Conv1d(dim, inner_dim * 2, 1),
            nn.GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            nn.SiLU(),
            nn.Conv1d(inner_dim, dim, 1),
            Permute(dims=(0, 2, 1)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Conformer Block

class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 2,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        context_size = -1,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, context_size = context_size)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn = PreNormAttn(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, context_size):
        x = self.ff1(x) + x
        x = self.attn(x, context_size) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x