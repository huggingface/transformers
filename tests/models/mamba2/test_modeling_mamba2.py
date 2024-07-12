# TODO
"""
add tests like this for einops -> torch (covers all cases that have exchanged einops for torch)

import torch
from einops import rearrange, repeat

x = torch.randn(size=(2, 6, 4))
z = torch.randn(size=(2, 1, 4))
d = torch.randn(size=(4,))
y = torch.randn(size=(2, 6, 4, 5))
w = torch.randn(size=(2, 3, 4, 5, 6))
v = torch.randn(size=(2, 4))


print(x.transpose(1, 2).equal(rearrange(x, "b l d -> b d l")))
print(z.squeeze(1).equal(rearrange(z, "d 1 w -> d w")))
print(rearrange(y, "b c l h -> b h c l").equal(y.permute(0, 3, 1, 2)))
print(y.unsqueeze(-1).expand(y.shape[0], y.shape[1], y.shape[2], y.shape[3], 5).equal(repeat(y, "... d -> ... d e", e=5)))
print(rearrange(w, "b c l h p -> b (c l) h p").equal(w.view(w.shape[0], -1, w.shape[-2], w.shape[-1])))
print(rearrange(x, "b (c l) ... -> b c l ...", l=3).equal(x.view(x.shape[0], -1, 3, x.shape[2])))
print(rearrange(y, "b (c l) ... -> b c l ...", l=3).equal(y.view(y.shape[0], -1, 3, y.shape[2], y.shape[3])))
print(rearrange(x, pattern="b l n -> b l 1 n").equal(x.unsqueeze(-2)))
print(rearrange(x, pattern="b l (h p) -> b l h p", p=2).equal(x.view(x.shape[0], x.shape[1], -1, 2)))
print(rearrange(y, "b l h p -> b l (h p)").equal(y.view(y.shape[0], y.shape[1], -1)))
print(v.view(v.shape[0], -1, 2).equal(rearrange(v, "b (h p) -> b h p", p=2)))
print(rearrange(v, "b h -> b h 1 1").equal(v.unsqueeze(-1).unsqueeze(-1)))
print(rearrange(x, "b h p -> b 1 (h p)").equal(x.view(x.shape[0], -1).unsqueeze(1)))
print(repeat(d, "h -> h p n", p=2, n=3).equal(d.unsqueeze(-1).unsqueeze(-1).expand(d.shape[0], 2, 3)))
print(repeat(z, "b 1 h -> b h p", p=2).equal(z.transpose(1, 2).expand(z.shape[0], z.shape[-1], 2)))
print(d.unsqueeze(-1).expand(d.shape[0], 3).equal(repeat(d, "h -> h p", p=3)))
"""
