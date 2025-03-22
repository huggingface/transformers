import torch

# from einops import rearrange, repeat
# from einops_exts import rearrange_many
from torch import einsum, nn


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def rearrange_einsum(out, h):
    # 使用torch.einsum重新排列维度
    out_einsum = torch.einsum("b h n d -> b n h d", out)
    # 合并h和d维度
    out_einsum = out_einsum.reshape(out_einsum.shape[0], out_einsum.shape[1], -1)
    return out_einsum


class SequenceCompressorAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, mask):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D);  n2: num of latent tokens
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(
            2, dim=-1
        )  # each: batch_size, max_protein_length+num_latents, dim_head*num_heads

        # q_raw, k_raw, v_raw = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q.view(q.size(0), q.size(1), h, -1).permute(0, 2, 1, 3)
        k = k.view(k.size(0), k.size(1), h, -1).permute(0, 2, 1, 3)
        v = v.view(v.size(0), v.size(1), h, -1).permute(0, 2, 1, 3)
        # assert torch.allclose(q_raw, q)
        # assert torch.allclose(k_raw, k)
        # assert torch.allclose(v_raw, v)
        q = q * self.scale  # batch_size, num_heads, num_latents, dim_head

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        bs, nh, skd, okd = sim.shape
        # mask_raw = repeat(mask, "bs okd -> bs nh skd okd", nh=nh, skd=skd)
        ones = torch.ones(nh, skd).to(mask.device)  # 创建一个全 1 的张量，形状为 (nh, skd)
        mask = torch.einsum("bk,oj -> bojk", mask, ones)
        # assert torch.allclose(mask_raw, mask)

        sim = sim.masked_fill((1 - mask).bool(), -1e4)
        # sim = sim + (1 - mask) * torch.tensor(float('-inf'), dtype=sim.dtype)  # 加上mask
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)

        # out_raw = rearrange(out, "b h n d -> b n (h d)", h=h)
        out_einsum = torch.einsum("b h n d -> b n h d", out)
        out = out_einsum.reshape(out_einsum.shape[0], out_einsum.shape[1], -1)
        # assert torch.allclose(out_raw, out)
        return self.to_out(out)


class SequenceCompressorResampler(nn.Module):
    def __init__(
        self,
        protein_repr_dim: int,
        output_repr_dim: int,
        depth: int,
        dim_head: int,
        heads: int,
        num_latents: int,
        ff_mult: int,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, protein_repr_dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SequenceCompressorAttention(dim=protein_repr_dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=protein_repr_dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(output_repr_dim)

        self.protein_projector = nn.Linear(protein_repr_dim, output_repr_dim)

        self.num_latents = num_latents

    @property
    def device(self):
        return self.latents.device

    @property
    def dtype(self):
        return self.latents.dtype

    def forward(self, embeds, mask):
        b = embeds.shape[0]

        bs, _ = mask.shape  # bs, max_protein_length
        latent_mask = torch.ones(bs, self.num_latents).to(mask.device)
        mask = torch.cat((mask, latent_mask), dim=1)  # bs, max_protein_length + num_latents

        # blocks
        # latents_raw = repeat(self.latents, "n d -> b n d", b=b)
        ones = torch.ones(b).to(self.latents.device)
        latents = torch.einsum("nd, b -> bnd", self.latents, ones)
        # assert torch.allclose(latents_raw, latents)
        for attn, ff in self.layers:
            latents = attn(embeds, latents, mask) + latents
            latents = ff(latents) + latents

        transformed_feature = self.protein_projector(latents)

        return self.norm(transformed_feature)


class MLPResampler(nn.Module):
    def __init__(
        self,
        protein_repr_dim,
        output_repr_dim,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(protein_repr_dim, output_repr_dim),
            nn.ReLU(),
            nn.Linear(output_repr_dim, output_repr_dim),
            nn.LayerNorm(output_repr_dim),
        )

    def forward(self, embeds, mask):
        return self.model(embeds)
