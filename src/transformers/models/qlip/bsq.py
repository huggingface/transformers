# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/QLIP/blob/main/LICENSE

# MIT License
# Based on https://github.com/zhaoyue-zephyrus/bsq-vit/blob/main/transcoder/models/quantizer/bsq.py

import torch
import torch.nn as nn
from einops import rearrange, reduce

_EPS = 1e-8


class DifferentiableEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        zb = (zq + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)
        cnt = torch.scatter_reduce(
            torch.zeros(2**K, device=zq.device, dtype=zq.dtype),
            0,
            zi.flatten(),
            torch.ones_like(zi.flatten()).to(zq.dtype),
            "sum",
        )
        prob = (cnt + eps) / (cnt + eps).sum()
        H = torch.special.entr(prob).sum()
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-8):
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


class BinarySphericalQuantizer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 18,
        group_size: int = 9,
        soft_entropy: bool = True,
        beta: float = 0.0,  # commit loss
        gamma_0: float = 1.0,  # entropy loss (E[H(q)])
        gamma_1: float = 1.0,  # entropy loss (H[E[q]])
        input_format: str = "bchw",
        persample_entropy_compute: str = "group",
        l2_norm: bool = True,
        inv_temperature: float = 100.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.group_size = group_size
        assert embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
        self.soft_entropy = soft_entropy
        self.beta = beta
        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        assert input_format in ["bchw", "blc"]
        self.input_format = input_format
        assert persample_entropy_compute in [
            "group",
            "analytical",
        ], "persample_entropy_compute must be either 'group' or 'analytical'"
        self.persample_entropy_compute = persample_entropy_compute
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature

        self.register_buffer("basis", 2 ** torch.arange(embed_dim - 1, -1, -1), persistent=False)
        self.register_buffer(
            "group_basis", 2 ** torch.arange(group_size - 1, -1, -1), persistent=False
        )

        group_codes = torch.arange(2**self.group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer("group_codebook", group_codebook, persistent=False)

    def quantize(self, z):
        assert (
            z.shape[-1] == self.embed_dim
        ), f"Expected {self.embed_dim} dimensions, got {z.shape[-1]}"
        zhat = torch.where(z > 0, torch.ones_like(z), -torch.ones_like(z))
        return z + (zhat - z).detach()

    def forward(self, z):
        if self.input_format == "bchw":
            z = rearrange(z, "b c h w -> b h w c")
        zq = self.quantize(z)

        indices = self.codes_to_indexes(zq.detach())
        group_indices = self.codes_to_group_indexes(zq.detach())

        if not self.training:
            used_codes = torch.unique(indices, return_counts=False)
        else:
            used_codes = None

        if self.soft_entropy:
            persample_entropy, cb_entropy = self.soft_entropy_loss(z)
        else:
            persample_entropy, cb_entropy = self.hard_entropy_loss(z)
        entropy_penalty = self.gamma_0 * persample_entropy - self.gamma_1 * cb_entropy

        q_scale = 1.0 / (self.embed_dim**0.5) if self.l2_norm else 1.0
        zq = zq * q_scale
        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        if self.input_format == "bchw":
            zq = rearrange(zq, "b h w c -> b c h w")

        return (
            zq,
            commit_loss + entropy_penalty / self.inv_temperature,
            {
                "H": cb_entropy,
                "used_codes": used_codes,
                "indices": indices,
                "group_indices": group_indices,
            },
        )

    def soft_entropy_loss(self, z):
        group_codebook = self.group_codebook / (self.embed_dim**0.5 if self.l2_norm else 1)
        divided_z = rearrange(z, "... (g c) -> ... g c", c=self.group_size)

        if self.persample_entropy_compute == "group":
            distance = -2 * torch.einsum("... g c, d c -> ... g d", divided_z, group_codebook)
            prob = (-distance * self.inv_temperature).softmax(dim=-1)
            persample_entropy = torch.special.entr(prob + _EPS).sum((-1, -2)).mean()
        else:
            p = torch.sigmoid(
                -4 * z / (self.embed_dim**0.5 if self.l2_norm else 1) * self.inv_temperature
            )
            prob = torch.stack([p, 1 - p], dim=-1)
            persample_entropy = torch.special.entr(prob + _EPS).sum((-1, -2)).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, "... g d -> g d", "mean")
        cb_entropy = torch.special.entr(avg_prob + _EPS).sum()

        return persample_entropy, cb_entropy

    def hard_entropy_loss(self, z):
        zb = ((z + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
        prob_per_dim = zb.sum(1) / zb.shape[1]
        prob = torch.stack([prob_per_dim, 1 - prob_per_dim], dim=-1)
        persample_entropy = torch.special.entr(prob + _EPS).sum((-1, -2)).mean()
        cb_entropy = codebook_entropy(z, self.basis, self.embed_dim)

        return persample_entropy, cb_entropy

    def codes_to_indexes(self, zhat):
        """Converts a `code` to an index in the codebook.
        Args:
            zhat: A tensor of shape (B, ..., C) containing the codes. must be in {-1, 1}
        """
        assert (
            zhat.shape[-1] == self.embed_dim
        ), f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}"
        return ((zhat.int() + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat):
        """Converts a `code` to a list of indexes (in groups) in the codebook.
        Args:
            zhat: A tensor of shape (B, ..., C) containing the codes. must be in {-1, 1}
        """
        zhat_in_group = rearrange(zhat, "b ... (g c) -> b ... g c", c=self.group_size)
        return ((zhat_in_group.int() + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        """Inverse of `codes_to_indexes`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(indices, self.basis), 2)
        return codes_non_centered * 2 - 1

    def group_indexes_to_codes(self, group_indices):
        """Inverse of `codes_to_group_indexes`."""
        group_indices = group_indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(torch.floor_divide(group_indices, self.group_basis), 2)
        codes_non_centered = rearrange(codes_non_centered, "b ... g c -> b ... (g c)")
        return codes_non_centered * 2 - 1

    def get_group_codebook_entry(self, group_indices, one_hot=False):
        """
        Args:
            group_indices: A tensor of shape (B, L, G, C) containing the group indices.
        """
        if one_hot:
            z_q = group_indices @ self.group_codebook
        else:
            z_q = self.group_indexes_to_codes(group_indices)
        q_scale = 1.0 / (self.embed_dim**0.5) if self.l2_norm else 1.0
        z_q = z_q * q_scale
        if self.input_format == "bchw":
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], "Invalid sequence length"
            z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h)
        return z_q

    def get_codebook_entry(self, indices, one_hot=False):
        """
        Args:
            group_indices: A tensor of shape (B, L, C) containing the indices.
        """
        if one_hot:
            assert self.embed_dim == self.group_size, "one_hot is only supported for group_size == embed_dim"
            z_q = indices @ self.group_codebook
        else:
            z_q = self.indexes_to_codes(indices)
        q_scale = 1.0 / (self.embed_dim**0.5) if self.l2_norm else 1.0
        z_q = z_q * q_scale
        if self.input_format == "bchw":
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], "Invalid sequence length"
            z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h)
        return z_q
