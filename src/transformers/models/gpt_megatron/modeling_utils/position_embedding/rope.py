"""Logic is copied from transformers.models.llama.modeling_utils with slight modifications"""

import math
from typing import Tuple

import torch
import torch.nn as nn


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations: int, dim: int, base: int = 10000, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: int, high_rot: int, dim: int, base: int = 10000, max_position_embeddings: int = 2048
) -> int:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min: float, max: float, dim: int) -> torch.Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        device: torch.device = None,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.mscale = 1

        self.initialize(device)

    @torch.no_grad()
    def initialize(self, device: torch.device) -> None:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    @torch.no_grad()
    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)

    def forward(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=dtype)

        cos = self.cos_cached[:seq_len].to(dtype)
        sin = self.sin_cached[:seq_len].to(dtype)

        return cos, sin


class YaRNScaledRoPE(RoPE):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
        scale: float = 1,
        original_max_position_embeddings: int = 2048,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        device: torch.device = None,
    ) -> None:
        nn.Module.__init__(self)

        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = _yarn_get_mscale(self.scale) * self.attn_factor

        self.initialize(device)

    def initialize(self, device: torch.device) -> None:
        pos_freqs = self.base ** (torch.arange(0, self.head_dim, 2).float().to(device) / self.head_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast, self.beta_slow, self.head_dim, self.base, self.original_max_position_embeddings
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, self.head_dim // 2).float().to(device)
        ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(self.max_position_embeddings, device=device, dtype=torch.get_default_dtype())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
