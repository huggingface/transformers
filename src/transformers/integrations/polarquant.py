# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PolarQuant KV cache quantization primitives.

Self-contained, pure PyTorch, no external dependencies beyond torch.

PolarQuant compresses KV cache vectors in three steps:

1. Per-channel z-score normalization: subtract the per-channel mean and divide
   by the per-channel standard deviation across the batch of vectors being
   quantized. This handles the heavy outliers and per-channel scale variance
   that real attention K/V tensors typically exhibit (the same problem that
   SmoothQuant, AWQ, and KIVI all address with per-channel handling).
2. Walsh-Hadamard rotation: an orthogonal transform that decorrelates the
   per-channel components. After step 1 the components are roughly per-channel
   Gaussian; after the rotation each rotated coordinate is a linear combination
   of unit-variance Gaussians, so the marginal of every output coordinate is
   itself approximately ``N(0, 1)``.
3. Lloyd-Max scalar quantization: each rotated coordinate is mapped to its
   nearest centroid in a hardcoded Lloyd-Max codebook for ``N(0, 1)``. The
   resulting integer codes are bit-packed into dense uint8 tensors.

Per-channel ``mean`` and ``std`` are stored as ``bfloat16`` alongside the
packed codes. They contribute a constant ``2 * head_dim * 2`` byte overhead
per quantize call, independent of how many vectors are being compressed -
unlike a per-vector L2-norm scheme whose overhead grows linearly with batch
size.

This module implements only the stateless quantize/dequantize primitives; the
residual-buffer and update bookkeeping live in :class:`PolarQuantizedLayer` in
``cache_utils.py``.

References:
    - Max, "Quantizing for Minimum Distortion", IEEE TIT, 1960.
    - Issue #45203 for the design discussion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..utils import is_torch_available


if is_torch_available():
    import torch


# ═══════════════════════════════════════════════════════════════════
# Lloyd-Max optimal centroids for N(0, 1)
#
# Precomputed offline with 200 iterations of Lloyd-Max on a standard
# normal distribution. These values are deterministic and well-known
# in the scalar quantization literature (Max 1960, Lloyd 1982).
# Hardcoding them removes any need for scipy at runtime.
# ═══════════════════════════════════════════════════════════════════

_CENTROIDS: dict[int, list[float]] = {
    2: [
        -1.5104176085,
        -0.4527800346,
        +0.4527800346,
        +1.5104176085,
    ],
    3: [
        -2.1519457045,
        -1.3439092785,
        -0.7560052812,
        -0.2450941789,
        +0.2450941789,
        +0.7560052812,
        +1.3439092785,
        +2.1519457045,
    ],
    4: [
        -2.7325895879,
        -2.0690172456,
        -1.6180464055,
        -1.2562312157,
        -0.9423404724,
        -0.6567591308,
        -0.3880483072,
        -0.1283950325,
        +0.1283950325,
        +0.3880483072,
        +0.6567591308,
        +0.9423404724,
        +1.2562312157,
        +1.6180464055,
        +2.0690172456,
        +2.7325895879,
    ],
    5: [
        -3.2674081093,
        -2.6986874661,
        -2.3258516118,
        -2.0371566611,
        -1.7957846116,
        -1.5847249154,
        -1.3946143602,
        -1.2196957216,
        -1.0561414115,
        -0.9012504182,
        -0.7530217640,
        -0.6099091973,
        -0.4706705565,
        -0.3342696950,
        -0.1998088561,
        -0.0664790561,
        +0.0664790561,
        +0.1998088561,
        +0.3342696950,
        +0.4706705565,
        +0.6099091973,
        +0.7530217640,
        +0.9012504182,
        +1.0561414115,
        +1.2196957216,
        +1.3946143602,
        +1.5847249154,
        +1.7957846116,
        +2.0371566611,
        +2.3258516118,
        +2.6986874661,
        +3.2674081093,
    ],
}

_centroid_tensor_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
_hadamard_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def get_centroids(nbits: int, device, dtype) -> torch.Tensor:
    """Return the hardcoded Lloyd-Max centroids for a given bit-width.

    Args:
        nbits: Number of bits per code. One of ``{2, 3, 4, 5}``.
        device: Torch device to place the tensor on.
        dtype: Torch dtype of the returned tensor.

    Returns:
        A 1-D tensor of length ``2 ** nbits`` containing the optimal centroids
        for a unit-variance Gaussian, sorted in ascending order.
    """
    if nbits not in _CENTROIDS:
        raise ValueError(f"PolarQuant centroids only defined for nbits in {{2, 3, 4, 5}}, got {nbits}")
    key = (nbits, str(device), dtype)
    cached = _centroid_tensor_cache.get(key)
    if cached is not None:
        return cached
    t = torch.tensor(_CENTROIDS[nbits], device=device, dtype=dtype)
    _centroid_tensor_cache[key] = t
    return t


# ═══════════════════════════════════════════════════════════════════
# Walsh-Hadamard matrix (Sylvester construction)
# ═══════════════════════════════════════════════════════════════════


def build_hadamard(n: int, device, dtype) -> torch.Tensor:
    """Construct an ``n × n`` orthogonal Walsh-Hadamard matrix.

    The matrix is built via the recursive Sylvester construction and normalized
    by ``1 / sqrt(n)`` so that ``H H^T = I``. Results are cached per
    ``(n, device, dtype)``.

    Args:
        n: Matrix side. Must be a power of two.
        device: Torch device.
        dtype: Torch floating-point dtype.

    Returns:
        An orthogonal Hadamard matrix of shape ``(n, n)``.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard dimension must be a positive power of two, got {n}")
    key = (n, str(device), dtype)
    cached = _hadamard_cache.get(key)
    if cached is not None:
        return cached

    def _build(sz: int) -> torch.Tensor:
        if sz == 1:
            return torch.tensor([[1.0]], dtype=torch.float64)
        half = _build(sz // 2)
        top = torch.cat([half, half], dim=1)
        bot = torch.cat([half, -half], dim=1)
        return torch.cat([top, bot], dim=0) / math.sqrt(2)

    H = _build(n).to(device=device, dtype=dtype)
    _hadamard_cache[key] = H
    return H


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two greater than or equal to ``n``."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ═══════════════════════════════════════════════════════════════════
# Bit-packing for 2/3/4/5-bit codes
#
# All layouts pack codes densely with no wasted bits at the alignment
# boundary. Input codes are (N, D) integer tensors with values in
# [0, 2^nbits - 1]; outputs are (N, packed_bytes) uint8 tensors.
# ═══════════════════════════════════════════════════════════════════


class BitPacker:
    """Dense bit-packing / unpacking for 2/3/4/5-bit code tensors."""

    @staticmethod
    def pack(codes: torch.Tensor, nbits: int) -> torch.Tensor:
        """Pack ``(N, D)`` integer codes into a ``(N, packed_bytes)`` uint8 tensor."""
        if codes.ndim != 2:
            raise ValueError(f"Expected 2-D codes tensor, got shape {tuple(codes.shape)}")
        N, D = codes.shape
        # Empty input — return an appropriately-shaped empty uint8 tensor.
        if N == 0:
            return torch.empty((0, BitPacker.packed_bytes(D, nbits)), dtype=torch.uint8, device=codes.device)
        c = codes.long()

        if nbits == 2:
            if D % 4 != 0:
                raise ValueError(f"nbits=2 requires D divisible by 4, got {D}")
            c = c.reshape(N, -1, 4)
            return ((c[:, :, 0] << 6) | (c[:, :, 1] << 4) | (c[:, :, 2] << 2) | c[:, :, 3]).to(torch.uint8)

        if nbits == 3:
            if D % 8 != 0:
                raise ValueError(f"nbits=3 requires D divisible by 8, got {D}")
            c = c.reshape(N, -1, 8)
            c0, c1, c2, c3, c4, c5, c6, c7 = (c[:, :, i] for i in range(8))
            b0 = (c0 << 5) | (c1 << 2) | (c2 >> 1)
            b1 = ((c2 & 1) << 7) | (c3 << 4) | (c4 << 1) | (c5 >> 2)
            b2 = ((c5 & 3) << 6) | (c6 << 3) | c7
            return torch.stack([b0, b1, b2], dim=-1).reshape(N, -1).to(torch.uint8)

        if nbits == 4:
            if D % 2 != 0:
                raise ValueError(f"nbits=4 requires D divisible by 2, got {D}")
            return ((c[:, 0::2] << 4) | c[:, 1::2]).to(torch.uint8)

        if nbits == 5:
            if D % 8 != 0:
                raise ValueError(f"nbits=5 requires D divisible by 8, got {D}")
            c = c.reshape(N, -1, 8)
            c0, c1, c2, c3, c4, c5, c6, c7 = (c[:, :, i] for i in range(8))
            # 8 codes × 5 bits = 40 bits = 5 bytes per group
            b0 = ((c0 & 0x1F) << 3) | ((c1 >> 2) & 0x07)
            b1 = ((c1 & 0x03) << 6) | ((c2 & 0x1F) << 1) | ((c3 >> 4) & 0x01)
            b2 = ((c3 & 0x0F) << 4) | ((c4 >> 1) & 0x0F)
            b3 = ((c4 & 0x01) << 7) | ((c5 & 0x1F) << 2) | ((c6 >> 3) & 0x03)
            b4 = ((c6 & 0x07) << 5) | (c7 & 0x1F)
            return torch.stack([b0, b1, b2, b3, b4], dim=-1).reshape(N, -1).to(torch.uint8)

        raise ValueError(f"Unsupported nbits: {nbits}")

    @staticmethod
    def unpack(packed: torch.Tensor, nbits: int, D: int) -> torch.Tensor:
        """Unpack a ``(N, packed_bytes)`` uint8 tensor into ``(N, D)`` long codes."""
        if packed.ndim != 2:
            raise ValueError(f"Expected 2-D packed tensor, got shape {tuple(packed.shape)}")
        N = packed.shape[0]
        # Empty input — return an appropriately-shaped empty long tensor.
        if N == 0:
            return torch.empty((0, D), dtype=torch.int64, device=packed.device)
        p = packed.long()

        if nbits == 2:
            return torch.stack([(p >> 6) & 3, (p >> 4) & 3, (p >> 2) & 3, p & 3], dim=-1).reshape(N, D)

        if nbits == 3:
            p3 = p.reshape(N, -1, 3)
            b0, b1, b2 = p3[:, :, 0], p3[:, :, 1], p3[:, :, 2]
            return torch.stack(
                [
                    (b0 >> 5) & 7,
                    (b0 >> 2) & 7,
                    ((b0 & 3) << 1) | ((b1 >> 7) & 1),
                    (b1 >> 4) & 7,
                    (b1 >> 1) & 7,
                    ((b1 & 1) << 2) | ((b2 >> 6) & 3),
                    (b2 >> 3) & 7,
                    b2 & 7,
                ],
                dim=-1,
            ).reshape(N, D)

        if nbits == 4:
            return torch.stack([(p >> 4) & 0xF, p & 0xF], dim=-1).reshape(N, D)

        if nbits == 5:
            p5 = p.reshape(N, -1, 5)
            b0, b1, b2, b3, b4 = (p5[:, :, i] for i in range(5))
            c0 = (b0 >> 3) & 0x1F
            c1 = ((b0 & 0x07) << 2) | ((b1 >> 6) & 0x03)
            c2 = (b1 >> 1) & 0x1F
            c3 = ((b1 & 0x01) << 4) | ((b2 >> 4) & 0x0F)
            c4 = ((b2 & 0x0F) << 1) | ((b3 >> 7) & 0x01)
            c5 = (b3 >> 2) & 0x1F
            c6 = ((b3 & 0x03) << 3) | ((b4 >> 5) & 0x07)
            c7 = b4 & 0x1F
            return torch.stack([c0, c1, c2, c3, c4, c5, c6, c7], dim=-1).reshape(N, D)

        raise ValueError(f"Unsupported nbits: {nbits}")

    @staticmethod
    def packed_bytes(D: int, nbits: int) -> int:
        """Return the number of packed bytes for a single ``D``-length vector."""
        if nbits == 2:
            return D // 4
        if nbits == 3:
            return (D // 8) * 3
        if nbits == 4:
            return D // 2
        if nbits == 5:
            return (D // 8) * 5
        raise ValueError(f"Unsupported nbits: {nbits}")


# ═══════════════════════════════════════════════════════════════════
# PolarQTensor — packed-tensor representation
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PolarQTensor:
    """Compressed representation of a KV tensor under PolarQuant.

    Attributes:
        packed: uint8 tensor of shape ``(N, packed_bytes)`` holding the bit-packed
            Lloyd-Max indices for every ``head_dim``-length vector.
        mean: bfloat16 tensor of shape ``(padded_dim,)`` with the per-channel
            mean computed over the batch of vectors being quantized.
        std: bfloat16 tensor of shape ``(padded_dim,)`` with the per-channel
            standard deviation. Used to invert the z-score during dequantization.
        shape: The original tensor shape prior to flattening (used to restore the
            output to the expected ``(B, H, S, D)`` layout after dequantization).
        nbits: The bit-width used to encode ``packed``. One of ``{2, 3, 4, 5}``.
    """

    packed: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    shape: tuple
    nbits: int


# ═══════════════════════════════════════════════════════════════════
# Quantize / Dequantize
# ═══════════════════════════════════════════════════════════════════


def polarquant_quantize(
    tensor: torch.Tensor,
    head_dim: int,
    padded_dim: int,
    nbits: int,
    centroids: torch.Tensor,
    hadamard: torch.Tensor,
    chunk_size: int = 4096,
) -> PolarQTensor:
    """Quantize a KV tensor to a :class:`PolarQTensor`.

    Pipeline:
    1. Reshape to ``(N, head_dim)`` and zero-pad to ``padded_dim`` when
       ``head_dim`` is not a power of two.
    2. Per-channel z-score: subtract the per-channel mean, divide by the
       per-channel standard deviation. Each channel becomes ``~ N(0, 1)``.
    3. Hadamard rotation: linearly mixes the components so each rotated
       coordinate has approximately the unit-Gaussian marginal that the
       Lloyd-Max codebook expects.
    4. Lloyd-Max nearest-centroid mapping per coordinate.
    5. Bit-pack the integer codes.

    Args:
        tensor: Input float tensor whose last dimension equals ``head_dim``.
        head_dim: The original (pre-padding) head dimension.
        padded_dim: The padded dimension used for the Hadamard rotation. Must be
            equal to ``head_dim`` when ``head_dim`` is a power of two, otherwise
            the next power of two.
        nbits: Number of bits per code. One of ``{2, 3, 4, 5}``.
        centroids: Lloyd-Max centroids for ``N(0, 1)``, shape ``(2 ** nbits,)``.
        hadamard: Orthogonal Hadamard matrix of shape ``(padded_dim, padded_dim)``.
        chunk_size: Row-wise chunk size for the nearest-centroid search. Controls
            a space / time tradeoff during quantization.

    Returns:
        A :class:`PolarQTensor` capturing the packed codes, per-channel mean and
        standard deviation, and the original shape.
    """
    orig_shape = tuple(tensor.shape)
    device = tensor.device

    flat = tensor.reshape(-1, head_dim).to(torch.float32)
    N = flat.shape[0]

    if padded_dim != head_dim:
        pad = torch.zeros(N, padded_dim - head_dim, device=device, dtype=flat.dtype)
        flat = torch.cat([flat, pad], dim=1)

    if N == 0:
        return PolarQTensor(
            packed=torch.empty((0, BitPacker.packed_bytes(padded_dim, nbits)), dtype=torch.uint8, device=device),
            mean=torch.zeros(padded_dim, dtype=torch.bfloat16, device=device),
            std=torch.ones(padded_dim, dtype=torch.bfloat16, device=device),
            shape=orig_shape,
            nbits=nbits,
        )

    # Per-channel z-score. ``unbiased=False`` so the std reduces to zero (and
    # is then clamped to a small constant) when there is only one vector,
    # rather than producing NaN.
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp(min=1e-6)
    normalized = (flat - mean) / std

    # Hadamard rotation. With per-channel-Gaussian inputs and ``H`` having
    # entries ``±1/sqrt(padded_dim)``, every rotated coordinate has variance 1
    # by linearity, so the codebook prior matches without an extra rescale.
    rotated = normalized @ hadamard

    # Nearest-centroid search in chunks to keep the temporary
    # (chunk, padded_dim, 2^nbits) distance tensor bounded in memory.
    codes = torch.empty(N, padded_dim, dtype=torch.int64, device=device)
    ct = centroids.view(1, 1, -1).to(device=device, dtype=torch.float32)
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        codes[i:j] = (rotated[i:j].unsqueeze(-1) - ct).abs().argmin(-1)

    packed = BitPacker.pack(codes, nbits)

    return PolarQTensor(
        packed=packed,
        mean=mean.to(torch.bfloat16),
        std=std.to(torch.bfloat16),
        shape=orig_shape,
        nbits=nbits,
    )


def polarquant_dequantize(
    qtensor: PolarQTensor,
    head_dim: int,
    padded_dim: int,
    centroids: torch.Tensor,
    hadamard: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct a dense tensor from a :class:`PolarQTensor`.

    The inverse pipeline is:
    1. Unpack and look up Lloyd-Max centroids.
    2. Apply the Hadamard matrix again. Walsh-Hadamard is symmetric and
       orthogonal so the inverse equals the matrix itself.
    3. Invert the per-channel z-score by multiplying by the stored std and
       adding the stored mean.
    4. Slice off the zero padding (if any) and reshape to the original shape.

    Args:
        qtensor: Packed representation produced by :func:`polarquant_quantize`.
        head_dim: Original head dimension.
        padded_dim: Padded dimension used during quantization.
        centroids: Lloyd-Max centroid lookup table.
        hadamard: Hadamard matrix used during quantization.
        output_dtype: Dtype of the returned dense tensor.

    Returns:
        A dense tensor with shape equal to ``qtensor.shape`` and dtype
        ``output_dtype``.
    """
    if qtensor.packed.shape[0] == 0:
        return torch.empty(qtensor.shape, dtype=output_dtype, device=qtensor.packed.device)

    codes = BitPacker.unpack(qtensor.packed, qtensor.nbits, padded_dim)
    values = centroids[codes]

    # Inverse Hadamard (orthogonal symmetric matrix), then inverse z-score.
    values = values @ hadamard
    values = values * qtensor.std.to(torch.float32) + qtensor.mean.to(torch.float32)

    if padded_dim != head_dim:
        values = values[:, :head_dim]

    return values.to(output_dtype).reshape(qtensor.shape)
