"""Minimal reproducer for a NaN that appears on B200 (SM100) but not on H100
(SM90) when calling DeepGEMM's `m_grouped_fp8_fp4_gemm_nt_contiguous` with
float32 scale factors (DSv3-style block-quantized SFs).

Setup:
  * FP8 weights (E, N, K) cast from a real bf16 tensor with proper
    per-(128, 128)-block amax scaling — i.e., dequant_w ≈ original.
  * Per-token FP8 activations (M, K) with proper float32 SFs.
  * Block-quantized float32 weight SF of shape (E, N/128, K/128).

Path:
  * On SM90: kernel uses the `(FP32, 128, 128)` recipe directly, dispatches
    `sm90_m_grouped_fp8_gemm_contiguous_1d2d`. Output is finite.
  * On SM100: kernel converts float SF → packed UE8M0 int32 internally via
    `index_select(broadcast)` + `transpose_and_pack_fp32_into_ue8m0`, then
    dispatches `sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d`. Output has
    millions of NaNs.

To run:
  H100:  python repro_nan_dsv3_b200.py
  B200:  CUDA_HOME=$HOME/cuda-12.9 python repro_nan_dsv3_b200.py
"""

from __future__ import annotations

import sys

import torch

from transformers.integrations.deepgemm import _load_deepgemm_kernel


_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max  # 448.0


def make_grouped_fp8(e: int, n: int, k: int, block_n: int, block_k: int, device):
    """Per-block-amax FP8 quantization. Returns (w_fp8, sf_float32)."""
    w_fp32 = torch.randn(e, n, k, device=device) * 0.1
    sf_n, sf_k = n // block_n, k // block_k
    blocks = w_fp32.view(e, sf_n, block_n, sf_k, block_k)
    amax = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-4)  # (e, sf_n, sf_k)
    sf = (amax / _FP8_MAX).to(torch.float32)
    sf_expanded = (
        sf.view(e, sf_n, 1, sf_k, 1).expand(-1, -1, block_n, -1, block_k).reshape(e, n, k)
    )
    w_fp8 = (w_fp32 / sf_expanded).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
    return w_fp8, sf


def make_per_token_fp8(x_bf16: torch.Tensor, gran_k: int = 128):
    """Per-row amax FP8 quantization. Returns (x_fp8, sf_float32)."""
    m, n = x_bf16.shape
    assert n % gran_k == 0
    x_view = x_bf16.float().view(m, n // gran_k, gran_k)
    amax = x_view.abs().amax(dim=2).clamp(min=1e-4)  # (m, n/gran_k)
    sf = (amax / _FP8_MAX).to(torch.float32)
    x_fp8 = (x_view / sf.unsqueeze(2)).clamp(-_FP8_MAX, _FP8_MAX).view(m, n).to(_FP8)
    return x_fp8, sf


def to_mn_major(sf: torch.Tensor) -> torch.Tensor:
    """Rewrite SF to MN-major + TMA-aligned strides (kernel requires this)."""
    elem = sf.element_size()
    align = 16 // elem
    mn = sf.size(-2)
    aligned_mn = -(-mn // align) * align
    if sf.dim() == 2:
        target = (1, aligned_mn)
    elif sf.dim() == 3:
        target = (sf.size(-1) * aligned_mn, 1, aligned_mn)
    else:
        raise ValueError(sf.dim())
    if tuple(sf.stride()) == target:
        return sf
    out = torch.empty_strided(sf.shape, target, dtype=sf.dtype, device=sf.device)
    out.copy_(sf)
    return out


def main() -> int:
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cap = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}  SM{cap[0]}{cap[1]}")

    dg = _load_deepgemm_kernel()
    torch.manual_seed(0)

    # Shapes — small but trigger the real codegen path.
    M, N, K, E = 512, 1024, 1024, 4
    block_n, block_k = 128, 128

    # FP8 weights with realistic per-(128,128)-block amax scales.
    w_fp8, w_sf_block = make_grouped_fp8(E, N, K, block_n, block_k, device)
    print(f"  w_fp8: {tuple(w_fp8.shape)} {w_fp8.dtype}, "
          f"w_sf_block: {tuple(w_sf_block.shape)} (block-quantized 128×128)")

    # Activations (one expert per token in this minimal case: round-robin).
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.1
    a_fp8, a_sf = make_per_token_fp8(x)
    print(f"  a_fp8: {tuple(a_fp8.shape)} {a_fp8.dtype}, "
          f"a_sf: {tuple(a_sf.shape)} per-token")

    # Grouped layout: equal split, per-row expert id (Hopper layout).
    grouped_layout = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=device), M // E
    )

    # Kernel call. Pass float32 SF to force the SM100 broadcast+pack path.
    d = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    dg.grouped_fp8_fp4_matmul(
        (a_fp8, to_mn_major(a_sf)),
        (w_fp8, to_mn_major(w_sf_block)),
        d,
        grouped_layout,
        # No `recipe` → kernel picks default `(1, 128, 128)` for (float, float).
        # No `use_psum_layout` → default False (per-row id grouped_layout).
    )

    nf = (~torch.isfinite(d)).sum().item()
    finite_pct = 100.0 * (1 - nf / d.numel())
    print(f"  output: shape={tuple(d.shape)}  "
          f"nonfinite={nf}/{d.numel()} ({100.0 - finite_pct:.2f}%)  "
          f"finite_pct={finite_pct:.2f}%")
    if nf > 0:
        print("  → REPRO: output has NaN/Inf.")
        return 1
    print("  → OK: output is finite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
