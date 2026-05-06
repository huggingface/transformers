"""Minimal isolated test of DeepGEMM's `get_mn_major_tma_aligned_packed_ue8m0_tensor`.

The full GEMM produces NaN on B200 with float32 SFs but is finite on H100.
Earlier I speculated the bug was in `transpose_and_pack_fp32_into_ue8m0` (the
JIT helper that converts float SF → packed UE8M0 int32 on SM100). This script
tests *only* that conversion against a byte-exact Python reference, with no
GEMM in the loop. Outcome:

  - bytes match on both archs   → conversion is fine, NaN is from the GEMM.
  - bytes diverge on B200       → confirmed conversion bug.

To run:
  H100:  python repro_nan_dsv3_b200.py
  B200:  CUDA_HOME=$HOME/cuda-12.9 python repro_nan_dsv3_b200.py
"""

from __future__ import annotations

import sys

import torch

from transformers.integrations.deepgemm import _load_deepgemm_kernel


def python_pack(sf_fp32: torch.Tensor) -> torch.Tensor:
    """Reference: extract biased exponent (bits [30:23]) of each float32 as a
    uint8, then pack 4 K-consecutive bytes into one int32 (LE, byte 0 = lowest
    K). Output shape: same as input but last dim shrunk 4×; layout: K-major.
    """
    byte = (sf_fp32.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    *batch, mn, k = byte.shape
    assert k % 4 == 0
    g = byte.view(*batch, mn, k // 4, 4).to(torch.int32)
    return g[..., 0] | (g[..., 1] << 8) | (g[..., 2] << 16) | (g[..., 3] << 24)


def compare(name: str, kernel_out: torch.Tensor, py_out: torch.Tensor) -> bool:
    # Compare byte-by-byte, ignoring TMA-alignment padding the kernel may add.
    *_, mn, kf = py_out.shape
    # Slice kernel output to the same shape (it can be wider in mn due to TMA align).
    if kernel_out.shape != py_out.shape:
        kernel_out = kernel_out[..., :mn, :kf].contiguous()
    py_bytes = py_out.contiguous().view(torch.uint8).flatten()
    k_bytes = kernel_out.contiguous().view(torch.uint8).flatten()
    n = min(py_bytes.numel(), k_bytes.numel())
    diff = (py_bytes[:n] != k_bytes[:n]).sum().item()
    print(f"  [{name}] kernel={tuple(kernel_out.shape)} stride={tuple(kernel_out.stride())} "
          f"py={tuple(py_out.shape)}  diff_bytes={diff}/{n}")
    return diff == 0


def main() -> int:
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cap = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}  SM{cap[0]}{cap[1]}")

    dg = _load_deepgemm_kernel()
    torch.manual_seed(0)

    cases = [
        ("act SF (per-token)",  (512, 8)),         # 2D, contig
        ("weight SF (grouped)", (4, 1024, 8)),     # 3D, per-row N
        ("weight SF (block)",   (4, 8, 8)),        # 3D, block-quant — needs broadcast in real path
    ]

    all_ok = True
    for name, shape in cases:
        sf = (torch.rand(*shape, device=device) * 0.05 + 0.001).to(torch.float32)
        kernel_out = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
        py_out = python_pack(sf)
        if not compare(name, kernel_out, py_out):
            all_ok = False

    if all_ok:
        print("\nResult: kernel pack matches Python reference byte-for-byte.")
        print("        → bug is NOT in `transpose_and_pack_fp32_into_ue8m0`.")
        return 0
    print("\nResult: kernel pack diverges from Python reference.")
    print("        → confirmed bug in `transpose_and_pack_fp32_into_ue8m0`.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
