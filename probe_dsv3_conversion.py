"""Compare the kernel's float→packed-UE8M0 conversion against a Python
equivalent, on the exact SF tensors that DSv3 feeds to the GEMM. Goal: find
out whether `transpose_and_pack_fp32_into_ue8m0` (the JIT helper the kernel
runs internally for DSv3 on SM100) produces values matching what we'd compute
ourselves.

If the kernel's output matches Python on every byte, the NaN comes from
somewhere else in the GEMM. If it doesn't match, the conversion itself is the
bug.
"""

from __future__ import annotations

import torch

from transformers.integrations.deepgemm import _load_deepgemm_kernel


def py_pack(sf_fp32: torch.Tensor) -> torch.Tensor:
    """Same as `pack_fp32_into_ue8m0`: extract the biased exponent (bits
    [30:23]) of each float as a uint8, then pack 4 K-consecutive bytes into
    one int32 (LSB = lowest K). Returns an MN-major int32 tensor.
    """
    # Extract biased exponent → uint8
    byte = (sf_fp32.view(torch.int32) >> 23).to(torch.uint8)
    # Reshape so K dim is divisible by 4
    *batch, mn, k = byte.shape
    assert k % 4 == 0
    # Pack each group of 4 K-bytes into 1 int32 in little-endian order
    grouped = byte.view(*batch, mn, k // 4, 4).to(torch.int32)
    packed = grouped[..., 0] | (grouped[..., 1] << 8) | (grouped[..., 2] << 16) | (grouped[..., 3] << 24)
    return packed  # shape (..., mn, k//4) K-major; caller should rewrite to MN-major


def main():
    torch.cuda.set_device(0)
    d = torch.device("cuda", 0)
    dg = _load_deepgemm_kernel()

    # Activation SF case: shape (M, K_blocks), per-row.
    print("=== activation SF: (3056, 8) float32 → kernel pack vs python pack ===")
    M, Kb = 3056, 8
    sf_a = (torch.rand(M, Kb, device=d) * 0.05 + 0.001).to(torch.float32)

    kernel_packed = dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_a)
    py_packed = py_pack(sf_a)
    print(f"  kernel out shape={tuple(kernel_packed.shape)} stride={tuple(kernel_packed.stride())} dtype={kernel_packed.dtype}")
    print(f"  python out shape={tuple(py_packed.shape)}")
    # Compare bytes
    kernel_bytes = kernel_packed.contiguous().view(torch.uint8).flatten()
    python_bytes = py_packed.contiguous().view(torch.uint8).flatten()
    n = min(kernel_bytes.numel(), python_bytes.numel())
    diff = (kernel_bytes[:n] != python_bytes[:n]).sum().item()
    print(f"  byte-equal count: {(n - diff)}/{n}  (diff={diff})")

    print("\n=== weight SF: (16, 8, 8) float32 → kernel broadcast+pack ===")
    E, sn, sk = 16, 8, 8
    N, K = 1024, 1024
    sf_w = (torch.rand(E, sn, sk, device=d) * 0.05 + 0.001).to(torch.float32)

    # Python broadcast: each block-row repeated 128 times along dim -2.
    sf_w_broadcast = sf_w.repeat_interleave(N // sn, dim=-2)  # (E, N, sk)
    py_packed_w = py_pack(sf_w_broadcast)  # (E, N, sk//4)
    print(f"  python broadcast+pack shape={tuple(py_packed_w.shape)}")

    # Kernel path: pass float SF to transform_sf_into_required_layout via the
    # recipe machinery. We don't have direct access; the closest helper is
    # the public one which expects per-row float input. Skip and just confirm
    # python pack is correct on the broadcasted form.

    # Sanity: print a slice of packed bytes for visual inspection.
    print(f"  python packed[0, 0, :] = {py_packed_w[0, 0, :].tolist()}")
    print(f"  python packed[0, 127, :] = {py_packed_w[0, 127, :].tolist()}  (same block as row 0)")
    print(f"  python packed[0, 128, :] = {py_packed_w[0, 128, :].tolist()}  (next block)")


if __name__ == "__main__":
    main()
