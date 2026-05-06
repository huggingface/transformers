"""Pinpoint the DSv3-on-B200 NaN.

DeepGEMM's `pack_fp32_into_ue8m0` doesn't convert fp32 to UE8M0 — it expects
the fp32 input to *already* be UE8M0-rounded (each value an exact power of 2,
mantissa bits all zero) and just repacks the exponent bytes. The kernel's
inner shifts (`>> 23`, `>> 15`, `>> 7`, `<< 1`) only cleanly extract the
biased exponent for the first lane; the rest leak mantissa bits into adjacent
byte slots when the mantissa isn't zero.

This script verifies that on raw arbitrary fp32 SFs (kernel output diverges
from a "biased-exponent only" reference) but matches byte-for-byte once the
input is rounded to powers of 2 via `ceil_to_ue8m0`.

Implication: on SM100 the kernel's `(FP32, x, gran_k)` → packed-int path
silently corrupts SFs unless the caller pre-rounds them. SM90 sidesteps this
because its FP8 path consumes raw fp32 SFs directly without going through
`pack_fp32_into_ue8m0`.

Run on H100 and B200; both should print:
  raw   → DIVERGES (kernel needs UE8M0-rounded inputs).
  ue8m0 → MATCHES.
"""

from __future__ import annotations

import sys

import torch

from transformers.integrations.deepgemm import _load_deepgemm_kernel


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round each positive float up to the nearest power of 2 representable as
    UE8M0 (mantissa zeroed out). Mirrors upstream's `deep_gemm.utils.math`.
    """
    return (
        (x.view(torch.int32) + ((1 << 23) - 1)).bitwise_and_(~((1 << 23) - 1)).view(torch.float)
    )


def python_pack_exponent_only(sf_fp32: torch.Tensor) -> torch.Tensor:
    """Reference assuming UE8M0 input: extract biased exponent (bits [30:23])
    of each float as a uint8, then pack 4 K-consecutive bytes into one int32
    LE. This *only* matches the kernel when the input has zero mantissa.
    """
    byte = (sf_fp32.contiguous().view(torch.int32) >> 23).to(torch.uint8)
    *batch, mn, k = byte.shape
    assert k % 4 == 0
    g = byte.view(*batch, mn, k // 4, 4).to(torch.int32)
    return g[..., 0] | (g[..., 1] << 8) | (g[..., 2] << 16) | (g[..., 3] << 24)


def compare(label: str, kernel_out: torch.Tensor, ref_out: torch.Tensor) -> int:
    if kernel_out.shape != ref_out.shape:
        kernel_out = kernel_out[..., : ref_out.size(-2), : ref_out.size(-1)].contiguous()
    py_b = ref_out.contiguous().view(torch.uint8).flatten()
    k_b = kernel_out.contiguous().view(torch.uint8).flatten()
    n = min(py_b.numel(), k_b.numel())
    diff = (py_b[:n] != k_b[:n]).sum().item()
    status = "MATCH" if diff == 0 else "DIVERGE"
    print(f"  [{label}]  diff_bytes={diff}/{n}  ({status})")
    return diff


def main() -> int:
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cap = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}  SM{cap[0]}{cap[1]}")

    dg = _load_deepgemm_kernel()
    torch.manual_seed(0)

    # One representative SF tensor — what the integration would feed for
    # block-quantized weight SFs in DSv3 inference.
    sf_raw = (torch.rand(4, 8, 8, device=device) * 0.05 + 0.001).to(torch.float32)
    sf_ue8m0 = ceil_to_ue8m0(sf_raw)

    print("\nfp32 SF → kernel pack vs python `extract biased exponent and pack 4` reference:\n")

    diff_raw = compare(
        "raw fp32 SF (mantissa != 0)",
        dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_raw),
        python_pack_exponent_only(sf_raw),
    )
    diff_ue8m0 = compare(
        "ceil_to_ue8m0 fp32 SF (mantissa == 0)",
        dg.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf_ue8m0),
        python_pack_exponent_only(sf_ue8m0),
    )

    print()
    print("Conclusion:")
    print(
        f"  raw:    {'DIVERGES' if diff_raw else 'MATCHES'}  "
        "(kernel reads mantissa bits when not zero)"
    )
    print(
        f"  ue8m0:  {'DIVERGES' if diff_ue8m0 else 'MATCHES'}  "
        "(kernel cleanly repacks exponent bytes)"
    )

    if diff_ue8m0 != 0:
        print("\nUnexpected: pack diverges even with UE8M0-rounded input — that is a real kernel bug.")
        return 1
    if diff_raw == 0:
        print("\nUnexpected: kernel matches without UE8M0 rounding — investigate.")
        return 1
    print("\nFix: in our integration, round float SFs via ceil_to_ue8m0 before passing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
