"""Print the actual SF shapes / strides / dtypes the DeepGEMM integration feeds
into `m_grouped_fp8_fp4_gemm_nt_contiguous`, for each test case.

When the kernel rejects an SF with `check_sf_layout` assertions, you usually
can't tell from the message *which* SF (activation or weight) failed and what
its actual layout was. This wraps `_coerce_sf_for_kernel` to log every call,
then runs the smoke tests so you can see the exact tensor metadata that hit
the kernel boundary right before the assertion fired.

Usage:
    CUDA_HOME=$HOME/cuda-12.9 python probe_deepgemm_sf.py
"""

from __future__ import annotations

import sys

import torch

import test_deepgemm as t
from transformers.integrations import deepgemm as di


_real_coerce = di._coerce_sf_for_kernel
_call_idx = [0]


def _verbose_coerce(sf: torch.Tensor) -> torch.Tensor:
    out = _real_coerce(sf)
    _call_idx[0] += 1
    nonfinite_in = (~torch.isfinite(sf.float())).sum().item() if sf.is_floating_point() else 0
    nonfinite_out = (~torch.isfinite(out.float())).sum().item() if out.is_floating_point() else 0
    print(
        f"  [#{_call_idx[0]}] in:  shape={tuple(sf.shape)} "
        f"stride={tuple(sf.stride())} dtype={sf.dtype} "
        f"min={sf.float().abs().min().item():.3e} max={sf.float().abs().max().item():.3e} "
        f"nonfinite={nonfinite_in}"
    )
    print(
        f"        out: shape={tuple(out.shape)} "
        f"stride={tuple(out.stride())} dtype={out.dtype} "
        f"nonfinite={nonfinite_out}"
    )
    return out


di._coerce_sf_for_kernel = _verbose_coerce


# Wrap the matmul itself: print output stats after the call so we can see
# where NaN actually appears in the pipeline.
_real_matmul = None


def _verbose_matmul(*args, **kwargs):
    global _real_matmul
    out_tensor = args[2]  # (a_pair, b_pair, d, ...)
    label = f"matmul (d.shape={tuple(out_tensor.shape)})"
    _real_matmul(*args, **kwargs)
    nf = (~torch.isfinite(out_tensor)).sum().item()
    print(
        f"  → {label}: nonfinite_count={nf} "
        f"min_abs={out_tensor.abs().min().item():.3e} "
        f"max_abs={out_tensor.abs().max().item():.3e}"
    )


def _patch_matmul():
    global _real_matmul
    deepgemm = di._load_deepgemm_kernel()
    _real_matmul = deepgemm.grouped_fp8_fp4_matmul

    # Replace the cached kernel's matmul with our wrapper.
    object.__setattr__(deepgemm, "grouped_fp8_fp4_matmul", _verbose_matmul)


_patch_matmul()


def _run(name: str, fn) -> bool:
    print(f"\n=== {name} ===")
    _call_idx[0] = 0
    try:
        fn(d)
        print(f"  → PASS")
        return True
    except BaseException as exc:
        print(f"  → FAIL: {type(exc).__name__}: {str(exc)[:300]}")
        return False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")
    torch.cuda.set_device(0)
    d = torch.device("cuda", 0)
    print(
        f"GPU: {torch.cuda.get_device_name(d)}  "
        f"SM{''.join(str(x) for x in torch.cuda.get_device_capability(d))}"
    )

    results = [
        _run("test_dsv3_fp8", t.test_dsv3_fp8),
        _run("test_dsv4_fp8", t.test_dsv4_fp8),
        _run("test_dsv4_fp4", t.test_dsv4_fp4),
    ]
    sys.exit(0 if all(results) else 1)
