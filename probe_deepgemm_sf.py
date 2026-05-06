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
    print(
        f"  [#{_call_idx[0]}] in:  shape={tuple(sf.shape)} "
        f"stride={tuple(sf.stride())} dtype={sf.dtype}"
    )
    print(
        f"        out: shape={tuple(out.shape)} "
        f"stride={tuple(out.stride())} dtype={out.dtype}"
    )
    return out


di._coerce_sf_for_kernel = _verbose_coerce


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
