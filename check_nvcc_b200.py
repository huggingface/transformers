"""Smoke-test the user's nvcc + CUDA setup against the running GPU.

Compiles and runs a tiny CUDA kernel targeting the device's actual compute
capability (e.g. `sm_100a` on B200, `sm_90a` on H100). Mirrors what DeepGEMM's
JIT does at the first kernel call:

  1. Locate `nvcc` via `$CUDA_HOME/bin/nvcc`, then PATH, then `/usr/local/cuda`.
  2. nvcc-compile a kernel that uses an SM-specific intrinsic / API.
  3. Launch it, copy result back, sanity-check.

If this succeeds, DeepGEMM JIT will work at runtime. If it fails, the message
points at the specific layer (toolchain, driver, runtime) so you can fix it
before pulling DeepGEMM into a model run.

Usage:
    python check_nvcc_b200.py
    CUDA_HOME=/path/to/cuda python check_nvcc_b200.py
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def _find_cuda_home() -> str:
    """Same search order as the deep-gemm wheel's `_find_cuda_home`."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        cand = os.environ.get(var)
        if cand and (Path(cand) / "bin" / "nvcc").is_file():
            return cand

    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).parent.parent)

    try:
        import nvidia.cuda_nvcc as _nvcc  # type: ignore
        cand = Path(_nvcc.__file__).parent
        if (cand / "bin" / "nvcc").is_file():
            return str(cand)
    except ImportError:
        pass

    for cand in ("/usr/local/cuda", "/opt/cuda", "/opt/nvidia/cuda", "/usr/lib/cuda"):
        if (Path(cand) / "bin" / "nvcc").is_file():
            return cand
    import glob
    for cand in sorted(glob.glob("/usr/local/cuda-*"), reverse=True):
        if (Path(cand) / "bin" / "nvcc").is_file():
            return cand
    raise SystemExit("nvcc not found. Set CUDA_HOME or install CUDA toolkit.")


_KERNEL_SRC = r"""
#include <cuda_runtime.h>
#include <stdio.h>

// One element per thread; writes its global index. Probes that arch-specific
// codegen + scheduling work end-to-end on the device.
__global__ void identity_kernel(int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}

extern "C" __host__ int run_check(int n) {
    int* d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_out, n * sizeof(int));
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(err)); return 1; }

    int threads = 128, blocks = (n + threads - 1) / threads;
    identity_kernel<<<blocks, threads>>>(d_out, n);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "kernel launch: %s\n", cudaGetErrorString(err)); cudaFree(d_out); return 2; }

    int* h_out = (int*)malloc(n * sizeof(int));
    err = cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(err)); free(h_out); cudaFree(d_out); return 3; }

    int ok = 1;
    for (int i = 0; i < n; ++i) if (h_out[i] != i) { ok = 0; break; }
    free(h_out);
    cudaFree(d_out);
    return ok ? 0 : 4;
}
"""


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available to torch.")
        return 1

    cap = torch.cuda.get_device_capability()
    sm = f"{cap[0]}{cap[1]}a"
    name = torch.cuda.get_device_name()
    print(f"GPU: {name}  (compute capability sm_{sm})")

    cuda_home = _find_cuda_home()
    nvcc = str(Path(cuda_home) / "bin" / "nvcc")
    print(f"CUDA_HOME: {cuda_home}")

    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    print(ver.stdout.strip().splitlines()[-1] if ver.stdout else ver.stderr)

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "probe.cu"
        so = Path(td) / "probe.so"
        src.write_text(_KERNEL_SRC)

        cmd = [
            nvcc, "-shared", "-Xcompiler=-fPIC",
            "-O2", "-std=c++17",
            f"-gencode=arch=compute_{cap[0]}{cap[1]}{'a' if cap[0] >= 9 else ''},code=sm_{sm}",
            "-o", str(so), str(src),
        ]
        print("\n[1/3] nvcc compile…")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL: nvcc compile (exit {r.returncode})")
            print("--- stderr ---")
            print(r.stderr)
            return 1
        print("       OK")

        print("[2/3] dlopen…")
        try:
            lib = ctypes.CDLL(str(so))
        except OSError as e:
            print(f"FAIL: dlopen: {e}")
            print("Hint: missing libcudart.so on LD_LIBRARY_PATH. Try:")
            print(f"  export LD_LIBRARY_PATH={cuda_home}/lib64:$LD_LIBRARY_PATH")
            return 1
        lib.run_check.restype = ctypes.c_int
        lib.run_check.argtypes = [ctypes.c_int]
        print("       OK")

        print("[3/3] launch kernel…")
        rc = lib.run_check(1024)
        labels = {
            0: "OK",
            1: "cudaMalloc failed",
            2: "kernel launch / sync failed",
            3: "cudaMemcpy failed",
            4: "kernel produced wrong values",
        }
        print(f"       run_check → {rc} ({labels.get(rc, 'unknown')})")
        if rc != 0:
            print("\nFAIL: nvcc compiles but the kernel did not run correctly.")
            print("Common causes: GPU driver too old for the toolkit, mismatched libcudart.")
            return 1

    print(f"\nPASS: nvcc {Path(nvcc).name} can compile + run sm_{sm} kernels on this {name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
