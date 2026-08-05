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
"""Finding a kernel that can compute on GGUF blocks, or reporting that there is none.

Keeping weights in GGUF blocks only pays off with such a kernel, so this is what decides whether
the quantizer swaps modules at all. Resolution is a capability question, not a device check: the
kernel says which quant types it implements a gemv for, and that is what gets asked.

The kernels come from the hub repo for the device, built by `kernel-builder` from a pinned subset of
llama.cpp's ggml. If there is none for this device, the caller dequantizes at load instead.
"""

import torch

from ...utils import logging
from .dequant import dequantize


logger = logging.get_logger(__name__)

_HUB_REPOS = {"cuda": "marcsun13/gguf-kernels"}
# `kernels` requires the API version to be pinned, so a repo cannot change under a release
_KERNEL_VERSION = 1

# `kernels` waives this only for repos under a publisher the Hub marks as trusted, which a personal
# repo is not — so loading one means opting into running code from it.
# TODO: move the kernels under `kernels-community` and drop this.
_TRUST_REMOTE_CODE = True


# Resolved once, then called as plain module-level functions. Reaching an op through a method on
# an object stored on the module makes dynamo give up and graph-break at every quantized linear,
# which costs the entire compile speedup.
_MUL_MAT_VEC = None
_DEQUANTIZE = None
# the device type the resolved kernel was built for, so a caller can ask whether it can read a tensor
# without naming a backend: a weight that stayed on CPU or got offloaded is not one the kernel can touch
_KERNEL_DEVICE = None

# Upstream's MMVQ_MAX_BATCH_SIZE, which the kernel publishes as `MAX_GEMV_ROWS` too. Kept here as a
# constant because `GgufLinear.forward` needs it at trace time, before any kernel is resolved.
MAX_GEMV_ROWS = 8
DEQUANT_CHUNK_ELEMS = 64 << 20  # ~128 MB of bf16 per unpacked chunk


def mul_mat_vec(weight, x, ggml_type: int, out_features: int):
    """(N, bytes_per_row) uint8 @ (M, K) -> (M, N). May return f32 whatever `x` is."""
    return _MUL_MAT_VEC(weight, x, ggml_type, out_features)


def kernel_can_read(tensor) -> bool:
    """Whether the resolved kernel can read this tensor's memory.

    False when the tensor is not where the kernel can reach it: no kernel is published for its device —
    an MPS model while only a CUDA one exists — or a `device_map` put it on the host. Asked instead of
    naming a backend, so a kernel for a device other than CUDA is used wherever the CUDA one would be.
    """
    return _KERNEL_DEVICE is not None and tensor.device.type == _KERNEL_DEVICE


def dequantize_blocks(weight, ggml_type: int, rows: int, cols: int, dtype):
    """(rows, bytes_per_row) uint8 -> (rows, cols) `dtype`, a row chunk at a time.

    ggml's kernel when the blocks are on its device: it writes `dtype` straight out, where the torch
    dequantizer produces f32 and leaves the caller to cast — fewer ops and half the transient for a
    bf16 model. The torch path covers the blocks no kernel can read, which is how a device with no
    kernel published for it still loads and runs.

    Chunked because rows are quantized independently, so a row slice of the blocks stands alone, and
    unpacking a large weight in one go is a multi-GB transient — three times its final size on the
    torch path, which holds f32 and the cast at once.
    """
    rows_per_chunk = max(1, DEQUANT_CHUNK_ELEMS // cols)
    if rows <= rows_per_chunk:
        return _dequantize_chunk(weight, ggml_type, rows, cols, dtype)
    out = torch.empty((rows, cols), dtype=dtype, device=weight.device)
    for start in range(0, rows, rows_per_chunk):
        chunk = weight[start : start + rows_per_chunk]
        out[start : start + rows_per_chunk] = _dequantize_chunk(chunk, ggml_type, chunk.shape[0], cols, dtype)
    return out


def _dequantize_chunk(weight, ggml_type: int, rows: int, cols: int, dtype):
    if kernel_can_read(weight):
        return _DEQUANTIZE(weight, ggml_type, rows, cols, dtype)
    return dequantize(weight, ggml_type, dtype).reshape(rows, cols)


class GgufKernel:
    """A resolved kernel: which quant types it implements a gemv for, as it reported them."""

    def __init__(self, gemv_types):
        self.gemv_types = gemv_types

    def supports(self, ggml_type: int) -> bool:
        return ggml_type in self.gemv_types


def get_gguf_kernel(device_type: str | None = None) -> GgufKernel | None:
    """A kernel for this device, or `None` if there is none (so the caller unpacks at load)."""
    global _MUL_MAT_VEC, _DEQUANTIZE, _KERNEL_DEVICE
    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    repo = _HUB_REPOS.get(device_type)
    if repo is None:
        return None
    try:
        from kernels import get_kernel

        module = get_kernel(repo, version=_KERNEL_VERSION, trust_remote_code=_TRUST_REMOTE_CODE)
        # Both ops or neither: the modules call them without checking, since they are only ever
        # installed when this function returned a kernel. A build missing either raises here and is
        # reported as "no kernel", so the weights are unpacked at load instead.
        _MUL_MAT_VEC = module.mul_mat_vec
        _DEQUANTIZE = module.dequantize
        _KERNEL_DEVICE = device_type
        return GgufKernel(module.GEMV_TYPES)
    except Exception as error:  # noqa: BLE001
        logger.info(
            f"No GGUF kernel available for {device_type} ({error}). Weights will be dequantized at "
            "load time instead of kept in GGUF blocks."
        )
        return None
