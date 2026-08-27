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
import torch

from ...utils import is_kernels_available, logging
from .dequant import dequantize


logger = logging.get_logger(__name__)

_gguf_kernel = None

# Upstream's MMVQ_MAX_BATCH_SIZE, which the kernel publishes as `MAX_GEMV_ROWS` too. Kept here as a
# constant because `GgufLinear.forward` needs it at trace time, before any kernel is resolved.
MAX_GEMV_ROWS = 8
DEQUANT_CHUNK_ELEMS = 64 << 20  # ~128 MB of bf16 per unpacked chunk


def mul_mat_vec(weight: torch.Tensor, x: torch.Tensor, ggml_type: int, out_features: int) -> torch.Tensor:
    """(N, bytes_per_row) uint8 @ (M, K) -> (M, N). May return f32 whatever `x` is.

    Not wrapped as a custom op: the kernel's ops are registered ops with their own fakes, so dynamo
    traces straight through them.
    """
    return _gguf_kernel.mul_mat_vec(weight, x, ggml_type, out_features)


def kernel_can_read(tensor) -> bool:
    """Whether the resolved kernel can read this tensor's memory.

    False when there is no kernel -- only reachable while converting weights at load, since a module
    that holds blocks is only built where a kernel can compute on them -- or when an explicit
    `device_map` left this weight on the host, which is the one place an accelerator build cannot
    reach. Asked instead of naming a backend, so whichever one was built for is used without saying
    which.

    Compared against the `False` sentinel rather than asked for truthiness: an embedding asks this in
    its forward, and dynamo cannot trace `bool()` on a `GgufKernel`, which breaks the graph at every
    lookup and makes `fullgraph=True` a hard error.
    """
    return get_gguf_kernel() is not False and tensor.device.type != "cpu"


def dequantize_blocks(weight, ggml_type: int, rows: int, cols: int, dtype, indices=None):
    """(rows, bytes_per_row) uint8 -> (rows, cols) `dtype`, a row chunk at a time.

    `indices` unpacks only the rows it names, gathering as it goes -- ggml's `get_rows`. A caller
    reading a handful of rows out of a large table should pass them rather than slicing first, which
    would copy their packed bytes for nothing.

    ggml's kernel when the blocks are on its device: it writes `dtype` straight out, where the torch
    dequantizer produces f32 and leaves the caller to cast — fewer ops and half the transient for a
    bf16 model. The torch path covers the blocks no kernel can read, which is how a device with no
    kernel published for it still loads and runs.
    """
    if indices is not None:
        return _dequantize_chunk(weight, ggml_type, rows, cols, dtype, indices)
    rows_per_chunk = max(1, DEQUANT_CHUNK_ELEMS // cols)
    if rows <= rows_per_chunk:
        return _dequantize_chunk(weight, ggml_type, rows, cols, dtype)
    out = torch.empty((rows, cols), dtype=dtype, device=weight.device)
    for start in range(0, rows, rows_per_chunk):
        chunk = weight[start : start + rows_per_chunk]
        out[start : start + rows_per_chunk] = _dequantize_chunk(chunk, ggml_type, chunk.shape[0], cols, dtype)
    return out


def _dequantize_chunk(weight, ggml_type: int, rows: int, cols: int, dtype, indices=None):
    if kernel_can_read(weight):
        if indices is not None:
            return _gguf_kernel.get_rows(weight, indices, ggml_type, cols, dtype)
        return _gguf_kernel.dequantize(weight, ggml_type, rows, cols, dtype)
    # No kernel: gather the packed rows first, then unpack the copy.
    if indices is not None:
        weight = weight.index_select(0, indices)
    return dequantize(weight, ggml_type, dtype).reshape(rows, cols)


class GgufKernel:
    """A resolved kernel: its two ops, and which quant types it implements a gemv for."""

    def __init__(self, module):
        self.mul_mat_vec = module.mul_mat_vec
        self.dequantize = module.dequantize
        self.get_rows = module.get_rows
        self.gemv_types = module.GEMV_TYPES

    def supports(self, ggml_type: int) -> bool:
        return ggml_type in self.gemv_types


def get_gguf_kernel() -> "GgufKernel | bool":
    """The published kernel for this machine, or `False` if there is none (so the caller unpacks).

    Hub-only: a kernel is a published `kernel-builder` build or nothing. `kernels` picks the variant
    for the current torch build and device itself.
    """
    global _gguf_kernel
    if _gguf_kernel is not None:
        return _gguf_kernel

    try:
        from ..hub_kernels import get_kernel

        module = get_kernel(
            "marcsun13/ggml-quantization",
            version=1,
            # TODO: move the kernels under `kernels-community` and drop this.
            allow_all_kernels=True,
        )
        _gguf_kernel = GgufKernel(module)
    except Exception as error:  # noqa: BLE001
        logger.info(
            f"No GGUF kernel available ({error}). The weights stay packed and every forward unpacks "
            "them, which costs speed but not memory."
        )
        _gguf_kernel = False
    return _gguf_kernel


def get_ggml_layer_mapping() -> dict:
    """The layers ggml has a fused kernel for, in `kernels`' mapping form."""
    from kernels import LayerRepository, Mode

    return {
        "Qwen3_5GatedDeltaNet": {
            "mps": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="marcsun13/ggml-gated-delta-net",
                    layer_name="Qwen3_5GatedDeltaNet",
                    version=1,
                    # TODO: drop once the kernels move under an allow-listed trusted publisher
                    trust_remote_code=True,
                )
            },
        },
    }


def kernelize_ggml_layers(model) -> None:
    """Graft ggml's layer kernels onto a GGUF model, and nothing else.

    Applied on load, and registered so that a later `kernelize` re-applies them: that walk restores the
    original forward of every layer it has no mapping for, which would otherwise undo this graft.
    """
    if not is_kernels_available():
        return

    from kernels import Mode, kernelize, register_kernel_mapping, use_kernel_mapping

    mapping = get_ggml_layer_mapping()
    register_kernel_mapping(mapping)
    try:
        # `inherit_mapping=False` narrows what `kernelize` can see to ggml's own layers. Inheriting would
        # add everything else registered for the device -- by another library, a user, a later transformers
        # release -- so this would fetch and run kernels nobody asked it for, and a failure in any of them
        # would take ggml's own down with it.
        with use_kernel_mapping(mapping, inherit_mapping=False):
            kernelize(model, mode=Mode.INFERENCE, device=model.device.type)
    except Exception as error:  # noqa: BLE001
        # `kernelize` stops where it raises, so with more than one entry this can leave the layers it had
        # already reached grafted. Harmless -- each is a drop-in for the implementation it replaced.
        logger.info(f"ggml layer kernels not fully grafted ({error}); the rest keeps the model's own layers.")
