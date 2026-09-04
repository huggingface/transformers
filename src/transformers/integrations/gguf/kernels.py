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


def mul_mat_id(blocks: torch.Tensor, x: torch.Tensor, ids: torch.Tensor, ggml_type: int, out_features: int):
    """One dispatch for a bank of routed experts -- ggml's `mul_mv_id`.

    `blocks` is `(n_experts, out_features, bytes_per_row)`, `x` is `(n_tokens, in_features)`, `ids` is
    `(n_tokens, n_used)`; the result is `(n_tokens, n_used, out_features)` f32. The alternative is a
    gemv per expert per layer, which is dispatch overhead rather than arithmetic.
    """
    return _gguf_kernel.mul_mat_id(blocks, x, ids, ggml_type, out_features)


def mul_mat_vec(weight: torch.Tensor, x: torch.Tensor, ggml_type: int, out_features: int) -> torch.Tensor:
    """(N, bytes_per_row) uint8 @ (M, K) -> (M, N). May return f32 whatever `x` is."""
    return _gguf_kernel.mul_mat_vec(weight, x, ggml_type, out_features)


def kernel_can_read(tensor) -> bool:
    """Whether the resolved kernel can read this tensor's memory.

    Compared against the `False` sentinel, not asked for truthiness: dynamo cannot trace `bool()` on a
    `GgufKernel`, which breaks the graph at every lookup.
    """
    return get_gguf_kernel() is not False and tensor.device.type != "cpu"


def dequantize_blocks(weight, ggml_type: int, rows: int, cols: int, dtype, indices=None):
    """(rows, bytes_per_row) uint8 -> (rows, cols) `dtype`, a row chunk at a time."""
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
        self.mul_mat_id = module.mul_mat_id
        self.gemv_types = module.GEMV_TYPES

    def supports(self, ggml_type: int) -> bool:
        return ggml_type in self.gemv_types


def get_gguf_kernel() -> "GgufKernel | bool":
    """The published kernel for this machine, or `False` if there is none (so the caller unpacks)."""
    global _gguf_kernel
    if _gguf_kernel is not None:
        return _gguf_kernel

    try:
        from ..hub_kernels import get_kernel

        module = get_kernel("transformers-community/ggml-quantization", version=1)
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
        # Named for the weight convention rather than the model: this is the norm that computes
        # `x * (1 + w)`, which the plain `RMSNorm` kernels would get silently wrong.
        # Named for the scoring function: the kernel rewrites the routing maths, exact for softmax only.
        "SoftmaxTopKRouter": {
            "mps": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="kernels-staging/topk",
                    layer_name="SoftmaxTopKRouter",
                    # TODO: `version=1` once kernels-staging/topk#1124 is merged and tagged
                    revision="pr-1124",
                )
            },
        },
        "RMSNormZeroCentered": {
            "mps": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="transformers-community/ggml-norm",
                    layer_name="RMSNormZeroCentered",
                    version=1,
                )
            },
        },
        "Qwen3_5GatedDeltaNet": {
            "mps": {
                Mode.INFERENCE: LayerRepository(
                    repo_id="transformers-community/ggml-gated-delta-net",
                    layer_name="Qwen3_5GatedDeltaNet",
                    version=1,
                )
            },
        },
    }


def kernelize_ggml_layers(model) -> None:
    """Graft ggml's layer kernels onto a GGUF model, and nothing else."""
    if not is_kernels_available():
        return

    from kernels import Mode, kernelize, register_kernel_mapping, use_kernel_mapping

    # Resolve first and keep what answers: `kernelize` stops at the first entry it cannot fetch, which
    # would cost every layer after it.
    mapping = {}
    for layer_name, devices in get_ggml_layer_mapping().items():
        try:
            for modes in devices.values():
                for repo in modes.values():
                    repo.load()
        except Exception as error:  # noqa: BLE001
            logger.info(f"no ggml kernel for {layer_name} ({error}); it keeps the model's own layer.")
            continue
        mapping[layer_name] = devices
    if not mapping:
        return

    register_kernel_mapping(mapping)
    try:
        # `inherit_mapping=False` narrows what `kernelize` can see to ggml's own layers. Inheriting would
        # add everything else registered for the device -- by another library, a user, a later transformers
        # release -- so this would fetch and run kernels nobody asked it for, and a failure in any of them
        # would take ggml's own down with it.
        with use_kernel_mapping(mapping, inherit_mapping=False):
            kernelize(model, mode=Mode.INFERENCE, device=model.device.type)
    except Exception as error:  # noqa: BLE001
        # Every entry resolved above, so reaching here is a graft that failed rather than a fetch.
        logger.info(f"ggml layer kernels not fully grafted ({error}); the rest keeps the model's own layers.")
