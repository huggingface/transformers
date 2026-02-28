# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Metal affine quantization integration for transformers.

This module provides:
  - ``MetalLinear``: a drop-in replacement for ``nn.Linear`` that stores weights
    as affine-quantized uint32 packed tensors and uses the ``quantization-mlx``
    Metal kernels for the forward pass.
  - ``replace_with_metal_linear``: walks a model and swaps every eligible
    ``nn.Linear`` with ``MetalLinear``.
  - ``MetalQuantize`` / ``MetalDequantize``: weight conversion operations that
    participate in the new ``WeightConverter`` pipeline.

Weight layout (transposed, matching ``affine_qmm_t``):
  - ``weight``: ``[N, K_packed]`` (``uint32``) -- K is the packed dimension.
  - ``scales``:  ``[N, K // group_size]`` (``float16 / bfloat16``)
  - ``qbiases``: ``[N, K // group_size]`` (same dtype as scales)

The kernel call is ``affine_qmm_t(x, weight, scales, qbiases, group_size, bits)``
which computes ``y = x @ dequant(weight).T``, identical to ``nn.Linear``.
"""

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)

_metal_kernel = None

# ---------------------------------------------------------------------------
# Locally-compiled Metal fallback for affine_qmm_t
# ---------------------------------------------------------------------------

_AFFINE_QMM_T_METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;

// Fused dequantize + matmul:  y = x @ dequant(w).T
// x: [M, K], w: [N, K_packed] (uint32), scales/biases: [N, n_groups], out: [M, N]
// Each thread computes one (m, n) output element.

kernel void affine_qmm_t_float(
    device const float* x        [[buffer(0)]],
    device const uint*  w        [[buffer(1)]],
    device const float* scales   [[buffer(2)]],
    device const float* biases   [[buffer(3)]],
    device float*       out      [[buffer(4)]],
    constant uint& M             [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    constant uint& group_size    [[buffer(8)]],
    constant uint& bits          [[buffer(9)]],
    uint2 tid                    [[thread_position_in_grid]])
{
    uint m = tid.y;
    uint n = tid.x;
    if (m >= M || n >= N) return;

    uint elems_per_int = 32 / bits;
    uint mask = (1u << bits) - 1u;
    uint K_packed = K / elems_per_int;
    uint n_groups = K / group_size;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        uint packed_val = w[n * K_packed + k / elems_per_int];
        float q = float((packed_val >> ((k % elems_per_int) * bits)) & mask);
        uint g = k / group_size;
        acc += x[m * K + k] * (q * scales[n * n_groups + g] + biases[n * n_groups + g]);
    }
    out[m * N + n] = acc;
}

kernel void affine_qmm_t_half(
    device const half*  x        [[buffer(0)]],
    device const uint*  w        [[buffer(1)]],
    device const half*  scales   [[buffer(2)]],
    device const half*  biases   [[buffer(3)]],
    device half*        out      [[buffer(4)]],
    constant uint& M             [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    constant uint& group_size    [[buffer(8)]],
    constant uint& bits          [[buffer(9)]],
    uint2 tid                    [[thread_position_in_grid]])
{
    uint m = tid.y;
    uint n = tid.x;
    if (m >= M || n >= N) return;

    uint elems_per_int = 32 / bits;
    uint mask = (1u << bits) - 1u;
    uint K_packed = K / elems_per_int;
    uint n_groups = K / group_size;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        uint packed_val = w[n * K_packed + k / elems_per_int];
        float q = float((packed_val >> ((k % elems_per_int) * bits)) & mask);
        uint g = k / group_size;
        acc += float(x[m * K + k]) * (q * float(scales[n * n_groups + g]) + float(biases[n * n_groups + g]));
    }
    out[m * N + n] = half(acc);
}

kernel void affine_qmm_t_bfloat(
    device const bfloat* x       [[buffer(0)]],
    device const uint*   w       [[buffer(1)]],
    device const bfloat* scales  [[buffer(2)]],
    device const bfloat* biases  [[buffer(3)]],
    device bfloat*       out     [[buffer(4)]],
    constant uint& M             [[buffer(5)]],
    constant uint& N             [[buffer(6)]],
    constant uint& K             [[buffer(7)]],
    constant uint& group_size    [[buffer(8)]],
    constant uint& bits          [[buffer(9)]],
    uint2 tid                    [[thread_position_in_grid]])
{
    uint m = tid.y;
    uint n = tid.x;
    if (m >= M || n >= N) return;

    uint elems_per_int = 32 / bits;
    uint mask = (1u << bits) - 1u;
    uint K_packed = K / elems_per_int;
    uint n_groups = K / group_size;

    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        uint packed_val = w[n * K_packed + k / elems_per_int];
        float q = float((packed_val >> ((k % elems_per_int) * bits)) & mask);
        uint g = k / group_size;
        acc += float(x[m * K + k]) * (q * float(scales[n * n_groups + g]) + float(biases[n * n_groups + g]));
    }
    out[m * N + n] = bfloat(acc);
}
"""

_compiled_shader_lib = None


class _LocalMetalKernel:
    """Wrapper that mimics the Hub kernel interface using ``torch.mps.compile_shader``."""

    def __init__(self):
        global _compiled_shader_lib
        if _compiled_shader_lib is None:
            _compiled_shader_lib = torch.mps.compile_shader(_AFFINE_QMM_T_METAL_SOURCE)
        self._lib = _compiled_shader_lib

    def affine_qmm_t(self, x, w, scales, biases, group_size, bits):
        K_packed = w.shape[1]
        N = w.shape[0]
        elems_per_int = 32 // bits
        K = K_packed * elems_per_int

        x_2d = x.reshape(-1, K).contiguous()
        M_total = x_2d.shape[0]
        out = torch.empty(M_total, N, dtype=x.dtype, device=x.device)

        M_t = torch.tensor(M_total, dtype=torch.uint32, device="mps")
        N_t = torch.tensor(N, dtype=torch.uint32, device="mps")
        K_t = torch.tensor(K, dtype=torch.uint32, device="mps")
        gs_t = torch.tensor(group_size, dtype=torch.uint32, device="mps")
        bits_t = torch.tensor(bits, dtype=torch.uint32, device="mps")

        if x.dtype == torch.float32:
            fn = self._lib.affine_qmm_t_float
        elif x.dtype == torch.float16:
            fn = self._lib.affine_qmm_t_half
        elif x.dtype == torch.bfloat16:
            fn = self._lib.affine_qmm_t_bfloat
        else:
            raise ValueError(f"Unsupported dtype {x.dtype} for Metal affine_qmm_t")

        fn(x_2d, w, scales, biases, out, M_t, N_t, K_t, gs_t, bits_t, threads=[N, M_total, 1])

        return out.reshape(*x.shape[:-1], N)


def _get_metal_kernel():
    """Lazily load the quantization-mlx kernel from Hugging Face Hub, falling back to a
    locally-compiled Metal shader if the Hub kernel is unavailable or incompatible."""
    global _metal_kernel
    if _metal_kernel is None:
        try:
            from .hub_kernels import get_kernel

            hub_kernel = get_kernel("kernels-community/mlx-quantization-metal-kernels")
            # Smoke-test: the pre-built metallib may target an MSL version newer
            # than the current OS supports.  A tiny matmul catches this at init
            # time rather than mid-inference.
            _x = torch.zeros(1, 64, dtype=torch.float32, device="mps")
            _w = torch.zeros(1, 2, dtype=torch.uint32, device="mps")  # K=64 at 8-bit → 2 packed
            _s = torch.ones(1, 1, dtype=torch.float32, device="mps")
            _b = torch.zeros(1, 1, dtype=torch.float32, device="mps")
            hub_kernel.affine_qmm_t(_x, _w, _s, _b, 64, 8)
            _metal_kernel = hub_kernel
        except Exception:
            logger.info(
                "Hub kernel 'kernels-community/mlx-quantization-metal-kernels' unavailable; "
                "using locally-compiled Metal shader fallback."
            )
            _metal_kernel = _LocalMetalKernel()
    return _metal_kernel


# ---------------------------------------------------------------------------
# MetalLinear -- the quantized nn.Linear replacement
# ---------------------------------------------------------------------------


class MetalLinear(nn.Linear):
    """
    A quantized linear layer that stores weights in affine uint32 packed format
    and uses the ``quantization-mlx`` Metal kernels for the forward pass.

    Parameters match ``nn.Linear`` with additional quantization metadata.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        dtype=torch.uint32,
        bits: int = 4,
        group_size: int = 128,
    ):
        nn.Module.__init__(self)

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        elems_per_int = 32 // bits
        k_packed = in_features // elems_per_int
        n_groups = in_features // group_size

        if dtype == torch.uint32:
            self.weight = nn.Parameter(torch.zeros(out_features, k_packed, dtype=torch.uint32), requires_grad=False)
        else:
            self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=dtype), requires_grad=False)

        scales_dtype = torch.float32 if dtype == torch.uint32 else None
        self.scales = nn.Parameter(torch.zeros(out_features, n_groups, dtype=scales_dtype), requires_grad=False)
        self.qbiases = nn.Parameter(torch.zeros(out_features, n_groups, dtype=scales_dtype), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # MLX-quantized models store quantization biases as "biases" instead of "qbiases"
        biases_key = prefix + "biases"
        qbiases_key = prefix + "qbiases"
        if biases_key in state_dict and qbiases_key not in state_dict:
            state_dict[qbiases_key] = state_dict.pop(biases_key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != torch.uint32:
            return nn.functional.linear(input, self.weight, self.bias)

        kernel = _get_metal_kernel()

        output = kernel.affine_qmm_t(
            input,
            self.weight,
            self.scales.to(input.dtype),
            self.qbiases.to(input.dtype),
            self.group_size,
            self.bits,
        )

        if self.bias is not None:
            output = output + self.bias
        return output


def replace_with_metal_linear(
    model,
    modules_to_not_convert: list[str] | None = None,
    quantization_config=None,
    pre_quantized: bool = False,
):
    """
    Replace every eligible ``nn.Linear`` with ``MetalLinear``.

    Args:
        model: the ``PreTrainedModel`` (on the meta device at this point).
        modules_to_not_convert: module names to leave untouched.
        quantization_config: the ``MetalConfig`` instance.
        pre_quantized: ``True`` when loading from a quantized checkpoint.
    """
    if quantization_config.dequantize:
        return model

    bits = quantization_config.bits
    group_size = quantization_config.group_size

    has_been_replaced = False

    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue

        if isinstance(module, nn.Linear):
            module_kwargs = {} if pre_quantized else {"dtype": None}
            new_module = MetalLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                bits=bits,
                group_size=group_size,
                **module_kwargs,
            )

            model.set_submodule(module_name, new_module)
            has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading a model with Metal quantization but no nn.Linear modules were found. "
            "Please double check your model architecture."
        )

    return model


def _affine_quantize_tensor(weight: torch.Tensor, group_size: int, bits: int):
    """
    Quantize a 2-D float weight ``[N, K]`` into packed uint32 + scales + biases.

    Returns ``(w_packed, scales, biases)`` with:
      - ``w_packed``: ``[N, K // (32 // bits)]`` uint32
      - ``scales``:   ``[N, K // group_size]`` float32/float16/bfloat16
      - ``biases``:   ``[N, K // group_size]`` float32/float16/bfloat16
    """
    N, K = weight.shape
    elems_per_int = 32 // bits
    max_val = (1 << bits) - 1
    n_groups = K // group_size

    w_grouped = weight.float().reshape(N, n_groups, group_size)
    w_min = w_grouped.min(dim=-1).values  # [N, n_groups]
    w_max = w_grouped.max(dim=-1).values

    scales = ((w_max - w_min) / max_val).clamp(min=1e-8)
    biases = w_min

    w_int = (w_grouped - biases.unsqueeze(-1)) / scales.unsqueeze(-1)
    w_int = w_int.round().clamp(0, max_val).to(torch.int32).reshape(N, K)

    # Pack into uint32
    k_packed = K // elems_per_int
    w_packed = torch.zeros(N, k_packed, dtype=torch.int32, device=weight.device)
    for i in range(elems_per_int):
        w_packed |= w_int[:, i::elems_per_int] << (bits * i)

    return w_packed.to(torch.uint32), scales, biases


def _affine_dequantize_tensor(
    w_packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, group_size: int, bits: int
):
    """
    Dequantize a packed uint32 weight ``[N, K_packed]`` back to float.

    Returns a ``[N, K]`` float32 tensor.
    """
    N = w_packed.shape[0]
    elems_per_int = 32 // bits
    max_val = (1 << bits) - 1
    K = w_packed.shape[1] * elems_per_int

    w_packed_i = w_packed.to(torch.int32)
    w_flat = torch.zeros(N, K, dtype=torch.float32, device=w_packed.device)
    for i in range(elems_per_int):
        w_flat[:, i::elems_per_int] = ((w_packed_i >> (bits * i)) & max_val).float()

    w_grouped = w_flat.reshape(N, -1, group_size)
    w_deq = w_grouped * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    return w_deq.reshape(N, K)


class MetalQuantize(ConversionOps):
    """
    Quantize a full-precision weight tensor into (weight, scales, qbiases).

    Used during quantize-on-the-fly.  The float ``weight`` is replaced in-place
    by the packed uint32 tensor.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: dict, **kwargs) -> dict:
        target_key, value = next(iter(input_dict.items()))
        value = value[0] if isinstance(value, list) else value

        bits = self.hf_quantizer.quantization_config.bits
        group_size = self.hf_quantizer.quantization_config.group_size

        w_packed, scales, biases = _affine_quantize_tensor(value, group_size, bits)

        base = target_key.rsplit(".", 1)[0] if "." in target_key else ""
        scale_key = f"{base}.scales" if base else "scales"
        bias_key = f"{base}.qbiases" if base else "qbiases"

        orig_dtype = value.dtype
        return {
            target_key: w_packed,
            scale_key: scales.to(orig_dtype),
            bias_key: biases.to(orig_dtype),
        }


class MetalDequantize(ConversionOps):
    """
    Dequantize (weight, scales, qbiases) back to a full-precision tensor.

    Used when ``dequantize=True`` is set in the config to fall back to a normal
    ``nn.Linear`` on devices without MPS.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: dict, full_layer_name: str | None = None, **kwargs) -> dict:
        bits = self.hf_quantizer.quantization_config.bits
        group_size = self.hf_quantizer.quantization_config.group_size

        # Use source_patterns from kwargs as dict keys (they are the keys in input_dict).
        # Fall back to the default patterns for backward compatibility.
        source_patterns = kwargs.get("source_patterns", ["weight$", "scales", "qbiases"])

        if len(input_dict) < 2:
            return {full_layer_name: input_dict[source_patterns[0]]}

        quantized = input_dict[source_patterns[0]][0]
        scales = input_dict[source_patterns[1]][0]
        qbiases = input_dict[source_patterns[2]][0]

        w_deq = _affine_dequantize_tensor(quantized, scales, qbiases, group_size, bits)
        return {full_layer_name: w_deq.to(scales.dtype)}
