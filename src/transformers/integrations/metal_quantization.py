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
  - ``MetalQuantize`` / ``MetalDequantize``: weight conversion operations that
    participate in the new ``WeightConverter`` pipeline.

Weight layout (transposed, matching ``affine_qmm_t``):
  - ``weight``: ``[N, K_packed]`` (``uint32``) -- K is the packed dimension.
  - ``scales``:  ``[N, K // group_size]`` (``float16 / bfloat16``)
  - ``qbiases``: ``[N, K // group_size]`` (same dtype as scales)

The kernel call is ``affine_qmm_t(x, weight, scales, qbiases, group_size, bits)``
which computes ``y = x @ dequant(weight).T``, identical to ``nn.Linear``.
"""

from ..core_model_loading import ConversionOps, _IdentityOp
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)

_metal_kernel = None


_decode_kernels = []


def _decode_kernel():
    """The pinned `metal-moe-decode` hub kernel, or None where it does not load."""
    if not _decode_kernels:
        from .hub_kernels import metal_kernel

        _decode_kernels.append(metal_kernel("metal-moe-decode"))
    return _decode_kernels[0]


def _get_metal_kernel():
    """Lazily load the quantization-mlx kernel from Hugging Face Hub."""
    global _metal_kernel
    if _metal_kernel is None:
        try:
            from .hub_kernels import get_kernel

            _metal_kernel = get_kernel("kernels-community/mlx-quantization-metal-kernels", version=1)
        except Exception as e:
            raise ImportError(
                f"Failed to load the quantization-mlx kernel from the Hub: {e}. "
                "Make sure you have `kernels` installed (`pip install kernels`) "
                "and are running on an Apple Silicon machine."
            ) from e
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


def metal_fp4_experts_available(device_map) -> bool:
    """Whether a `device_map` puts the model on MPS and the `metal-moe-decode` kernels load: NVFP4 / MXFP8 experts then
    stay quantized (`MetalFp4Experts`) instead of being dequantized"""
    devices = set(device_map.values()) if isinstance(device_map, dict) else {device_map}
    on_mps = all(str(device).startswith("mps") for device in devices)
    return on_mps and _decode_kernel() is not None and hasattr(_decode_kernel(), "fp4_gate_up_act")


def metal_fp4_module(name: str, module: nn.Module, fmt: str) -> nn.Module:
    """The MPS module of a pre-quantized NVFP4 / MXFP8 module: experts run on `MetalFp4Experts`"""
    if name.endswith(".experts"):
        return MetalFp4Experts(module, fmt)
    raise NotImplementedError(
        f"Only MoE experts of a {fmt.upper()} checkpoint run quantized on MPS, not `{name}` ({type(module).__name__}). "
        "Load it with `dequantize=True` in its quantization config."
    )


class MetalFp4Experts(nn.Module):
    """
    Experts kept in their checkpoint's NVFP4 or MXFP8 banks (no dequantized copy), run by the `metal-moe-decode`
    kernels: gate/up + activation in one dispatch, down + routing sum in another, for any number of tokens. The params
    have the names of the plan's NVFP4 / FP8 experts, so the same converters load them:

        nvfp4: `{proj}` uint8 [E, N, K / 2], `{proj}_scale` e4m3 bytes [E, N, K / 16], `{proj}_scale_2` float32 global
               scales ([E, 2] for gate / up, [E] for down)
        mxfp8: `{proj}` float8_e4m3fn [E, N, K], `{proj}_scale_inv` e8m0 [E, N, K / 32]

    `gate_up_proj` is `[gate; up]`. The activation is the experts' own: a softcapped SwiGLU (`gate_softcap` /
    `up_softcap` on its `act_fn`, e.g. Inkling2) or SiLU(gate) * up.
    """

    def __init__(self, experts: nn.Module, fmt: str = "nvfp4"):
        super().__init__()
        self.num_experts, self.fmt = experts.num_experts, fmt
        hidden, inter = experts.hidden_dim, experts.intermediate_dim
        self.act_fn = experts.act_fn
        self.gate_cap = float(getattr(getattr(self.act_fn, "gate_softcap", None), "softcap", 0.0))
        self.up_cap = float(getattr(getattr(self.act_fn, "up_softcap", None), "softcap", 0.0))
        nvfp4 = fmt == "nvfp4"
        for name, (rows, columns) in {"gate_up_proj": (2 * inter, hidden), "down_proj": (hidden, inter)}.items():
            if nvfp4:
                params = {
                    name: torch.empty(self.num_experts, rows, columns // 2, dtype=torch.uint8),
                    f"{name}_scale": torch.empty(self.num_experts, rows, columns // 16, dtype=torch.uint8),
                    f"{name}_scale_2": torch.ones(self.num_experts, *((2,) if name == "gate_up_proj" else ())),
                }
            else:
                params = {
                    name: torch.empty(self.num_experts, rows, columns, dtype=torch.float8_e4m3fn),
                    f"{name}_scale_inv": torch.empty(self.num_experts, rows, columns // 32, dtype=torch.uint8),
                }
            for key, tensor in params.items():
                self.register_parameter(key, nn.Parameter(tensor, requires_grad=False))
        if not nvfp4:  # the kernels take a global scale argument, unused for MXFP8
            self.register_buffer("_no_scale_2", torch.ones(1), persistent=False)

    def _bank(self, name):
        if self.fmt == "nvfp4":
            return getattr(self, name), getattr(self, f"{name}_scale"), getattr(self, f"{name}_scale_2")
        return getattr(self, name), getattr(self, f"{name}_scale_inv"), self._no_scale_2

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        # every (token, route) pair is a gathered matvec straight from the banks: no dequantized copy, no host sync
        kernel = _decode_kernel()
        ids, weights = top_k_index.int(), top_k_weights.float()
        act = kernel.fp4_gate_up_act(
            hidden_states, *self._bank("gate_up_proj"), ids, weights, self.gate_cap, self.up_cap
        )
        return kernel.fp4_down_sum(act, *self._bank("down_proj"), ids, weights)


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

        # A model converter collects several weights (e.g. per expert): `*.weight` comes with `*.scales`, `*.qbiases`
        result = {}
        for key, weights in input_dict.items():
            prefix = key.removesuffix("weight")
            if key.endswith(("scales", "qbiases")):
                continue
            if prefix + "scales" not in input_dict:
                result[key] = weights
                continue
            scales, qbiases = input_dict[prefix + "scales"], input_dict[prefix + "qbiases"]
            result[key] = [
                _affine_dequantize_tensor(w, s, b, group_size, bits).to(s.dtype)
                for w, s, b in zip(weights, scales, qbiases)
            ]
        if "weight" in result:  # the general converter: one module
            return {full_layer_name: result["weight"][0]}
        return result

    @property
    def reverse_op(self) -> "ConversionOps":
        return _IdentityOp()
