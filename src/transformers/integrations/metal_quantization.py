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


def _get_metal_kernel():
    """Lazily load the quantization-mlx kernel from Hugging Face Hub."""
    global _metal_kernel
    if _metal_kernel is None:
        try:
            from .hub_kernels import get_kernel

            try:  # v2 has `gather_qmm` and the builds for recent torch versions
                _metal_kernel = get_kernel("kernels-community/mlx-quantization-metal-kernels", version=2)
            except Exception:
                _metal_kernel = get_kernel("kernels-community/mlx-quantization-metal-kernels", version=1)
        except Exception as e:
            raise ImportError(
                f"Failed to load the quantization-mlx kernel from the Hub: {e}. "
                "Make sure you have `kernels` installed (`pip install kernels`) "
                "and are running on an Apple Silicon machine."
            ) from e
        _register_fake_kernels(_metal_kernel)
    return _metal_kernel


def _register_fake_kernels(kernel):
    """Shape-only implementations of the kernel's ops, so `torch.compile` traces through them (one graph per decode
    step) instead of breaking the graph at every quantized matmul."""
    namespace = kernel.ops.__name__.rpartition(".")[2]

    def register(op, fake):
        try:
            torch.library.register_fake(f"{namespace}::{op}")(fake)
        except (RuntimeError, AttributeError):  # already registered, or an older build without the op
            pass

    def quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode):
        # transpose: w is [..., N, K * bits / 32] -> [..., N]; else w is [..., K, N * bits / 32]
        n = w.shape[-2] if transpose else w.shape[-1] * 32 // (bits or 4)
        return x.new_empty(*torch.broadcast_shapes(x.shape[:-2], w.shape[:-2]), x.shape[-2], n)

    def gather_qmm(
        x, w, scales, biases, lhs_indices, rhs_indices, transpose, group_size, bits, mode, global_scale, sorted_indices
    ):
        n = w.shape[-2] if transpose else w.shape[-1] * 32 // (bits or 4)
        indices = [i.shape for i in (lhs_indices, rhs_indices) if i is not None]
        batch = torch.broadcast_shapes(*indices) if indices else x.shape[:-2]
        return x.new_empty(*batch, x.shape[-2], n)

    register("quantized_matmul", quantized_matmul)
    register("gather_qmm", gather_qmm)


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

        short = min(self.in_features, self.out_features) <= 1024 and self.group_size in (32, 64)
        if self.bits == 4 and short and input.numel() == self.in_features and input.device.type == "mps":
            # One token through a matrix with a short side: MLX's qmv leaves lanes idle there
            from .metal_qmv import qmv4

            output = qmv4(input, self.weight, self.scales, self.qbiases, self.group_size)
            return output if self.bias is None else output + self.bias

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


class MetalExperts(nn.Module):
    """
    Quantized experts (3D banks) run by the ``gather_qmm`` Metal kernel: each route multiplies its token by the packed
    weights of its expert, no dequantized copy. Replaces an experts module (`gate_up_proj` / `down_proj` banks of
    shape `[num_experts, N, K]`) with `[num_experts, N, K_packed]` uint32 banks and their `_scales` / `_qbiases`.

    Expert ids `>= num_experts` are expert-parallel / streamed-experts sentinels: their routes contribute zero.
    """

    PROJECTIONS = ("gate_up_proj", "down_proj")

    def __init__(self, experts: nn.Module, bits: int = 4, group_size: int = 64, dtype=torch.uint32):
        super().__init__()
        self.num_experts = experts.num_experts
        self.bits = bits
        self.group_size = group_size
        self.act_fn = experts.act_fn
        self.config = getattr(experts, "config", None)
        for flag in ("has_gate", "has_bias", "is_concatenated", "is_transposed"):
            setattr(self, flag, getattr(experts, flag, None))
        self._apply_gate_fn = type(experts)._apply_gate
        # The Metal decode down+routing kernel: 4 bits, no bias, group a multiple of 32 (one scale per uint4 block)
        self._fused_decode = bits == 4 and not getattr(experts, "has_bias", False) and group_size in (32, 64)
        for name in self.PROJECTIONS:
            num_experts, rows, columns = getattr(experts, name).shape
            groups = columns // group_size
            if dtype == torch.uint32:
                weight = torch.zeros(num_experts, rows, columns * bits // 32, dtype=torch.uint32)
            else:  # quantized on the fly: the float bank is loaded first
                weight = torch.zeros(num_experts, rows, columns, dtype=dtype)
            self.register_parameter(name, nn.Parameter(weight, requires_grad=False))
            for suffix in ("scales", "qbiases"):
                # The activations' dtype (the model's default): the kernel takes them as they are, no cast per call
                tensor = torch.zeros(num_experts, rows, groups)
                self.register_parameter(f"{name}_{suffix}", nn.Parameter(tensor, requires_grad=False))

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        return self._apply_gate_fn(self, gate_up_out)

    def _qmm(self, x, name, ids, sorted_indices):
        scales = getattr(self, f"{name}_scales")
        if x.device.type == "cuda":  # same packed layout, Triton kernel
            from .triton_affine_moe import gather_qmv

            biases = getattr(self, f"{name}_qbiases")
            return gather_qmv(x, getattr(self, name), scales, biases, ids, self.group_size, self.bits)
        kernel = _get_metal_kernel()
        out = kernel.gather_qmm(
            x[:, None, :],
            getattr(self, name),
            scales.to(x.dtype),
            getattr(self, f"{name}_qbiases").to(x.dtype),
            rhs_indices=ids,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            sorted_indices=sorted_indices,
        )
        return out.squeeze(1)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        num_tokens, top_k = top_k_index.shape
        ids = top_k_index.reshape(-1)
        # Sentinels (expert parallel, streamed misses) get weight 0; their clamped slot holds finite values (a pool's
        # unfilled slots are zeros), so the product is 0
        weights = top_k_weights.reshape(-1) * (ids < self.num_experts)
        ids = ids.clamp(max=self.num_experts - 1).to(torch.int32)
        if num_tokens == 1 and self._fused_decode and hidden_states.device.type == "mps":
            # One token: MLX gate/up for every route, then one dispatch for the short-row down projections + routing sum
            from .metal_qmv import experts_down_sum

            act = self._apply_gate(self._qmm(hidden_states.expand(top_k, -1), "gate_up_proj", ids, False))
            down = (self.down_proj, self.down_proj_scales, self.down_proj_qbiases)
            return experts_down_sum(act.contiguous(), *down, ids.contiguous(), weights.float(), self.group_size)
        if num_tokens == 1:
            # One token: every route reads the same row, and the routes reduce with a weighted sum (no scatter)
            x = hidden_states.expand(top_k, -1)
            out = self._qmm(self._apply_gate(self._qmm(x, "gate_up_proj", ids, False)), "down_proj", ids, False)
            return (out * weights[:, None].to(out.dtype)).sum(0, keepdim=True)
        rows = torch.arange(num_tokens * top_k, device=ids.device) // top_k
        # MLX picks its batched kernel for sorted ids (prefill); decode rows stay in place
        sorted_indices = ids.numel() >= 16
        if sorted_indices:
            ids, order = torch.sort(ids, stable=True)
            rows, weights = rows[order], weights[order]
        x = hidden_states[rows]
        out = self._qmm(
            self._apply_gate(self._qmm(x, "gate_up_proj", ids, sorted_indices)), "down_proj", ids, sorted_indices
        )
        out = out * weights[:, None].to(out.dtype)
        return torch.zeros_like(hidden_states).index_add_(0, rows, out)


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

        # Experts banks `[num_experts, N, K]` are quantized row by row like a `[num_experts * N, K]` weight
        w_packed, scales, biases = _affine_quantize_tensor(value.reshape(-1, value.shape[-1]), group_size, bits)
        lead = value.shape[:-1]
        w_packed, scales, biases = (t.reshape(*lead, -1) for t in (w_packed, scales, biases))

        base, _, name = target_key.rpartition(".")
        if name == "weight":
            scale_key = f"{base}.scales" if base else "scales"
            bias_key = f"{base}.qbiases" if base else "qbiases"
        else:  # several quantized params in one module (experts): `gate_up_proj_scales`, ...
            scale_key, bias_key = f"{target_key}_scales", f"{target_key}_qbiases"

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
