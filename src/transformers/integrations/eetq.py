# Copyright 2024 NetEase, Inc. and the HuggingFace Inc. team. All rights reserved.
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
from ..core_model_loading import ConversionOps
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


class EetqQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self, input_dict: dict[str, list[torch.Tensor]], full_layer_name: str | None = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        _, value = tuple(input_dict.items())[0]
        value = value[0]

        value_device = value.device
        int8_weight = torch.t(value).contiguous().cpu()
        int8_weight, scales = eetq_kernels_hub.quant_weights(int8_weight, torch.int8, False)

        int8_weight = int8_weight.to(value_device)
        scales = scales.to(value_device)

        return {full_layer_name: int8_weight, f"{full_layer_name}_scales": scales}


class EetqDequantize(ConversionOps):
    """Dequantized weights from the serialized int8 `weight` and its `weight_scales`. The int8 weight is stored in the
    CUTLASS mixed gemm layout: the kernel dequantizes it with a gemm with the identity, and without the kernel (or a GPU)
    the layout is undone in torch."""

    def __init__(self, hf_quantizer):
        from .hub_kernels import get_kernel

        self.hf_quantizer = hf_quantizer
        try:
            self.kernel = (
                get_kernel("kernels-community/quantization-eetq", version=1) if torch.cuda.is_available() else None
            )
        except Exception:
            self.kernel = None

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        output = {}
        for key, weights in input_dict.items():
            if not key.endswith("weight"):
                continue  # `weight_scales`
            weights = weights if isinstance(weights, list) else [weights]
            scales = input_dict.get(f"{key}_scales")
            if scales is not None:
                scales = scales if isinstance(scales, list) else [scales]
                weights = [self._dequantize(weight, scale) for weight, scale in zip(weights, scales)]
            output[key] = weights
        return output

    def _dequantize(self, weight, scales):
        if self.kernel is not None:
            # `weight` is (in_features, out_features): `identity @ weight` is the transposed dequantized weight
            identity = torch.eye(weight.shape[0], dtype=scales.dtype, device="cuda")
            return self.kernel.w8_a16_gemm(identity, weight.cuda(), scales.cuda()).t().to(weight.device)
        # Inverse of EETQ's `preprocess_weights_for_mixed_gemm` for sm75-sm89, in reverse order:
        # +128 bias and swap of bytes 1 and 2 in every 4, then 2-column interleave of 64-row tiles of the transposed
        # (column-major) weight, then the permutation of each 16 rows (`permute_B_rows_for_mixed_gemm`)
        in_features, out_features = weight.shape
        weight = weight.view(torch.uint8).reshape(-1, 4)[:, [0, 2, 1, 3]].to(torch.int16) - 128
        weight = weight.reshape(out_features // 2, in_features // 64, 2, 64).transpose(1, 2)
        rows = torch.tensor([0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15], device=weight.device)
        weight = weight.reshape(out_features, in_features // 16, 16)[..., rows].reshape(out_features, in_features)
        return weight.to(scales.dtype) * scales[:, None].to(weight.device)


class EetqLinearMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scales, bias=None):
        # The forward pass can use ctx.
        ctx.save_for_backward(x, weight, scales, bias)
        output = eetq_kernels_hub.w8_a16_gemm(x, weight, scales)
        output = output + bias if bias is not None else output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, scales, bias = ctx.saved_tensors
        identity = torch.eye(weight.shape[0]).to(weight.device).to(input.dtype)

        # Dequantize the weight
        weight = eetq_kernels_hub.w8_a16_gemm(identity, weight, scales)

        if ctx.needs_input_grad[0]:
            # 2D matrix multiplication, unsqueeze to 3D
            grad_input = grad_output.squeeze(0).matmul(weight.transpose(0, 1)).unsqueeze(0)

        return grad_input, None, None, None


class EetqLinear(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.int8, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_features, out_features), dtype=dtype), requires_grad=False)
        self.weight_scales = nn.Parameter(torch.empty((out_features), dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features), dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, input):
        output = EetqLinearMMFunction.apply(input, self.weight, self.weight_scales, self.bias)
        return output


def load_eetq_kernels():
    """Loads the EETQ hub kernels used by `EetqLinear` and `EetqQuantize`."""
    from .hub_kernels import get_kernel

    global eetq_kernels_hub
    eetq_kernels_hub = get_kernel("kernels-community/quantization-eetq", version=1)
