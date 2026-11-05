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
