# coding=utf-8
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
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn

if is_accelerate_available():
    from accelerate import init_empty_weights

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


def replace_with_eetq_linear(
    model: torch.nn.Module, modules_to_not_convert: list[str] | None = None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `EetqLinear` modules.
    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `modules_to_not_convert` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Each weight will be quantized along the channel.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `None`):
            Names of the modules to not convert in `EetqLinear`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`list[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    from kernels import get_kernel

    global eetq_kernels_hub
    eetq_kernels_hub = get_kernel("kernels-community/quantization-eetq")

    has_been_replaced = False
    # we need this to correctly materialize the weights during quantization
    module_kwargs = {} if pre_quantized else {"dtype": None}
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with init_empty_weights():
            if isinstance(module, nn.Linear):
                new_module = EetqLinear(
                    module.in_features, module.out_features, bias=module.bias is not None, **module_kwargs
                )
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
