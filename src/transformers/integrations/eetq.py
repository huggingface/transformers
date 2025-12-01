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
from ..utils import is_accelerate_available, is_torch_available, logging

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name
    
if is_torch_available():
    import torch
    import torch.nn as nn

if is_accelerate_available():
    from accelerate import init_empty_weights

logger = logging.get_logger(__name__)

class EetqQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict: dict[str, list[torch.Tensor]], full_layer_name: str | None = None, **kwargs) -> dict[str, torch.Tensor]:
        _, value = tuple(input_dict.items())[0]
        value = value[0]
    
        value_device = value.device
        int8_weight = torch.t(value).contiguous().cpu()
        int8_weight, scales = eetq_kernels_hub.quant_weights(int8_weight, torch.int8, False)
        int8_weight = int8_weight.to(value_device)
        scales = scales.to(value_device)
        
        # fix when pre-quantized
        # int8_weight = eetq_kernels_hub.preprocess_weights(int8_weight)

        return {full_layer_name: int8_weight,
                f"{full_layer_name}_scales": scales}

class EetqLinearMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        scales,
        bias=None
    ):
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
            grad_input = grad_output.squeeze(0).matmul(
                weight.transpose(0, 1)
            ).unsqueeze(0)

        return grad_input, None, None, None
    
class EetqLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device="cuda:0"):
        super().__init__()
        self.register_buffer("weight", torch.zeros((in_features, out_features), dtype=torch.int8, device=device))
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=torch.float16, device=device))
        else:
            self.bias = None
        self.register_buffer("weight_scales", torch.zeros((out_features), dtype=torch.float16, device=device))

    def forward(self, input):
        output = EetqLinearMMFunction.apply(input, self.weight, self.weight_scales, self.bias)
        return output
    
    
def _replace_with_eetq_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successful or not.
    """
    import eetq
    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        current_key_name.append(name)

        if (isinstance(module, nn.Linear)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features
                    model._modules[name] = eetq.EetqLinear(
                        in_features, out_features, module.bias is not None, module.weight.device
                    )
                    if pre_quantized:
                        model._modules[name].register_scale(module.weight.device)
                    has_been_replaced = True

                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_eetq_linear(
                module,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                pre_quantized=pre_quantized,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_eetq_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `eetq.EetqLinear` modules from the `eetq`
    library. This will enable running your models using high performance int8 weight-only gemm kerner from
    FasterTransformer and TensorRT-LLM. Make sure `eetq` compiled with the correct CUDA
    version of your hardware is installed before running this function. EETQ shall be installed via the source
    'https://github.com/NetEase-FuXi/EETQ'

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Each weight will be quantized along the channel.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `["lm_head"]`):
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

    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert

    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_eetq_linear(
        model, modules_to_not_convert, current_key_name, quantization_config, pre_quantized=pre_quantized
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
