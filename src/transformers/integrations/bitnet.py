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
from ..utils import is_accelerate_available, logging
import torch
import torch.nn as nn
import torch.nn.functional as F


if is_accelerate_available():
    from accelerate import init_empty_weights

logger = logging.get_logger(__name__)

@torch.compile
def unpack_weights(packed: torch.Tensor, bits: int = 2) -> torch.Tensor:
    values_per_item = 8 // bits
    packed_shape = packed.shape

    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * values_per_item
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * values_per_item
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)

    for i in range(values_per_item):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = (3 << (2 * i))
        unpacked[start:end] = (packed & mask) >> (2 * i)

    unpacked = unpacked.to(torch.bfloat16) - 1
    return unpacked.to(torch.bfloat16)

class BitLinear158(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = False, input_bits: int = 8,
                 device=None, dtype=None, config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_bits = input_bits
        self.register_buffer(
            "weight",
            torch.zeros(
                (out_features // 4, in_features),
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(
                (1),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        if bias :
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.out_features),
                    dtype=torch.bfloat16,
                    device=device
                ))
        else :
            self.bias = None

    @torch.compile
    def activation_quant(self, x, num_bits=8):
        Qn = -(2**(num_bits - 1))
        Qp = 2**(num_bits - 1) - 1
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp)
        return result.type(torch.int8), s

    @torch.compile
    def post_quant_process(self, input, si, sw):
        out = input / si
        out = out / sw
        return out

    def forward(self, x):
        w = self.weight
        w_unpacked = unpack_weights(w)
        w_quant = w_unpacked.to(torch.bfloat16)
        x_quant, x_scale = self.activation_quant(x)
        y = F.linear(x_quant.to(torch.bfloat16), w_quant)
        y = self.post_quant_process(y, self.weight_scale, x_scale)
        if self.bias is not None :
            y += self.bias.view(1, -1).expand_as(y)
        return y

def _replace_with_bitnet_linear(
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    pre_quantized=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """

    if current_key_name is None:
        current_key_name = []

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        # Check if the current key is not in the `modules_to_not_convert`
        if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
            with init_empty_weights() :
                if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = BitLinear158(
                        in_features=in_features,
                        out_features=out_features,
                        bias=module.bias is not None,
                        device=module.weight.device,
                    )
                    has_been_replaced = True
                    model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bitnet_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bitnet_linear(
    model, modules_to_not_convert=None, current_key_name=None, quantization_config=None, pre_quantized=False
):
    """
    A helper function to replace all `torch.nn.Linear` modules by `BitLinear158` modules`.

    The function will be run recursively and replace all `torch.nn.Linear` modules except for the `lm_head` that should
    be kept as a `torch.nn.Linear` module. The replacement is done under `init_empty_weights` context manager so no
    CPU/GPU memory is required to run this function. Each weight will be quantized along the channel.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`List[`str`]`, *optional*, defaults to `["lm_head"]`):
            Names of the modules to not convert in `EetqLinear`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
        current_key_name (`List[`str`]`, *optional*):
            An array to track the current key of the recursion. This is used to check whether the current key (part of
            it) is not in the list of modules to not convert (for instances modules that are offloaded to `cpu` or
            `disk`).
    """
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    if quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_bitnet_linear(
        model, modules_to_not_convert, current_key_name, quantization_config, pre_quantized=pre_quantized
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using bitnet but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
