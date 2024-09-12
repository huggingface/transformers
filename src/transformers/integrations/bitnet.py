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


if is_accelerate_available():
    from accelerate import init_empty_weights

if is_torch_available():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

logger = logging.get_logger(__name__)


def pack_weights(quantized_weights: torch.Tensor) -> torch.Tensor:
    """
    Packs a tensor of quantized weights into a compact format using 2 bits per value.

    Parameters:
    -----------
    quantized_weights : torch.Tensor
        A tensor containing ternary quantized weights with values in {-1, 0, 1}. These values are adjusted to
        {0, 1, 2} before being packed.

    Returns:
    --------
    torch.Tensor
        A packed tensor where each element stores 4 quantized values (each using 2 bits) in an 8-bit format.
    """

    quantized_weights += 1
    original_shape = quantized_weights.shape
    values_per_item = 4
    row_dim = (original_shape[0] + values_per_item - 1) // values_per_item

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    packed = torch.zeros(
        packed_tensor_shape, device=quantized_weights.device, dtype=torch.uint8
    )
    unpacked = quantized_weights.to(torch.uint8)

    def lshift(t: torch.Tensor, bits: int):
        return t << bits

    it = min(values_per_item, (original_shape[0] // row_dim) + 1)
    for i in range(it):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        packed[: (end - start)] |= lshift(unpacked[start:end], 2 * i)

    return packed


@torch.compile
def unpack_weights(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Unpacks a tensor of quantized weights that were stored in a packed format using 2 bits per value.

    Parameters:
    -----------
    packed : torch.Tensor
        A tensor containing packed weights where each element represents 4 quantized values (using 2 bits per value).
    dtype : torch.dtype
        The dtype of the returned Tensor
    Returns:
    --------
    torch.Tensor
        A tensor of unpacked weights, where each value is converted from its packed 2-bit representation.

    Example:
    --------
    packed = torch.tensor([[0b10100001, 0b00011000],
                           [0b10010000, 0b00001010]], dtype=torch.uint8)

    # Unpack the values
    unpacked = unpack_weights(packed)

    # Resulting unpacked tensor
    print(unpacked)
    # Output: tensor([[ 0, -1],
                      [-1,  1],
                      [-1,  1],
                      [-1,  1],
                      [ 1,  0],
                      [ 0, -1],
                      [ 1, -1],
                      [ 1, -1]])

    Explanation of the example:
    ---------------------------
    Let's take the first value for example 0b10100001, we we will only focus on the first column,
    because every element is unpacked across the first dimension
    - First 2 bits: `01` → 0 at [0][0]
    - Second 2 bits: `00` → -1 at [0][2]
    - Third 2 bits: `10` → 1 at [0][4]
    - Fourth 2 bits: `10` → 1 at [0][6]
    the second value of the same row (0b10010000) will give the values for [0][1], [0][3], [0][5], [0][7]

    We subtract 1 because during the packing process, it's easier to work with values like 0, 1, and 2. To make this possible,
    we add 1 to the original ternary weights (which are typically -1, 0, and 1) when packing them. When unpacking, we reverse
    this by subtracting 1 to restore the original ternary values.
    """
    values_per_item = 4
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
        mask = 3 << (2 * i)
        unpacked[start:end] = (packed & mask) >> (2 * i)

    unpacked = unpacked.to(dtype) - 1
    return unpacked.to(dtype)


class BitLinear(nn.Module):

    def __init__(
        self, in_features: int, out_features: int, bias: bool, device=None, dtype=None
    ):
        super().__init__()
        self.dtype = dtype
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
                dtype=dtype,
                device=device,
            ),
        )
        if bias:
            self.register_buffer(
                "bias", torch.zeros((out_features), dtype=dtype, device=device)
            )
        else:
            self.bias = None

    @torch.compile
    def activation_quant(self, x, num_bits=8):
        """
        Activation function : Performs symmetric, per-channel quantization on the input activations.
        Parameters:
        -----------
        x : torch.Tensor
            Input activations to be quantized.
        num_bits : int, optional (default=8)
            Number of bits to use for quantization, determining the quantization range.

        Returns:
        --------
        result : torch.Tensor
            Quantized activation tensor, with values mapped to an `int8` range.
        s : torch.Tensor
            The per-channel scaling factors used to quantize the tensor.
        """
        Qn = -(2 ** (num_bits - 1))
        Qp = 2 ** (num_bits - 1) - 1
        s = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        result = (x * s).round().clamp(Qn, Qp)
        return result.to(torch.int8), s

    @torch.compile
    def post_quant_process(self, input, si, sw):
        out = input / si
        out = out / sw
        return out

    def forward(self, x):
        w = self.weight
        w_quant = unpack_weights(w, dtype=self.dtype)
        x_quant, x_scale = self.activation_quant(x)
        y = F.linear(x_quant.to(self.dtype), w_quant)
        y = self.post_quant_process(y, self.weight_scale, x_scale)
        if self.bias is not None:
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
            with init_empty_weights():
                if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
                    in_features = module.in_features
                    out_features = module.out_features
                    model._modules[name] = BitLinear(
                        in_features=in_features,
                        out_features=out_features,
                        bias=module.bias is not None,
                        device=module.weight.device,
                        dtype=module.weight.dtype,
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
    model,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    pre_quantized=False,
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
    modules_to_not_convert = (
        ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    )
    if quantization_config and quantization_config.modules_to_not_convert is not None:
        modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
    modules_to_not_convert = list(set(modules_to_not_convert))
    model, has_been_replaced = _replace_with_bitnet_linear(
        model,
        modules_to_not_convert,
        current_key_name,
        quantization_config,
        pre_quantized=pre_quantized,
    )

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using bitnet but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
