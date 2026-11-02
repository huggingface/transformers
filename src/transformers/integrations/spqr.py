# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"SpQR (Sparse-Quantized Representation) integration file"

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_spqr_available, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)


def dequantize_spqr(dense_weights, row_offsets, col_vals, in_perm, rows, cols, bits, beta1, beta2) -> "torch.Tensor":
    """Same as the host `dequantize_compressed` of `spqr_quant`, in fp16, then with the input permutation undone."""
    # One int64 per row of a `beta1 x beta2` tile: its scale, zero point and `beta2` weights of `bits` bits each, then a
    # chunk of the tile's second order statistics
    tiles = dense_weights.view(rows // beta1, cols // beta2, beta1)
    shifts = torch.arange(0, bits * (beta2 + 2), bits, device=tiles.device)
    codes = ((tiles.unsqueeze(-1) >> shifts) & (2**bits - 1)).half()
    chunks = (tiles >> bits * (beta1 + 2)) & (2 ** (64 - bits * (beta1 + 2)) - 1)
    second_order = torch.zeros_like(tiles[..., 0])
    for k in range(beta1):
        second_order |= chunks[..., k] << 8 * (k // 2)
    # The scale and zero point of the scale, then of the zero point
    ss, zs, sz, zz = second_order.unsqueeze(-1).view(torch.float16).unbind(-1)
    scale = ss.unsqueeze(-1) * (codes[..., 0] - sz.unsqueeze(-1))
    zero = zs.unsqueeze(-1) * (codes[..., 1] - zz.unsqueeze(-1))
    weight = scale.unsqueeze(-1) * (codes[..., 2:] - zero.unsqueeze(-1))
    weight = weight.float().permute(0, 2, 1, 3).reshape(rows, cols)

    # The outliers: a CSR matrix, or one CSR row per 16 rows (PTCSR), of int32 `column | fp16 value << 16`
    col_vals = col_vals.view(torch.int16).view(-1, 2)
    row_ids = torch.repeat_interleave(
        torch.arange(len(row_offsets) - 1, device=weight.device), row_offsets.diff().long()
    )
    if len(row_offsets) != rows + 1:
        row_ids = row_ids * 16 + torch.arange(len(row_ids), device=weight.device) % 16
    values = col_vals[:, 1].contiguous().view(torch.float16).float()
    weight.index_put_((row_ids, col_vals[:, 0].long() & 0xFFFF), values, accumulate=True)
    weight = weight.half()

    # The kernel multiplies with `x[in_perm]`
    if in_perm is not None and in_perm.numel() > 0:
        weight = torch.empty_like(weight).index_copy_(1, in_perm.view(torch.int16).long() & 0xFFFF, weight)
    return weight


class SpQRDequantize(ConversionOps):
    """Dense weights from the SpQR `dense_weights` tiles, the outliers of `row_offsets` / `col_vals` and `in_perm`."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        config = self.hf_quantizer.quantization_config
        output = {}
        for key, all_dense_weights in input_dict.items():
            if not key.endswith("dense_weights"):
                continue
            prefix = key.removesuffix("dense_weights")
            rows, cols = model.get_submodule(full_layer_name.rpartition(".")[0]).weight.shape
            all_dense_weights = all_dense_weights if isinstance(all_dense_weights, list) else [all_dense_weights]
            others = [input_dict.get(prefix + name) for name in ("row_offsets", "col_vals", "in_perm")]
            others = [other if isinstance(other, list) else [other] * len(all_dense_weights) for other in others]
            output[prefix + "weight"] = [
                dequantize_spqr(*tensors, rows, cols, config.bits, config.beta1, config.beta2)
                for tensors in zip(all_dense_weights, *others)
            ]
        return output


def replace_with_spqr_linear(model, modules_to_not_convert: list[str] | None = None, quantization_config=None):
    """
    Public method that replaces the Linear layers of the given model with SPQR quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `None`):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        quantization_config (`SpQRConfig`):
            The quantization config object that contains the quantization parameters.
    """
    if is_spqr_available():
        from spqr_quant import QuantizedLinear

    has_been_replaced = False
    # we need this to correctly materialize the weights during quantization
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with torch.device("meta"):
            if isinstance(module, nn.Linear):
                shapes = quantization_config.shapes

                new_module = QuantizedLinear.create_placeholder(
                    rows=module.out_features,
                    cols=module.in_features,
                    bits=quantization_config.bits,
                    beta1=quantization_config.beta1,
                    beta2=quantization_config.beta2,
                    dense_weights_shape=shapes[f"{module_name}.dense_weights.shape"],
                    row_offsets_shape=shapes[f"{module_name}.row_offsets.shape"],
                    col_vals_shape=shapes[f"{module_name}.col_vals.shape"],
                    in_perm_shape=shapes[f"{module_name}.in_perm.shape"],
                )
                # Force requires grad to False to avoid unexpected errors
                model._modules[module_name].requires_grad_(False)
                model.set_submodule(module_name, new_module)
                has_been_replaced = True
    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
