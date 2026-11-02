# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"FP-Quant integration file"

import torch

from ..utils import (
    is_fp_quant_available,
)


if is_fp_quant_available():
    from fp_quant import FPQuantConfig as FPQuantLinearConfig
    from fp_quant import FPQuantDtype

from transformers.utils.quantization_config import FPQuantConfig

from ..core_model_loading import ConversionOps, expand_target_names
from ..quantizers.quantizers_utils import get_module_from_name
from .mxfp4 import FP4_VALUES


class FpQuantQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: torch.Tensor,
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_key, value = tuple(expand_target_names(input_dict, full_layer_name).items())[0]
        value = value[0]
        # Loading master weights or an unquantized checkpoint
        weight = torch.nn.Parameter(value)
        module, _ = get_module_from_name(model, target_key)
        module.weight = weight

        # Let pre-forward handle the quantization and set None where necessary
        # This operation will quantize the weights internally
        torch_accelerator_module = getattr(torch, value.device.type, torch.cuda)
        with torch_accelerator_module.device(value.device):
            module.pre_forward()

        prefix_target_key = target_key.rsplit(".", 1)[0]

        # keys are set inside the module.pre_forward() method, we don't need remove them from the missing keys list
        missing_keys.discard(target_key)
        missing_keys.discard(f"{prefix_target_key}.backward_hadamard_matrix")
        missing_keys.discard(f"{prefix_target_key}.forward_hadamard_matrix")
        missing_keys.discard(f"{prefix_target_key}.act_global_scale")
        missing_keys.discard(f"{prefix_target_key}.weight_global_scale")
        missing_keys.discard(f"{prefix_target_key}.qweight")
        missing_keys.discard(f"{prefix_target_key}.scales")
        missing_keys.discard(f"{prefix_target_key}.dqweight")
        return {}


class FpQuantDeserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: torch.Tensor,
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        output = {key: value[0] if isinstance(value, list) else value for key, value in input_dict.items()}
        # The module holds `weight`, `qweight`, `scales` and `dqweight`, but only uses the ones of the checkpoint:
        #  * `qweight` and `scales` when `pseudoquantization=False`
        #  * `dqweight` when `pseudoquantization=True`
        # the others are expected in the model even though they are not used, so we set them to empty tensors
        for name in ("weight", "qweight", "scales", "dqweight"):
            output.setdefault(name, torch.nn.Parameter(torch.zeros(0), requires_grad=False))
        return output


class FpQuantDequantize(ConversionOps):
    """
    Dense weights from the forward-quantized ones: `dqweight` (pseudo-quantization), or the FP4 `qweight` with its
    per-block `scales` (E8M0 for MXFP4, E4M3 for NVFP4) divided by `weight_global_scale` (the GEMM scales its output by
    `1 / (weight_global_scale * act_global_scale)`). Both are rotated by `forward_hadamard_matrix` per group along the
    input dim, so the dense weight is rotated back by its transpose.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def dequantize_fp4(self, qweight, scales, global_scale):
        lut = torch.tensor(FP4_VALUES, device=qweight.device)
        qweight = qweight.view(torch.uint8)
        values = torch.stack([lut[(qweight & 0xF).long()], lut[(qweight >> 4).long()]], dim=-1).flatten(-2)
        if self.hf_quantizer.quantization_config.forward_dtype == "mxfp4":
            scales = torch.exp2(scales.view(torch.uint8).float() - 127)
        else:
            scales = scales.view(torch.float8_e4m3fn).float()
        values = values.unflatten(-1, (scales.shape[-1], -1)) * scales[..., None]
        return values.flatten(-2) / global_scale.float()

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        pseudoquantization = self.hf_quantizer.quantization_config.pseudoquantization
        anchor = "dqweight" if pseudoquantization else "qweight"
        output = {}
        for key, weights in input_dict.items():
            if not key.endswith(anchor):
                continue  # `scales`, `weight_global_scale` and `forward_hadamard_matrix`
            prefix = key.removesuffix(anchor)

            def get(name):
                value = input_dict[f"{prefix}{name}"]
                return value if isinstance(value, list) else [value]

            weights = weights if isinstance(weights, list) else [weights]
            if not pseudoquantization:
                weights = [
                    self.dequantize_fp4(*args) for args in zip(weights, get("scales"), get("weight_global_scale"))
                ]
            output[f"{prefix}weight"] = [
                (weight.float().unflatten(-1, (-1, hadamard.shape[0])) @ hadamard.float().T)
                .flatten(-2)
                .to(hadamard.dtype)
                for weight, hadamard in zip(weights, get("forward_hadamard_matrix"))
            ]
        return output


def adapt_fp_quant_config(config: FPQuantConfig):
    if config.forward_dtype == "mxfp4":
        forward_dtype = FPQuantDtype.MXFP4
    elif config.forward_dtype == "nvfp4":
        forward_dtype = FPQuantDtype.NVFP4
    else:
        raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")

    if config.backward_dtype == "bf16":
        backward_dtype = FPQuantDtype.BF16
    elif config.backward_dtype == "mxfp8":
        backward_dtype = FPQuantDtype.MXFP8
    elif config.backward_dtype == "mxfp4":
        backward_dtype = FPQuantDtype.MXFP4
    else:
        raise ValueError(f"Unsupported backward dtype: {config.backward_dtype}")

    return FPQuantLinearConfig(
        forward_dtype=forward_dtype,
        forward_method=config.forward_method,
        backward_dtype=backward_dtype,
        store_master_weights=config.store_master_weights,
        hadamard_group_size=config.hadamard_group_size,
        pseudoquantization=config.pseudoquantization,
        transform_init=config.transform_init,
        modules_to_not_convert=config.modules_to_not_convert,
    )
