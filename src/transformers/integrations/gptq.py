# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"GPTQ integration file: loading of the int32 packed GPTQ format into gptqmodel layers or dense ones, shared with AWQ and AutoRound"

from ..core_model_loading import ConversionOps
from ..utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn


def unpack_int32(packed: "torch.Tensor", bits: int, dim: int) -> "torch.Tensor":
    """The `32 // bits` codes of each int32 of `packed`, lowest bits first, laid out along `dim`."""
    if bits not in (2, 4, 8):
        raise NotImplementedError(f"Dequantizing {bits}-bit packed weights is not supported.")
    shifts = torch.arange(0, 32, bits, dtype=torch.int32, device=packed.device)
    codes = packed.unsqueeze(dim + 1) >> shifts.view(-1, *[1] * (packed.dim() - 1 - dim))
    return (codes & (2**bits - 1)).flatten(dim, dim + 1)


def dequantize_groups(weight, zeros, scales, group_size, g_idx=None) -> "torch.Tensor":
    """
    `scales * (weight - zeros)` with one scale and zero point per group of input features, from the `(in, out)` codes
    to an `(out, in)` weight. `g_idx` maps each input feature to its group, e.g. when quantized with `desc_act`.
    """
    if g_idx is None:
        g_idx = torch.arange(weight.shape[0], device=weight.device) // (
            group_size if group_size != -1 else weight.shape[0]
        )
    g_idx = g_idx.long()
    zeros = zeros[:, : scales.shape[1]]
    return (scales[g_idx] * (weight - zeros[g_idx])).T.contiguous()


class GptqDeserialize(ConversionOps):
    """
    Loads the packed tensors of a layer into its gptqmodel `QuantLinear`, then runs the per-layer part of the library's
    post-load init (`hf_convert_gptq_v1_to_v2_format` + `hf_gptqmodel_post_init`) right away: the v1 -> v2 zero points
    when the kernel needs them, then the kernel's `post_init` (e.g. marlin repacks `qweight`, `scales` and the bias and
    sorts `g_idx`). What has to wait for the other layers is left to `gptqmodel_post_init`.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, torch.Tensor]:
        module_name = full_layer_name.rpartition(".")[0]
        module = model.get_submodule(module_name)
        for name, value in input_dict.items():
            value = value[0] if isinstance(value, list) else value
            if value.is_floating_point():  # as the loader would cast it without a load op, e.g. `scales`
                value = value.to(getattr(module, name).dtype)
            setattr(module, name, nn.Parameter(value, requires_grad=False) if name in module._parameters else value)
        if self.hf_quantizer.quantization_config.format == "gptq" and getattr(module, "REQUIRES_FORMAT_V2", False):
            from gptqmodel.utils.model import convert_gptq_v1_to_v2_format_module

            convert_gptq_v1_to_v2_format_module(module, bits=module.bits, pack_dtype=module.pack_dtype)
        bias = getattr(module, "bias", None)
        # Exllamav2 layers slice a scratch space shared by all layers. Some kernels (e.g. marlin) repack the bias,
        # which may load after this op (e.g. from another shard)
        if hasattr(module, "scratch_space_fixed") or (bias is not None and bias.device.type == "meta"):
            module._hf_needs_post_init = True
        else:
            module.post_init()
        # The loader sets the post-init tensors, e.g. marlin's repacked `qweight`
        return {f"{module_name}.{name}": getattr(module, name) for name in input_dict}


def gptqmodel_post_init(model, use_act_order: bool):
    """
    The model-wide part of `hf_gptqmodel_post_init`, for the layers `GptqDeserialize` could not `post_init`: exllamav2
    layers slice one scratch space per device, sized for the largest of them, so it can only be allocated once all of
    them are loaded. The library allocates it for the layers it is given and `post_init`s them.
    """
    from gptqmodel.utils.model import hf_gptqmodel_post_init

    pending = nn.ModuleList([module for module in model.modules() if getattr(module, "_hf_needs_post_init", False)])
    if len(pending) == 0:
        return
    hf_gptqmodel_post_init(pending, use_act_order=use_act_order)
    if hasattr(pending, "device_tensors"):
        model.device_tensors = pending.device_tensors  # the scratch spaces must outlive the call
    for module in pending:
        del module._hf_needs_post_init


class GptqDequantize(ConversionOps):
    """Dense weights from `qweight` (input features packed in int32), `qzeros` (packed), `scales` and `g_idx`."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer
        config = hf_quantizer.quantization_config
        # AutoRound saves e.g. `auto_round:auto_gptq` (gptq v1) or `auto_round` / `auto_round:gptqmodel` (no offset)
        checkpoint_format = getattr(config, "packing_format", None) or config.format
        if checkpoint_format not in ("gptq", "gptq_v2") and not checkpoint_format.startswith("auto_round"):
            raise NotImplementedError(f"Dequantizing the `{checkpoint_format}` GPTQ format is not supported.")
        # gptq v1 checkpoints store the zero points minus one
        self.zero_offset = int(checkpoint_format == "gptq" or checkpoint_format.endswith(("auto_gptq", ":gptq")))

    def convert(self, input_dict, model=None, full_layer_name=None, **kwargs) -> dict[str, list[torch.Tensor]]:
        config = self.hf_quantizer.quantization_config
        output = {}
        for key, qweights in input_dict.items():
            if not key.endswith("qweight"):
                continue
            prefix = key.removesuffix("qweight")
            qweights = qweights if isinstance(qweights, list) else [qweights]
            others = [input_dict.get(prefix + name) for name in ("qzeros", "scales", "g_idx")]
            others = [other if isinstance(other, list) else [other] * len(qweights) for other in others]
            output[prefix + "weight"] = [
                dequantize_groups(
                    unpack_int32(qweight, config.bits, dim=0),
                    unpack_int32(qzeros, config.bits, dim=1) + self.zero_offset,
                    scales,
                    config.group_size,
                    g_idx,
                )
                for qweight, qzeros, scales, g_idx in zip(qweights, *others)
            ]
        return output
