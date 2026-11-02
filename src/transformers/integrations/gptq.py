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
"GPTQ integration file: dequantization of the int32 packed GPTQ format, shared with AWQ and AutoRound"

from ..core_model_loading import ConversionOps
from ..utils import is_torch_available


if is_torch_available():
    import torch


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
