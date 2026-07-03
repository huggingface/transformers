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
"CompressedTensors integration file"

import torch
from torch import nn

from ..core_model_loading import ConversionOps


class DecompressExperts(ConversionOps):
    """
    Dequantize MoE layers when they are in new layout, because they aren't `nn.Module` anymore!

    Takes packed weights and scales from the loaded state dict, creates a dummy Module
    to take advantage of higher-lvl API `decompress_module` and dequantizes all weights.

    Requires MoE conversion to be defined on conversion mapping, so that decompressed weights
    are stacked/merged for all experts.
    """

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        from compressed_tensors.compressors import BaseCompressor
        from compressed_tensors.compressors.format import infer_module_format

        ct_quantization_config = self.hf_quantizer.compressor.quantization_config

        quantization_scheme = list(ct_quantization_config.config_groups.values())[0]
        format = quantization_scheme.format or infer_module_format(nn.Linear, quantization_scheme)
        compressor = BaseCompressor.get_value_from_registry(format)

        class DummyModule(nn.Module):
            def __init__(self, weight, scale, shape):
                super().__init__()
                self.weight_packed = nn.Parameter(weight, requires_grad=False)
                self.weight_scale = nn.Parameter(scale, requires_grad=False)
                self.weight_shape = nn.Parameter(shape, requires_grad=False)

        # `pack_factor` low-bit weights are packed per int32 along the packed dim.
        pack_factor = 32 // quantization_scheme.weights.num_bits

        # Per-expert compressed projections of size (input-dim; output-dim)
        processed_out = {}
        for key, value in input_dict.items():
            if "weight_packed" not in key:
                continue
            quantized = value
            scales = input_dict[key.replace("weight_packed", "weight_scale")]

            # Pre-allocate the stacked output buffer to reduce cuda mem fragmentation
            # Without pre-allocation the loop accumulates N tensors per expert and next
            # `MergeModulelist` stacks the full list for MoE kernels compatipility, i.e. x2 memory
            output = None
            for i, (quant, scale) in enumerate(zip(quantized, scales)):
                # The checkpoint's `weight_shape` is a 2D tensor of `(out-dim, in-dim)`
                # Under TP/EP sharding it leaves the 2-element `weight_shape` empty on most ranks
                # Packed tensor can be used instead to rebuild `weight_shape`
                shape = torch.tensor([quant.shape[0], quant.shape[1] * pack_factor])
                module = DummyModule(quant, scale, shape)
                module.quantization_scheme = quantization_scheme
                compressor.decompress_module(module)

                if output is None:
                    # Use the first expert's decompressed shape/dtype to allocate full buffer.
                    output = torch.empty(
                        (len(quantized), *module.weight.shape),
                        dtype=module.weight.dtype,
                        device=module.weight.device,
                    )
                output[i].copy_(module.weight)
                # explicitly free intermediate tensors so it does not accumulate across iterations
                del module

            del quantized, scales
            if output is not None:
                # Return a single pre-stacked tensor instead of a list. `MergeModulelist`
                # passes it through without an extra `torch.stack` copy -> no x2 memory overhead
                processed_out[key] = output

        return processed_out

    @property
    def reverse_op(self) -> "ConversionOps":
        return None  # FIXME
