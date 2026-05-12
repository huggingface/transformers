# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from ..core_model_loading import ConversionOps, WeightConverter
from ..utils import is_compressed_tensors_available, is_torch_available, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package.  Loads and restores models to
    quantized state with compressed_tensors
    """

    requires_calibration = True
    quantization_config: CompressedTensorsConfig

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )

        # Call post_init here to ensure proper config setup when `run_compressed`
        # is provided directly via CompressedTensorsConfig, and to avoid duplicate logging.

        quantization_config.post_init()
        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)
        self.run_compressed = quantization_config.run_compressed
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype != torch.float16:
            logger.info("We suggest you to set `dtype=torch.float16` for better efficiency with compressed_tensors.")
        return dtype

    def _process_model_before_weight_loading(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config

        # Always initialize compressed wrappers to match the checkpoint
        apply_quantization_config(model, ct_quantization_config, self.run_compressed)
        if (
            self.quantization_config.is_quantization_compressed
            or self.quantization_config.is_sparsification_compressed
        ):
            self.compressor.compress_model(model=model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Decompress loaded model if necessary - need for qat"""

        if (self.quantization_config.is_quantization_compressed and not self.run_compressed) or (
            self.quantization_config.is_sparsification_compressed
        ):
            self.compressor.decompress_model(model=model)

    # NOTE: TP plan override for compressed tensors removed - unsupported styles were used.
    # TODO: Implement proper TP support for compressed tensors quantization
    def update_tp_plan(self, config):
        additional_plan = {
            "layers.*.feed_forward.experts.*.gate_proj.weight": "colwise",
            "layers.*.feed_forward.experts.*.gate_proj.weight_scale": "colwise",
            "layers.*.feed_forward.experts.*.up_proj.weight": "colwise",
            "layers.*.feed_forward.experts.*.up_proj.weight_scale": "colwise",
            "layers.*.feed_forward.experts.*.down_proj.weight": "rowwise",
        }
        if config.get_text_config() is not None and config.get_text_config().base_model_tp_plan is not None:
            config.get_text_config().base_model_tp_plan.update(additional_plan)

        return config

    @property
    def is_trainable(self):
        return True

    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        # models need to be decompressed carry out qat
        return not self.run_compressed or not self.quantization_config.is_quantization_compressed

    def is_serializable(self) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True

    def get_weight_conversions(self):
        # FIXME: can't always just add conversions blindly! Should check that loaded scheme's `target`
        # matches param names/classes (i.e. `is_moe_proj_in_config_scheme`) and that model already has simple
        # MoE conversion in the mapping (i.e. `has_moe_conversion(self)`). Then we remove simple MoE and replace it
        # with `DecompressExperts` below

        # Only models that have already been quantized can be loaded atm, so we can
        # assume that if `hasattr(self, hf_quantizer)` and `has_moe_conversion(self)` and `is_moe_proj_in_config_scheme`
        # then it needs special dequantization for MoE projections
        # NOTE: MoE conversion should happen AFTER decompression! Already hardcoded in convesion
        dequant_conversions = [
            WeightConverter(
                source_patterns=[
                    r"mlp.experts.\d+.down_proj.weight_packed",
                    r"mlp.experts.\d+.down_proj.weight_scale",
                    r"mlp.experts.\d+.down_proj.weight_shape",
                ],
                target_patterns=r"mlp.experts.down_proj",
                operations=[DecompressExperts(self)],
            ),
            WeightConverter(
                source_patterns=[
                    r"mlp.experts.\d+.gate_proj.weight_packed",
                    r"mlp.experts.\d+.gate_proj.weight_scale",
                    r"mlp.experts.\d+.gate_proj.weight_shape",
                    r"mlp.experts.\d+.up_proj.weight_packed",
                    r"mlp.experts.\d+.up_proj.weight_scale",
                    r"mlp.experts.\d+.up_proj.weight_shape",
                ],
                target_patterns=r"mlp.experts.gate_up_proj",
                operations=[DecompressExperts(self)],
            ),
        ]

        return dequant_conversions


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
        from torch import nn

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

        # Per-expert compressed projections of size (input-dim; output-dim)
        processed_out = []
        input_dict = self.group_input_dict(input_dict)
        for value in input_dict.values():
            quantized, scales, shapes = (
                value["weight_packed"],
                value["weight_scale"],
                value["weight_shape"],
            )

            # Create a dummy module to not rely on low-lvl API and iter over each expert
            decompessed_tensors = []
            for quant, scale, shape in zip(quantized, scales, shapes):
                module = DummyModule(quant, scale, shape)

                module.quantization_scheme = quantization_scheme
                compressor.decompress_module(module)
                decompessed_tensors.append(module.weight)

            del quantized, scales, shapes, module
            processed_out.append(torch.stack(decompessed_tensors, dim=0))

        processed_out = torch.cat(processed_out, dim=1)
        return {target_patterns[0]: processed_out}

    def group_input_dict(self, input_dict: dict) -> dict:
        weight_grouped = {}
        for key in input_dict:
            if "weight_packed" in key:
                layername = key.replace(".weight_packed", "")
                weight_grouped.setdefault(layername, {})["weight_packed"] = input_dict[key]
            elif "weight_scale" in key:
                layername = key.replace(".weight_scale", "")
                weight_grouped.setdefault(layername, {})["weight_scale"] = input_dict[key]
            elif "weight_shape" in key:
                layername = key.replace(".weight_shape", "")
                weight_grouped.setdefault(layername, {})["weight_shape"] = input_dict[key]

        return weight_grouped

    @property
    def reverse_op(self) -> "ConversionOps":
        return None  # FIXME
