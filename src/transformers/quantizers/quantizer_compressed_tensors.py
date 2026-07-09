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

from copy import deepcopy

from ..utils import is_compressed_tensors_available, is_torch_available, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter
    from ..integrations.compressed_tensors import DecompressExperts, get_experts_scheme


logger = logging.get_logger(__name__)


def _is_fp8_scheme(scheme) -> bool:
    """Whether a compressed-tensors quantization scheme quantizes weights to FP8."""
    weights = scheme.weights
    return weights is not None and weights.type == "float" and weights.num_bits == 8


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package. Loads and restores models to
    quantized state with compressed_tensors.

    FP8 checkpoints are kept in FP8 and matmuls run through row-wise FP8 kernels
    (torch._scaled_mm) via `CompressedTensorsFP8Linear` when FP8 matmul hardware is
    available (CUDA SM89+ or XPU). Otherwise the model is dequantized at load time
    through the regular compressed-tensors route.
    """

    requires_calibration = True
    quantization_config: CompressedTensorsConfig

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        # Call post_init here to ensure proper config setup when `run_compressed`
        # is provided directly via CompressedTensorsConfig, and to avoid duplicate logging.

        quantization_config.post_init()
        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires compressed-tensors>=0.15.0: "
                "`pip install compressed-tensors>=0.15.0`"
            )

        # The FP8 kernel path needs FP8 matmul hardware (XPU, or CUDA SM89+). Without it, fall
        # back to dequantizing at load time (regular compressed-tensors route).
        if self.quantization_config.has_fp8_modules and not self.quantization_config.dequantize:
            has_fp8_hardware = torch.xpu.is_available() or (
                torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
            )
            if not has_fp8_hardware:
                logger.warning_once(
                    "FP8 compressed-tensors models need a CUDA GPU with compute capability >= 8.9 (e.g. "
                    "4090/H100) or an Intel XPU to run FP8 matmul kernels; none was found. Defaulting to "
                    "dequantizing the model to the original dtype."
                )
                self.quantization_config.dequantize = True

        if not self.quantization_config.has_fp8_modules and not self.quantization_config.dequantize:
            logger.info(
                "Dequantizing the model as there are no kernels for this scheme. We only support fp8 for now for compressed tensors. "
            )
            self.quantization_config.dequantize = True

        self.use_fp8_kernel = self.quantization_config.has_fp8_modules and not self.quantization_config.dequantize

    def _process_model_before_weight_loading(self, model, **kwargs):
        ct_config = self.compressor.quantization_config
        remaining_groups = dict(ct_config.config_groups)

        if self.use_fp8_kernel:
            from ..integrations.compressed_tensors import replace_with_compressed_tensors_fp8_linear

            # Quantization may only target a subset of the layers: each config group scopes
            # its scheme with `targets` (class names or `re:` regexes) minus `ignore`.
            fp8_groups = {name: group for name, group in remaining_groups.items() if _is_fp8_scheme(group)}
            remaining_groups = {name: group for name, group in remaining_groups.items() if name not in fp8_groups}

            self.modules_to_not_convert = self.get_modules_to_not_convert(model, None, model._keep_in_fp32_modules)
            targets = [target for group in fp8_groups.values() for target in group.targets]
            replace_with_compressed_tensors_fp8_linear(
                model,
                targets=targets,
                ignore=ct_config.ignore,
                modules_to_not_convert=self.modules_to_not_convert,
            )

        if not remaining_groups:
            return

        from compressed_tensors.quantization import apply_quantization_config

        # Layers quantized with the remaining (non-FP8) schemes go through the regular
        # compressed-tensors wrappers and are dequantized after loading — there are no
        # kernels for those schemes.
        remaining_config = deepcopy(ct_config)
        remaining_config.config_groups = remaining_groups

        apply_quantization_config(model, remaining_config, run_compressed=False)
        # Packed formats (e.g. int4 `weight_packed`) need the compressed module layout to
        # receive the checkpoint tensors.
        if self.quantization_config.is_quantization_compressed:
            self.compressor.compress_model(model=model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Decompress the layers loaded through the compressed-tensors wrappers. FP8-kernel
        modules were never wrapped (their weights loaded directly in FP8), so decompression
        does not touch them."""
        if self.quantization_config.is_quantization_compressed:
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
        # The FP8 kernel path is inference-only; load with `dequantize=True` to fine-tune.
        return not self.use_fp8_kernel

    @property
    def is_compileable(self) -> bool:
        return True

    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        if self.use_fp8_kernel:
            return False
        # models need to be decompressed carry out qat
        return self.quantization_config.dequantize or not self.quantization_config.is_quantization_compressed

    def is_serializable(self) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True

    def update_weight_conversions(self, weight_conversions):
        """Attach the quantization sources (scales, packed weights) of MoE expert converters
        to their bucket and prepend a :class:`DecompressExperts` op, so the per-expert
        (weight, scale) pairs are dequantized *before* the merge / concat ops collapse the
        per-expert structure. FP8 checkpoints keep the plain ``weight`` name; packed formats
        use ``weight_packed`` / ``weight_shape``.
        """
        updated: list = []
        for conv in weight_conversions:
            # Only WeightConverter for experts have ``.operations`` to extend with the dequant op
            if not isinstance(conv, WeightConverter) or any("experts" not in p for p in conv.source_patterns):
                updated.append(conv)
                continue
            weight_sources = [p for p in conv.source_patterns if p.endswith(".weight")]
            if weight_sources:
                scheme = get_experts_scheme(self.quantization_config.quantization_config)
                scale_sources = [p + "_scale$" for p in weight_sources]
                other = [p for p in conv.source_patterns if not p.endswith(".weight")]
                if _is_fp8_scheme(scheme):
                    # Merged experts cannot stay FP8 (they are not nn.Linear): they are
                    # dequantized to the model dtype before the merge. The weight patterns
                    # must be anchored with `$`: patterns are regex-searched, so unanchored
                    # `.weight` would also match the `.weight_scale` keys.
                    new_sources = [p + "$" for p in weight_sources] + scale_sources + other
                else:
                    packed_weight = [p + "_packed$" for p in weight_sources]
                    shape_sources = [p + "_shape$" for p in weight_sources]
                    new_sources = packed_weight + scale_sources + shape_sources + other
                new_ops = [DecompressExperts(self, scheme=scheme)] + list(conv.operations)
                conv = WeightConverter(
                    source_patterns=new_sources,
                    target_patterns=conv._original_target_patterns,
                    operations=new_ops,
                )
            updated.append(conv)
        return updated
