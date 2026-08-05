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

    With `use_optimized_inference=True`, FP8 checkpoints are kept in FP8 and their matmuls run through
    row-wise FP8 kernels (`torch.nn.functional.scaled_mm`) via `CompressedTensorsFP8Linear`, when FP8 matmul
    hardware is available (CUDA SM89+ or XPU). This is opt-in and inference only.

    Otherwise the model goes through the regular compressed-tensors route: `dequantize=True`
    dequantizes the weights at load time, while `dequantize=False` leaves them compressed and lets
    compressed-tensors decompress them on the first forward pass.
    """

    requires_calibration = True
    quantization_config: CompressedTensorsConfig

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        super().__init__(quantization_config, **kwargs)

        # Call post_init here to ensure proper config setup when `dequantize` /
        # `use_optimized_inference` are provided directly via CompressedTensorsConfig, and to avoid
        # duplicate logging.

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

        # `use_optimized_inference` has been resolved against the checkpoint by
        # `CompressedTensorsConfig.post_init`; what is left to check is whether the hardware can run
        # the kernels. When it cannot, the model goes through the regular compressed-tensors route.
        self.use_fp8_kernel = self.quantization_config.use_optimized_inference
        if self.use_fp8_kernel and not (
            torch.xpu.is_available() or (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9))
        ):
            logger.warning_once(
                "Ignoring `use_optimized_inference=True`: FP8 matmul kernels need a CUDA GPU with compute capability "
                ">= 8.9 (e.g. 4090/H100) or an Intel XPU, and none was found."
            )
            self.use_fp8_kernel = False

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

        remaining_config = deepcopy(ct_config)
        remaining_config.config_groups = remaining_groups

        apply_quantization_config(model, remaining_config, run_compressed=False)
        # Packed formats (e.g. int4 `weight_packed`) need the compressed module layout to
        # receive the checkpoint tensors.
        if self.quantization_config.is_quantization_compressed:
            self.compressor.compress_model(model=model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Dequantize the layers loaded through the compressed-tensors wrappers, but only when asked
        for: with `dequantize=False` the weights are left compressed, and the hook `compress_model`
        registered decompresses them on the first forward pass instead.

        FP8-kernel modules were never wrapped (their weights loaded directly in FP8) and hold no
        compressed-tensors scheme, so neither the call below nor that hook touches them."""
        if self.quantization_config.dequantize and self.quantization_config.is_quantization_compressed:
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

    def get_weight_conversions(self):
        """On the FP8 kernel path, a generic converter reshapes the checkpoint ``weight_scale``
        tensors into the row-wise kernel layout once at load time — see
        :class:`ConvertFP8LinearScale`."""
        if not self.use_fp8_kernel:
            return []

        from ..integrations.compressed_tensors import ConvertFP8LinearScale

        return [
            WeightConverter(
                source_patterns=["weight_scale"],
                target_patterns=["weight_scale"],
                operations=[ConvertFP8LinearScale()],
            )
        ]

    def update_weight_conversions(self, weight_conversions):
        """Attach the quantization sources (scales, packed weights) of MoE expert converters
        to their bucket and prepend a :class:`DecompressExperts` op, so the per-expert
        (weight, scale) pairs are dequantized *before* the merge / concat ops collapse the
        per-expert structure. FP8 checkpoints keep the plain ``weight`` name; packed formats
        use ``weight_packed`` / ``weight_shape``.

        The generic converters from :meth:`get_weight_conversions` are appended at the end:
        converters are matched in order, so the expert converters keep their scale keys.
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

        updated.extend(self.get_weight_conversions())
        return updated
