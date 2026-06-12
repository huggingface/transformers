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


from ..utils import is_compressed_tensors_available, is_torch_available, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package.  Loads and restores models to
    quantized state with compressed_tensors.

    When FP8 quantization is detected and a GPU/XPU is available, uses row-wise
    FP8 matmul kernels (torch._scaled_mm) for acceleration.
    Otherwise falls back to the default compressed-tensors dequantize path.
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
        self.run_compressed = quantization_config.run_compressed
        self.quantization_config = quantization_config

        # Detect FP8
        self.is_fp8 = False
        self._activation_scheme = "dynamic"
        self._modules_to_not_convert_ct = []

        if quantization_config.is_fp8:
            self.is_fp8 = True
            ct_qconfig = quantization_config.quantization_config
            if ct_qconfig and ct_qconfig.ignore:
                self._modules_to_not_convert_ct = list(ct_qconfig.ignore)
            # Parse activation scheme
            if ct_qconfig:
                for group in ct_qconfig.config_groups.values():
                    act = group.input_activations
                    if act is not None:
                        self._activation_scheme = "dynamic" if act.dynamic else "static"
                    break

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires compressed-tensors>=0.15.0: "
                "`pip install compressed-tensors>=0.15.0`"
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        return dtype

    def param_needs_quantization(self, model, param_name: str, **kwargs) -> bool:
        # FP8 checkpoints are always pre-quantized; we never quantize on the fly.
        # Online FP8 quantization is intentionally unsupported (use finegrained-fp8).
        return False

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.is_fp8:
            self._process_model_before_weight_loading_fp8(model, **kwargs)
        else:
            self._process_model_before_weight_loading_default(model, **kwargs)

    def _process_model_before_weight_loading_default(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config

        # Always initialize compressed wrappers to match the checkpoint
        apply_quantization_config(model, ct_quantization_config, self.run_compressed)
        if self.quantization_config.is_quantization_compressed:
            self.compressor.compress_model(model=model)

    def _process_model_before_weight_loading_fp8(self, model, **kwargs):
        from ..integrations.compressed_tensors_fp8 import replace_with_compressed_tensors_fp8_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(model, None, model._keep_in_fp32_modules)
        if self._modules_to_not_convert_ct:
            self.modules_to_not_convert = list(set(self.modules_to_not_convert + self._modules_to_not_convert_ct))

        replace_with_compressed_tensors_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            activation_scheme=self._activation_scheme,
            dequantize=self.quantization_config.dequantize,
            pre_quantized=self.pre_quantized,
        )

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Decompress loaded model if necessary - need for qat"""
        if self.is_fp8:
            return

        if self.quantization_config.is_quantization_compressed and not self.run_compressed:
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
        if self.is_fp8:
            # Only trainable when we dequantize back to BF16; the FP8 kernel path is not.
            return self.quantization_config.dequantize
        return True

    @property
    def is_compileable(self) -> bool:
        return True

    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        if self.is_fp8:
            # Only the dequantized BF16 path can be trained further.
            return self.quantization_config.dequantize
        # models need to be decompressed carry out qat
        return not self.run_compressed or not self.quantization_config.is_quantization_compressed

    def is_serializable(self) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True

    def get_quantize_ops(self):
        # Online FP8 quantization is not supported (use finegrained-fp8 instead),
        # so there is never an on-the-fly quantization op to apply.
        return None

    def get_weight_conversions(self):
        """Generic fallback converter for plain ``nn.Linear`` FP8 weights.

        - ``dequantize=False`` (default): keep the weight in FP8 and only reshape
          ``weight_scale`` for the row-wise kernel (the name is kept identical to
          the checkpoint), so the layer runs through
          :class:`CompressedTensorsFP8Linear`.
        - ``dequantize=True``: fold ``weight_scale`` into the FP8 weight to produce
          a plain BF16 ``nn.Linear`` (e.g. for further training / saving in BF16).
        """
        if not self.is_fp8 or not self.pre_quantized:
            return []

        from ..core_model_loading import WeightConverter
        from ..integrations.compressed_tensors_fp8 import (
            CompressedTensorsFp8Dequantize,
            CompressedTensorsScaleConvert,
        )

        if self.quantization_config.dequantize:
            return [
                WeightConverter(
                    source_patterns=["weight$", "weight_scale$"],
                    target_patterns=["weight"],
                    operations=[CompressedTensorsFp8Dequantize()],
                ),
            ]

        return [
            WeightConverter(
                source_patterns=["weight_scale$"],
                target_patterns=["weight_scale"],
                operations=[CompressedTensorsScaleConvert()],
            ),
        ]

    def update_weight_conversions(self, weight_conversions):
        """Walk the model-provided conversion pipeline so FP8 scales are handled
        alongside their weights.

        Plain ``nn.Linear`` weights are handled by the generic converter from
        :meth:`get_weight_conversions` (kept in FP8 by default, or dequantized to
        BF16 when ``dequantize=True``).

        For models whose converters *merge* weights (e.g. MoE experts via a
        ``MergeModulelist`` / concat op), the merged tensor cannot be an FP8
        ``nn.Linear`` and is therefore not replaced by
        :class:`CompressedTensorsFP8Linear`. For each such converter we anchor the
        weight patterns, attach the sibling ``*.weight_scale`` sources to the same
        bucket, and prepend a :class:`CompressedTensorsFp8Dequantize` op so the
        per-expert (weight, scale) pairs are folded into BF16 *before* the merge /
        concat ops collapse the per-expert structure.
        """
        if not self.is_fp8 or not self.pre_quantized:
            return weight_conversions + self.get_weight_conversions()

        from ..core_model_loading import WeightConverter
        from ..integrations.compressed_tensors_fp8 import CompressedTensorsFp8Dequantize

        updated: list = []
        for conv in weight_conversions:
            # Only WeightConverter has ``.operations`` to extend; WeightRenaming
            # rules just pass through untouched.
            if not isinstance(conv, WeightConverter):
                updated.append(conv)
                continue

            weight_sources = [p for p in conv.source_patterns if p.endswith(".weight")]
            if weight_sources:
                anchored_weight = [p + "$" for p in weight_sources]
                scale_sources = [p[: -len(".weight")] + ".weight_scale$" for p in weight_sources]
                other = [p for p in conv.source_patterns if not p.endswith(".weight")]
                new_sources = anchored_weight + scale_sources + other
                new_ops = [CompressedTensorsFp8Dequantize()] + list(conv.operations)
                conv = WeightConverter(
                    source_patterns=new_sources,
                    target_patterns=conv._original_target_patterns,
                    operations=new_ops,
                )
            updated.append(conv)

        # Generic fallback for plain nn.Linear FP8 weights (kept in FP8).
        updated.extend(self.get_weight_conversions())
        return updated
