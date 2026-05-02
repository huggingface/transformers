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


from ..utils import is_compressed_tensors_available, is_torch_available, is_torch_xpu_available, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def _is_ct_fp8_config(quantization_config: CompressedTensorsConfig) -> bool:
    """Check if a CompressedTensorsConfig describes FP8 quantization."""
    ct_qconfig = quantization_config.quantization_config
    if ct_qconfig is None:
        return False
    for group in ct_qconfig.config_groups.values():
        weights = group.weights
        if weights is not None and weights.type == "float" and weights.num_bits == 8:
            return True
    return False


class CompressedTensorsHfQuantizer(HfQuantizer):
    """
    Quantizer for the compressed_tensors package.  Loads and restores models to
    quantized state with compressed_tensors.

    When FP8 quantization is detected and a GPU/XPU is available, uses row-wise
    FP8 matmul kernels (torch._scaled_mm on XPU, fbgemm on CUDA) for acceleration.
    Otherwise falls back to the default compressed-tensors dequantize path.
    """

    requires_calibration = True
    quantization_config: CompressedTensorsConfig

    def __init__(self, quantization_config: CompressedTensorsConfig, **kwargs):
        # For FP8 with GPU/XPU, we don't require calibration (online quantization is supported)
        if _is_ct_fp8_config(quantization_config) and (
            is_torch_available() and (torch.cuda.is_available() or is_torch_xpu_available())
        ):
            self.requires_calibration = False

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

        # Detect FP8 and decide whether to use FP8 kernel path
        self._use_fp8_kernel = False
        self._fp8_dequantize = False
        self._activation_scheme = "dynamic"
        self._modules_to_not_convert_ct = []

        if _is_ct_fp8_config(quantization_config):
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

            if torch.cuda.is_available() or is_torch_xpu_available():
                self._use_fp8_kernel = True
                self.requires_calibration = False
                logger.info("Compressed-tensors FP8 detected — using FP8 kernel path for acceleration.")
            else:
                logger.info(
                    "Compressed-tensors FP8 detected but no GPU/XPU found. "
                    "Falling back to default compressed-tensors path."
                )

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires the compressed-tensors library: "
                "`pip install compressed-tensors`"
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if not self._use_fp8_kernel and dtype != torch.float16:
            logger.info("We suggest you to set `dtype=torch.float16` for better efficiency with compressed_tensors.")
        return dtype

    def param_needs_quantization(self, model, param_name: str, **kwargs) -> bool:
        if not self._use_fp8_kernel:
            return False
        from ..integrations.compressed_tensors_fp8 import CTFP8Linear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, CTFP8Linear):
            if self.pre_quantized or tensor_name == "bias":
                return False
            return True
        return False

    def param_element_size(self, model, param_name: str, param: "torch.Tensor") -> float:
        if self._use_fp8_kernel and self.param_needs_quantization(model, param_name):
            return 1  # 8-bit
        return super().param_element_size(model, param_name, param)

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self._use_fp8_kernel:
            self._process_model_before_weight_loading_fp8(model, **kwargs)
        else:
            self._process_model_before_weight_loading_default(model, **kwargs)

    def _process_model_before_weight_loading_default(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        ct_quantization_config = self.compressor.quantization_config

        # Always initialize compressed wrappers to match the checkpoint
        apply_quantization_config(model, ct_quantization_config, self.run_compressed)
        if (
            self.quantization_config.is_quantization_compressed
            or self.quantization_config.is_sparsification_compressed
        ):
            self.compressor.compress_model(model=model)

    def _process_model_before_weight_loading_fp8(self, model, **kwargs):
        from ..integrations.compressed_tensors_fp8 import replace_with_ct_fp8_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(model, None, model._keep_in_fp32_modules)
        if self._modules_to_not_convert_ct:
            self.modules_to_not_convert = list(set(self.modules_to_not_convert + self._modules_to_not_convert_ct))

        replace_with_ct_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            activation_scheme=self._activation_scheme,
            dequantize=self._fp8_dequantize,
            pre_quantized=self.pre_quantized,
        )

    def _process_model_after_weight_loading(self, model, **kwargs):
        """Decompress loaded model if necessary - need for qat"""
        if self._use_fp8_kernel:
            return

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
        if self._use_fp8_kernel:
            return False
        return True

    @property
    def is_compileable(self) -> bool:
        return True

    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        if self._use_fp8_kernel:
            return False
        # models need to be decompressed carry out qat
        return not self.run_compressed or not self.quantization_config.is_quantization_compressed

    def is_serializable(self) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return True

    def get_quantize_ops(self):
        if not self._use_fp8_kernel or self.pre_quantized:
            return None
        from ..integrations.compressed_tensors_fp8 import CTFP8PerRowQuantize

        return CTFP8PerRowQuantize(self)

    def get_weight_conversions(self):
        if not self._use_fp8_kernel:
            return []

        from ..core_model_loading import WeightConverter
        from ..integrations.compressed_tensors_fp8 import (
            CompressedTensorsFp8Dequantize,
            CompressedTensorsScaleConvert,
        )

        if self.pre_quantized and self._fp8_dequantize:
            return [
                WeightConverter(
                    source_patterns=["weight$", "weight_scale"],
                    target_patterns="weight",
                    operations=[CompressedTensorsFp8Dequantize(self)],
                )
            ]

        if self.pre_quantized:
            return [
                WeightConverter(
                    source_patterns=["weight_scale$"],
                    target_patterns=["weight_scale_inv"],
                    operations=[CompressedTensorsScaleConvert(self)],
                ),
            ]

        return []
