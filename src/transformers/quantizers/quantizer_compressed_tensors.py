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

from ..utils import is_compressed_tensors_available, is_torch_available, is_torch_greater_or_equal, logging
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter
    from ..integrations.compressed_tensors import CompressedTensorsDequantize


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
    hardware is available (CUDA SM89+, XPU, or MPS with torch >= 2.15). This is opt-in and inference only.

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
        # MPS has FP8 tensors and a `torch._scaled_mm_v2` kernel from torch 2.15 on.
        has_fp8_kernel = (
            torch.xpu.is_available()
            or (torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9))
            or (torch.backends.mps.is_available() and is_torch_greater_or_equal("2.15", accept_dev=True))
        )
        if self.use_fp8_kernel and not has_fp8_kernel:
            logger.warning_once(
                "Ignoring `use_optimized_inference=True`: FP8 matmul kernels need a CUDA GPU with compute capability "
                ">= 8.9 (e.g. 4090/H100), an Intel XPU, or Apple silicon (MPS) with torch >= 2.15, and none was found."
            )
            self.use_fp8_kernel = False

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.quantization_config.dequantize:
            # The model stays unquantized: the `dequantize` load operation runs on every checkpoint weight
            return
        from compressed_tensors.quantization import apply_quantization_config

        ct_config = deepcopy(self.compressor.quantization_config)
        if self.use_fp8_kernel:
            from ..integrations.compressed_tensors import replace_with_compressed_tensors_fp8_linear

            # Quantization may only target a subset of the layers: each config group scopes
            # its scheme with `targets` (class names or `re:` regexes) minus `ignore`.
            fp8_groups = {name: group for name, group in ct_config.config_groups.items() if _is_fp8_scheme(group)}
            ct_config.config_groups = {
                name: group for name, group in ct_config.config_groups.items() if name not in fp8_groups
            }
            self.modules_to_not_convert = self.get_modules_to_not_convert(model, None, model._keep_in_fp32_modules)
            replace_with_compressed_tensors_fp8_linear(
                model,
                targets=[target for group in fp8_groups.values() for target in group.targets],
                ignore=ct_config.ignore,
                modules_to_not_convert=self.modules_to_not_convert,
            )

        apply_quantization_config(model, ct_config, run_compressed=False)
        # Packed formats (e.g. int4 `weight_packed`) need the compressed module layout to
        # receive the checkpoint tensors.
        if self.quantization_config.is_quantization_compressed:
            self.compressor.compress_model(model=model)

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
            return super().get_weight_conversions()

        from ..integrations.compressed_tensors import ConvertFP8LinearScale

        return [
            WeightConverter(
                source_patterns=["weight_scale"],
                target_patterns=["weight_scale"],
                operations=[ConvertFP8LinearScale()],
            )
        ]

    # Every name a compressed-tensors format stores for a weight; a checkpoint only has its own format's
    serialized_names = {
        "weight": [
            "weight",
            "weight_scale",
            "weight_zero_point",
            "weight_global_scale",
            "weight_packed",
            "weight_shape",
        ]
    }

    def get_load_operations(self):
        # No `deserialize`: the serialized tensors load as is into the params compressed-tensors creates
        return {"dequantize": CompressedTensorsDequantize}
