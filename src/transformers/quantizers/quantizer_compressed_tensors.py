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

import re
from copy import deepcopy

from ..utils import (
    is_compressed_tensors_available,
    is_torch_available,
    is_torch_greater_or_equal,
    is_torchao_available,
    logging,
)
from ..utils.quantization_config import CompressedTensorsConfig
from .base import HfQuantizer


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter, build_glob_alternation
    from ..integrations.compressed_tensors import CompressedTensorsDequantize, CompressedTensorsNVFP4Deserialize


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

    With `weight_only=True`, NVFP4 checkpoints use torchao weights with BF16 activations.
    This changes activation precision, does not select a fused kernel, and is inference only.

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
        self.use_nvfp4_weight_only = quantization_config.weight_only
        self.use_fp8_kernel = quantization_config.use_optimized_inference
        self.nvfp4_modules: set[str] = set()
        self.nvfp4_base_model_prefix = ""
        if self.use_nvfp4_weight_only:
            self.serialized_names = {"weight": [*self.serialized_names["weight"], "input_global_scale"]}
        from compressed_tensors.compressors import ModelCompressor

        self.compressor = ModelCompressor.from_compression_config(quantization_config)

    def validate_environment(self, *args, **kwargs):
        if not is_compressed_tensors_available():
            raise ImportError(
                "Using `compressed_tensors` quantized models requires compressed-tensors>=0.15.0: "
                "`pip install compressed-tensors>=0.15.0`"
            )

        if self.use_nvfp4_weight_only:
            # NVFP4Tensor renamed `_is_swizzled_scales` in torchao 0.16.
            if not is_torchao_available(min_version="0.16.0"):
                raise ImportError(
                    "NVFP4 weight-only loading requires torchao>=0.16.0. "
                    "Install a compatible build with `pip install 'torchao>=0.16.0'`."
                )
            try:
                from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor  # noqa: F401
            except ImportError as error:
                raise ImportError(
                    "NVFP4 weight-only loading requires a torchao build providing "
                    "`torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor`. Upgrade torchao."
                ) from error
            device_map = kwargs.get("device_map")
            if device_map is not None:
                if not isinstance(device_map, dict) or set(device_map) != {""}:
                    raise ValueError("NVFP4 weight-only loading requires an explicit whole-model device_map.")
                device = device_map[""]
                if device == "disk" or torch.device(device).type not in ("cpu", "cuda", "xpu"):
                    raise ValueError(
                        "NVFP4 weight-only loading supports a single CPU, CUDA or XPU device, without offload."
                    )
            logger.warning_once(
                "NVFP4 weight-only loading disables activation quantization and uses BF16 activations (W4A16). "
                "Torchao currently dequantizes weights for each Linear call; this is not a fused FP4 kernel."
            )
            return

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

    def update_dtype(self, dtype):
        if self.use_nvfp4_weight_only and dtype != torch.bfloat16:
            raise ValueError("NVFP4 weight-only loading requires `dtype=torch.bfloat16`.")
        return dtype

    def param_element_size(self, model, param_name: str, param: "torch.Tensor") -> float:
        module_name, _, attribute = param_name.rpartition(".")
        if self.use_nvfp4_weight_only and attribute == "weight" and module_name in self.nvfp4_modules:
            # Packed FP4, one FP8 scale per 16 weights, and one FP32 global scale per Linear.
            return 0.5 + 1 / 16 + 4 / param.numel()
        return super().param_element_size(model, param_name, param)

    def _process_model_before_weight_loading(self, model, **kwargs):
        if self.use_nvfp4_weight_only:
            from compressed_tensors.utils.match import match_named_modules

            if getattr(model.config, "distributed_config", None) is not None:
                raise ValueError("NVFP4 weight-only loading does not support tensor or expert parallelism.")
            config = self.quantization_config.quantization_config
            if config is None:
                raise ValueError("NVFP4 weight-only loading requires a checkpoint quantization config.")
            self.nvfp4_modules = set()
            self.nvfp4_base_model_prefix = getattr(model, "base_model_prefix", "")
            fp32_pattern = None
            if model._keep_in_fp32_modules_strict:
                fp32_pattern, _, _ = build_glob_alternation(list(model._keep_in_fp32_modules_strict))
            for group in config.config_groups.values():
                for name, module in match_named_modules(model, group.targets, ignore=config.ignore):
                    if type(module) is not torch.nn.Linear:
                        raise ValueError(
                            f"NVFP4 weight-only loading supports ordinary nn.Linear modules only: {name}."
                        )
                    if (
                        module.in_features <= 0
                        or module.out_features <= 0
                        or module.in_features % 16
                        or module.weight.dtype != torch.bfloat16
                    ):
                        raise ValueError(
                            f"NVFP4 module {name} requires BF16 weights, positive dimensions and in_features divisible by 16."
                        )
                    if fp32_pattern is not None and fp32_pattern.search(f"{name}.weight"):
                        raise ValueError(
                            f"NVFP4 weight-only loading cannot quantize an FP32-preserved weight: {name}."
                        )
                    self.nvfp4_modules.add(name)
            if not self.nvfp4_modules:
                raise ValueError("No Linear modules matched the NVFP4 weight-only targets and ignore patterns.")
            tied_weights = getattr(model, "all_tied_weights_keys", {})
            tied_quantized = {f"{name}.weight" for name in self.nvfp4_modules} & (
                set(tied_weights) | set(tied_weights.values())
            )
            if tied_quantized:
                raise ValueError(
                    f"NVFP4 weight-only loading does not support tied quantized weights: {sorted(tied_quantized)}."
                )
            selected_weights = {id(model.get_submodule(name).weight) for name in self.nvfp4_modules}
            seen = set()
            for name, parameter in model.named_parameters(remove_duplicate=False):
                if id(parameter) in selected_weights:
                    if id(parameter) in seen:
                        raise ValueError(f"NVFP4 weight-only loading does not support tied quantized weights: {name}.")
                    seen.add(id(parameter))
            # No CT wrappers or companion parameters: the deserializer constructs the torchao weights.
            return
        if self.quantization_config.dequantize:
            # The model stays dense: the `dequantize` load operation runs on every checkpoint weight
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

    def _process_model_after_weight_loading(self, model, **kwargs):
        if self.use_nvfp4_weight_only:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

            missing = [
                name for name in self.nvfp4_modules if not isinstance(model.get_submodule(name).weight, NVFP4Tensor)
            ]
            if missing:
                raise ValueError(f"Missing NVFP4 packed weights for selected modules: {missing}.")
        return model

    @property
    def is_trainable(self):
        # The FP8 kernel path is inference-only; load with `dequantize=True` to fine-tune.
        return not (self.use_fp8_kernel or self.use_nvfp4_weight_only)

    @property
    def is_compileable(self) -> bool:
        return not self.use_nvfp4_weight_only

    @property
    def is_qat_trainable(self) -> bool:
        """Loaded Models can carry out quantization aware training"""
        if self.use_fp8_kernel or self.use_nvfp4_weight_only:
            return False
        # models need to be decompressed carry out qat
        return self.quantization_config.dequantize or not self.quantization_config.is_quantization_compressed

    def is_serializable(self) -> bool:
        """Models quantized using compressed tensors can be saved to disk"""
        return not self.use_nvfp4_weight_only

    def update_weight_conversions(self, weight_conversions, model):
        if self.use_nvfp4_weight_only:
            if any(isinstance(converter, WeightConverter) for converter in weight_conversions):
                raise ValueError(
                    "NVFP4 weight-only loading does not support model-level WeightConverters, "
                    "including fused or grouped weight conversions, even for unquantized modules."
                )
            return weight_conversions + self.get_weight_conversions()
        return super().update_weight_conversions(weight_conversions, model)

    def get_weight_conversions(self):
        """On the FP8 kernel path, a generic converter reshapes the checkpoint ``weight_scale``
        tensors into the row-wise kernel layout once at load time — see
        :class:`ConvertFP8LinearScale`."""
        if self.use_nvfp4_weight_only:
            converters = []
            for name in sorted(self.nvfp4_modules):
                name = name.removeprefix(f"{self.nvfp4_base_model_prefix}.")
                converter = WeightConverter(
                    source_patterns=[rf"^{re.escape(name)}\.{param}$" for param in self.serialized_names["weight"]],
                    target_patterns=[f"{name}.weight"],
                    operations=[CompressedTensorsNVFP4Deserialize(self)],
                )
                # Match exact module names with or without the checkpoint's base-model prefix.
                converter.scope_prefix = ""
                converter.base_model_prefix = self.nvfp4_base_model_prefix
                converters.append(converter)
            return converters
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
        if self.use_nvfp4_weight_only:
            return {"deserialize": CompressedTensorsNVFP4Deserialize}
        # No `deserialize`: the serialized tensors load as is into the params compressed-tensors creates
        return {"dequantize": CompressedTensorsDequantize}
