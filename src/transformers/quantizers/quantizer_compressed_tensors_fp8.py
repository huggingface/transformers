from typing import TYPE_CHECKING

from ..utils import is_accelerate_available, is_torch_available, is_torch_xpu_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import FineGrainedFP8Config

logger = logging.get_logger(__name__)


class CompressedTensorsFP8HfQuantizer(HfQuantizer):
    """
    Quantizer for loading compressed-tensors FP8 checkpoints.

    Supports per-channel and per-tensor FP8 weight quantization from
    compressed-tensors format (weight_scale, input_scale naming convention).
    Weights are stored in FP8 and dequantized on-the-fly during forward.
    """

    requires_calibration = False
    quantization_config: "FineGrainedFP8Config"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self._ct_config = getattr(quantization_config, "_original_ct_config", None)

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a compressed-tensors FP8 model requires accelerate (`pip install accelerate`)")

        if self.quantization_config.dequantize:
            return

        if not torch.cuda.is_available() and not is_torch_xpu_available():
            if self.pre_quantized:
                logger.warning_once(
                    "Using compressed-tensors FP8 models requires a GPU or XPU. "
                    "Defaulting to dequantize mode (BF16) since no GPU/XPU is available."
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError("No GPU or XPU found. A GPU or XPU is needed for FP8 quantization.")

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA or XPU device available, "
                "please set device_map='cuda' or 'xpu'."
            )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from ..integrations.compressed_tensors_fp8 import CTFP8Linear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, CTFP8Linear):
            if self.pre_quantized or tensor_name == "bias":
                return False
            return True
        return False

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param: "torch.Tensor") -> float:
        if self.param_needs_quantization(model, param_name):
            return 1  # 8-bit
        return super().param_element_size(model, param_name, param)

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations.compressed_tensors_fp8 import replace_with_ct_fp8_linear

        # Auto-detect activation scheme and weight strategy from CT config
        if self._ct_config is not None:
            ct_groups = self._ct_config.get("config_groups", {})
            for group in ct_groups.values():
                if isinstance(group, dict):
                    act_cfg = group.get("input_activations")
                    if act_cfg:
                        if act_cfg.get("dynamic", True):
                            self.quantization_config.activation_scheme = "dynamic"
                        else:
                            self.quantization_config.activation_scheme = "static"

                    weight_cfg = group.get("weights")
                    if weight_cfg:
                        strategy = weight_cfg.get("strategy", "tensor")
                        if strategy in ("channel", "tensor"):
                            self.quantization_config.weight_block_size = None
                    break

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        # Also respect the ignore list from compressed-tensors config
        if self._ct_config is not None:
            ct_ignore = self._ct_config.get("ignore", [])
            if ct_ignore:
                self.modules_to_not_convert = list(set(self.modules_to_not_convert + ct_ignore))

        model = replace_with_ct_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def get_weight_conversions(self):
        from ..core_model_loading import WeightConverter
        from ..integrations.compressed_tensors_fp8 import (
            CompressedTensorsFp8Dequantize,
            CompressedTensorsScaleConvert,
        )

        if self.pre_quantized and self.quantization_config.dequantize:
            # Dequantize CT FP8 to BF16
            source_patterns = ["weight$", "weight_scale"]
            return [
                WeightConverter(
                    source_patterns=source_patterns,
                    target_patterns="weight",
                    operations=[CompressedTensorsFp8Dequantize(self)],
                )
            ]

        if self.pre_quantized:
            # Native FP8 loading: rename weight_scale -> weight_scale_inv
            return [
                WeightConverter(
                    source_patterns=["weight_scale$"],
                    target_patterns=["weight_scale_inv"],
                    operations=[CompressedTensorsScaleConvert(self)],
                ),
            ]

        return []

