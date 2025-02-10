import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from packaging import version

from ..utils import is_accelerate_available, is_torch_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)


class FineGrainedFP8HfQuantizer(HfQuantizer):
    """
    FP8 quantization implementation supporting both standard and MoE models.
    Supports both e4m3fn formats based on platform.
    """

    requires_parameters_quantization = True
    requires_calibration = False
    required_packages = ["accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading an FP8 quantized model requires accelerate (`pip install accelerate`)")

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into FP8 weights from tf/flax weights is currently not supported, "
                "please make sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for FP8 quantization.")

        compute_capability = torch.cuda.get_device_capability()
        major, minor = compute_capability
        if major < 9:
            raise ValueError(
                "FP8 quantized models is only supported on GPUs with compute capability >= 9.0 (e.g H100)"
            )
        torch_version = version.parse(importlib.metadata.version("torch"))
        if torch_version < version.parse("2.1.0"):
            raise RuntimeError(
                "float8_e4m3fn is only supported in torch versions >= 2.1.0, please upgrade your pytorch version"
            )
        device_map = kwargs.get("device_map", None)
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model. To remove this warning, pass device_map = 'cuda'. "
            )
        elif device_map is not None:
            if (
                not self.pre_quantized
                and isinstance(device_map, dict)
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load an FP8 model with a device_map that contains a cpu/disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the cpu/disk device from the device_map."
                )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            logger.info("Setting torch_dtype to torch.float32 as no torch_dtype was specified in from_pretrained")
            torch_dtype = torch.float32
        return torch_dtype

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: Optional[List[str]] = None,
    ):
        """
        Quantizes weights to FP8 format using either:
        - Block-wise quantization when weight_block_size is provided
        - Per-tensor quantization when weight_block_size is None
        """
        from accelerate.utils import set_module_tensor_to_device

        set_module_tensor_to_device(model, param_name, target_device, param_value)

        module, tensor_name = get_module_from_name(model, param_name)

        # Get FP8 min/max values
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        block_size_m, block_size_n = self.quantization_config.weight_block_size

        rows, cols = param_value.shape[-2:]

        if rows % block_size_m != 0 or cols % block_size_n != 0:
            raise ValueError(
                f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
            )
        param_value_orig_shape = param_value.shape

        param_value = param_value.reshape(
            -1, rows // block_size_m, block_size_m, cols // block_size_n, block_size_n
        ).permute(0, 1, 3, 2, 4)

        # Calculate scaling factor for each block
        max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
        scale = fp8_max / max_abs
        scale_orig_shape = scale.shape
        scale = scale.unsqueeze(-1).unsqueeze(-1)

        # Quantize the weights
        quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        quantized_param = quantized_param.permute(0, 1, 3, 2, 4)
        # Reshape back to matrix shape
        quantized_param = quantized_param.reshape(param_value_orig_shape)

        # Reshape scale to match the number of blocks
        scale = scale.reshape(scale_orig_shape).squeeze().reciprocal()

        module._buffers[tensor_name] = quantized_param.to(target_device)
        module._buffers["weight_scale_inv"] = scale.to(target_device)

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        from ..integrations.finegrained_fp8 import FP8Linear

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, FP8Linear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != torch.float8_e4m3fn:
                    raise ValueError("Expect quantized weights but got an unquantized weight")
                return False
            else:
                if tensor_name == "weight_scale_inv":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
        return False

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        modules_to_not_convert: List[str] = [],
        **kwargs,
    ):
        from ..integrations.finegrained_fp8 import replace_with_fp8_linear

        self.modules_to_not_convert = ["lm_head"] + modules_to_not_convert

        if self.quantization_config.modules_to_not_convert:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)

        model = replace_with_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        from ..integrations import FP8Linear

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, FP8Linear):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return False
