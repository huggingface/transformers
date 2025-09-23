from typing import TYPE_CHECKING, Any, Optional

from ..utils import is_accelerate_available, is_torch_available, is_torch_xpu_available, logging
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
        if not is_torch_available():
            raise ImportError(
                "Using fp8 quantization requires torch >= 2.1.0"
                "Please install the latest version of torch ( pip install --upgrade torch )"
            )

        if not is_accelerate_available():
            raise ImportError("Loading an FP8 quantized model requires accelerate (`pip install accelerate`)")

        if not (torch.cuda.is_available() or is_torch_xpu_available()):
            raise RuntimeError("No GPU or XPU found. A GPU or XPU is needed for FP8 quantization.")

        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            major, minor = compute_capability
            if (major < 8) or (major == 8 and minor < 9):
                raise ValueError(
                    "FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100)"
                    f", actual = `{major}.{minor}`"
                )

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA or XPU device available, make sure to set "
                "your model on a GPU or XPU device in order to run your model. To remove this warning, "
                "pass device_map = 'cuda' or 'xpu'. "
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

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype is None:
            logger.info("Setting dtype to torch.float32 as no dtype was specified in from_pretrained")
            dtype = torch.float32
        return dtype

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any],
    ):
        """
        Quantizes weights to FP8 format using Block-wise quantization
        """
        from ..modeling_utils import _load_parameter_into_model

        param_value = param_value.to(target_device)

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

        # Load into the model
        _load_parameter_into_model(model, param_name, quantized_param)
        _load_parameter_into_model(model, param_name.rsplit(".", 1)[0] + ".weight_scale_inv", scale)

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
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
        keep_in_fp32_modules: Optional[list[str]] = None,
        **kwargs,
    ):
        from ..integrations.finegrained_fp8 import replace_with_fp8_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules
        )

        model = replace_with_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def update_missing_keys(self, model, missing_keys: list[str], prefix: str) -> list[str]:
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

    def update_tp_plan(self, config):
        if "Qwen3" in config.__class__.__name__:
            text_plan = {
                "layers.*.self_attn.q_proj.weight": "local_colwise",
                "layers.*.self_attn.q_proj.weight_scale_inv": "local_colwise",
                "layers.*.self_attn.k_proj.weight": "local_colwise",
                "layers.*.self_attn.k_proj.weight_scale_inv": "local_colwise",
                "layers.*.self_attn.v_proj.weight": "local_colwise",
                "layers.*.self_attn.v_proj.weight_scale_inv": "local_colwise",
                "layers.*.self_attn.o_proj.weight": "local_rowwise",
                "layers.*.self_attn.o_proj.weight_scale_inv": "local_rowwise",
                "layers.*.self_attn": "gather",
                "layers.*.mlp.gate_proj.weight": "local_colwise",
                "layers.*.mlp.gate_proj.weight_scale_inv": "local_colwise",
                "layers.*.mlp.up_proj.weight": "local_colwise",
                "layers.*.mlp.up_proj.weight_scale_inv": "local_colwise",
                "layers.*.mlp.down_proj.weight": "local_rowwise",
                "layers.*.mlp.down_proj.weight_scale_inv": "local_rowwise",
                "layers.*.mlp": "gather",
            }

            config.base_model_tp_plan = text_plan

        return config

    def is_serializable(self, safe_serialization=None):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    def get_accelerator_warm_up_factor(self):
        # Pre-processing is done cleanly, so we can allocate everything here
        return 2
