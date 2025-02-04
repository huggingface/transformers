from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import torch
from torch.nn.parameter import Parameter as Parameter
from .base import HfQuantizer
from ..utils import is_accelerate_available, logging
from .quantizers_utils import get_module_from_name

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

logger = logging.get_logger(__name__)

class FP8HfQuantizer(HfQuantizer):
    """
    FP8 quantization implementation supporting both standard and MoE models.
    Supports both e4m3fn and e4m3fnuz formats based on platform.
    """
    
    requires_parameters_quantization = False
    requires_calibration = False
    required_packages = ["accelerate"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config
        self.is_moe_model = kwargs.get("is_moe_model", False)

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

        device_map = kwargs.get("device_map", None)
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA device available. "
                "Make sure to set your model on a GPU device to run your model."
            )
        elif isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
            raise ValueError(
                "FP8 models do not support CPU or disk offloading in the device map. "
                "Please remove CPU/disk devices from the device map."
            )

    def update_torch_dtype(self, torch_dtype):
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
        module, tensor_name = get_module_from_name(model, param_name)
        print("######################### in quantizer_fp8.py #########################")
        # Get FP8 min/max values
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        
        if self.quantization_config.weight_block_size is not None:

            block_size_m, block_size_n = self.quantization_config.weight_block_size
            
            rows, cols = param_value.shape[-2:]
            
            # Check if dimensions are divisible by block sizes
            if rows % block_size_m != 0 or cols % block_size_n != 0:
                raise ValueError(
                    f"Matrix dimensions ({rows}, {cols}) must be divisible by block sizes ({block_size_m}, {block_size_n})"
                )
            
            
            # Create blocks using unfold
            param_value = param_value.unfold(-2, block_size_m, block_size_m)
            param_value = param_value.unfold(-2, block_size_n, block_size_n)
            
            # Calculate scaling factor for each block
            max_abs = torch.amax(torch.abs(param_value), dim=(-1, -2))
            scale = fp8_max / max_abs
            
            # Expand scale to match block dimensions for multiplication
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            
            # Quantize the weights
            quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
            
            # Reshape back to matrix shape
            quantized_param = quantized_param.reshape(param_value.shape[:-4] + (rows, cols))
            
            # Reshape scale to match the number of blocks
            scale = scale.reshape(scale.shape[:-2] + (-1,)).reciprocal()
            
        else:
            # Per-tensor quantization
            max_abs = torch.max(torch.abs(param_value))
            print("###################max_abs#################", max_abs)
            scale = fp8_max / max_abs
            print("###################scale#################", scale)
            # Quantize the weights
            quantized_param = torch.clamp(param_value * scale, min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
            # For per-tensor quantization, we just need a single scale value
            scale = torch.tensor([[scale]]).reciprocal()
        # Store the quantized weights and scales in the module
        print(f"tensor_name in create_quantized_param: {tensor_name} {target_device}")
        module._buffers[tensor_name] = quantized_param.to(target_device)
        module._buffers["weight_scale"] = scale.to(target_device)

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        from ..integrations.fp8 import FP8Linear, FP8MoELinear
        
        module, tensor_name = get_module_from_name(model, param_name)
        
        if isinstance(module, FP8Linear):
            if self.pre_quantized or tensor_name == "bias":
                if tensor_name == "weight" and param_value.dtype != torch.int8:
                    raise ValueError("Expect quantized weights but got an unquantized weight")
                return False
            else:
                if tensor_name == "weight_scale":
                    raise ValueError("Expect unquantized weights but got a quantized weight_scale")
                return True
        return False

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from ..integrations.fp8 import replace_with_fp8_linear
        
        self.modules_to_not_convert = ["lm_head"] + keep_in_fp32_modules
        
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
    
    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def is_serializable(self, safe_serialization=None):
        return True

    @property 
    def is_trainable(self) -> bool:
        return False  # FP8 quantization is typically used for inference only