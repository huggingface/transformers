from typing import TYPE_CHECKING, Any, Dict, List, Union
from .base import HfQuantizer

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_torch_available, logging
from ..integrations import replace_with_hqq_linear


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def is_hqq_available():
	available = True
	try:
		import hqq 
	except:
		available = False
	return available

class HQQHfQuantizer(HfQuantizer):
    """
	#TODO: 
    """

    use_keep_in_fp32_modules = False
    requires_parameters_quantization = False
    requires_calibration = False

    required_packages = ["hqq"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not (is_hqq_available()):
            raise ImportError("Using `HQQ` quantization requires `pip install hqq`")

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

        device_map = kwargs.get("device_map", None)


    def check_quantized_param(
        self, model: "PreTrainedModel", param_value: "torch.Tensor", param_name: str, state_dict: Dict[str, Any]
    ) -> bool:
        from hqq.core.quantize import HQQLinear

        if isinstance(module, HQQLinear):
        	return True
        else:
        	return False

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
        return torch_dtype

    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_before_weight_loading
    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        #from ..integrations import get_keys_to_not_convert
        #from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
        
        #TOdo: how to get device from device_map
        device        = 'cuda'
        compute_dtype =  self.update_torch_dtype(None)
        model  = replace_with_hqq_linear(model, quantization_config=self.quantization_config, 
        										modules_to_not_convert=self.modules_to_not_convert, 
        										compute_dtype=compute_dtype, device=device)

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_hqq_quantized = True
        model.is_hqq_serializable = self.is_serializable
        return model

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True
