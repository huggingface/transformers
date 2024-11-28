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
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

# if is_torch_available():
import torch

from ..integrations import HiggsLinear, quantize_with_higgs, replace_with_higgs_linear
from ..utils import is_accelerate_available, is_flute_available, is_hadamard_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


logger = logging.get_logger(__name__)


# Finds the parent of a node module named "name"
def find_parent(model, name):
    module_tree = name.split(".")[:-1]
    parent = model
    for m in module_tree:
        parent = parent._modules[m]
    return parent


class HiggsHfQuantizer(HfQuantizer):
    """
    Quantizer of the HIGGS method. Enables the loading of prequantized models.
    """

    requires_calibration = False
    requires_parameters_quantization = True
    required_packages = ["flute-kernel", "fast_hadamard_transform"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Using `higgs` quantization requires Accelerate: `pip install accelerate`")

        if not is_flute_available():
            raise ImportError("Using `higgs` quantization requires FLUTE: `pip install flute-kernel`")

        if not is_hadamard_available():
            raise ImportError(
                "Using `higgs` quantization requires fast_hadamard_transform: `pip install fast_hadamard_transform`"
            )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info(
                    "CUDA available. Assuming HIGGS inference on GPU and loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually."
                )
            else:
                raise NotImplementedError(
                    "HIGGS quantization is only supported on GPU. Please use a different quantizer."
                )
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
        Quantizes weights into weight and weight_scale
        """
        flute_dict = quantize_with_higgs(
            param_value.to(target_device),
            self.quantization_config.bits,
            self.quantization_config.p,
        )

        del param_value

        module, tensor_name = get_module_from_name(model, param_name)
        for key, value in flute_dict.items():
            if key in module._parameters:
                module._parameters[key] = torch.nn.Parameter(value, requires_grad=False)
            elif key in module._buffers:
                module._buffers[key] = torch.nn.Buffer(value)
            else:
                raise ValueError(f"Unexpected key {key} in module {module}")

        if unexpected_keys is not None and param_name in unexpected_keys:
            unexpected_keys.remove(param_name)

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        replace_with_higgs_linear(
            model,
            quantization_config=self.quantization_config,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        import flute.utils

        flute_workspaces = {}
        for name, module in model.named_modules():
            if isinstance(module, HiggsLinear):
                if module.weight.device not in flute_workspaces:
                    flute_workspaces[module.weight.device] = flute.utils.make_workspace_streamk(
                        device=module.weight.device
                    )
                module.workspace = flute_workspaces[module.weight.device]

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        from ..integrations import HiggsLinear

        not_missing_keys = []
        for name, module in model.named_modules():
            if isinstance(module, HiggsLinear):
                for missing in missing_keys:
                    if (
                        (name in missing or name in f"{prefix}.{missing}")
                        and not missing.endswith(".weight")
                        and not missing.endswith(".bias")
                    ):
                        not_missing_keys.append(missing)
        return [k for k in missing_keys if k not in not_missing_keys]

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    def is_serializable(self, safe_serialization=None):
        return True

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ) -> bool:
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, HiggsLinear) and tensor_name == "weight" and param_value.dtype != torch.int16:
            # Add here check for loaded components' dtypes once serialization is implemented
            return True
        else:
            return False
