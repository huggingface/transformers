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

from ..utils.logging import tqdm
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_flute_available, is_hadamard_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


def get_num_sms_from_device(device):
    target_device_cc = torch.cuda.get_device_capability(device=device)
    if target_device_cc == (8, 6):
        return 84
    elif target_device_cc == (8, 0):
        return 108
    elif target_device_cc == (8, 9):
        return 128
    else:
        raise NotImplementedError(
            f"Device capability {target_device_cc} not supported for FLUTE (yet?) to verify your device capability check out https://developer.nvidia.com/cuda-gpus"
        )


class HiggsHfQuantizer(HfQuantizer):
    """
    Quantizer of the HIGGS method. Enables the loading of prequantized models and in-flight quantization of full-precision models.
    """

    requires_calibration = False
    requires_parameters_quantization = True
    required_packages = ["flute-kernel", "fast_hadamard_transform"]

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, device_map, **kwargs):
        if not torch.cuda.is_available():
            raise NotImplementedError("HIGGS quantization is only supported on GPU. Please use a different quantizer.")

        if not is_accelerate_available():
            raise ImportError("Using `higgs` quantization requires Accelerate: `pip install accelerate`")

        if not is_flute_available():
            raise ImportError("Using `higgs` quantization requires FLUTE: `pip install flute-kernel>=0.3.0`")

        if not is_hadamard_available():
            raise ImportError(
                "Using `higgs` quantization requires fast_hadamard_transform: `pip install fast_hadamard_transform`"
            )

        if device_map is None:
            raise ValueError(
                "You are attempting to load a HIGGS model without setting device_map."
                " Please set device_map comprised of 'cuda' devices."
            )
        elif isinstance(device_map, dict) and ("cpu" in device_map.values() or "disk" in device_map.values()):
            raise ValueError(
                "You are attempting to load a HIGGS model with a device_map that contains a CPU or disk device."
                " This is not supported. Please remove the CPU or disk device from the device_map."
            )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            logger.info("`torch_dtype` is None. Setting `torch_dtype=torch.float16` for FLUTE compatibility.")
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16 and torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Invalid `torch_dtype` {torch_dtype}. HIGGS quantization only supports `torch_dtype=torch.float16` or `torch_dtype=torch.bfloat16`."
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
        from ..integrations import quantize_with_higgs

        """
        Quantizes weights into weight and weight_scale
        """
        flute_dict = quantize_with_higgs(
            param_value.to(target_device),
            self.quantization_config.bits,
            self.quantization_config.p,
            self.quantization_config.group_size,
            self.quantization_config.hadamard_size,
        )
        del param_value

        module, _ = get_module_from_name(model, param_name)
        module_name = ".".join(param_name.split(".")[:-1])
        for key, value in flute_dict.items():
            if key in module._parameters:
                module._parameters[key] = torch.nn.Parameter(value, requires_grad=False)
            elif key in module._buffers:
                module._buffers[key] = torch.nn.Buffer(value)
            elif key == "tune_metadata":
                module.tune_metadata = value
                self.quantization_config.tune_metadata[module_name] = value.to_dict()
            else:
                raise ValueError(f"Unexpected key {key} in module {module}")

        if unexpected_keys is not None and param_name in unexpected_keys:
            unexpected_keys.remove(param_name)

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from ..integrations import replace_with_higgs_linear

        replace_with_higgs_linear(
            model,
            quantization_config=self.quantization_config,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from flute.tune import TuneMetaData, maybe_tune_and_repack
        from flute.utils import make_workspace_streamk

        from ..integrations import HiggsLinear

        flute_workspaces = {}
        flute_modules = {name: module for name, module in model.named_modules() if isinstance(module, HiggsLinear)}
        for name, module in tqdm(flute_modules.items(), desc="Repacking HIGGS modules", leave=False):
            # Every HiggsLinear needs a "workspace": a buffer for the unpacking operation.
            # This buffer needs to be on the same device as the weights, but can be reused across modules otherwise.
            if module.weight.device not in flute_workspaces:
                flute_workspaces[module.weight.device] = make_workspace_streamk(device=module.weight.device)
            module.workspace = flute_workspaces[module.weight.device]

            # FLUTE weights are packed in a way that is optimized for a specific number of SMs (GPU streaming multiprocessors).
            # If the model is loaded on a different device than the one it was saved on, we need to repack the weights.
            module.tune_metadata = TuneMetaData.from_dict(self.quantization_config.tune_metadata[name])
            module.weight.data, module.tune_metadata = maybe_tune_and_repack(
                weight=module.weight.data,
                scales=module.scales.data,
                metadata=module.tune_metadata,
            )
            self.quantization_config.tune_metadata[name] = module.tune_metadata.to_dict()

    def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        from ..integrations import HiggsLinear

        higgs_names = {name for name, module in model.named_modules() if isinstance(module, HiggsLinear)}

        def should_update(key: str) -> bool:
            if key.endswith('.weight') or key.endswith('.bias'):
                return False
            full_key = f"{prefix}.{key}"
            return any(name in key or name in full_key for name in higgs_names)

        return [key for key in missing_keys if not should_update(key)]


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
        from ..integrations import HiggsLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, HiggsLinear) and tensor_name == "weight" and param_value.dtype != torch.int16:
            # Only quantize weights of HiggsLinear modules that are not already quantized
            return True
        else:
            return False

    def _dequantize(self, model):
        from ..integrations import dequantize_higgs

        model = dequantize_higgs(model)
        return model
