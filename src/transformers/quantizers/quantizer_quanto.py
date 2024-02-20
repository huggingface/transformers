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
from typing import TYPE_CHECKING, Any, Dict, Optional

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..integrations import replace_with_quanto_layers
from ..utils import is_quanto_available, is_torch_available, logging
from ..utils.quantization_config import QuantoConfig


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantoHfQuantizer(HfQuantizer):
    """
    Quantizer for the quanto library
    """

    required_packages = ["quanto", "accelerate"]

    def __init__(self, quantization_config: QuantoConfig, **kwargs):
        self.requires_calibration = quantization_config.activations is not None
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_quanto_available():
            raise ImportError("Loading a quanto quantized model requires quanto library (`pip install quanto`)")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            # TODO: Discuss if we should do that for quanto. I think that we should probably not do that and let the user cast the torch_dtype by themselves.
            # since in this case, quanto can also work on cpu.
            # If a user have both a cpu and cuda and he wants to play with quanto on cpu, he will have a specify manually torch_dtype to torch.float32.
            if torch.cuda.is_available():
                torch_dtype = torch.float16
                logger.info(
                    "CUDA available. Assuming Quanto inference on GPU and loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually."
                )
            else:
                torch_dtype = torch.float32
                logger.info(
                    "CUDA is unavailable. Assuming AQLM inference on CPU and loading the model in `torch.float32`. To overwrite it, set `torch_dtype` manually."
                )
        return torch_dtype

    def update_weights_only_kwarg(self,weights_only_kwarg: Dict[str,Any]) -> Dict[str,Any]:
        weights_only_kwarg["weights_only"]=False
        return weights_only_kwarg

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            model, _ = replace_with_quanto_layers(model, quantization_config=self.quantization_config)
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model):
        if not self.pre_quantized:
            from quanto import freeze, qfloat8_e4m3fn, qfloat8_e5m2, qint8, quantize
            w_mapping = {"int8": qint8}
            a_mapping = {None: None, "int8": qint8, "fp8_e5m2": qfloat8_e5m2, "fp8_e4m3": qfloat8_e4m3fn}
            quantize(
                model,
                weights=w_mapping[self.quantization_config.weights],
                activations=a_mapping[self.quantization_config.activations],
            )
            freeze(model)

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    @property
    def is_serializable(self):
        return True
