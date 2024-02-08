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
from typing import TYPE_CHECKING, Optional

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch
    from torch import nn

logger = logging.get_logger(__name__)


class AqlmHfQuantizer(HfQuantizer):
    """
    Quantizer of the AQLM method. Enables the loading of prequantized models.
    """

    requires_calibration = False
    required_packages = ["aqlm"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Using `aqlm` quantization requires Accelerate: `pip install accelerate`")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            raise RuntimeError(
                "AQLM only supports float16 for now. Please set `torch_dtype=torch.float16` when loading the model."
            )
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        _replace_with_aqlm_linear(
            model,
            linear_weights_not_to_quantize=self.quantization_config.linear_weights_not_to_quantize,
            quantization_config=self.quantization_config,
        )
        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model._is_quantized_training_enabled = False
        return model

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    @property
    def is_serializable(self):
        return True


def _replace_with_aqlm_linear(
    model,
    linear_weights_not_to_quantize=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    from accelerate import init_empty_weights
    from aqlm import QuantizedLinear

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `linear_weights_not_to_quantize`
            if ".".join(current_key_name) + ".weight" not in linear_weights_not_to_quantize:
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = QuantizedLinear(
                        in_features,
                        out_features,
                        bias=module.bias is not None,
                        in_group_size=quantization_config.in_group_size,
                        out_group_size=quantization_config.out_group_size,
                        num_codebooks=quantization_config.num_codebooks,
                        nbits_per_codebook=quantization_config.nbits_per_codebook,
                    )
                    has_been_replaced = True

                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_aqlm_linear(
                module,
                linear_weights_not_to_quantize,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced
