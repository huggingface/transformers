# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name, should_convert_module
from ..utils import is_entropack_available, is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

if is_entropack_available():
    import entropack

logger = logging.get_logger(__name__)


def is_entropack_linear(module) -> bool:
    return is_entropack_available() and isinstance(module, entropack.CompressedLinear)


def prepare_entropack_model_for_ddp(model):
    """Exclude frozen EntroPack weights from DDP synchronization after adapter injection."""
    weight_ids = {id(module.weight) for module in model.modules() if is_entropack_linear(module)}
    ignored = {name for name, parameter in model.named_parameters() if id(parameter) in weight_ids}
    ignored.update(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
    model._ddp_params_and_buffers_to_ignore = sorted(ignored)
    return model


def get_entropack_fsdp_ignored_modules_pattern(model) -> str:
    """Return an Accelerate `FSDP_IGNORED_MODULES` regex for EntroPack layers."""
    names = [re.escape(name) for name, module in model.named_modules() if is_entropack_linear(module)]
    if not names:
        raise ValueError("The model contains no EntroPack linear layers")
    return "(?:" + "|".join(names) + ")"


def replace_with_entropack_linear(
    model,
    quantization_config,
    modules_to_not_convert: list[str] | None = None,
):
    codec_options = dict(quantization_config.entropack_lossy_config or {})
    codec_options.update(target_bpp=quantization_config.target_bpp, execution_backend="cuda")
    codec_config = entropack.EntroPackLossyConfig(**codec_options)
    linear_class = {
        "default": entropack.CompressedLinear,
        "fp8": entropack.CompressedFP8Linear,
        "int8": entropack.CompressedINT8Linear,
    }[quantization_config.linear_type]
    has_been_replaced = False

    for module_name, module in list(model.named_modules()):
        if type(module) is not nn.Linear or not should_convert_module(module_name, modules_to_not_convert):
            continue

        with torch.device("meta"):
            new_module = linear_class(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                config=codec_config,
                dtype=module.weight.dtype,
            )
            # Device-map planning needs the logical dense shape before packed bytes exist.
            new_module.weight = nn.Parameter(torch.empty_like(module.weight, device="meta"), requires_grad=False)
        # Model-specific initializers treat this Linear subclass as a dense layer.
        new_module._is_hf_initialized = True
        model.set_submodule(module_name, new_module)
        has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading a model with entropack but no convertible `nn.Linear` module was found. Double "
            "check the model architecture and `modules_to_not_convert`."
        )
    return model


class EntroPackQuantize(ConversionOps):
    """Compress a dense checkpoint weight directly into its target module."""

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        model: nn.Module,
        missing_keys: set[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        for param_name, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            module, _ = get_module_from_name(model, param_name)

            module.compress_weight(tensor.detach())
            for buffer_name in module.state_buffer_names:
                getattr(module, buffer_name)._is_hf_initialized = True

            # compress_weight bypasses set_param_for_module, so clear the keys it initialized.
            if missing_keys is not None:
                missing_keys.discard(param_name)
                module_name = param_name.rpartition(".")[0]
                for buffer_name in module.state_buffer_names:
                    missing_keys.discard(f"{module_name}.{buffer_name}")

        return {}


class EntroPackDeserialize(ConversionOps):
    """Restore a compressed weight from its packed checkpoint entry."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        value = input_dict["weight.entropack_quant_state"]
        tensor = value[0] if isinstance(value, list) else value
        container = entropack.CompressedTensor.from_state_dict({"entropack_quant_state": tensor})
        return {"weight": nn.Parameter(container, requires_grad=False)}
