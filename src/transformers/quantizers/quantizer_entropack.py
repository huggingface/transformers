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
from typing import TYPE_CHECKING

from ..utils import is_entropack_available
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..utils.quantization_config import EntroPackConfig


class EntroPackHfQuantizer(HfQuantizer):
    """Quantizer for EntroPack entropy-coded weight compression."""

    requires_calibration = False
    serialized_names = {"weight": ["weight.entropack_quant_state"]}
    quantization_config: "EntroPackConfig"

    def validate_environment(self, *args, **kwargs):
        if not is_entropack_available():
            raise ImportError(
                "Loading an EntroPack model requires entropack>=0.1.3. Install it with "
                "`pip install entropack[cuda12]` (or `[cuda13]` for CUDA 13)."
            )

        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict) and "disk" in device_map.values():
            raise ValueError("EntroPack does not support disk offloading; remove the `disk` device from `device_map`.")

        from entropack.backends.cuda import probe

        reason = probe()
        if reason is not None:
            raise RuntimeError(f"The entropack CUDA codec is unavailable: {reason}")

    def update_parallel_plan(self, model, plan):
        distributed_config = getattr(model.config, "distributed_config", None)
        tp_size = getattr(distributed_config, "tp_size", None)
        if tp_size is not None and tp_size > 1:
            raise ValueError(
                "EntroPack does not support tensor parallelism because its entropy-coded weight bitstreams do not "
                "have a sharding plan."
            )

    def _process_model_before_weight_loading(self, model, **kwargs):
        from ..integrations.entropack import replace_with_entropack_linear

        modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        replace_with_entropack_linear(model, self.quantization_config, modules_to_not_convert)

    def param_needs_quantization(self, model, param_name: str, **kwargs) -> bool:
        from ..integrations.entropack import is_entropack_linear

        module, tensor_name = get_module_from_name(model, param_name)
        return is_entropack_linear(module) and tensor_name == "weight"

    def get_quantize_ops(self):
        from ..integrations.entropack import EntroPackQuantize

        return EntroPackQuantize()

    def get_load_operations(self):
        from ..integrations.entropack import EntroPackDeserialize

        return {"deserialize": EntroPackDeserialize}

    @property
    def quantized_element_size(self) -> float:
        return self.quantization_config.target_bpp / 8

    def _process_model_after_weight_loading(self, model, **kwargs):
        import entropack

        from ..integrations.entropack import EntroPackDeserialize, is_entropack_linear

        # Remove EntroPack's load-only converter: state_dict already emits `weight.entropack_quant_state`, and
        # reversing the converter would rename every `.weight` key.
        conversions = getattr(model, "_weight_conversions", None)
        if conversions is not None:
            model._weight_conversions = [
                conversion
                for conversion in conversions
                if not any(isinstance(op, EntroPackDeserialize) for op in getattr(conversion, "operations", ()))
            ]

        for name, module in model.named_modules():
            if not is_entropack_linear(module):
                continue
            container = module.weight
            if not isinstance(container, entropack.CompressedTensor):
                raise RuntimeError(f"EntroPack layer {name} has no packed weight")

            expected_shape = (module.out_features, module.in_features)
            if tuple(container.shape) != expected_shape:
                raise RuntimeError(
                    f"EntroPack layer {name} declares shape {tuple(container.shape)}, expected {expected_shape}"
                )

            for buffer_name in module.state_buffer_names:
                if not getattr(getattr(module, buffer_name), "_is_hf_initialized", False):
                    raise RuntimeError(f"EntroPack layer {name} did not load {buffer_name}")

        return model

    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return True
