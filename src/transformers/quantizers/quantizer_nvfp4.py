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
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..utils import is_kernels_available, is_torch_available, is_torch_xpu_available, logging
from ..utils.import_utils import KERNELS_MAX_VERSION, KERNELS_MIN_VERSION
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import NVFP4Config


class NVFP4HfQuantizer(HfQuantizer):
    """NVFP4 linears and MoE experts, from a ModelOpt export or quantized on the fly. FP8 layers load dequantized."""

    requires_calibration = False
    quantization_config: NVFP4Config
    serialized_names = {"weight": ["weight", "weight_scale_inv"]}

    @property
    def quantized_element_size(self) -> float:
        return 0.5

    def validate_environment(self, device_map, **kwargs):
        if self.quantization_config.dequantize:
            return
        missing = None
        if not is_kernels_available():
            missing = f"the `kernels` package ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION})"
        elif not torch.cuda.is_available() and not is_torch_xpu_available():
            missing = "a GPU or XPU"
        # Triton's FP8 type needs compute capability 8.9 on NVIDIA
        elif torch.cuda.is_available() and torch.version.hip is None and torch.cuda.get_device_capability() < (8, 9):
            missing = "a GPU with compute capability 8.9 or newer"
        if missing is not None:
            if not self.pre_quantized:
                raise RuntimeError(f"NVFP4 quantization requires {missing}.")
            logger.warning_once(f"NVFP4 kernels require {missing}. Loading the model dequantized to its dtype.")
            self.quantization_config.dequantize = True
            return
        devices = set(device_map.values()) if isinstance(device_map, dict) else {device_map}
        if devices & {"cpu", "disk"}:
            raise ValueError("NVFP4 does not support CPU or disk offload.")

    def update_parallel_plan(self, model, plan):
        from ..integrations.nvfp4 import update_nvfp4_parallel_plan

        super().update_parallel_plan(model, plan)
        update_nvfp4_parallel_plan(model, plan)

    def param_needs_quantization(self, model: PreTrainedModel, param_name: str, **kwargs) -> bool:
        from ..integrations.nvfp4 import EXPERT_PROJECTIONS, NVFP4Experts, NVFP4Linear

        if self.pre_quantized or self.quantization_config.dequantize:
            return False
        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, NVFP4Experts):
            return tensor_name in EXPERT_PROJECTIONS
        return isinstance(module, NVFP4Linear) and tensor_name == "weight"

    def _module_formats(self, model: PreTrainedModel) -> dict[str, str] | None:
        """Return the ModelOpt format of each module of `model`, or `None` when every eligible module is NVFP4.

        `quantized_layers` names checkpoint modules. A module matches by its model or checkpoint name. Fused experts
        also match their per-expert entries, such as `experts.3.w1`.
        """
        from ..conversion_mapping import get_model_conversion_mapping
        from ..core_model_loading import WeightRenaming

        layers = self.quantization_config.quantized_layers
        if layers is None:
            return None
        # the model's renames, reversed, give a module's checkpoint name
        reverse = [
            rule.reverse_transform()
            for rule in get_model_conversion_mapping(model, add_legacy=False)[::-1]
            if isinstance(rule, WeightRenaming)
        ]
        formats = {}
        for name, _ in model.named_modules():
            key = f"{name}.weight"
            for rule in reverse:
                key, _ = rule.rename_source_key(key)
            names = {name, key.removesuffix(".weight")}
            entries = [algo for layer, algo in layers.items() for prefix in names if layer == prefix]
            if name.endswith(".experts"):
                entries += [
                    algo for layer, algo in layers.items() for prefix in names if layer.startswith(f"{prefix}.")
                ]
            if len(set(entries)) > 1:
                raise ValueError(
                    f"{name!r} is one module, but the checkpoint gives it formats {sorted(set(entries))}."
                )
            if entries:
                formats[name] = entries[0]
        return formats

    def _process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs):
        from ..integrations.nvfp4 import NVFP4Experts, NVFP4Linear, replace_with_nvfp4

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        self._nvfp4_shapes = {}
        # no slot for calibrated `input_scale`s or FP8 KV-cache scales
        model._keys_to_ignore_on_load_unexpected = {
            *(model._keys_to_ignore_on_load_unexpected or ()),
            r"[._]input(_global)?_scale$",
            r"\.[kv]_scale$",
        }
        if self.quantization_config.dequantize:
            return
        formats = self._module_formats(model)
        weight_only = self.quantization_config.activation_format == "bf16"
        # tensor parallelism splits a dense scale grid across its swizzled tiles, so it stays row-major
        swizzle_linears = getattr(model.config, "distributed_config", None) is None
        if not replace_with_nvfp4(
            model, self.modules_to_not_convert, formats, self.pre_quantized, weight_only, swizzle_linears
        ):
            raise ValueError("NVFP4 found no linear or experts module to quantize in this model.")
        # the loader accepts any shape for a quantized tensor
        self._nvfp4_shapes = {
            name: param.shape
            for name, param in model.named_parameters()
            if self.pre_quantized
            and isinstance(model.get_submodule(name.rpartition(".")[0]), (NVFP4Experts, NVFP4Linear))
        }

    def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        from ..core_model_loading import WeightRenaming

        conversions = getattr(model, "_weight_conversions", None) or []
        if not self.pre_quantized:
            # scales from on-the-fly quantization save through the conversions of their weights
            conversions = self.update_weight_conversions(conversions, model, for_save=True)
        # save the modules' own names, not ModelOpt's
        model._weight_conversions = [
            conversion
            for conversion in conversions
            if not (
                isinstance(conversion, WeightRenaming)
                and conversion._original_source_patterns[0].endswith(("_scale$", "_scale_2$"))
            )
        ]
        # a distributed load holds shards, so only a full load checks the shapes
        if getattr(model.config, "distributed_config", None) is None:
            for name, param in model.named_parameters():
                expected = self._nvfp4_shapes.get(name)
                if expected is not None and param.shape != expected:
                    raise ValueError(
                        f"{name} loaded with shape {tuple(param.shape)}, but its module holds {tuple(expected)}."
                    )
        return model

    def postprocess_model(self, model: PreTrainedModel, **kwargs):
        if self.quantization_config.dequantize:
            self.remove_quantization_config(model)
            return model
        return super().postprocess_model(model, **kwargs)

    def get_quantize_ops(self):
        from ..integrations.nvfp4 import NVFP4Quantize

        return NVFP4Quantize(self)

    def get_load_operations(self):
        from ..integrations.nvfp4 import NVFP4Dequantize, NVFP4Deserialize

        return {"deserialize": NVFP4Deserialize, "dequantize": NVFP4Dequantize}

    def get_param_ops(self, model: PreTrainedModel, param_name: str) -> list:
        """Return the layout ops a tensor runs when it loads into `param_name`. Each op is reversed on save.

        The ops are the packed FP4 view, the gate|up interleave and the scale swizzle.
        """
        from ..integrations.nvfp4 import (
            EXPERT_PROJECTIONS,
            NVFP4Experts,
            NVFP4InterleaveGateUp,
            NVFP4Linear,
            NVFP4SwizzleScales,
            NVFP4ViewPackedInt8,
        )

        module, name = get_module_from_name(model, param_name)
        if not isinstance(module, (NVFP4Linear, NVFP4Experts)):
            return []
        ops = []
        if name in ("weight", *EXPERT_PROJECTIONS):
            ops.append(NVFP4ViewPackedInt8(self))
        if isinstance(module, NVFP4Experts) and module.has_gate and name in ("gate_up_proj", "gate_up_proj_scale_inv"):
            ops.append(NVFP4InterleaveGateUp(self))
        if name.endswith("scale_inv"):
            ops.append(NVFP4SwizzleScales(self))
        return ops

    def update_weight_conversions(self, weight_conversions, model, for_save: bool = False):
        from ..core_model_loading import WeightConverter
        from ..integrations.nvfp4 import NVFP4Deserialize, nvfp4_checkpoint_conversions

        conversions = super().update_weight_conversions(weight_conversions, model)
        if not self.pre_quantized and not for_save:
            return conversions
        conversions = nvfp4_checkpoint_conversions(self, conversions, dequantize=self.quantization_config.dequantize)
        if self.quantization_config.dequantize:
            return conversions
        # A fused stack in the checkpoint, or a dense scale, loads through its own converter. It comes first, as the
        # first matching converter wins. On save, skip the params a model converter writes back, as they would match
        # first there.
        reverse = [c.reverse_transform() for c in conversions if isinstance(c, WeightConverter)] if for_save else []
        per_param = [
            WeightConverter(re.escape(name), name, [NVFP4Deserialize(self)])
            for name, _ in model.named_parameters()
            if self.get_param_ops(model, name) and not any(c.rename_source_key(name)[1] for c in reverse)
        ]
        renames = [c for c in conversions if not isinstance(c, WeightConverter)]
        return renames + per_param + [c for c in conversions if isinstance(c, WeightConverter)]

    def is_serializable(self) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True
