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

    @property
    def serialized_names(self) -> dict[str, list[str]]:
        """The tensors a weight ships as. A dequantized load also folds each global in.

        Expert stacks come first, as `weight_global_scale` also ends a stack's `{proj}_weight_global_scale`. They keep
        their `experts.` prefix, so a dense MLP's `down_proj` does not match.
        """
        from ..integrations.nvfp4 import EXPERT_PROJECTIONS

        globals_ = self.quantization_config.dequantize
        stacks = {
            f"experts.{proj}": [
                f"experts.{proj}",
                f"experts.{proj}_scale_inv",
                *([f"experts.{proj}_weight_global_scale"] if globals_ else []),
            ]
            for proj in EXPERT_PROJECTIONS
        }
        return {**stacks, "weight": ["weight", "weight_scale_inv", *(["weight_global_scale"] if globals_ else [])]}

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

    def _module_formats(self, model: PreTrainedModel) -> dict[str, str] | None:
        """Return how each module of `model` loads, from `quantized_layers`, or `None` when every eligible module is NVFP4.

        A module matches by its model or checkpoint name. Exact names win, and fused experts also match their per-expert
        entries, such as `experts.3.w1`. Otherwise the first pattern that matches applies.
        """
        layers = self.quantization_config.quantized_layers
        if layers is None:
            return None
        # the model's renames, reversed, give a module's checkpoint name
        reverse = [rule.reverse_transform() for rule in self.model_renames[::-1]]
        patterns = [(re.compile(layer), loads) for layer, loads in layers.items()]
        formats = {}
        for name, _ in model.named_modules():
            key = f"{name}.weight"
            for rule in reverse:
                key, _ = rule.rename_source_key(key)
            names = {name, key.removesuffix(".weight")}
            entries = [loads for layer, loads in layers.items() if layer in names]
            if not entries and name.endswith(".experts"):
                entries = [
                    loads for layer, loads in layers.items() for prefix in names if layer.startswith(f"{prefix}.")
                ]
            if not entries:
                entries = [loads for pattern, loads in patterns if any(pattern.fullmatch(n) for n in names)][:1]
            if len(set(entries)) > 1:
                raise ValueError(
                    f"{name!r} is one module, but the checkpoint gives it formats {sorted(set(entries))}."
                )
            if entries:
                formats[name] = entries[0]
        return formats

    def _process_model_before_weight_loading(self, model: PreTrainedModel, **kwargs):
        from ..conversion_mapping import get_model_conversion_mapping
        from ..core_model_loading import WeightRenaming
        from ..integrations.nvfp4 import replace_with_nvfp4

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        # the model's renames map its names to the checkpoint's and back
        mapping = get_model_conversion_mapping(model, add_legacy=False)
        self.model_renames = [rule for rule in mapping if isinstance(rule, WeightRenaming)]
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
        if not replace_with_nvfp4(model, self.modules_to_not_convert, formats, self.pre_quantized, weight_only):
            raise ValueError("NVFP4 found no linear or experts module to quantize in this model.")

    def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        from ..integrations.nvfp4 import checkpoint_renames, dequantize_fp8_linears

        dequantize_fp8_linears(model)
        # the dequantized layers are now in the model dtype, so a save lists them no more
        if self.quantization_config.quantized_layers is not None:
            self.quantization_config.quantized_layers = {
                layer: loads
                for layer, loads in self.quantization_config.quantized_layers.items()
                if loads != "dequantize"
            }

        conversions = getattr(model, "_weight_conversions", None) or []
        if not self.pre_quantized:
            # scales from on-the-fly quantization save through the conversions of their weights
            conversions = self.update_weight_conversions(conversions, model, for_save=True)
        # Save the modules' own names, which load as they are. A save cannot reverse the ModelOpt renames, as on save
        # the renames run before the converters.
        shipped = {tuple(rename._original_source_patterns) for rename in checkpoint_renames()}
        kept = [c for c in conversions if tuple(c._original_source_patterns) not in shipped]
        model._weight_conversions = kept  # type: ignore[unresolved-attribute]
        return model

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
            NVFP4Reciprocal,
            NVFP4SwizzleScales,
            NVFP4ViewPackedInt8,
            _reciprocal_globals,
        )

        module, name = get_module_from_name(model, param_name)
        if not isinstance(module, (NVFP4Linear, NVFP4Experts)):
            return []
        if name.endswith("global_scale"):
            return [NVFP4Reciprocal(self)] if _reciprocal_globals(self) else []
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
        from ..integrations.nvfp4 import (
            EXPERT_PROJECTIONS,
            NVFP4Deserialize,
            NVFP4Quantize,
            checkpoint_renames,
            nvfp4_checkpoint_conversions,
        )

        conversions = super().update_weight_conversions(weight_conversions, model)
        if self.quantization_config.dequantize:
            return [*checkpoint_renames(), *conversions]
        quantize = not self.pre_quantized and not for_save
        op = NVFP4Quantize if quantize else NVFP4Deserialize
        renames = checkpoint_renames() if self.pre_quantized and not for_save else []
        conversions = [*renames, *nvfp4_checkpoint_conversions(self, conversions, op)]
        # A fused stack in the checkpoint, or a dense scale, loads through its own converter. It comes first, as the
        # first matching converter wins. On save, skip the params a model converter writes back, as they would match
        # first there.
        reverse = [c.reverse_transform() for c in conversions if isinstance(c, WeightConverter)] if for_save else []
        # An FP8 layer's weight and scale load into their own slots too, so the scale shards by its own plan.
        fp8_linears = {
            name
            for name, module in model.named_modules()
            if type(module) is torch.nn.Linear and "weight_scale_inv" in module._parameters
        }
        per_param = [
            WeightConverter(re.escape(name), name, [op(self)])
            for name, _ in model.named_parameters()
            if (self.get_param_ops(model, name) or name.rpartition(".")[0] in fp8_linears)
            and not any(c.rename_source_key(name)[1] for c in reverse)
            # quantizing on the fly only loads the weights, their scales come out of the quantize op
            and (not quantize or name.rpartition(".")[2] in ("weight", *EXPERT_PROJECTIONS))
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
