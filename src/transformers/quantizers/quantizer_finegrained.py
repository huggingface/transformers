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
from typing import TYPE_CHECKING

from ..utils import is_accelerate_available, is_torch_available, is_torch_xpu_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import FineGrainedConfig

logger = logging.get_logger(__name__)


class FineGrainedHfQuantizer(HfQuantizer):
    """Quantizer for the fine-grained family the `kernels-community/finegrained-kernels` package
    serves: block-FP8 (fp32 or UE8M0 scales), MXFP8, MXFP4 and NVFP4, for dense linears, embeddings
    and MoE experts alike.

    It never declares the weight format — that is resolved from the checkpoint tensors themselves,
    the way the kernels resolve it. What it does own is the vocabulary each producer ships its
    scales under (modelopt's `weight_scale_2` / `input_scale`, GPT-OSS's `_blocks` / `_scales`) and
    the conversion ops that bring them to the layout `FineGrainedExperts` holds.
    """

    requires_calibration = False
    quantization_config: "FineGrainedConfig"

    def _assert_dequantize_supported(self) -> None:
        """The dequantize chain folds a weight into full precision with its per-block scale alone.
        NVFP4 carries a second level (modelopt's `weight_scale_2`) that the chain has no slot for,
        so refuse rather than hand back a weight scaled by `1 / global`."""
        if self._quant_method() in ("nvfp4", "modelopt"):
            raise NotImplementedError(
                f"`dequantize=True` is not supported for {self._quant_method()!r} checkpoints: their "
                "two-level scales cannot be folded into a full-precision weight by this path. Load "
                "them quantized on a GPU that serves NVFP4, or start from a bf16 checkpoint."
            )

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading an FP8 quantized model requires accelerate (`pip install accelerate`)")

        if self.quantization_config.dequantize:
            self._assert_dequantize_supported()
            return

        if not torch.cuda.is_available() and not is_torch_xpu_available():
            if self.pre_quantized:
                self._assert_dequantize_supported()
                logger.warning_once(
                    "Using FP8 quantized models requires a GPU or XPU, we will default to dequantizing the model to bf16 since no GPU or XPU is available"
                )
                self.quantization_config.dequantize = True
                return
            else:
                raise RuntimeError("No GPU or XPU found. A GPU or XPU is needed for FP8 quantization.")

        if torch.cuda.is_available():
            compute_capability = torch.cuda.get_device_capability()
            major, minor = compute_capability
            if (major < 8) or (major == 8 and minor < 9):
                self._assert_dequantize_supported()
                logger.warning_once(
                    "FP8 quantized models is only supported on GPUs with compute capability >= 8.9 (e.g 4090/H100)"
                    f", actual = `{major}.{minor}`. We will default to dequantizing the model to bf16. Feel free "
                    f"to use a different quantization method like bitsandbytes or torchao"
                )
                self.quantization_config.dequantize = True
                return

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded an FP8 model on CPU and have a CUDA or XPU device available, make sure to set "
                "your model on a GPU or XPU device in order to run your model. To remove this warning, "
                "pass device_map = 'cuda' or 'xpu'. "
            )
        elif isinstance(device_map, dict):
            if (
                not self.pre_quantized
                and len(device_map) > 1
                and ("cpu" in device_map.values() or "disk" in device_map.values())
            ):
                raise ValueError(
                    "You are attempting to load an FP8 model with a device_map that contains a cpu/disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the cpu/disk device from the device_map."
                )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        # `FP8GroupedLinear` is a subclass of `FineGrainedLinear`, so the tuple covers it implicitly.
        from ..integrations.finegrained import FineGrainedExperts, FineGrainedLinear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, (FineGrainedLinear, FineGrainedExperts)):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        return False

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param: "torch.Tensor") -> float:
        "Return the element size (in bytes) for `param_name`."
        if self.param_needs_quantization(model, param_name):
            # 8 bit, this is needed as when `pre_quantized`` is False, we don't set the dtype of the FineGrainedLinear in order to correctly load the weights
            return 1
        return super().param_element_size(model, param_name, param)

    def _normalize_modules_to_not_convert(self, model: "PreTrainedModel"):
        """Rewrite the skip-list to the model's own module tree.
        For models that were already released, if they have a list of modules to not quantize
        we need to apply the weight renaming / weight conversion opérations to get the actual
        layer name of the model in `transformers`.
        """
        skip = self.quantization_config.modules_to_not_convert
        if not skip:
            return

        from ..conversion_mapping import get_model_conversion_mapping

        renamings = get_model_conversion_mapping(model)
        remapped = []
        for name in skip:
            renamed = name
            for rename in renamings:
                renamed, _ = rename.rename_source_key(renamed)
            remapped.append(renamed)
        self.quantization_config.modules_to_not_convert = remapped

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from ..integrations.finegrained import replace_with_finegrained_embedding, replace_with_finegrained_layer

        self._normalize_modules_to_not_convert(model)
        if self._quant_method() == "mxfp4" and getattr(self.quantization_config, "activation_format", None) is None:
            # GPT-OSS MXFP4 runs weight-only (W4A16): raw bf16 activations against packed
            # weights. The kernels' weight-native default would quantize activations to mxfp4.
            self.quantization_config.activation_format = "bf16"
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        if self.pre_quantized and self.quantization_config.modules_to_convert:
            replace_with_finegrained_embedding(
                model, self.quantization_config.modules_to_convert, self.modules_to_not_convert
            )
        replace_with_finegrained_layer(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        if getattr(self.quantization_config, "activation_format", None) == "bf16":
            # Weight-only: the experts hold no activation global, so a calibrated checkpoint's
            # `input_scale` has no slot to load into. It is not an unexpected key, it is one this
            # run has no use for — a later W4A4 run of the same checkpoint reads it.
            model._keys_to_ignore_on_load_unexpected = set(model._keys_to_ignore_on_load_unexpected or []) | {
                r"input_scale$"
            }

    def _process_model_after_weight_loading(self, model, **kwargs):
        from ..integrations.finegrained import disable_deepgemm_on_multi_device

        disable_deepgemm_on_multi_device(model)
        return model

    def update_tp_plan(self, config):
        if "Qwen3" in config.__class__.__name__:
            text_plan = {
                "layers.*.self_attn.q_proj.weight": "colwise",
                "layers.*.self_attn.q_proj.weight_scale_inv": "colwise",
                "layers.*.self_attn.k_proj.weight": "colwise",
                "layers.*.self_attn.k_proj.weight_scale_inv": "colwise",
                "layers.*.self_attn.v_proj.weight": "colwise",
                "layers.*.self_attn.v_proj.weight_scale_inv": "colwise",
                "layers.*.self_attn.o_proj.weight": "rowwise",
                "layers.*.self_attn.o_proj.weight_scale_inv": "rowwise",
                "layers.*.mlp.gate_proj.weight": "colwise",
                "layers.*.mlp.gate_proj.weight_scale_inv": "colwise",
                "layers.*.mlp.up_proj.weight": "colwise",
                "layers.*.mlp.up_proj.weight_scale_inv": "colwise",
                "layers.*.mlp.down_proj.weight": "rowwise",
                "layers.*.mlp.down_proj.weight_scale_inv": "rowwise",
            }

            config.base_model_tp_plan = text_plan

        # Per-impl rewrite of the experts parallel-layer kind. Applied LAST so it composes
        # on top of any plan written above (e.g. the Qwen3 dense plan). Models carry the
        # experts mapping under `base_model_tp_plan` and/or `base_model_ep_plan` — rewrite
        # both. See `FineGrainedExperts._impl_tp_layer_overrides`.
        from ..integrations.finegrained import FineGrainedExperts

        impl = getattr(config, "_experts_implementation", None)
        layer_overrides = FineGrainedExperts._impl_tp_layer_overrides.get(impl)
        if layer_overrides:
            for plan_attr in ("base_model_tp_plan", "base_model_ep_plan"):
                base_plan = getattr(config, plan_attr, None) or {}
                updated_plan = {k: layer_overrides.get(v, v) for k, v in base_plan.items()}
                if updated_plan != base_plan:
                    setattr(config, plan_attr, updated_plan)

        return config

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return False

    @property
    def is_compileable(self) -> bool:
        return True

    def get_quantize_ops(self):
        from ..integrations.finegrained import FineGrainedQuantize

        return FineGrainedQuantize(self)

    def _quant_method(self) -> str:
        method = getattr(self.quantization_config, "quant_method", None)
        return getattr(method, "value", method)

    def get_weight_conversions(self):
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained import (
            FineGrainedDequantize,
            FineGrainedInterleaveGateUp,
            FineGrainedPackedBlocks,
            FineGrainedScaleContainer,
            FineGrainedSwizzleScales,
        )

        # {proj}_blocks + {proj}_scales checkpoints (GPT-OSS): the blocks regroup into the packed
        # weight, the exponent bytes take the scale container op; then the usual layout ops. Bare
        # patterns (no `experts.` scope), so they ride here rather than via `_with_expert_layout_ops`.
        if self.pre_quantized and self._quant_method() == "mxfp4" and not self.quantization_config.dequantize:
            converters = []
            for proj in ("gate_up_proj", "down_proj"):
                interleave = [FineGrainedInterleaveGateUp(self)] if proj == "gate_up_proj" else []
                converters += [
                    WeightConverter(
                        source_patterns=rf"{proj}_blocks$",
                        target_patterns=proj,
                        operations=[FineGrainedPackedBlocks(self), *interleave, FineGrainedSwizzleScales(self)],
                    ),
                    WeightConverter(
                        source_patterns=rf"{proj}_scales$",
                        target_patterns=f"{proj}_scale_inv",
                        operations=[*interleave, FineGrainedScaleContainer(self), FineGrainedSwizzleScales(self)],
                    ),
                ]
            return converters

        # The same GPT-OSS keys under `dequantize=True`: the blocks regroup into the packed rows
        # and the exponent-byte scales come back out of them as bf16, in the (E, hidden, 2I)
        # orientation the unquantized experts hold.
        if self.pre_quantized and self._quant_method() == "mxfp4" and self.quantization_config.dequantize:
            from ..core_model_loading import Transpose

            return [
                WeightConverter(
                    source_patterns=[rf"{proj}_blocks$", rf"{proj}_scales$"],
                    target_patterns=proj,
                    operations=[FineGrainedPackedBlocks(self), FineGrainedDequantize(self), Transpose(1, 2)],
                )
                for proj in ("gate_up_proj", "down_proj")
            ]

        # modelopt NVFP4 scales (`weight_scale`, `weight_scale_2`, `input_scale`), in both layouts
        # a modelopt checkpoint ships them in: per expert per projection (GLM-5.2-NVFP4) or already
        # stacked per layer (the fused vLLM layout, Muse-Spark). A checkpoint matches one set and
        # the other never fires. Routed experts only — a generic `weight_scale*` rename would run
        # before converter collection and mangle these keys.
        if self.pre_quantized and self._quant_method() == "modelopt" and not self.quantization_config.dequantize:
            # A weight-only run keeps activations bf16, so its experts hold no activation global
            # and the checkpoint's calibrated `input_scale` has nowhere to land: leave those keys
            # to the unexpected-key filter below rather than converting them onto a module slot
            # that does not exist.
            calibrated = getattr(self.quantization_config, "activation_format", None) != "bf16"
            from ..core_model_loading import Concatenate, MergeModulelist
            from ..integrations.finegrained import (
                FineGrainedInputGlobals,
                FineGrainedScaleContainer,
                FineGrainedViewPackedInt8,
                FineGrainedWeightGlobals,
            )

            # MergeModulelist is what stamps each source with its expert index, which is what lets
            # expert parallelism keep only this rank's experts (see `tensor_idx` in
            # core_model_loading). Without it every rank collects all E values and the forward
            # asserts on the per-expert count.
            per_expert = [
                WeightConverter(
                    source_patterns=[
                        "mlp.experts.*.gate_proj.weight_scale$",
                        "mlp.experts.*.up_proj.weight_scale$",
                    ],
                    target_patterns="mlp.experts.gate_up_proj_scale_inv",
                    operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
                ),
                WeightConverter(
                    source_patterns="mlp.experts.*.down_proj.weight_scale$",
                    target_patterns="mlp.experts.down_proj_scale_inv",
                    operations=[MergeModulelist(dim=0)],
                ),
                # every second-level global of a layer in ONE converter: the gate|up stack's two
                # calibrated halves merge into one global per expert by folding the up half's onto
                # the down projection, so the weight globals and the down's input scale have to be
                # decided together (`FineGrainedWeightGlobals`)
                WeightConverter(
                    source_patterns=[
                        "mlp.experts.*.gate_proj.weight_scale_2",
                        "mlp.experts.*.up_proj.weight_scale_2",
                        "mlp.experts.*.down_proj.weight_scale_2",
                        *(["mlp.experts.*.down_proj.input_scale"] if calibrated else []),
                    ],
                    target_patterns=[
                        "mlp.experts.gate_up_proj_weight_global_scale",
                        "mlp.experts.down_proj_weight_global_scale",
                        *(["mlp.experts.down_proj_input_global_scale"] if calibrated else []),
                    ],
                    operations=[MergeModulelist(dim=0), FineGrainedWeightGlobals(self)],
                ),
            ]
            if calibrated:
                per_expert.append(
                    WeightConverter(
                        source_patterns=[
                            "mlp.experts.*.gate_proj.input_scale",
                            "mlp.experts.*.up_proj.input_scale",
                        ],
                        target_patterns="mlp.experts.gate_up_proj_input_global_scale",
                        operations=[MergeModulelist(dim=0), FineGrainedInputGlobals(self)],
                    )
                )
            fused = [
                # the container cast the module's scale dtype needs; `_with_expert_layout_ops`
                # adds the interleave and the swizzle on top (it re-adds the cast, which is
                # idempotent)
                WeightConverter(
                    source_patterns=rf"experts\.{proj}_weight_scale$",
                    target_patterns=f"experts.{proj}_scale_inv",
                    operations=[FineGrainedScaleContainer(self)],
                )
                for proj in ("gate_up_proj", "down_proj")
            ]
            fused += [
                # the same one-converter rule as the per-expert layout above, over the keys the
                # fused layout ships
                WeightConverter(
                    source_patterns=[
                        r"experts\.gate_up_proj_weight_scale_2$",
                        r"experts\.down_proj_weight_scale_2$",
                        *([r"experts\.down_proj_input_scale$"] if calibrated else []),
                    ],
                    target_patterns=[
                        "experts.gate_up_proj_weight_global_scale",
                        "experts.down_proj_weight_global_scale",
                        *(["experts.down_proj_input_global_scale"] if calibrated else []),
                    ],
                    operations=[FineGrainedWeightGlobals(self)],
                ),
            ]
            if calibrated:
                fused.append(
                    WeightConverter(
                        source_patterns=r"experts\.gate_up_proj_input_scale$",
                        target_patterns="experts.gate_up_proj_input_global_scale",
                        operations=[FineGrainedInputGlobals(self)],
                    )
                )
            # the fused expert WEIGHTS take the packed uint8 -> int8 view the `.weight`-anchored
            # rule below gives the per-expert layout; the layout ops ride on top of both
            fused += [
                WeightConverter(
                    source_patterns=rf"experts\.{proj}$",
                    target_patterns=f"experts.{proj}",
                    operations=[FineGrainedViewPackedInt8(self)],
                )
                for proj in ("gate_up_proj", "down_proj")
            ]
            return per_expert + fused

        if self.pre_quantized and self.quantization_config.dequantize:
            return [
                # anchored `weight$` so the scale keys land in the scale slots; the activation
                # scales are collected only to be dropped
                WeightConverter(
                    source_patterns=["weight$", "weight_scale_inv", "activation_scale"],
                    target_patterns="weight",
                    operations=[FineGrainedDequantize(self)],
                )
            ]
        return []

    def update_weight_conversions(self, weight_conversions):
        """Rewrite the model's conversion plan for this checkpoint:

        - every mode: ``*.scale`` -> ``*.weight_scale_inv`` (DeepSeek-V4-Flash ships per-block
          scales under ``.scale``; kept here rather than in each model's mapping so non-FP8
          round-trips never see the rule);
        - ``dequantize=True``: a :class:`FineGrainedDequantize` runs first in every converter with a
          ``.weight`` source (anchored, with the sibling ``.weight_scale_inv`` collected alongside),
          so per-expert (weight, scale) pairs fold into full-precision tensors before merge/concat
          ops collapse the per-expert structure;
        - every mode: sharded ``.weight`` converter targets are anchored, since they become source
          patterns on save and would otherwise match their scale keys;
        - modelopt: the arch plan's ``.weight`` sources are anchored so they don't swallow the
          ``weight_scale`` / ``weight_scale_2`` keys, and the packed weights get the uint8 -> int8 view;
        - otherwise the expert converters get the module layout ops (``_with_expert_layout_ops``).

        :meth:`get_weight_conversions` is appended in every mode."""
        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..integrations.finegrained import FineGrainedDequantize, FineGrainedViewPackedInt8

        scale_rename = WeightRenaming(source_patterns=r"^(.+)\.scale$", target_patterns=r"\1.weight_scale_inv")
        weight_conversions = [scale_rename, *weight_conversions]
        # a converter's targets become source patterns on save: anchor sharded `.weight` targets
        # (Qwen4-Exp's `ngram_embedding`) so they don't also match their scale keys then
        for conv in weight_conversions:
            if isinstance(conv, WeightConverter):
                conv._original_target_patterns = [
                    f"{p}$" if p.endswith(".weight") else p for p in conv._original_target_patterns
                ]

        if self.pre_quantized and self.quantization_config.dequantize:
            updated = []
            for conv in weight_conversions:
                weight_sources = [p for p in conv.source_patterns if p.endswith(".weight")]
                if isinstance(conv, WeightConverter) and weight_sources:
                    conv = WeightConverter(
                        source_patterns=[p + "$" for p in weight_sources]
                        + [p[: -len(".weight")] + ".weight_scale_inv$" for p in weight_sources]
                        + [p for p in conv.source_patterns if not p.endswith(".weight")],
                        target_patterns=conv._original_target_patterns,
                        operations=[FineGrainedDequantize(self), *conv.operations],
                    )
                updated.append(conv)
            return updated + self.get_weight_conversions()

        if self.pre_quantized and self._quant_method() == "modelopt":
            updated = []
            for conv in weight_conversions:
                if isinstance(conv, WeightConverter) and any(p.endswith(".weight") for p in conv.source_patterns):
                    conv = WeightConverter(
                        source_patterns=[p + "$" if p.endswith(".weight") else p for p in conv.source_patterns],
                        target_patterns=conv.target_patterns,
                        operations=[*conv.operations, FineGrainedViewPackedInt8(self)],
                    )
                updated.append(conv)
            weight_conversions = updated

        return self._with_expert_layout_ops(weight_conversions + self.get_weight_conversions())

    def _with_expert_layout_ops(self, weight_conversions):
        """Give every converter that produces expert tensors the layout ``FineGrainedExperts`` holds
        (see its ``__init__``), in chain order: gate|up rows interleaved (weight, scale grid and bias
        share the row axis), the scale in the held container dtype, then the swizzled scale layout.
        Each op decides from the model it runs against and each has a reverse, so saving restores
        the checkpoint layout. Catch-all converters at the end take keys that arrive already under
        the fused names (a plain rename, a transformers-format checkpoint) and dense linears' scale
        keys, so they get the same treatment."""
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained import (
            FineGrainedInterleaveGateUp,
            FineGrainedScaleContainer,
            FineGrainedSwizzleScales,
        )

        def layout_ops(target: str) -> list:
            ops = []
            if "gate_up_proj" in target:
                ops.append(FineGrainedInterleaveGateUp(self))
            if "_bias" not in target:
                ops += [
                    FineGrainedScaleContainer(self),
                    FineGrainedSwizzleScales(self),
                ]  # the swizzle reads 1-byte scales
            return ops

        updated = []
        for conv in weight_conversions:
            targets = conv.target_patterns if isinstance(conv, WeightConverter) else []
            expert_targets = [
                t for t in targets if re.search(r"experts\.(gate_up_proj|down_proj)(_scale_inv|_bias)?\$?$", t)
            ]
            if expert_targets:
                conv = WeightConverter(
                    source_patterns=conv.source_patterns,
                    target_patterns=conv.target_patterns,
                    operations=list(conv.operations) + layout_ops(expert_targets[0]),
                )
            updated.append(conv)
        # dense linears have no converter: their scale keys take the container cast alone
        updated.append(
            WeightConverter(
                source_patterns=r"weight_scale_inv$",
                target_patterns="weight_scale_inv",
                operations=[FineGrainedScaleContainer(self)],
            )
        )
        for name in ("gate_up_proj", "gate_up_proj_scale_inv", "gate_up_proj_bias", "down_proj_scale_inv"):
            if layout_ops(name):
                # the target is replacement text (op outputs are keyed by it), so no regex escapes
                updated.append(
                    WeightConverter(
                        source_patterns=rf"experts\.{name}$",
                        target_patterns=f"experts.{name}",
                        operations=layout_ops(name),
                    )
                )
        return updated
