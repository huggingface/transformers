import re
from typing import TYPE_CHECKING

from ..utils import is_accelerate_available, is_torch_available, is_torch_xpu_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import FineGrainedFP8Config

logger = logging.get_logger(__name__)


class FineGrainedHfQuantizer(HfQuantizer):
    """
    FP8 quantization implementation supporting both standard and MoE models.
    Supports both e4m3fn formats based on platform.
    """

    requires_calibration = False
    quantization_config: "FineGrainedFP8Config"

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading an FP8 quantized model requires accelerate (`pip install accelerate`)")

        if self.quantization_config.dequantize:
            return

        if not torch.cuda.is_available() and not is_torch_xpu_available():
            if self.pre_quantized:
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

        # modelopt NVFP4 (per-expert gate/up, E4M3 group-16 `weight_scale`, fp32 `weight_scale_2`
        # globals): the arch plan stacks the expert weights; these stack the scales and globals the
        # same way. Only the routed experts are quantized, so no dense converters — a generic
        # `weight_scale*` rename would run before converter collection and mangle these keys.
        # Calibrated `input_scale` keys stay unconsumed (dynamic act quant) and surface as unexpected.
        if self.pre_quantized and self._quant_method() == "modelopt" and not self.quantization_config.dequantize:
            from ..core_model_loading import Concatenate, MergeModulelist
            from ..integrations.finegrained import FineGrainedFuseEqualGlobals

            return [
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
                WeightConverter(
                    source_patterns=[
                        "mlp.experts.*.gate_proj.weight_scale_2",
                        "mlp.experts.*.up_proj.weight_scale_2",
                    ],
                    target_patterns="mlp.experts.gate_up_proj_global_scale",
                    # MergeModulelist is what stamps each source with its expert index, which
                    # is what lets expert parallelism keep only this rank's experts (see
                    # `tensor_idx` in core_model_loading). Without it every rank collects all
                    # E globals and the forward asserts on the per-expert count.
                    operations=[MergeModulelist(dim=0), FineGrainedFuseEqualGlobals(self)],
                ),
                WeightConverter(
                    source_patterns="mlp.experts.*.down_proj.weight_scale_2",
                    target_patterns="mlp.experts.down_proj_global_scale",
                    operations=[MergeModulelist(dim=0), FineGrainedFuseEqualGlobals(self)],
                ),
            ]

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
