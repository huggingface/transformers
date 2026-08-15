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

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

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
        from ..integrations.finegrained import replace_with_finegrained_layer

        self._normalize_modules_to_not_convert(model)
        if self._quant_method() == "mxfp4" and getattr(self.quantization_config, "activation_format", None) is None:
            # GPT-OSS MXFP4 runs weight-only (W4A16): raw bf16 activations against packed
            # weights. The kernels' weight-native default would quantize activations to mxfp4.
            self.quantization_config.activation_format = "bf16"
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_finegrained_layer(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

    def _process_model_after_weight_loading(self, model, **kwargs):
        # dsv4-flash-base stores its (power-of-two) ue8m0 scales in a float32 container under
        # `.scale`; those renamed keys keep the on-disk float32 dtype, so cast them to the UE8M0
        # dtype the kernels expect (exact, since the values are powers of two). Checkpoints that
        # already ship the native float8 E8M0 dtype (e.g. dsv4-flash) are left untouched.
        if self.quantization_config.scale_fmt == "ue8m0":
            from ..integrations.finegrained import _get_ue8m0_dtype

            ue8m0 = _get_ue8m0_dtype()
            float32_scales = [
                name
                for name, param in model.named_parameters()
                if name.endswith("_scale_inv") and param.dtype == torch.float32
            ]
            for name in float32_scales:
                module_name, _, attr = name.rpartition(".")
                module = model.get_submodule(module_name)
                scale = getattr(module, attr)
                setattr(module, attr, torch.nn.Parameter(scale.data.to(ue8m0), requires_grad=False))

        # Single-process multi-device is unsafe for DeepGEMM (its kernels are bound to one CUDA
        # context); route those models through Triton/grouped_mm instead.
        from ..integrations.finegrained import _disable_deepgemm_on_multi_device

        _disable_deepgemm_on_multi_device(model)

        from ..integrations.finegrained import swizzle_scales_after_loading

        swizzle_scales_after_loading(model)
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
            FineGrainedGateUpBiasDeinterleave,
            FineGrainedMxfp4Deserialize,
        )

        # GPT-OSS-style MXFP4 checkpoints ship {proj}_blocks + {proj}_scales (interleaved
        # gate|up rows); deserialize them into the packed weight + e8m0 scale pair the
        # finegrained kernels read, de-interleaving gate_up rows (and the bias to match).
        if self.pre_quantized and self._quant_method() == "mxfp4" and not self.quantization_config.dequantize:
            return [
                WeightConverter(
                    source_patterns=["gate_up_proj_blocks", "gate_up_proj_scales"],
                    target_patterns=r"gate_up_proj$",
                    operations=[FineGrainedMxfp4Deserialize(self)],
                ),
                WeightConverter(
                    source_patterns=["down_proj_blocks", "down_proj_scales"],
                    target_patterns=r"down_proj$",
                    operations=[FineGrainedMxfp4Deserialize(self)],
                ),
                WeightConverter(
                    source_patterns=[r"gate_up_proj_bias$"],
                    target_patterns=r"gate_up_proj_bias$",
                    operations=[FineGrainedGateUpBiasDeinterleave(self)],
                ),
            ]

        # NVIDIA modelopt NVFP4 checkpoints: per-expert split gate/up projections with
        # packed-uint8 weights, E4M3 group-16 `weight_scale`, and fp32 `weight_scale_2`
        # second-level globals (bit-identical across the gate/up halves — asserted by the
        # fuse op). The arch conversion plan already stacks the expert WEIGHTS; these add
        # the same stacking for scales and globals, plus the uint8 -> int8 packed bitcast.
        # Calibrated `input_scale` entries are left unconsumed (activations run the
        # kernels' dynamic quant); they surface as unexpected keys, which is intended.
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
            # No converters for non-expert modules: this checkpoint family quantizes ONLY
            # the routed experts (every attention / shared-expert / dense-MLP module sits
            # in the ignore list and ships bf16). Generic weight_scale* renames here would
            # run BEFORE converter collection and mangle the expert keys out from under
            # the converters above (per-expert `weight_global_scale` orphans and
            # `_scale_inv_inv` double-suffix targets — the saturation bug of round 8).
            # Quantized-dense modelopt checkpoints need scoped per-module converters when
            # one shows up.

        if self.pre_quantized and self.quantization_config.dequantize:
            return [
                # either use the dollar sign, or permute the source patterns to start matching against the scales first
                # We also collect the activation scales, they will not be used
                WeightConverter(
                    source_patterns=["weight$", "weight_scale_inv", "activation_scale"],
                    target_patterns="weight",
                    operations=[FineGrainedDequantize(self)],
                )
            ]
        return []

    def update_weight_conversions(self, weight_conversions):
        """When loading with ``dequantize=True``, attach an :class:`FineGrainedDequantize` op to
        every existing :class:`WeightConverter` so that per-block scales are folded into
        the weight *before* any later merge/concat ops collapse the per-expert structure.

        For each model-supplied converter that has a ``.weight`` source, we:
          1. anchor the existing weight patterns with ``$`` so they don't accidentally
             also match the ``.weight_scale_inv`` keys (the regex is searched, so the
             unanchored prefix would match both, sending scales to the wrong bucket);
          2. add anchored ``*.weight_scale_inv`` sources next to each weight pattern so
             the loader collects scale tensors alongside the weight tensors into the
             *same* converter bucket (both keys rewrite to the same target);
          3. prepend a fresh :class:`FineGrainedDequantize` op so dequant runs first, before
             any merge/concat collapses the per-expert structure.

        The generic ``weight$ + weight_scale_inv → weight`` converter from
        :meth:`get_weight_conversions` is still appended at the end as a fallback for
        plain ``nn.Linear`` weights with no model-specific converter.
        """
        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..integrations.finegrained import FineGrainedDequantize

        # `*.scale` → `*.weight_scale_inv`. Some FP8 checkpoints (e.g. DeepSeek-V4-Flash)
        # ship per-block scales under `.scale`; the model expects `.weight_scale_inv`.
        # Lives here (not in each model's `conversion_mapping`) so non-FP8 round-trips
        # don't see a stray rule. Needed in both dequantize modes — `dequantize=False`
        # loads scales as parameters, `dequantize=True` feeds them into `FineGrainedDequantize`.
        scale_rename = WeightRenaming(source_patterns=r"^(.+)\.scale$", target_patterns=r"\1.weight_scale_inv")
        weight_conversions = [scale_rename] + list(weight_conversions)

        if self.pre_quantized and self._quant_method() == "modelopt" and not self.quantization_config.dequantize:
            # Anchor the arch plan's `.weight` sources so they don't also swallow the
            # modelopt `weight_scale`/`weight_scale_2` keys into the weight merge chain
            # (unanchored patterns are searched — same trap the dequantize path guards).
            from ..integrations.finegrained import FineGrainedViewPackedInt8

            updated = []
            for conv in weight_conversions:
                if isinstance(conv, WeightConverter) and any(p.endswith(".weight") for p in conv.source_patterns):
                    conv = WeightConverter(
                        source_patterns=[p + "$" if p.endswith(".weight") else p for p in conv.source_patterns],
                        target_patterns=conv.target_patterns,
                        # the stacked packed-fp4 expert weights need the same uint8 -> int8
                        # bitcast the standalone-linear converter applies (pass-through on
                        # any unquantized weights the plan also routes here)
                        operations=list(conv.operations) + [FineGrainedViewPackedInt8(self)],
                    )
                updated.append(conv)
            return updated + self.get_weight_conversions()
        if not (self.pre_quantized and self.quantization_config.dequantize):
            return weight_conversions + self.get_weight_conversions()

        updated: list = []
        for conv in weight_conversions:
            # Only WeightConverter has ``.operations`` to extend with the dequant op;
            # WeightRenaming (e.g. the ``scale_rename`` we prepended) just passes through.
            if not isinstance(conv, WeightConverter):
                updated.append(conv)
                continue
            weight_sources = [p for p in conv.source_patterns if p.endswith(".weight")]
            if weight_sources:
                anchored_weight = [p + "$" for p in weight_sources]
                scale_sources = [p[: -len(".weight")] + ".weight_scale_inv$" for p in weight_sources]
                other = [p for p in conv.source_patterns if not p.endswith(".weight")]
                new_sources = anchored_weight + scale_sources + other
                new_ops = [FineGrainedDequantize(self)] + list(conv.operations)
                conv = WeightConverter(
                    source_patterns=new_sources,
                    target_patterns=conv._original_target_patterns,
                    operations=new_ops,
                )
            updated.append(conv)
        # Generic fallback for plain ``nn.Linear`` weights with no model-specific converter.
        updated.extend(self.get_weight_conversions())
        return updated
