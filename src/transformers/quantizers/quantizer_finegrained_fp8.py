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


class FineGrainedFP8HfQuantizer(HfQuantizer):
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
                and "cpu" in device_map.values()
                or "disk" in device_map.values()
            ):
                raise ValueError(
                    "You are attempting to load an FP8 model with a device_map that contains a cpu/disk device."
                    "This is not supported when the model is quantized on the fly. "
                    "Please use a quantized checkpoint or remove the cpu/disk device from the device_map."
                )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from ..integrations.finegrained_fp8 import FP8Experts, FP8Linear

        module, tensor_name = get_module_from_name(model, param_name)
        if isinstance(module, (FP8Linear, FP8Experts)):
            if self.pre_quantized or tensor_name == "bias":
                return False
            else:
                return True
        return False

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param: "torch.Tensor") -> float:
        "Return the element size (in bytes) for `param_name`."
        if self.param_needs_quantization(model, param_name):
            # 8 bit, this is neeed as when `pre_quantized`` is False, we don't set the dtype of the FP8Linear in order to correctly load the weights
            return 1
        return super().param_element_size(model, param_name, param)

    def _normalize_modules_to_not_convert(self, model: "PreTrainedModel"):
        """Rewrite the skip-list to the model's own module tree.

        ``modules_to_not_convert`` (and its ``ignored_layers`` alias) ships inside the checkpoint's
        ``quantization_config`` using the *checkpoint's* key layout. Composite / renamed models expose a
        different module tree (e.g. a VL model nests its decoder under ``model.language_model`` and renames
        ``block_sparse_moe`` to ``mlp``), and some weights are *fused* by weight converters (e.g.
        ``gate_proj`` + ``up_proj`` -> ``gate_up_proj``), so a raw checkpoint name may point at a module that
        no longer exists under that name and would be quantized anyway. Run every skip name through the
        model's own ordered conversion chain — the same transforms the weight loader uses — so it lands on
        the real (possibly fused / renamed) module. Names already in model-tree form match no source pattern
        and pass through unchanged.
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
        from ..integrations.finegrained_fp8 import replace_with_fp8_linear

        self._normalize_modules_to_not_convert(model)
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_fp8_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
            pre_quantized=self.pre_quantized,
        )

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
        from ..integrations.finegrained_fp8 import Fp8Quantize

        return Fp8Quantize(self)

    def get_weight_conversions(self):
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained_fp8 import Fp8Dequantize

        if self.pre_quantized and self.quantization_config.dequantize:
            return [
                # either use the dollar sign, or permute the source patterns to start matching against the scales first
                # We also collect the activation scales, they will not be used
                WeightConverter(
                    source_patterns=["weight$", "weight_scale_inv", "activation_scale"],
                    target_patterns="weight",
                    operations=[Fp8Dequantize(self)],
                )
            ]
        return []

    def _is_mxfp8(self) -> bool:
        """MXFP8 checkpoints ship E8M0 (uint8) per-block scales; plain FP8 ships float32."""
        quant_method = getattr(self.quantization_config, "quant_method", None)
        return quant_method == "mxfp8"

    def _update_weight_conversions_mxfp8(self, weight_conversions):
        """Native MXFP8 path: prepend an :class:`Fp8DecodeScale` op so the uint8 E8M0
        scales are decoded to float32 ``2 ** (byte - 127)`` *before* any merge/concat op
        collapses the per-expert structure, and add a generic fallback converter that
        decodes the scales of plain ``FP8Linear`` weights (attention / dense projections)
        which have no model-specific converter.
        """
        from ..core_model_loading import WeightConverter
        from ..integrations.finegrained_fp8 import Fp8DecodeScale

        updated: list = []
        for conv in weight_conversions:
            if isinstance(conv, WeightConverter) and any(p.endswith(".weight") for p in conv.source_patterns):
                conv = WeightConverter(
                    source_patterns=conv.source_patterns,
                    target_patterns=conv._original_target_patterns,
                    operations=[Fp8DecodeScale(self)] + list(conv.operations),
                )
            updated.append(conv)
        # Generic fallback for plain ``nn.Linear`` scales with no model-specific converter.
        # Listed last so the model converters above win the first-match for expert/dense scales.
        updated.append(
            WeightConverter(
                source_patterns=["weight_scale_inv"],
                target_patterns="weight_scale_inv",
                operations=[Fp8DecodeScale(self)],
            )
        )
        return updated

    def update_weight_conversions(self, weight_conversions):
        """When loading with ``dequantize=True``, attach an :class:`Fp8Dequantize` op to
        every existing :class:`WeightConverter` so that per-block scales are folded into
        the weight *before* any later merge/concat ops collapse the per-expert structure.

        For each model-supplied converter that has a ``.weight`` source, we:
          1. anchor the existing weight patterns with ``$`` so they don't accidentally
             also match the ``.weight_scale_inv`` keys (the regex is searched, so the
             unanchored prefix would match both, sending scales to the wrong bucket);
          2. add anchored ``*.weight_scale_inv`` sources next to each weight pattern so
             the loader collects scale tensors alongside the weight tensors into the
             *same* converter bucket (both keys rewrite to the same target);
          3. prepend a fresh :class:`Fp8Dequantize` op so dequant runs first, before
             any merge/concat collapses the per-expert structure.

        The generic ``weight$ + weight_scale_inv → weight`` converter from
        :meth:`get_weight_conversions` is still appended at the end as a fallback for
        plain ``nn.Linear`` weights with no model-specific converter.
        """
        # Native (``dequantize=False``) path: the weights stay in ``float8_e4m3fn`` and
        # the model's own converters route the sibling ``*.weight_scale_inv`` keys through
        # the same substring match + suffix-preserving rename as ``*.weight`` (see
        # huggingface/transformers#45634). MXFP8 checkpoints ship those per-block scales as
        # ``uint8`` E8M0 exponents (real scale = ``2 ** (byte - 127)``), but the FP8 compute
        # path expects float32 multiplicative scales — so decode them at conversion time.
        if not (self.pre_quantized and self.quantization_config.dequantize):
            if self.pre_quantized and self._is_mxfp8():
                return self._update_weight_conversions_mxfp8(weight_conversions)
            return weight_conversions + self.get_weight_conversions()

        from ..core_model_loading import WeightConverter, WeightRenaming
        from ..integrations.finegrained_fp8 import Fp8Dequantize

        # Some upstream FP8 checkpoints (e.g. DeepSeek-V4-Flash) ship per-block scales
        # under a ``.scale`` suffix instead of HF's canonical ``.weight_scale_inv``.
        # Prepending the rename here (instead of in each model's conversion_mapping)
        # keeps the model-side mapping clean — the rename only kicks in when FP8 dequant
        # is actually active, so a non-FP8 save / load round-trip doesn't see a stray
        # rule that ``test_reverse_loading_mapping`` can't match.
        scale_rename = WeightRenaming(source_patterns=r"^(.+)\.scale$", target_patterns=r"\1.weight_scale_inv")
        weight_conversions = [scale_rename] + list(weight_conversions)

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
                new_ops = [Fp8Dequantize(self)] + list(conv.operations)
                conv = WeightConverter(
                    source_patterns=new_sources,
                    target_patterns=conv._original_target_patterns,
                    operations=new_ops,
                )
            updated.append(conv)
        # Generic fallback for plain ``nn.Linear`` weights with no model-specific converter.
        updated.extend(self.get_weight_conversions())
        return updated
